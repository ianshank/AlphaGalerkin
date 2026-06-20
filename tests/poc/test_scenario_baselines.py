"""Tests for the PoC-scenario baseline harness (WS2).

Covers schema validation, JSON load/save round-trip, schema migration
(unversioned -> v1, including a Hypothesis property test), the
``from_observed`` builder, direction-aware regression classification, the
tolerance boundary, and ``observed_from_result_dicts``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.poc.baselines import (
    POC_BASELINE_DOCUMENT_SCHEMA_VERSION,
    ScenarioBaselineDocument,
    ScenarioBaselineEntry,
    ScenarioBaselineRegistry,
    migrate_baseline_document,
    observed_from_result_dicts,
    regression_pct,
)

# --------------------------------------------------------------------- schema


def test_entry_key_is_scenario_dot_metric() -> None:
    entry = ScenarioBaselineEntry(
        scenario_name="scaling_law",
        metric_name="residual_fit_r2",
        value=0.9,
        direction="higher_better",
        tolerance_pct=10.0,
    )
    assert entry.key == "scaling_law.residual_fit_r2"


def test_entry_rejects_negative_tolerance() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        ScenarioBaselineEntry(
            scenario_name="s",
            metric_name="m",
            value=1.0,
            direction="lower_better",
            tolerance_pct=-1.0,
        )


def test_document_forward_compat_ignores_unknown_fields() -> None:
    doc = ScenarioBaselineDocument.model_validate(
        {"description": "x", "entries": [], "future_field": 123}
    )
    assert doc.description == "x"
    assert not hasattr(doc, "future_field")


# ------------------------------------------------------------------ migration


def test_migrate_unversioned_adds_v1() -> None:
    migrated = migrate_baseline_document({"entries": []})
    assert migrated["schema_version"] == POC_BASELINE_DOCUMENT_SCHEMA_VERSION


def test_migrate_does_not_mutate_caller() -> None:
    raw = {"entries": []}
    migrate_baseline_document(raw)
    assert "schema_version" not in raw


def test_migrate_future_schema_raises() -> None:
    with pytest.raises(ValueError, match="newer than this binary"):
        migrate_baseline_document({"schema_version": 999, "entries": []})


@given(
    version=st.one_of(
        st.none(), st.integers(min_value=1, max_value=POC_BASELINE_DOCUMENT_SCHEMA_VERSION)
    )
)
def test_migrate_idempotent_for_known_versions(version: int | None) -> None:
    raw: dict[str, object] = {"entries": []}
    if version is not None:
        raw["schema_version"] = version
    migrated = migrate_baseline_document(raw)
    assert migrated["schema_version"] == (version or POC_BASELINE_DOCUMENT_SCHEMA_VERSION)
    # Loadable into the model.
    ScenarioBaselineDocument.model_validate(migrated)


# ----------------------------------------------------------------- regression_pct


def test_regression_pct_lower_better_rise_is_positive() -> None:
    # residual rose 0.5 -> 0.6 => 20% worse for a lower-better metric.
    assert regression_pct(baseline=0.5, observed=0.6, higher_is_better=False) == pytest.approx(20.0)


def test_regression_pct_higher_better_drop_is_positive() -> None:
    # solved_fraction dropped 1.0 -> 0.8 => 20% worse for a higher-better metric.
    assert regression_pct(baseline=1.0, observed=0.8, higher_is_better=True) == pytest.approx(20.0)


def test_regression_pct_zero_baseline_uses_floor() -> None:
    # Must not divide by zero.
    val = regression_pct(baseline=0.0, observed=1.0, higher_is_better=False)
    assert val > 0 and val != float("inf")


# ------------------------------------------------------------ from_observed/compare


def _observed() -> dict[str, dict[str, float]]:
    return {
        "scaling_law": {"residual_fit_r2": 0.9, "residual_scaling_exponent": -0.3},
        "research_loop": {"solved_fraction": 1.0},
    }


def test_from_observed_records_directions() -> None:
    reg = ScenarioBaselineRegistry.from_observed(
        _observed(),
        higher_better_metrics={"residual_fit_r2", "solved_fraction"},
        tolerance_pct=10.0,
    )
    assert reg.get("scaling_law", "residual_fit_r2").direction == "higher_better"
    assert reg.get("scaling_law", "residual_scaling_exponent").direction == "lower_better"
    assert reg.get("research_loop", "solved_fraction").direction == "higher_better"


def test_compare_self_is_clean() -> None:
    reg = ScenarioBaselineRegistry.from_observed(_observed(), tolerance_pct=10.0)
    report = reg.compare(_observed())
    assert not report.has_regressions
    assert all(d.status == "ok" for d in report.diffs)


def test_compare_detects_lower_better_regression() -> None:
    reg = ScenarioBaselineRegistry.from_observed({"s": {"residual": 0.5}}, tolerance_pct=10.0)
    # residual rose 0.5 -> 0.7 (40% worse) => regression.
    report = reg.compare({"s": {"residual": 0.7}})
    assert report.has_regressions
    assert report.regressions[0].metric_name == "residual"


def test_compare_detects_higher_better_regression() -> None:
    reg = ScenarioBaselineRegistry.from_observed(
        {"s": {"solved_fraction": 1.0}},
        higher_better_metrics={"solved_fraction"},
        tolerance_pct=10.0,
    )
    report = reg.compare({"s": {"solved_fraction": 0.5}})
    assert report.has_regressions


def test_compare_improvement_classified() -> None:
    reg = ScenarioBaselineRegistry.from_observed({"s": {"residual": 0.5}}, tolerance_pct=10.0)
    report = reg.compare({"s": {"residual": 0.1}})
    assert not report.has_regressions
    assert len(report.improvements) == 1


def test_compare_within_tolerance_is_ok() -> None:
    reg = ScenarioBaselineRegistry.from_observed({"s": {"residual": 1.0}}, tolerance_pct=10.0)
    report = reg.compare({"s": {"residual": 1.05}})  # 5% < 10% tolerance
    assert not report.has_regressions
    assert report.diffs[0].status == "ok"


def test_compare_missing_observed_metric_is_not_regression() -> None:
    reg = ScenarioBaselineRegistry.from_observed({"s": {"residual": 1.0}}, tolerance_pct=10.0)
    report = reg.compare({"s": {}})
    assert not report.has_regressions
    assert report.missing_in_observed == ["s.residual"]


# ------------------------------------------------------------------- round-trip


def test_save_load_round_trip(tmp_path: Path) -> None:
    reg = ScenarioBaselineRegistry.from_observed(
        _observed(),
        higher_better_metrics={"residual_fit_r2", "solved_fraction"},
        tolerance_pct=12.5,
        hardware_tag="RTX 5060 Ti",
        git_sha="deadbeef",
        llm_backend="vllm",
    )
    path = tmp_path / "base.json"
    reg.save(path)
    loaded = ScenarioBaselineRegistry.load(path)
    assert loaded.document.hardware_tag == "RTX 5060 Ti"
    assert loaded.document.llm_backend == "vllm"
    # Self-compare clean after a round-trip.
    assert not loaded.compare(_observed()).has_regressions


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ScenarioBaselineRegistry.load(tmp_path / "nope.json")


def test_load_non_object_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        ScenarioBaselineRegistry.load(path)


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        ScenarioBaselineRegistry.load(path)


# ------------------------------------------------------ observed_from_result_dicts


def test_compare_missing_entire_scenario_is_not_regression() -> None:
    reg = ScenarioBaselineRegistry.from_observed({"s": {"residual": 1.0}}, tolerance_pct=10.0)
    report = reg.compare({"other_scenario": {"residual": 1.0}})
    assert not report.has_regressions
    assert report.missing_in_observed == ["s.residual"]


def test_report_and_diff_to_dict_round_trip() -> None:
    reg = ScenarioBaselineRegistry.from_observed({"s": {"residual": 0.5}}, tolerance_pct=10.0)
    report = reg.compare({"s": {"residual": 0.7}}, baseline_path="b.json")
    payload = report.to_dict()
    assert payload["baseline_path"] == "b.json"
    assert payload["n_regressions"] == 1
    assert payload["has_regressions"] is True
    assert payload["diffs"][0]["metric_name"] == "residual"


def test_observed_from_result_dicts_extracts_metrics() -> None:
    result_dicts = [
        {"scenario_name": "scaling_law", "metrics": {"residual_fit_r2": 0.9, "junk": "nan-str"}},
        {"scenario_name": "research_loop", "metrics": {"solved_fraction": 1.0}},
        {"metrics": {"ignored": 1.0}},  # no scenario_name -> skipped
    ]
    observed = observed_from_result_dicts(result_dicts)
    assert observed["scaling_law"]["residual_fit_r2"] == 0.9
    assert "junk" not in observed["scaling_law"]  # non-numeric dropped
    assert observed["research_loop"]["solved_fraction"] == 1.0
    assert "" not in observed


def test_observed_from_result_dicts_skips_non_dict_metrics() -> None:
    result_dicts = [
        {"scenario_name": "a", "metrics": ["not", "a", "dict"]},  # list -> skipped
        {"scenario_name": "b", "metrics": "oops"},  # str -> skipped
        {"scenario_name": "c"},  # metrics absent -> skipped
        {"scenario_name": "d", "metrics": {"x": 1.0}},  # valid
    ]
    observed = observed_from_result_dicts(result_dicts)
    assert "a" not in observed
    assert "b" not in observed
    assert "c" not in observed
    assert observed["d"] == {"x": 1.0}
