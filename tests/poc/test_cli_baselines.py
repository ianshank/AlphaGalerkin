"""Tests for the poc-CLI baseline subcommands (WS2 Part C)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.poc.cli import (
    _load_run_result_dicts,
    _resolve_higher_better,
    _split_csv,
    cmd_diff,
    cmd_record_baseline,
)


def _write_run(output_dir: Path, run_id: str, metrics: dict[str, float]) -> None:
    run_dir = output_dir / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scaling_law_abc.json").write_text(
        json.dumps({"scenario_name": "scaling_law", "metrics": metrics})
    )


def test_split_csv() -> None:
    assert _split_csv("a, b ,,c") == ["a", "b", "c"]
    assert _split_csv("") == []
    assert _split_csv(None) == []


def test_resolve_higher_better_uses_suffixes_and_extras() -> None:
    observed = {"s": {"random_residual_fit_r2": 0.9, "residual_median_b4": 0.5, "custom": 1.0}}
    higher = _resolve_higher_better(observed, extra_names=["custom"], extra_suffixes=[])
    assert "random_residual_fit_r2" in higher  # suffix _fit_r2
    assert "custom" in higher  # explicit
    assert "residual_median_b4" not in higher


def test_load_run_result_dicts(tmp_path: Path) -> None:
    _write_run(tmp_path, "run1", {"residual_fit_r2": 0.9})
    dicts = _load_run_result_dicts(str(tmp_path), "run1")
    assert dicts[0]["scenario_name"] == "scaling_law"


def test_load_run_result_dicts_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_run_result_dicts(str(tmp_path), "absent")


def test_load_run_result_dicts_research_layout(tmp_path: Path) -> None:
    """Research-loop layout: {output_dir}/{run_id}/result.json (no results/ segment)."""
    run_dir = tmp_path / "rl1"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps({"name": "research_loop", "metrics": {"solved_fraction": 1.0}})
    )
    dicts = _load_run_result_dicts(str(tmp_path), "rl1")
    assert dicts[0]["name"] == "research_loop"


def test_load_run_result_dicts_corrupt_json_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "bad.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        _load_run_result_dicts(str(tmp_path), "run1")


def test_record_baseline_from_research_layout(tmp_path: Path) -> None:
    """record-baseline works end-to-end on a persisted research-loop run."""
    run_dir = tmp_path / "rl1"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps({"name": "research_loop", "metrics": {"solved_fraction": 0.66}})
    )
    out = tmp_path / "base.json"
    rc = cmd_record_baseline(
        argparse.Namespace(
            run_id="rl1",
            out=str(out),
            output_dir=str(tmp_path),
            tolerance_pct=10.0,
            higher_better="solved_fraction",
            higher_better_suffix="",
            description="",
            hardware_tag="",
            git_sha="",
            llm_backend="",
        )
    )
    assert rc == 0
    assert out.exists()


def test_record_then_diff_clean(tmp_path: Path) -> None:
    _write_run(tmp_path, "run1", {"residual_fit_r2": 0.9, "residual_median_b4": 0.5})
    out = tmp_path / "base.json"
    rc = cmd_record_baseline(
        argparse.Namespace(
            run_id="run1",
            out=str(out),
            output_dir=str(tmp_path),
            tolerance_pct=10.0,
            higher_better="",
            higher_better_suffix="",
            description="",
            hardware_tag="",
            git_sha="",
            llm_backend="",
        )
    )
    assert rc == 0
    assert out.exists()
    # Self-diff is clean (exit 0).
    rc_diff = cmd_diff(
        argparse.Namespace(baseline=str(out), run_id="run1", output_dir=str(tmp_path))
    )
    assert rc_diff == 0


def test_diff_detects_regression(tmp_path: Path) -> None:
    _write_run(tmp_path, "run1", {"residual_median_b4": 0.5})
    out = tmp_path / "base.json"
    cmd_record_baseline(
        argparse.Namespace(
            run_id="run1",
            out=str(out),
            output_dir=str(tmp_path),
            tolerance_pct=10.0,
            higher_better="",
            higher_better_suffix="",
            description="",
            hardware_tag="",
            git_sha="",
            llm_backend="",
        )
    )
    # Second run regressed: residual rose 0.5 -> 0.8 (lower-better) => exit 1.
    _write_run(tmp_path, "run2", {"residual_median_b4": 0.8})
    rc_diff = cmd_diff(
        argparse.Namespace(baseline=str(out), run_id="run2", output_dir=str(tmp_path))
    )
    assert rc_diff == 1


def test_record_no_metrics_returns_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "empty"
    run_dir.mkdir(parents=True)
    (run_dir / "x.json").write_text(json.dumps({"scenario_name": "s", "metrics": {}}))
    rc = cmd_record_baseline(
        argparse.Namespace(
            run_id="empty",
            out=str(tmp_path / "b.json"),
            output_dir=str(tmp_path),
            tolerance_pct=10.0,
            higher_better="",
            higher_better_suffix="",
            description="",
            hardware_tag="",
            git_sha="",
            llm_backend="",
        )
    )
    assert rc == 1
