"""CLI tests for ``scripts.run_stochastic_galerkin_compare`` (AC8).

Covers YAML loading, override merging, typed-config dispatch, the live
micro-run exit code, and the record-baseline / self-diff round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.run_stochastic_galerkin_compare import (
    STABLE_BASELINE_METRICS,
    _stable_metrics,
    apply_overrides,
    build_config,
    build_parser,
    load_scenario_dict,
    main,
)
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    SCENARIO_NAME,
    StochasticGalerkinCompareConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YAML = REPO_ROOT / "config" / "scenarios" / "stochastic_galerkin_compare_ci.yaml"

MICRO_ARGS = [
    "--grid-n",
    "12",
    "--n-epochs",
    "1",
    "--log-level",
    "WARNING",
]


class TestLoading:
    def test_load_scenario_dict_from_ci_yaml(self):
        data = load_scenario_dict(CI_YAML)
        assert data["name"] == SCENARIO_NAME

    def test_load_scenario_dict_bare_mapping(self, tmp_path):
        path = tmp_path / "bare.yaml"
        path.write_text(yaml.safe_dump({"name": SCENARIO_NAME, "grid_n": 8}))
        assert load_scenario_dict(path)["grid_n"] == 8

    def test_load_scenario_dict_missing_raises(self, tmp_path):
        path = tmp_path / "other.yaml"
        path.write_text(yaml.safe_dump({"scenarios": [{"name": "transfer"}]}))
        with pytest.raises(ValueError, match=SCENARIO_NAME):
            load_scenario_dict(path)

    def test_apply_overrides_only_non_none(self):
        args = build_parser().parse_args(["--grid-n", "24"])
        merged = apply_overrides({"name": SCENARIO_NAME, "n_epochs": 7}, args)
        assert merged["grid_n"] == 24
        assert merged["n_epochs"] == 7  # untouched

    def test_build_config_returns_typed_config(self):
        args = build_parser().parse_args([])
        config = build_config(CI_YAML, args)
        assert type(config).__name__ == StochasticGalerkinCompareConfig.__name__


class TestStableMetrics:
    def test_filters_to_declared_names(self):
        metrics = {
            "stochastic_density_mse": 1e-8,
            "deterministic_density_mse": 1e-2,
            "deterministic_density_mse_median": 1e-2,
            "stochastic_wall_clock_s": 0.1,
            "stochastic_vs_deterministic_mse_ratio": 1e-6,
            "deterministic_n_params": 100.0,
        }
        stable = _stable_metrics(metrics)
        assert set(stable) == set(STABLE_BASELINE_METRICS)


class TestMainJourneys:
    def test_micro_run_exits_zero(self, tmp_path):
        exit_code = main([*MICRO_ARGS, "--output-dir", str(tmp_path)])
        assert exit_code == 0
        assert (tmp_path / "stochastic_galerkin_compare.csv").exists()

    def test_record_baseline_then_self_diff(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        record_exit = main(
            [
                *MICRO_ARGS,
                "--output-dir",
                str(tmp_path),
                "--record-baseline",
                str(baseline_path),
            ]
        )
        assert record_exit == 0
        document = json.loads(baseline_path.read_text())
        recorded = {entry["metric_name"] for entry in document["entries"]}
        assert recorded == set(STABLE_BASELINE_METRICS)
        diff_exit = main(
            [*MICRO_ARGS, "--output-dir", str(tmp_path), "--baseline", str(baseline_path)]
        )
        assert diff_exit == 0
