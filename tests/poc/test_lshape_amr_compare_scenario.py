"""Tests for LShapeAMRCompareScenario (real CPU micro-run + config dispatch).

The scenario has no arm gating (dorfler + mcts are always available on CPU).
The load-bearing assertions are that both comparison ratios are recorded and
finite and that the CSV/PNG artifacts are registered — not a specific
wall-clock-gated pass/fail (which would be flaky).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.poc.config import ScenarioStatus, load_config_from_dict
from src.poc.scenarios.lshape_amr_compare import LShapeAMRCompareScenario
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME,
    LShapeAMRCompareConfig,
)

pytest.importorskip("scipy", reason="scipy required for the masked FD solve")


def _config(tmp_path, **overrides: object) -> LShapeAMRCompareConfig:  # type: ignore[no-untyped-def]
    params: dict[str, object] = {
        "name": SCENARIO_NAME,
        "device": "cpu",
        "initial_side": 4,
        "max_dof": 120,
        "max_steps": 6,
        "n_candidate_elements": 4,
        "n_simulations": 4,
        "n_seeds": 2,
        "add_noise": False,
        "output_dir": str(tmp_path),
    }
    params.update(overrides)
    return LShapeAMRCompareConfig(**params)  # type: ignore[arg-type]


class TestConfigDispatch:
    def test_load_config_from_dict_returns_config(self) -> None:
        cfg = load_config_from_dict({"name": SCENARIO_NAME, "device": "cpu"})
        assert isinstance(cfg, LShapeAMRCompareConfig)
        assert cfg.name == SCENARIO_NAME


class TestMicroRun:
    def test_records_ratio_metrics_and_artifacts(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        scenario = LShapeAMRCompareScenario(_config(tmp_path))
        result = scenario.run()

        # Never an ERROR (a genuine crash); PASSED or FAILED is acceptable.
        assert result.status in {ScenarioStatus.PASSED, ScenarioStatus.FAILED}

        # Both honest comparison ratios must be present and finite.
        assert "l2_error_ratio_at_matched_dof" in result.metrics
        assert "error_per_dof_ratio_mcts_over_dorfler" in result.metrics
        assert np.isfinite(result.metrics["l2_error_ratio_at_matched_dof"])
        assert np.isfinite(result.metrics["error_per_dof_ratio_mcts_over_dorfler"])

        # The primary threshold must have been evaluated.
        assert "l2_error_ratio_at_matched_dof" in result.threshold_results

        # Multi-seed aggregation metrics are recorded on the run.
        for key in ("mcts_win_fraction", "l2_ratio_seed_std", "n_seeds"):
            assert key in result.metrics
            assert np.isfinite(result.metrics[key])
        assert result.metrics["n_seeds"] == pytest.approx(2.0)
        assert 0.0 <= result.metrics["mcts_win_fraction"] <= 1.0

        # CSV + PNG artifacts registered and written.
        assert "csv" in result.artifacts
        assert "png" in result.artifacts
        from pathlib import Path

        assert Path(result.artifacts["csv"]).exists()
        assert Path(result.artifacts["png"]).exists()

    def test_per_arm_final_metrics_present(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        scenario = LShapeAMRCompareScenario(_config(tmp_path))
        result = scenario.run()
        for key in (
            "dorfler_final_dof",
            "mcts_final_dof",
            "dorfler_final_l2",
            "mcts_final_l2",
            "matched_dof",
        ):
            assert key in result.metrics
            assert np.isfinite(result.metrics[key])
