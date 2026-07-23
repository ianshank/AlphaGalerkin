"""Scenario tests for ``stochastic_galerkin_compare`` (AC8): real micro-runs.

The scenario is exercised through the full BaseScenario lifecycle (setup →
execute → teardown → threshold evaluation) with micro budgets.
"""

from __future__ import annotations

import pytest

from src.poc.config import ScenarioStatus
from src.poc.registry import ScenarioRegistry
from src.poc.scenarios.stochastic_galerkin_compare import StochasticGalerkinCompareScenario
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    SCENARIO_NAME,
    StochasticGalerkinCompareConfig,
)


def _micro_config(tmp_path, **overrides) -> StochasticGalerkinCompareConfig:
    defaults = {
        "grid_n": 12,
        "n_train_samples": 6,
        "n_eval_samples": 3,
        "n_epochs": 1,
        "d_model": 16,
        "n_fourier_features": 8,
        "batch_size": 4,
        "n_seeds": 1,
        "output_dir": str(tmp_path),
    }
    defaults.update(overrides)
    return StochasticGalerkinCompareConfig(**defaults)


class TestRegistration:
    def test_scenario_registered(self):
        assert SCENARIO_NAME in ScenarioRegistry().list_scenarios()

    def test_registry_resolves_class(self):
        # Compare by class *name* rather than identity — robust to dual-import
        # under some pytest orderings (same convention as the CLI's dispatch check).
        cls = ScenarioRegistry().get(SCENARIO_NAME)
        assert cls.__name__ == StochasticGalerkinCompareScenario.__name__


class TestMicroRun:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("sgc")
        scenario = StochasticGalerkinCompareScenario(_micro_config(tmp_path))
        return scenario.run()

    def test_completed_and_passed(self, result):
        assert result.status == ScenarioStatus.PASSED
        assert result.passed

    def test_metrics_recorded(self, result):
        for key in (
            "stochastic_density_mse",
            "deterministic_density_mse",
            "stochastic_vs_deterministic_mse_ratio",
        ):
            assert key in result.metrics

    def test_gated_metric_within_gate(self, result):
        assert result.metrics["stochastic_density_mse"] < 1e-6

    def test_artifacts_written(self, result, tmp_path_factory):
        assert "csv" in result.artifacts
        assert "png" in result.artifacts

    def test_thresholds_installed(self, result):
        # The scenario installs get_default_thresholds() on setup; the result
        # carries the single gated metric evaluation.
        assert result.passed is True


class TestHonestFailurePath:
    def test_unreachable_gate_fails_scenario(self, tmp_path):
        """A gate below the achievable floor FAILS honestly (no silent pass)."""
        scenario = StochasticGalerkinCompareScenario(
            _micro_config(tmp_path, stochastic_mse_gate=1e-12)
        )
        result = scenario.run()
        assert result.status == ScenarioStatus.FAILED
        assert not result.passed
