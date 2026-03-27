"""Coverage tests for the hyperparameter tuner.

Targets uncovered lines in src/poc/tuning/tuner.py:
    - HyperparameterTuner initialization
    - tune() loop with trials
    - _run_trial
    - _is_better (minimize/maximize)
    - save_results
    - create_tuner factory
    - TrialResult / TuningResult dataclasses
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.poc.config import (
    BaseScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.registry import BaseScenario, ScenarioRegistry, scenario
from src.poc.tuning.config import SearchSpace, TuningConfig
from src.poc.tuning.tuner import (
    HyperparameterTuner,
    TrialResult,
    TuningResult,
    create_tuner,
)


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    ScenarioRegistry().clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_scenario_cls():
    """Return a simple scenario class for tuning."""

    @scenario("tuner_test")
    class TunerTestScenario(BaseScenario):
        def execute(self) -> ScenarioResult:
            # Use d_model from config as a proxy metric
            d = getattr(self.config, "d_model", 64)
            self.record_metric("mse", 1.0 / d)
            return self._create_result(ScenarioStatus.PASSED)

    return TunerTestScenario


def _simple_search_space() -> dict[str, SearchSpace]:
    return {
        "d_model": SearchSpace(type="int", low=16, high=128),
    }


def _simple_tuning_config(n_trials: int = 3) -> TuningConfig:
    return TuningConfig(
        n_trials=n_trials,
        sampler="random",
        search_space=_simple_search_space(),
        objective_metric="mse",
        direction="minimize",
        seed=42,
        study_name="test_study",
    )


# ---------------------------------------------------------------------------
# Tests: TrialResult / TuningResult
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_trial_result_creation(self) -> None:
        tr = TrialResult(
            trial_id=0,
            params={"lr": 0.01},
            objective_value=0.5,
            metrics={"mse": 0.5},
            duration_seconds=1.0,
            status="completed",
        )
        assert tr.trial_id == 0
        assert tr.timestamp  # auto-generated

    def test_tuning_result_creation(self) -> None:
        tr = TuningResult(
            best_params={"lr": 0.01},
            best_value=0.1,
            n_trials_completed=5,
            n_trials_pruned=0,
            n_trials_failed=1,
            trials=[],
            search_space={},
            duration_seconds=10.0,
            study_name="test",
        )
        assert tr.best_value == 0.1
        assert tr.n_trials_completed == 5


# ---------------------------------------------------------------------------
# Tests: HyperparameterTuner init
# ---------------------------------------------------------------------------


class TestTunerInit:
    def test_basic_init(self) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config()
        tuner = HyperparameterTuner(config, cls)

        assert tuner.config == config
        assert tuner.scenario_cls is cls
        assert tuner.trials == []
        assert tuner._best_value is None

    def test_init_with_base_config(self) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config()
        base = {"name": "tuner_test", "description": "base"}
        tuner = HyperparameterTuner(config, cls, base_scenario_config=base)
        assert tuner.base_scenario_config == base


# ---------------------------------------------------------------------------
# Tests: _is_better
# ---------------------------------------------------------------------------


class TestIsBetter:
    def test_first_value_always_better(self) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config()
        tuner = HyperparameterTuner(config, cls)
        assert tuner._is_better(999.0) is True

    def test_minimize_lower_is_better(self) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config()
        tuner = HyperparameterTuner(config, cls)
        tuner._best_value = 0.5
        assert tuner._is_better(0.3) is True
        assert tuner._is_better(0.7) is False

    def test_maximize_higher_is_better(self) -> None:
        cls = _dummy_scenario_cls()
        config = TuningConfig(
            n_trials=2,
            sampler="random",
            search_space=_simple_search_space(),
            direction="maximize",
            seed=42,
        )
        tuner = HyperparameterTuner(config, cls)
        tuner._best_value = 0.5
        assert tuner._is_better(0.7) is True
        assert tuner._is_better(0.3) is False


# ---------------------------------------------------------------------------
# Tests: tune()
# ---------------------------------------------------------------------------


class TestTune:
    def test_tune_completes(self) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config(n_trials=2)
        tuner = HyperparameterTuner(
            config,
            cls,
            base_scenario_config={"name": "tuner_test", "description": "t"},
        )

        result = tuner.tune()

        assert isinstance(result, TuningResult)
        assert result.study_name == "test_study"
        assert result.duration_seconds > 0
        assert len(result.trials) <= 2

    def test_tune_tracks_best(self) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config(n_trials=3)
        tuner = HyperparameterTuner(
            config,
            cls,
            base_scenario_config={"name": "tuner_test", "description": "t"},
        )

        result = tuner.tune()
        assert result.best_value != float("inf")

    def test_tune_with_failing_trial(self) -> None:
        """If a trial raises, it should be counted as failed."""
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config(n_trials=2)
        tuner = HyperparameterTuner(config, cls)

        # Make _run_trial raise on first call
        original = tuner._run_trial

        call_count = 0

        def failing_trial(trial_id: int) -> TrialResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return original(trial_id)

        tuner._run_trial = failing_trial

        result = tuner.tune()
        assert result.n_trials_failed >= 1


# ---------------------------------------------------------------------------
# Tests: save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_save_results_creates_file(self, tmp_path: Path) -> None:
        cls = _dummy_scenario_cls()
        config = _simple_tuning_config(n_trials=1)
        tuner = HyperparameterTuner(
            config,
            cls,
            base_scenario_config={"name": "tuner_test", "description": "t"},
        )
        tuner.tune()

        out = tmp_path / "results.json"
        tuner.save_results(out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "best_params" in data
        assert "trials" in data
        assert "config" in data


# ---------------------------------------------------------------------------
# Tests: create_tuner factory
# ---------------------------------------------------------------------------


class TestCreateTuner:
    def test_factory(self) -> None:
        cls = _dummy_scenario_cls()
        tuner = create_tuner(
            scenario_cls=cls,
            search_space={
                "d_model": {"type": "int", "low": 16, "high": 64},
            },
            n_trials=5,
            seed=123,
        )
        assert isinstance(tuner, HyperparameterTuner)
        assert tuner.config.n_trials == 5
        assert tuner.config.seed == 123
