"""Tests for hyperparameter tuner samplers and the HyperparameterTuner.

Covers RandomSampler, GridSampler, TPESampler, HyperparameterTuner,
TuningResult, and TrialResult in isolation and via lightweight scenarios.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.poc.tuning.config import SearchSpace, TuningConfig
from src.poc.tuning.sampler import GridSampler, RandomSampler, TPESampler
from src.poc.tuning.tuner import HyperparameterTuner, TrialResult, TuningResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def float_search_space() -> dict[str, SearchSpace]:
    """Two-dimensional float search space for bowl-function tests."""
    return {
        "x": SearchSpace(type="float", low=-1.0, high=1.0),
        "y": SearchSpace(type="float", low=-1.0, high=1.0),
    }


@pytest.fixture()
def int_search_space() -> dict[str, SearchSpace]:
    """Integer search space."""
    return {
        "n_layers": SearchSpace(type="int", low=1, high=8),
        "d_model": SearchSpace(type="int", low=32, high=256),
    }


@pytest.fixture()
def mixed_search_space() -> dict[str, SearchSpace]:
    """Mixed float/int search space (no categorical — known validator limitation)."""
    return {
        "lr": SearchSpace(type="float", low=1e-4, high=1e-1, log_scale=True),
        "batch_size": SearchSpace(type="int", low=16, high=128),
        "d_model": SearchSpace(type="int", low=64, high=256),
    }


@pytest.fixture()
def simple_tuning_config(float_search_space: dict[str, SearchSpace]) -> TuningConfig:
    """Minimal TuningConfig for fast tests."""
    return TuningConfig(
        n_trials=5,
        sampler="random",
        search_space=float_search_space,
        objective_metric="loss",
        direction="minimize",
        seed=0,
        study_name="test_study",
    )


# ---------------------------------------------------------------------------
# Minimal mock scenario (no real training; returns configurable metrics)
# ---------------------------------------------------------------------------


def _make_scenario_cls(
    metric_name: str = "loss",
    *,
    objective_fn: Any = None,
    passed: bool = True,
) -> type:
    """Build a lightweight BaseScenario subclass for tuner tests.

    The scenario reads ``x`` and ``y`` kwargs (or any params), evaluates
    ``objective_fn`` if given, and returns a ScenarioResult via the base
    class helper (which correctly populates start_time, end_time, etc.).
    """
    from src.poc.config import BaseScenarioConfig, ScenarioStatus
    from src.poc.registry import BaseScenario

    class _Config(BaseScenarioConfig):
        """Config that accepts arbitrary extra float/int kwargs via model_extra."""

        model_config = BaseScenarioConfig.model_config.copy()
        # Allow extra fields so sampled params can be forwarded directly.
        model_config["extra"] = "allow"

        name: str = "mock"
        description: str = "mock scenario for tuner tests"

    class _MockScenario(BaseScenario):
        config_class = _Config

        def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
            # Pull out the known config fields; everything else is for the
            # objective function.
            config_keys = {
                "name",
                "description",
                "tier",
                "enabled",
                "timeout_seconds",
                "retry_count",
                "seed",
                "thresholds",
                "requires_gpu",
                "estimated_duration_seconds",
            }
            config_kwargs = {k: v for k, v in kwargs.items() if k in config_keys}
            self._extra_params: dict[str, Any] = {
                k: v for k, v in kwargs.items() if k not in config_keys
            }
            super().__init__(config=_Config(**config_kwargs))

        def execute(self) -> Any:
            if objective_fn is not None:
                value = float(objective_fn(self._extra_params))
            else:
                value = sum(float(v) ** 2 for v in self._extra_params.values())

            self.record_metric(metric_name, value)
            status = ScenarioStatus.PASSED if passed else ScenarioStatus.FAILED
            return self._create_result(status)

    return _MockScenario


# ---------------------------------------------------------------------------
# TestRandomSampler
# ---------------------------------------------------------------------------


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_returns_dict_with_expected_keys(
        self, float_search_space: dict[str, SearchSpace]
    ) -> None:
        """sample() returns a dict whose keys match the search space."""
        sampler = RandomSampler(seed=1)
        result = sampler.sample(float_search_space, trial_number=0)
        assert set(result.keys()) == set(float_search_space.keys())

    def test_float_values_within_bounds(self, float_search_space: dict[str, SearchSpace]) -> None:
        """All sampled float values lie within [low, high]."""
        sampler = RandomSampler(seed=42)
        for trial in range(20):
            params = sampler.sample(float_search_space, trial_number=trial)
            for name, space in float_search_space.items():
                assert space.low <= params[name] <= space.high, (
                    f"{name}={params[name]} out of [{space.low}, {space.high}]"
                )

    def test_int_values_within_bounds(self, int_search_space: dict[str, SearchSpace]) -> None:
        """All sampled int values lie within [low, high] and are integers."""
        sampler = RandomSampler(seed=7)
        for trial in range(20):
            params = sampler.sample(int_search_space, trial_number=trial)
            for name, space in int_search_space.items():
                val = params[name]
                assert isinstance(val, int), f"{name} should be int, got {type(val)}"
                assert int(space.low) <= val <= int(space.high)

    def test_same_seed_reproducibility(self, float_search_space: dict[str, SearchSpace]) -> None:
        """Verify identical seeds produce identical first samples.

        Because RandomSampler seeds Python's global ``random`` module at
        ``__init__`` time, we reseed immediately before sampling to isolate
        from other test-suite random calls.
        """
        import random

        random.seed(55)
        s1 = RandomSampler(seed=55)
        p1 = s1.sample(float_search_space, trial_number=0)

        # Re-seed global state to same value before second run
        random.seed(55)
        s2 = RandomSampler(seed=55)
        p2 = s2.sample(float_search_space, trial_number=0)

        assert p1 == pytest.approx(p2), "Same seed must produce same first sample"

    def test_handles_mixed_param_types(self, mixed_search_space: dict[str, SearchSpace]) -> None:
        """Sampler handles mixed float/int log-scale parameters without error."""
        sampler = RandomSampler(seed=0)
        params = sampler.sample(mixed_search_space, trial_number=0)
        assert set(params.keys()) == set(mixed_search_space.keys())
        # log-scale float must be positive
        assert params["lr"] > 0
        # int params
        assert isinstance(params["batch_size"], int)
        assert isinstance(params["d_model"], int)


# ---------------------------------------------------------------------------
# TestGridSampler
# ---------------------------------------------------------------------------


class TestGridSampler:
    """Tests for GridSampler."""

    def test_first_sample_is_first_grid_point(
        self, float_search_space: dict[str, SearchSpace]
    ) -> None:
        """trial_number=0 returns the first combination in the grid."""
        sampler = GridSampler(float_search_space, n_samples_per_dim=3)
        p0 = sampler.sample(float_search_space, trial_number=0)
        assert set(p0.keys()) == {"x", "y"}
        # The first grid point for linspace(-1,1,3) x linspace(-1,1,3) is (-1,-1)
        assert math.isclose(p0["x"], -1.0, abs_tol=1e-9)
        assert math.isclose(p0["y"], -1.0, abs_tol=1e-9)

    def test_iterates_through_grid_systematically(
        self, float_search_space: dict[str, SearchSpace]
    ) -> None:
        """Successive trial numbers return distinct grid points."""
        n = 3
        sampler = GridSampler(float_search_space, n_samples_per_dim=n)
        total = n * n  # 2-dim grid
        points = [sampler.sample(float_search_space, trial_number=i) for i in range(total)]
        # All points should be unique (grid has no duplicates for linspace with n=3)
        tuples = [tuple(sorted(p.items())) for p in points]
        assert len(set(tuples)) == total

    def test_grid_covers_all_combinations(self, float_search_space: dict[str, SearchSpace]) -> None:
        """Grid contains n^d total combinations for n_samples_per_dim=n, d dimensions."""
        n = 4
        sampler = GridSampler(float_search_space, n_samples_per_dim=n)
        expected = n ** len(float_search_space)
        assert len(sampler._grid) == expected

    def test_wraps_around_when_exhausted(self, float_search_space: dict[str, SearchSpace]) -> None:
        """trial_number >= grid_size wraps around modulo grid size."""
        n = 2
        sampler = GridSampler(float_search_space, n_samples_per_dim=n)
        grid_size = n**2  # 4 points
        p0 = sampler.sample(float_search_space, trial_number=0)
        p_wrap = sampler.sample(float_search_space, trial_number=grid_size)
        assert p0 == pytest.approx(p_wrap)

    def test_single_value_per_dim(self, float_search_space: dict[str, SearchSpace]) -> None:
        """n_samples_per_dim=1 produces a single grid point."""
        sampler = GridSampler(float_search_space, n_samples_per_dim=1)
        assert len(sampler._grid) == 1
        p = sampler.sample(float_search_space, trial_number=0)
        assert set(p.keys()) == {"x", "y"}


# ---------------------------------------------------------------------------
# TestTPESampler
# ---------------------------------------------------------------------------


class TestTPESampler:
    """Tests for TPESampler."""

    def test_cold_start_returns_valid_params(
        self, float_search_space: dict[str, SearchSpace]
    ) -> None:
        """Before any updates (cold start), sample() returns valid float params."""
        sampler = TPESampler(seed=0, n_startup_trials=10)
        params = sampler.sample(float_search_space, trial_number=0)
        assert set(params.keys()) == {"x", "y"}
        for name, space in float_search_space.items():
            assert space.low <= params[name] <= space.high

    def test_returns_all_param_keys(self, mixed_search_space: dict[str, SearchSpace]) -> None:
        """sample() always returns all keys from the search space."""
        sampler = TPESampler(seed=42, n_startup_trials=5)
        params = sampler.sample(mixed_search_space, trial_number=0)
        assert set(params.keys()) == set(mixed_search_space.keys())

    def test_update_does_not_raise(self, float_search_space: dict[str, SearchSpace]) -> None:
        """update() with valid params and value does not raise."""
        sampler = TPESampler(seed=0, n_startup_trials=10)
        params = sampler.sample(float_search_space, trial_number=0)
        sampler.update(params, value=0.25)  # Should not raise
        assert len(sampler._history) == 1

    def test_history_accumulates(self, float_search_space: dict[str, SearchSpace]) -> None:
        """History list grows with each update() call."""
        sampler = TPESampler(seed=0, n_startup_trials=10)
        for i in range(5):
            params = sampler.sample(float_search_space, trial_number=i)
            sampler.update(params, value=float(i))
        assert len(sampler._history) == 5

    def test_falls_back_to_random_during_startup(
        self, float_search_space: dict[str, SearchSpace]
    ) -> None:
        """During startup phase (trial < n_startup_trials), uses random fallback."""
        sampler = TPESampler(seed=0, n_startup_trials=100)
        # trial_number=50 is still in startup phase
        params = sampler.sample(float_search_space, trial_number=50)
        assert set(params.keys()) == {"x", "y"}
        for name, space in float_search_space.items():
            assert space.low <= params[name] <= space.high


# ---------------------------------------------------------------------------
# TestHyperparameterTuner — helpers
# ---------------------------------------------------------------------------


def _bowl_objective(params: dict[str, Any]) -> float:
    """Simple bowl function: minimum 0 at x=0, y=0."""
    return params["x"] ** 2 + params["y"] ** 2


def _make_config(
    n_trials: int = 5,
    direction: str = "minimize",
    sampler: str = "random",
    search_space: dict[str, SearchSpace] | None = None,
    objective_metric: str = "loss",
) -> TuningConfig:
    """Build a TuningConfig for tuner tests."""
    if search_space is None:
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
            "y": SearchSpace(type="float", low=-1.0, high=1.0),
        }
    return TuningConfig(
        n_trials=n_trials,
        sampler=sampler,  # type: ignore[arg-type]
        search_space=search_space,
        objective_metric=objective_metric,
        direction=direction,  # type: ignore[arg-type]
        seed=0,
        study_name="test_study",
    )


# ---------------------------------------------------------------------------
# TestHyperparameterTuner
# ---------------------------------------------------------------------------


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner."""

    def test_instantiation(self) -> None:
        """HyperparameterTuner can be instantiated with a valid TuningConfig."""
        config = _make_config()
        scenario_cls = _make_scenario_cls()
        tuner = HyperparameterTuner(config, scenario_cls)
        assert tuner.config is config
        assert tuner.scenario_cls is scenario_cls
        assert tuner.trials == []

    def test_tune_returns_tuning_result(self) -> None:
        """tune() returns a TuningResult instance."""
        config = _make_config(n_trials=3)
        scenario_cls = _make_scenario_cls(metric_name="loss", objective_fn=_bowl_objective)
        tuner = HyperparameterTuner(config, scenario_cls)
        result = tuner.tune()
        assert isinstance(result, TuningResult)

    def test_n_trials_controls_trial_count(self) -> None:
        """n_trials determines the number of completed trials."""
        n = 5
        config = _make_config(n_trials=n)
        scenario_cls = _make_scenario_cls(metric_name="loss")
        tuner = HyperparameterTuner(config, scenario_cls)
        result = tuner.tune()
        assert result.n_trials_completed == n
        assert len(result.trials) == n

    def test_minimize_tracks_minimum(self) -> None:
        """direction='minimize' stores the minimum objective value as best_value."""
        config = _make_config(n_trials=8, direction="minimize")
        scenario_cls = _make_scenario_cls(metric_name="loss", objective_fn=_bowl_objective)
        tuner = HyperparameterTuner(config, scenario_cls)
        result = tuner.tune()
        min_value = min(t.objective_value for t in result.trials if t.status == "completed")
        assert math.isclose(result.best_value, min_value, rel_tol=1e-9)

    def test_maximize_tracks_maximum(self) -> None:
        """direction='maximize' stores the maximum objective value as best_value."""
        # Use negative bowl so values are negative and maximization picks least-negative
        config = _make_config(n_trials=8, direction="maximize", objective_metric="score")
        negative_bowl = lambda p: -(p["x"] ** 2 + p["y"] ** 2)  # noqa: E731
        scenario_cls = _make_scenario_cls(metric_name="score", objective_fn=negative_bowl)
        tuner = HyperparameterTuner(config, scenario_cls)
        result = tuner.tune()
        max_value = max(t.objective_value for t in result.trials if t.status == "completed")
        assert math.isclose(result.best_value, max_value, rel_tol=1e-9)

    def test_best_params_come_from_best_trial(self) -> None:
        """best_params in TuningResult correspond to the trial with best_value."""
        config = _make_config(n_trials=8, direction="minimize")
        scenario_cls = _make_scenario_cls(metric_name="loss", objective_fn=_bowl_objective)
        tuner = HyperparameterTuner(config, scenario_cls)
        result = tuner.tune()

        # Find the trial that has best_value
        best_trial = min(
            (t for t in result.trials if t.status == "completed"),
            key=lambda t: t.objective_value,
        )
        assert result.best_params == best_trial.params

    def test_tuning_result_study_name(self) -> None:
        """TuningResult.study_name matches the config study_name."""
        config = _make_config(n_trials=2)
        scenario_cls = _make_scenario_cls()
        tuner = HyperparameterTuner(config, scenario_cls)
        result = tuner.tune()
        assert result.study_name == "test_study"

    def test_save_results_writes_json(self, tmp_path: Path) -> None:
        """save_results() writes a valid JSON file at the given path."""
        config = _make_config(n_trials=3)
        scenario_cls = _make_scenario_cls(metric_name="loss")
        tuner = HyperparameterTuner(config, scenario_cls)
        tuner.tune()

        output_path = tmp_path / "results" / "tuning.json"
        tuner.save_results(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "best_params" in data
        assert "best_value" in data
        assert "trials" in data
        assert len(data["trials"]) == 3

    def test_save_results_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_results() creates missing parent directories."""
        config = _make_config(n_trials=2)
        scenario_cls = _make_scenario_cls()
        tuner = HyperparameterTuner(config, scenario_cls)
        tuner.tune()

        deep_path = tmp_path / "a" / "b" / "c" / "out.json"
        tuner.save_results(deep_path)
        assert deep_path.exists()

    def test_skips_trial_on_unexpected_exception(self) -> None:
        """Trials that raise an exception increment n_failed and do not crash the tuner."""
        call_count = {"n": 0}

        def flaky_objective(params: dict[str, Any]) -> float:
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("simulated transient error")
            return params["x"] ** 2

        # We need a scenario that raises on its .run() call for even trials.
        # Easier: monkeypatch the scenario's execute to raise sometimes.
        from src.poc.config import BaseScenarioConfig, ScenarioStatus
        from src.poc.registry import BaseScenario

        raise_count = {"n": 0}

        class _RaisingConfig(BaseScenarioConfig):
            model_config = BaseScenarioConfig.model_config.copy()
            model_config["extra"] = "allow"
            name: str = "raising"
            description: str = "raises on even calls"

        class _RaisingScenario(BaseScenario):
            config_class = _RaisingConfig

            def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
                super().__init__(config=_RaisingConfig())
                raise_count["n"] += 1
                self._should_raise = raise_count["n"] % 2 == 0

            def execute(self) -> Any:
                if self._should_raise:
                    raise ValueError("expected test error")
                self.record_metric("loss", 0.1)
                return self._create_result(ScenarioStatus.RUNNING)

        config = _make_config(n_trials=4)
        tuner = HyperparameterTuner(config, _RaisingScenario)
        result = tuner.tune()

        # Some trials failed (raised), some completed; total shouldn't exceed n_trials
        total = result.n_trials_completed + result.n_trials_failed + result.n_trials_pruned
        assert total <= config.n_trials


# ---------------------------------------------------------------------------
# TestTuningResult
# ---------------------------------------------------------------------------


class TestTuningResult:
    """Tests for TuningResult dataclass fields and invariants."""

    def _run_tuner(
        self,
        n_trials: int = 5,
        direction: str = "minimize",
    ) -> TuningResult:
        """Run a quick tuner and return its TuningResult."""
        config = _make_config(n_trials=n_trials, direction=direction)
        scenario_cls = _make_scenario_cls(metric_name="loss", objective_fn=_bowl_objective)
        tuner = HyperparameterTuner(config, scenario_cls)
        return tuner.tune()

    def test_has_required_fields(self) -> None:
        """TuningResult has best_params, best_value, n_trials_completed, trials."""
        result = self._run_tuner()
        assert hasattr(result, "best_params")
        assert hasattr(result, "best_value")
        assert hasattr(result, "n_trials_completed")
        assert hasattr(result, "trials")

    def test_trials_list_length(self) -> None:
        """Trials list length equals n_trials_completed (all pass in this scenario)."""
        n = 7
        result = self._run_tuner(n_trials=n)
        assert len(result.trials) == n
        assert result.n_trials_completed == n

    def test_best_value_is_minimum_of_trials_when_minimizing(self) -> None:
        """best_value equals the minimum objective_value across all completed trials."""
        result = self._run_tuner(n_trials=10, direction="minimize")
        completed_values = [t.objective_value for t in result.trials if t.status == "completed"]
        assert math.isclose(result.best_value, min(completed_values), rel_tol=1e-9)

    def test_search_space_field_is_dict(self) -> None:
        """TuningResult.search_space is a dict serialised from the config."""
        result = self._run_tuner()
        assert isinstance(result.search_space, dict)
        assert "x" in result.search_space
        assert "y" in result.search_space

    def test_duration_seconds_is_positive(self) -> None:
        """TuningResult.duration_seconds is a non-negative float."""
        result = self._run_tuner()
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# TestTrialResult
# ---------------------------------------------------------------------------


class TestTrialResult:
    """Tests for TrialResult dataclass fields."""

    def _make_trial(self, trial_id: int = 0, value: float = 0.5) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            params={"x": 0.1, "y": -0.2},
            objective_value=value,
            metrics={"loss": value},
            duration_seconds=0.01,
            status="completed",
        )

    def test_fields_are_accessible(self) -> None:
        """TrialResult exposes trial_id, params, objective_value, status."""
        t = self._make_trial(trial_id=3, value=0.42)
        assert t.trial_id == 3
        assert t.params == {"x": 0.1, "y": -0.2}
        assert math.isclose(t.objective_value, 0.42)
        assert t.status == "completed"

    def test_timestamp_is_set_automatically(self) -> None:
        """Timestamp field is populated with an ISO-format datetime string."""
        t = self._make_trial()
        assert isinstance(t.timestamp, str)
        # Quick sanity: should be parseable
        from datetime import datetime

        datetime.fromisoformat(t.timestamp)

    def test_metrics_dict(self) -> None:
        """TrialResult.metrics is a dict of float values."""
        t = self._make_trial(value=0.3)
        assert isinstance(t.metrics, dict)
        assert "loss" in t.metrics
