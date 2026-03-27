"""Tests for hyperparameter samplers.

Validates:
    - RandomSampler produces values within bounds
    - GridSampler covers the search space
    - TPESampler interface and fallback behavior
    - Property-based: all samples stay within declared bounds
    - create_sampler factory function
"""

from __future__ import annotations

from typing import Any

import pytest

from src.poc.tuning.config import SearchSpace, TuningConfig
from src.poc.tuning.sampler import (
    GridSampler,
    RandomSampler,
    TPESampler,
    create_sampler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture()
def float_space() -> SearchSpace:
    return SearchSpace(type="float", low=0.01, high=10.0)


@pytest.fixture()
def int_space() -> SearchSpace:
    return SearchSpace(type="int", low=1, high=50)


@pytest.fixture()
def categorical_space() -> SearchSpace:
    # Use model_construct to bypass field_validator ordering issue in SearchSpace
    # (the validator on 'type' runs before 'choices' is available in info.data)
    return SearchSpace.model_construct(
        type="categorical",
        choices=["adam", "sgd", "adamw"],
        low=None,
        high=None,
        log_scale=False,
        step=None,
        default=None,
    )


@pytest.fixture()
def log_float_space() -> SearchSpace:
    return SearchSpace(type="float", low=1e-5, high=1e-1, log_scale=True)


@pytest.fixture()
def mixed_search_space(
    float_space: SearchSpace,
    int_space: SearchSpace,
    categorical_space: SearchSpace,
) -> dict[str, SearchSpace]:
    return {
        "learning_rate": float_space,
        "n_layers": int_space,
        "optimizer": categorical_space,
    }


# ---------------------------------------------------------------------------
# RandomSampler
# ---------------------------------------------------------------------------


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_sample_returns_all_params(
        self, mixed_search_space: dict[str, SearchSpace]
    ) -> None:
        """Sample should return a value for every parameter."""
        sampler = RandomSampler(seed=SEED)
        result = sampler.sample(mixed_search_space, trial_number=0)
        assert set(result.keys()) == set(mixed_search_space.keys())

    @pytest.mark.parametrize("trial", range(20))
    def test_float_within_bounds(
        self, float_space: SearchSpace, trial: int
    ) -> None:
        """Float samples should be within [low, high]."""
        sampler = RandomSampler(seed=SEED + trial)
        space = {"param": float_space}
        result = sampler.sample(space, trial_number=trial)
        assert float_space.low <= result["param"] <= float_space.high

    @pytest.mark.parametrize("trial", range(20))
    def test_int_within_bounds(
        self, int_space: SearchSpace, trial: int
    ) -> None:
        """Int samples should be within [low, high] and integer-typed."""
        sampler = RandomSampler(seed=SEED + trial)
        space = {"param": int_space}
        result = sampler.sample(space, trial_number=trial)
        assert int_space.low <= result["param"] <= int_space.high
        assert isinstance(result["param"], int)

    @pytest.mark.parametrize("trial", range(20))
    def test_categorical_in_choices(
        self, categorical_space: SearchSpace, trial: int
    ) -> None:
        """Categorical samples should come from the choices list."""
        sampler = RandomSampler(seed=SEED + trial)
        space = {"param": categorical_space}
        result = sampler.sample(space, trial_number=trial)
        assert result["param"] in categorical_space.choices

    @pytest.mark.parametrize("trial", range(20))
    def test_log_float_within_bounds(
        self, log_float_space: SearchSpace, trial: int
    ) -> None:
        """Log-scale float samples should be within [low, high]."""
        sampler = RandomSampler(seed=SEED + trial)
        space = {"param": log_float_space}
        result = sampler.sample(space, trial_number=trial)
        assert log_float_space.low <= result["param"] <= log_float_space.high

    def test_deterministic_with_same_seed(
        self, float_space: SearchSpace
    ) -> None:
        """Same seed should produce same first sample for single-param space.

        Note: RandomSampler uses global random.seed(), so creating two
        samplers sequentially resets the global state. We test determinism
        by creating a sampler, sampling, then re-creating with same seed.
        """
        space = {"param": float_space}

        sampler1 = RandomSampler(seed=SEED)
        result1 = sampler1.sample(space, trial_number=0)

        sampler2 = RandomSampler(seed=SEED)
        result2 = sampler2.sample(space, trial_number=0)

        assert result1 == result2


# ---------------------------------------------------------------------------
# GridSampler
# ---------------------------------------------------------------------------


class TestGridSampler:
    """Tests for GridSampler."""

    def test_grid_covers_space(self, float_space: SearchSpace) -> None:
        """Grid should include boundary values (approximately)."""
        n_per_dim = 5
        space = {"param": float_space}
        sampler = GridSampler(space, n_samples_per_dim=n_per_dim)

        values = [sampler.sample(space, t)["param"] for t in range(n_per_dim)]

        # First and last should be near boundaries
        assert min(values) == pytest.approx(float_space.low, rel=0.1)
        assert max(values) == pytest.approx(float_space.high, rel=0.1)

    def test_grid_size_matches(self) -> None:
        """Grid size should be product of per-dimension counts."""
        n_per_dim = 3
        space = {
            "a": SearchSpace(type="float", low=0.0, high=1.0),
            "b": SearchSpace(type="int", low=1, high=10),
        }
        sampler = GridSampler(space, n_samples_per_dim=n_per_dim)

        expected_size = n_per_dim * n_per_dim
        assert len(sampler._grid) == expected_size

    def test_categorical_exhaustive(self, categorical_space: SearchSpace) -> None:
        """Grid should include all categorical choices."""
        space = {"act": categorical_space}
        sampler = GridSampler(space, n_samples_per_dim=10)

        all_values = {sampler.sample(space, t)["act"] for t in range(len(sampler._grid))}
        assert set(categorical_space.choices) == all_values

    def test_wraps_around(self) -> None:
        """Sampling beyond grid size should wrap around."""
        space = {"x": SearchSpace(type="float", low=0.0, high=1.0)}
        sampler = GridSampler(space, n_samples_per_dim=3)
        grid_size = len(sampler._grid)

        val_at_0 = sampler.sample(space, 0)
        val_at_wrap = sampler.sample(space, grid_size)
        assert val_at_0 == val_at_wrap

    def test_all_samples_within_bounds(self, float_space: SearchSpace) -> None:
        """Every grid point should be within bounds."""
        space = {"param": float_space}
        sampler = GridSampler(space, n_samples_per_dim=10)

        for t in range(len(sampler._grid)):
            val = sampler.sample(space, t)["param"]
            assert float_space.low <= val <= float_space.high

    def test_log_scale_grid(self) -> None:
        """Log-scale grid should space values geometrically."""
        space = {"lr": SearchSpace(type="float", low=1e-4, high=1e-1, log_scale=True)}
        sampler = GridSampler(space, n_samples_per_dim=4)

        values = sorted(sampler.sample(space, t)["lr"] for t in range(4))
        # Ratios between consecutive values should be roughly equal (geometric)
        ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
        for i in range(len(ratios) - 1):
            assert ratios[i] == pytest.approx(ratios[i + 1], rel=0.3)


# ---------------------------------------------------------------------------
# TPESampler
# ---------------------------------------------------------------------------


class TestTPESampler:
    """Tests for TPESampler interface."""

    def test_startup_uses_random(
        self, mixed_search_space: dict[str, SearchSpace]
    ) -> None:
        """During startup phase, TPE should fall back to random sampling."""
        n_startup = 5
        sampler = TPESampler(seed=SEED, n_startup_trials=n_startup)

        # All startup trials should work without error
        for trial in range(n_startup):
            result = sampler.sample(mixed_search_space, trial)
            assert set(result.keys()) == set(mixed_search_space.keys())

    def test_samples_within_bounds(
        self, mixed_search_space: dict[str, SearchSpace]
    ) -> None:
        """All samples should respect declared bounds."""
        sampler = TPESampler(seed=SEED, n_startup_trials=3)

        for trial in range(5):
            result = sampler.sample(mixed_search_space, trial)

            lr = result["learning_rate"]
            assert mixed_search_space["learning_rate"].low <= lr
            assert lr <= mixed_search_space["learning_rate"].high

            n_layers = result["n_layers"]
            assert mixed_search_space["n_layers"].low <= n_layers
            assert n_layers <= mixed_search_space["n_layers"].high

            assert result["optimizer"] in mixed_search_space["optimizer"].choices

    def test_update_records_history(self) -> None:
        """update() should append to internal history."""
        sampler = TPESampler(seed=SEED)
        params = {"lr": 0.01}
        sampler.update(params, 0.5)

        assert len(sampler._history) == 1
        assert sampler._history[0] == (params, 0.5)

    def test_multiple_updates(self) -> None:
        """Multiple updates should accumulate in history."""
        sampler = TPESampler(seed=SEED)
        for i in range(5):
            sampler.update({"lr": 0.01 * i}, float(i))

        assert len(sampler._history) == 5


# ---------------------------------------------------------------------------
# Property-based: all samples within bounds
# ---------------------------------------------------------------------------


class TestSamplerBoundsProperty:
    """Property-based tests ensuring all samplers respect bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_random_sampler_bounds(
        self, mixed_search_space: dict[str, SearchSpace], seed: int
    ) -> None:
        """RandomSampler must always stay in bounds."""
        sampler = RandomSampler(seed=seed)
        for trial in range(20):
            result = sampler.sample(mixed_search_space, trial)
            self._assert_within_bounds(result, mixed_search_space)

    @pytest.mark.parametrize("seed", range(5))
    def test_tpe_sampler_bounds(
        self, mixed_search_space: dict[str, SearchSpace], seed: int
    ) -> None:
        """TPESampler must always stay in bounds."""
        sampler = TPESampler(seed=seed, n_startup_trials=3)
        for trial in range(10):
            result = sampler.sample(mixed_search_space, trial)
            self._assert_within_bounds(result, mixed_search_space)

    def _assert_within_bounds(
        self,
        result: dict[str, Any],
        space: dict[str, SearchSpace],
    ) -> None:
        for name, spec in space.items():
            value = result[name]
            if spec.type in ("float", "int"):
                assert spec.low <= value <= spec.high, (
                    f"{name}={value} outside [{spec.low}, {spec.high}]"
                )
            elif spec.type == "categorical":
                assert value in spec.choices, (
                    f"{name}={value} not in {spec.choices}"
                )


# ---------------------------------------------------------------------------
# create_sampler factory
# ---------------------------------------------------------------------------


class TestCreateSampler:
    """Tests for the create_sampler factory function."""

    def test_random(self) -> None:
        config = TuningConfig(sampler="random", seed=SEED)
        sampler = create_sampler(config)
        assert isinstance(sampler, RandomSampler)

    def test_grid(self) -> None:
        config = TuningConfig(
            sampler="grid",
            search_space={
                "x": {"type": "float", "low": 0.0, "high": 1.0},
            },
        )
        sampler = create_sampler(config)
        assert isinstance(sampler, GridSampler)

    def test_tpe(self) -> None:
        config = TuningConfig(sampler="tpe", seed=SEED)
        sampler = create_sampler(config)
        assert isinstance(sampler, TPESampler)

    def test_unknown_falls_back_to_random(self) -> None:
        """Cmaes sampler should fall back to RandomSampler."""
        config = TuningConfig(sampler="cmaes", seed=SEED)
        sampler = create_sampler(config)
        assert isinstance(sampler, RandomSampler)
