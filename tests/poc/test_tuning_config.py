"""Tests for hyperparameter tuning configuration schemas.

Validates:
    - SearchSpace validation (bounds, log_scale, categorical)
    - TuningConfig defaults and constraint enforcement
    - Invalid configurations raise ValidationError
    - SearchSpace random sampling
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from pydantic import ValidationError

from src.poc.tuning.config import (
    SearchSpace,
    TuningConfig,
    create_search_space_from_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture()
def float_space() -> SearchSpace:
    """A float search space with default bounds."""
    return SearchSpace(type="float", low=0.0001, high=1.0)


@pytest.fixture()
def int_space() -> SearchSpace:
    """An int search space."""
    return SearchSpace(type="int", low=1, high=100)


@pytest.fixture()
def categorical_space() -> SearchSpace:
    """A categorical search space."""
    return SearchSpace(type="categorical", choices=["relu", "gelu", "silu"])


@pytest.fixture()
def log_float_space() -> SearchSpace:
    """A float search space with log scale."""
    return SearchSpace(type="float", low=1e-5, high=1e-1, log_scale=True)


@pytest.fixture()
def sample_search_space(
    float_space: SearchSpace, int_space: SearchSpace, categorical_space: SearchSpace
) -> dict[str, SearchSpace]:
    """A multi-parameter search space."""
    return {
        "learning_rate": float_space,
        "n_layers": int_space,
        "activation": categorical_space,
    }


# ---------------------------------------------------------------------------
# SearchSpace validation
# ---------------------------------------------------------------------------


class TestSearchSpaceValidation:
    """Tests for SearchSpace Pydantic validation."""

    def test_float_space_valid(self, float_space: SearchSpace) -> None:
        """Valid float space should be created without error."""
        assert float_space.type == "float"
        assert float_space.low == 0.0001
        assert float_space.high == 1.0

    def test_int_space_valid(self, int_space: SearchSpace) -> None:
        """Valid int space should be created without error."""
        assert int_space.type == "int"
        assert int_space.low == 1
        assert int_space.high == 100

    def test_categorical_space_valid(self, categorical_space: SearchSpace) -> None:
        """Valid categorical space should be created without error."""
        assert categorical_space.type == "categorical"
        assert len(categorical_space.choices) == 3

    def test_log_scale_valid(self, log_float_space: SearchSpace) -> None:
        """Log scale with positive bounds should be valid."""
        assert log_float_space.log_scale is True
        assert log_float_space.low > 0

    def test_float_low_ge_high_invalid(self) -> None:
        """Float space with low >= high should raise."""
        with pytest.raises(ValidationError, match="low.*must be.*high"):
            SearchSpace(type="float", low=1.0, high=0.5)

    def test_float_low_eq_high_invalid(self) -> None:
        """Float space with low == high should raise."""
        with pytest.raises(ValidationError, match="low.*must be.*high"):
            SearchSpace(type="float", low=1.0, high=1.0)

    def test_int_low_ge_high_invalid(self) -> None:
        """Int space with low >= high should raise."""
        with pytest.raises(ValidationError, match="low.*must be.*high"):
            SearchSpace(type="int", low=10, high=5)

    def test_log_scale_non_positive_low_invalid(self) -> None:
        """Log scale with low <= 0 should raise."""
        with pytest.raises(ValidationError, match="log_scale requires low > 0"):
            SearchSpace(type="float", low=-1.0, high=1.0, log_scale=True)

    def test_log_scale_zero_low_invalid(self) -> None:
        """Log scale with low == 0 should raise."""
        with pytest.raises(ValidationError, match="log_scale requires low > 0"):
            SearchSpace(type="float", low=0.0, high=1.0, log_scale=True)

    def test_categorical_no_choices_invalid(self) -> None:
        """Categorical without choices should raise."""
        with pytest.raises(ValidationError, match="choices"):
            SearchSpace(type="categorical")

    def test_categorical_empty_choices_invalid(self) -> None:
        """Categorical with empty choices should raise."""
        with pytest.raises(ValidationError, match="choices"):
            SearchSpace(type="categorical", choices=[])

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchSpace(type="float", low=0.0, high=1.0, unknown=True)  # type: ignore[call-arg]

    def test_default_log_scale_false(self) -> None:
        """log_scale should default to False."""
        space = SearchSpace(type="float", low=0.1, high=1.0)
        assert space.log_scale is False


# ---------------------------------------------------------------------------
# SearchSpace.sample_random
# ---------------------------------------------------------------------------


class TestSearchSpaceSampling:
    """Tests for SearchSpace.sample_random."""

    @pytest.mark.parametrize("trial", range(10))
    def test_float_sample_in_bounds(self, float_space: SearchSpace, trial: int) -> None:
        """Float samples should be within [low, high]."""
        random.seed(SEED + trial)
        value = float_space.sample_random()
        assert float_space.low <= value <= float_space.high

    @pytest.mark.parametrize("trial", range(10))
    def test_int_sample_in_bounds(self, int_space: SearchSpace, trial: int) -> None:
        """Int samples should be within [low, high] and integer-typed."""
        random.seed(SEED + trial)
        value = int_space.sample_random()
        assert int_space.low <= value <= int_space.high
        assert isinstance(value, int)

    @pytest.mark.parametrize("trial", range(10))
    def test_categorical_sample_in_choices(
        self, categorical_space: SearchSpace, trial: int
    ) -> None:
        """Categorical samples should be from choices list."""
        random.seed(SEED + trial)
        value = categorical_space.sample_random()
        assert value in categorical_space.choices

    @pytest.mark.parametrize("trial", range(10))
    def test_log_float_sample_in_bounds(
        self, log_float_space: SearchSpace, trial: int
    ) -> None:
        """Log-scale float samples should be within [low, high]."""
        random.seed(SEED + trial)
        value = log_float_space.sample_random()
        assert log_float_space.low <= value <= log_float_space.high

    def test_log_int_sample_in_bounds(self) -> None:
        """Log-scale int samples should be within [low, high]."""
        space = SearchSpace(type="int", low=1, high=1000, log_scale=True)
        random.seed(SEED)
        for _ in range(20):
            value = space.sample_random()
            assert space.low <= value <= space.high
            assert isinstance(value, int)

    def test_default_value_field(self) -> None:
        """default field should be stored but not affect sampling."""
        space = SearchSpace(type="float", low=0.0, high=1.0, default=0.5)
        assert space.default == 0.5
        # sample_random does not use default
        random.seed(SEED)
        value = space.sample_random()
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# TuningConfig defaults and validation
# ---------------------------------------------------------------------------


class TestTuningConfigDefaults:
    """Tests for TuningConfig default values."""

    def test_defaults(self) -> None:
        """TuningConfig should have sensible defaults."""
        config = TuningConfig()
        assert config.n_trials == 100
        assert config.sampler == "tpe"
        assert config.pruner == "median"
        assert config.direction == "minimize"
        assert config.seed == 42
        assert config.parallel_trials == 1

    def test_custom_values(self) -> None:
        """TuningConfig should accept custom values."""
        config = TuningConfig(
            n_trials=50,
            sampler="random",
            direction="maximize",
            seed=123,
            objective_metric="accuracy",
        )
        assert config.n_trials == 50
        assert config.sampler == "random"
        assert config.direction == "maximize"
        assert config.seed == 123
        assert config.objective_metric == "accuracy"


class TestTuningConfigValidation:
    """Tests for TuningConfig validation constraints."""

    def test_n_trials_must_be_positive(self) -> None:
        """n_trials must be >= 1."""
        with pytest.raises(ValidationError):
            TuningConfig(n_trials=0)

    def test_n_trials_negative_invalid(self) -> None:
        """Negative n_trials should raise."""
        with pytest.raises(ValidationError):
            TuningConfig(n_trials=-5)

    def test_parallel_trials_must_be_positive(self) -> None:
        """parallel_trials must be >= 1."""
        with pytest.raises(ValidationError):
            TuningConfig(parallel_trials=0)

    @pytest.mark.parametrize("sampler", ["random", "grid", "tpe", "cmaes"])
    def test_valid_samplers(self, sampler: str) -> None:
        """All valid sampler names should be accepted."""
        config = TuningConfig(sampler=sampler)  # type: ignore[arg-type]
        assert config.sampler == sampler

    def test_invalid_sampler(self) -> None:
        """Unknown sampler should raise."""
        with pytest.raises(ValidationError):
            TuningConfig(sampler="bayesian")  # type: ignore[arg-type]

    @pytest.mark.parametrize("pruner", ["none", "median", "hyperband", "percentile"])
    def test_valid_pruners(self, pruner: str) -> None:
        """All valid pruner names should be accepted."""
        config = TuningConfig(pruner=pruner)  # type: ignore[arg-type]
        assert config.pruner == pruner

    def test_invalid_pruner(self) -> None:
        """Unknown pruner should raise."""
        with pytest.raises(ValidationError):
            TuningConfig(pruner="early_stop")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            TuningConfig(unknown_param=True)  # type: ignore[call-arg]

    def test_timeout_per_trial_must_be_positive(self) -> None:
        """timeout_per_trial must be >= 1 if specified."""
        with pytest.raises(ValidationError):
            TuningConfig(timeout_per_trial=0)

    def test_pruner_percentile_bounds(self) -> None:
        """pruner_percentile must be in [0, 100]."""
        with pytest.raises(ValidationError):
            TuningConfig(pruner_percentile=101.0)

        with pytest.raises(ValidationError):
            TuningConfig(pruner_percentile=-1.0)


# ---------------------------------------------------------------------------
# TuningConfig with search space
# ---------------------------------------------------------------------------


class TestTuningConfigSearchSpace:
    """Tests for TuningConfig search_space handling."""

    def test_search_space_from_dict(self) -> None:
        """Search space should be parsed from raw dicts."""
        config = TuningConfig(
            search_space={
                "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log_scale": True},
                "layers": {"type": "int", "low": 1, "high": 10},
            }
        )
        assert len(config.search_space) == 2
        assert isinstance(config.search_space["lr"], SearchSpace)
        assert config.search_space["lr"].log_scale is True

    def test_search_space_from_objects(
        self, sample_search_space: dict[str, SearchSpace]
    ) -> None:
        """Search space should accept SearchSpace objects directly."""
        config = TuningConfig(search_space=sample_search_space)
        assert len(config.search_space) == 3

    def test_empty_search_space(self) -> None:
        """Empty search space should be accepted."""
        config = TuningConfig(search_space={})
        assert config.search_space == {}

    def test_invalid_search_space_entry(self) -> None:
        """Invalid search space values should raise."""
        with pytest.raises(ValidationError):
            TuningConfig(
                search_space={
                    "bad": {"type": "float", "low": 10.0, "high": 1.0},
                }
            )


# ---------------------------------------------------------------------------
# create_search_space_from_dict
# ---------------------------------------------------------------------------


class TestCreateSearchSpaceFromDict:
    """Tests for the factory function."""

    def test_from_raw_dicts(self) -> None:
        """Should convert raw dicts to SearchSpace objects."""
        data = {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1},
            "act": {"type": "categorical", "choices": ["relu", "gelu"]},
        }
        result = create_search_space_from_dict(data)
        assert len(result) == 2
        assert isinstance(result["lr"], SearchSpace)
        assert isinstance(result["act"], SearchSpace)

    def test_passthrough_search_space_objects(
        self, float_space: SearchSpace
    ) -> None:
        """SearchSpace objects should pass through unchanged."""
        data = {"lr": float_space}
        result = create_search_space_from_dict(data)
        assert result["lr"] is float_space

    def test_empty_dict(self) -> None:
        """Empty dict should return empty dict."""
        assert create_search_space_from_dict({}) == {}
