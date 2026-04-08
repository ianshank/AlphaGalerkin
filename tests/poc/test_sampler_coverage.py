"""Additional tests for sampler.py to cover missed lines.

Covers:
- BaseSampler abstract method (NotImplementedError on line 39)
- GridSampler with categorical, int log_scale, float log_scale spaces
- TPESampler post-startup fallback path (optuna import mock)
- TPESampler._sample_tpe / _to_optuna_distributions with mocked optuna
- create_sampler factory for grid, tpe, and default branches
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.poc.tuning.config import SearchSpace, TuningConfig
from src.poc.tuning.sampler import (
    BaseSampler,
    GridSampler,
    RandomSampler,
    TPESampler,
    create_sampler,
)


def _categorical_space(choices: list[Any]) -> SearchSpace:
    """Create a categorical SearchSpace, bypassing field-order validator issue."""
    return SearchSpace.model_construct(
        type="categorical",
        choices=choices,
        low=None,
        high=None,
        log_scale=False,
        step=None,
        default=None,
    )


# ---------------------------------------------------------------------------
# BaseSampler abstract method
# ---------------------------------------------------------------------------


class TestBaseSamplerAbstract:
    """Test BaseSampler.sample() raises NotImplementedError when called directly."""

    def test_abstract_sample_raises_not_implemented(self) -> None:
        """Calling BaseSampler.sample() via super() raises NotImplementedError."""

        class _ConcreteSampler(BaseSampler):
            def sample(
                self,
                search_space: dict[str, SearchSpace],
                trial_number: int,
            ) -> dict[str, Any]:
                # Delegate to super to hit the NotImplementedError
                return super().sample(search_space, trial_number)

        sampler = _ConcreteSampler()
        with pytest.raises(NotImplementedError):
            sampler.sample({}, 0)


# ---------------------------------------------------------------------------
# GridSampler with categorical, int log_scale, float log_scale
# ---------------------------------------------------------------------------


class TestGridSamplerExtraSpaces:
    """Test GridSampler with space types not covered in existing tests."""

    def test_categorical_space(self) -> None:
        """Grid sampler enumerates all categorical choices."""
        search_space = {
            "optimizer": _categorical_space(["adam", "sgd", "rmsprop"]),
        }
        sampler = GridSampler(search_space, n_samples_per_dim=5)
        # Categorical ignores n_samples_per_dim; grid has len(choices) entries
        assert len(sampler._grid) == 3
        optimizers = {point["optimizer"] for point in sampler._grid}
        assert optimizers == {"adam", "sgd", "rmsprop"}

    def test_int_log_scale_space(self) -> None:
        """Grid sampler handles int type with log_scale=True."""
        search_space = {
            "d_model": SearchSpace(type="int", low=8, high=256, log_scale=True),
        }
        sampler = GridSampler(search_space, n_samples_per_dim=5)
        assert len(sampler._grid) == 5
        for point in sampler._grid:
            val = point["d_model"]
            assert isinstance(val, int)
            assert 8 <= val <= 256

        # Check that values are geometrically spaced (not linear)
        values = [p["d_model"] for p in sampler._grid]
        assert values[0] == 8
        assert values[-1] == 256
        # In geomspace, the second value should be much less than midpoint of linear
        # For geomspace(8, 256, 5): ~8, 16, 32, 64, 128 style progression
        # rather than 8, 70, 132, 194, 256 for linspace
        assert values[1] < 70, f"Expected geometric spacing, got {values}"

    def test_int_linear_scale_space(self) -> None:
        """Grid sampler handles int type with log_scale=False (linspace)."""
        search_space = {
            "n_layers": SearchSpace(type="int", low=1, high=10, log_scale=False),
        }
        sampler = GridSampler(search_space, n_samples_per_dim=4)
        assert len(sampler._grid) == 4
        for point in sampler._grid:
            val = point["n_layers"]
            assert isinstance(val, (int, np.integer))
            assert 1 <= val <= 10

    def test_float_log_scale_space(self) -> None:
        """Grid sampler handles float type with log_scale=True."""
        search_space = {
            "lr": SearchSpace(type="float", low=1e-4, high=1e-1, log_scale=True),
        }
        sampler = GridSampler(search_space, n_samples_per_dim=4)
        assert len(sampler._grid) == 4
        for point in sampler._grid:
            val = point["lr"]
            assert isinstance(val, float)
            assert 1e-4 <= val <= 1e-1

        # First and last should be the endpoints
        values = [p["lr"] for p in sampler._grid]
        assert values[0] == pytest.approx(1e-4, rel=1e-6)
        assert values[-1] == pytest.approx(1e-1, rel=1e-6)

    def test_mixed_categorical_and_numeric(self) -> None:
        """Grid sampler handles mix of categorical and numeric spaces."""
        search_space = {
            "optimizer": _categorical_space(["adam", "sgd"]),
            "lr": SearchSpace(type="float", low=0.001, high=0.1),
        }
        sampler = GridSampler(search_space, n_samples_per_dim=3)
        # 2 categorical * 3 float = 6 combinations
        assert len(sampler._grid) == 6
        # Check all have the expected keys
        for point in sampler._grid:
            assert "optimizer" in point
            assert "lr" in point
            assert point["optimizer"] in ["adam", "sgd"]
            assert 0.001 <= point["lr"] <= 0.1


# ---------------------------------------------------------------------------
# TPESampler post-startup and fallback paths
# ---------------------------------------------------------------------------


class TestTPESamplerPostStartup:
    """Test TPE sampler after startup phase (lines 200-207)."""

    def test_post_startup_falls_back_on_import_error(self) -> None:
        """When optuna is not available, TPE falls back to random sampling."""
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
        }
        sampler = TPESampler(seed=42, n_startup_trials=2)

        # Add some history so we're past startup
        for i in range(3):
            params = sampler.sample(search_space, trial_number=i)
            sampler.update(params, value=float(i) * 0.1)

        # Now mock _sample_tpe to raise ImportError (simulating no optuna)
        with patch.object(sampler, "_sample_tpe", side_effect=ImportError("no optuna")):
            result = sampler.sample(search_space, trial_number=5)

        assert "x" in result
        assert -1.0 <= result["x"] <= 1.0

    def test_post_startup_tries_tpe_first(self) -> None:
        """After startup, sample() calls _sample_tpe before falling back."""
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
        }
        sampler = TPESampler(seed=42, n_startup_trials=0)

        mock_result = {"x": 0.5}
        with patch.object(sampler, "_sample_tpe", return_value=mock_result) as mock_tpe:
            result = sampler.sample(search_space, trial_number=0)

        mock_tpe.assert_called_once_with(search_space)
        assert result == {"x": 0.5}


class TestTPESampleTPEMethod:
    """Test TPESampler._sample_tpe and _to_optuna_distributions with mocked optuna."""

    def test_sample_tpe_with_mocked_optuna(self) -> None:
        """_sample_tpe creates an optuna study and returns params."""
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
            "n": SearchSpace(type="int", low=1, high=10),
        }
        sampler = TPESampler(seed=42, n_startup_trials=0)

        # Build mock optuna module
        mock_trial = MagicMock()
        mock_trial.params = {"x": 0.3, "n": 5}

        mock_study = MagicMock()
        mock_study.ask.return_value = mock_trial

        mock_optuna = MagicMock()
        mock_optuna.samplers.TPESampler.return_value = MagicMock()
        mock_optuna.create_study.return_value = mock_study

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            result = sampler._sample_tpe(search_space)

        assert result == {"x": 0.3, "n": 5}

    def test_sample_tpe_with_history(self) -> None:
        """_sample_tpe adds history trials to the study."""
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
        }
        sampler = TPESampler(seed=42, n_startup_trials=0)
        sampler._history = [
            ({"x": 0.1}, 0.5),
            ({"x": -0.3}, 0.2),
        ]

        mock_trial_obj = MagicMock()
        mock_trial_obj.params = {"x": 0.7}

        mock_study = MagicMock()
        mock_study.ask.return_value = mock_trial_obj

        mock_create_trial = MagicMock()

        mock_optuna = MagicMock()
        mock_optuna.samplers.TPESampler.return_value = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.trial.create_trial = mock_create_trial

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            result = sampler._sample_tpe(search_space)

        # Verify that add_trial was called for each history entry
        assert mock_study.add_trial.call_count == 2
        assert result == {"x": 0.7}

    def test_to_optuna_distributions_float(self) -> None:
        """_to_optuna_distributions handles float type."""
        search_space = {
            "lr": SearchSpace(type="float", low=1e-4, high=1e-1, log_scale=True),
        }
        sampler = TPESampler(seed=0)

        mock_float_dist = MagicMock()
        mock_optuna = MagicMock()
        mock_optuna.distributions.FloatDistribution.return_value = mock_float_dist

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            dists = sampler._to_optuna_distributions(search_space)

        assert "lr" in dists
        mock_optuna.distributions.FloatDistribution.assert_called_once_with(
            low=1e-4, high=1e-1, log=True
        )

    def test_to_optuna_distributions_int(self) -> None:
        """_to_optuna_distributions handles int type."""
        search_space = {
            "n_layers": SearchSpace(type="int", low=1, high=8, log_scale=False),
        }
        sampler = TPESampler(seed=0)

        mock_int_dist = MagicMock()
        mock_optuna = MagicMock()
        mock_optuna.distributions.IntDistribution.return_value = mock_int_dist

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            dists = sampler._to_optuna_distributions(search_space)

        assert "n_layers" in dists
        mock_optuna.distributions.IntDistribution.assert_called_once_with(low=1, high=8, log=False)

    def test_to_optuna_distributions_categorical(self) -> None:
        """_to_optuna_distributions handles categorical type."""
        search_space = {
            "opt": _categorical_space(["adam", "sgd"]),
        }
        sampler = TPESampler(seed=0)

        mock_cat_dist = MagicMock()
        mock_optuna = MagicMock()
        mock_optuna.distributions.CategoricalDistribution.return_value = mock_cat_dist

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            dists = sampler._to_optuna_distributions(search_space)

        assert "opt" in dists
        mock_optuna.distributions.CategoricalDistribution.assert_called_once_with(
            choices=["adam", "sgd"]
        )

    def test_to_optuna_distributions_all_types(self) -> None:
        """_to_optuna_distributions handles all three types in a single space."""
        search_space = {
            "lr": SearchSpace(type="float", low=1e-4, high=1e-1),
            "n": SearchSpace(type="int", low=1, high=10),
            "opt": _categorical_space(["a", "b"]),
        }
        sampler = TPESampler(seed=0)

        mock_optuna = MagicMock()
        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            dists = sampler._to_optuna_distributions(search_space)

        assert set(dists.keys()) == {"lr", "n", "opt"}


# ---------------------------------------------------------------------------
# create_sampler factory function
# ---------------------------------------------------------------------------


class TestCreateSampler:
    """Test create_sampler factory (lines 288-306)."""

    def _make_config(self, sampler_type: str) -> TuningConfig:
        """Build a TuningConfig with the given sampler type."""
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
        }
        return TuningConfig(
            n_trials=5,
            sampler=sampler_type,  # type: ignore[arg-type]
            search_space=search_space,
            seed=42,
            study_name="factory_test",
        )

    def test_create_random_sampler(self) -> None:
        """create_sampler returns RandomSampler for 'random'."""
        config = self._make_config("random")
        sampler = create_sampler(config)
        assert isinstance(sampler, RandomSampler)

    def test_create_grid_sampler(self) -> None:
        """create_sampler returns GridSampler for 'grid'."""
        config = self._make_config("grid")
        sampler = create_sampler(config)
        assert isinstance(sampler, GridSampler)

    def test_create_tpe_sampler(self) -> None:
        """create_sampler returns TPESampler for 'tpe'."""
        config = self._make_config("tpe")
        sampler = create_sampler(config)
        assert isinstance(sampler, TPESampler)

    def test_create_sampler_unknown_falls_back_to_random(self) -> None:
        """create_sampler returns RandomSampler for unknown sampler type."""
        search_space = {
            "x": SearchSpace(type="float", low=-1.0, high=1.0),
        }
        # Build a config with an unsupported sampler by bypassing validation
        config = TuningConfig(
            n_trials=5,
            sampler="cmaes",  # type: ignore[arg-type]
            search_space=search_space,
            seed=42,
            study_name="factory_test",
        )
        sampler = create_sampler(config)
        # cmaes is not handled in create_sampler -> defaults to RandomSampler
        assert isinstance(sampler, RandomSampler)

    def test_grid_sampler_seed_is_from_config(self) -> None:
        """GridSampler created by factory receives the search space from config."""
        config = self._make_config("grid")
        sampler = create_sampler(config)
        assert isinstance(sampler, GridSampler)
        # Grid should have been built from the config's search_space
        assert len(sampler._grid) > 0

    def test_tpe_sampler_seed_is_from_config(self) -> None:
        """TPESampler created by factory uses seed from config."""
        config = self._make_config("tpe")
        sampler = create_sampler(config)
        assert isinstance(sampler, TPESampler)
        assert sampler.seed == 42
