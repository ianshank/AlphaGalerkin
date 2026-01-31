"""Hyperparameter samplers for tuning.

This module provides various sampling strategies for hyperparameter
search including random, grid, and Bayesian methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.poc.tuning.config import SearchSpace, TuningConfig

logger = structlog.get_logger(__name__)


class BaseSampler(ABC):
    """Abstract base class for hyperparameter samplers."""

    @abstractmethod
    def sample(
        self,
        search_space: dict[str, SearchSpace],
        trial_number: int,
    ) -> dict[str, Any]:
        """Sample a set of hyperparameters.

        Args:
            search_space: Parameter search spaces.
            trial_number: Current trial number.

        Returns:
            Dictionary of sampled parameters.

        """
        raise NotImplementedError


class RandomSampler(BaseSampler):
    """Random sampling strategy."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize random sampler.

        Args:
            seed: Random seed.

        """
        self.seed = seed
        import random

        random.seed(seed)

    def sample(
        self,
        search_space: dict[str, SearchSpace],
        trial_number: int,
    ) -> dict[str, Any]:
        """Sample random values from search space.

        Args:
            search_space: Parameter search spaces.
            trial_number: Current trial number (unused).

        Returns:
            Dictionary of sampled parameters.

        """
        return {name: space.sample_random() for name, space in search_space.items()}


class GridSampler(BaseSampler):
    """Grid search sampling strategy."""

    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        n_samples_per_dim: int = 10,
    ) -> None:
        """Initialize grid sampler.

        Args:
            search_space: Parameter search spaces.
            n_samples_per_dim: Samples per dimension.

        """
        self.n_samples_per_dim = n_samples_per_dim
        self._grid = self._create_grid(search_space)
        self._grid_index = 0

    def _create_grid(
        self,
        search_space: dict[str, SearchSpace],
    ) -> list[dict[str, Any]]:
        """Create full parameter grid.

        Args:
            search_space: Parameter search spaces.

        Returns:
            List of parameter combinations.

        """
        import itertools
        import numpy as np

        param_values: dict[str, list[Any]] = {}

        for name, space in search_space.items():
            if space.type == "categorical":
                param_values[name] = list(space.choices)
            elif space.type == "int":
                if space.log_scale:
                    values = np.geomspace(space.low, space.high, self.n_samples_per_dim)
                    param_values[name] = [int(round(v)) for v in values]
                else:
                    param_values[name] = list(
                        np.linspace(space.low, space.high, self.n_samples_per_dim, dtype=int)
                    )
            else:  # float
                if space.log_scale:
                    param_values[name] = list(
                        np.geomspace(space.low, space.high, self.n_samples_per_dim)
                    )
                else:
                    param_values[name] = list(
                        np.linspace(space.low, space.high, self.n_samples_per_dim)
                    )

        # Create all combinations
        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def sample(
        self,
        search_space: dict[str, SearchSpace],
        trial_number: int,
    ) -> dict[str, Any]:
        """Get next grid point.

        Args:
            search_space: Parameter search spaces (unused, grid already created).
            trial_number: Current trial number.

        Returns:
            Next parameter combination from grid.

        """
        if trial_number >= len(self._grid):
            # Wrap around
            return self._grid[trial_number % len(self._grid)]
        return self._grid[trial_number]


class TPESampler(BaseSampler):
    """Tree-structured Parzen Estimator sampling.

    Uses Optuna's TPE sampler if available, falls back to random.
    """

    def __init__(
        self,
        seed: int = 42,
        n_startup_trials: int = 10,
    ) -> None:
        """Initialize TPE sampler.

        Args:
            seed: Random seed.
            n_startup_trials: Random trials before TPE kicks in.

        """
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self._history: list[tuple[dict[str, Any], float]] = []
        self._fallback = RandomSampler(seed)

    def sample(
        self,
        search_space: dict[str, SearchSpace],
        trial_number: int,
    ) -> dict[str, Any]:
        """Sample using TPE or random.

        Args:
            search_space: Parameter search spaces.
            trial_number: Current trial number.

        Returns:
            Sampled parameters.

        """
        if trial_number < self.n_startup_trials:
            return self._fallback.sample(search_space, trial_number)

        # Try to use Optuna's TPE
        try:
            return self._sample_tpe(search_space)
        except ImportError:
            return self._fallback.sample(search_space, trial_number)

    def _sample_tpe(
        self,
        search_space: dict[str, SearchSpace],
    ) -> dict[str, Any]:
        """Sample using Optuna TPE.

        Args:
            search_space: Parameter search spaces.

        Returns:
            Sampled parameters.

        """
        import optuna

        # Create temporary study
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(sampler=sampler)

        # Add history
        for params, value in self._history:
            trial = optuna.trial.create_trial(
                params=params,
                distributions=self._to_optuna_distributions(search_space),
                values=[value],
            )
            study.add_trial(trial)

        # Suggest new params
        trial = study.ask(self._to_optuna_distributions(search_space))

        return {name: trial.params[name] for name in search_space}

    def _to_optuna_distributions(
        self,
        search_space: dict[str, SearchSpace],
    ) -> dict[str, Any]:
        """Convert search space to Optuna distributions.

        Args:
            search_space: Our search space format.

        Returns:
            Optuna distribution format.

        """
        import optuna

        distributions = {}
        for name, space in search_space.items():
            if space.type == "categorical":
                distributions[name] = optuna.distributions.CategoricalDistribution(
                    choices=space.choices
                )
            elif space.type == "int":
                distributions[name] = optuna.distributions.IntDistribution(
                    low=int(space.low),
                    high=int(space.high),
                    log=space.log_scale,
                )
            else:
                distributions[name] = optuna.distributions.FloatDistribution(
                    low=space.low,
                    high=space.high,
                    log=space.log_scale,
                )
        return distributions

    def update(self, params: dict[str, Any], value: float) -> None:
        """Update history with trial result.

        Args:
            params: Trial parameters.
            value: Objective value.

        """
        self._history.append((params, value))


def create_sampler(config: TuningConfig) -> BaseSampler:
    """Create sampler from configuration.

    Args:
        config: Tuning configuration.

    Returns:
        Configured sampler instance.

    """
    if config.sampler == "random":
        return RandomSampler(seed=config.seed)
    elif config.sampler == "grid":
        return GridSampler(config.search_space)
    elif config.sampler == "tpe":
        return TPESampler(seed=config.seed)
    else:
        # Default to random
        return RandomSampler(seed=config.seed)
