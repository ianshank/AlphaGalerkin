"""Hyperparameter tuner for PoC scenarios.

This module provides the main tuning orchestrator that coordinates
trials, tracks results, and identifies optimal configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.poc.tuning.config import TuningConfig
from src.poc.tuning.sampler import TPESampler, create_sampler

if TYPE_CHECKING:
    from src.poc.registry import BaseScenario

logger = structlog.get_logger(__name__)


@dataclass
class TrialResult:
    """Result from a single tuning trial."""

    trial_id: int
    params: dict[str, Any]
    objective_value: float
    metrics: dict[str, float]
    duration_seconds: float
    status: str  # "completed", "pruned", "failed"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TuningResult:
    """Result from complete tuning run."""

    best_params: dict[str, Any]
    best_value: float
    n_trials_completed: int
    n_trials_pruned: int
    n_trials_failed: int
    trials: list[TrialResult]
    search_space: dict[str, Any]
    duration_seconds: float
    study_name: str


class HyperparameterTuner:
    """Orchestrates hyperparameter tuning for scenarios.

    Coordinates sampling, trial execution, and result tracking
    to find optimal configurations.

    Attributes:
        config: Tuning configuration.
        scenario_cls: Scenario class to tune.
        base_scenario_config: Base scenario configuration.

    """

    def __init__(
        self,
        config: TuningConfig,
        scenario_cls: type[BaseScenario],
        base_scenario_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tuner.

        Args:
            config: Tuning configuration.
            scenario_cls: Scenario class to instantiate and run.
            base_scenario_config: Base configuration for scenario.

        """
        self.config = config
        self.scenario_cls = scenario_cls
        self.base_scenario_config = base_scenario_config or {}

        self.sampler = create_sampler(config)
        self.trials: list[TrialResult] = []

        self._best_value: float | None = None
        self._best_params: dict[str, Any] | None = None
        self._n_pruned = 0
        self._n_failed = 0

        self._logger = structlog.get_logger(__name__).bind(
            study_name=config.study_name,
            n_trials=config.n_trials,
            sampler=config.sampler,
        )

    def tune(self) -> TuningResult:
        """Run the tuning process.

        Returns:
            TuningResult with best parameters and all trial results.

        """
        import time

        start_time = time.time()

        self._logger.info("tuning_started")

        for trial_id in range(self.config.n_trials):
            self._logger.info(
                "trial_starting",
                trial_id=trial_id,
                best_so_far=self._best_value,
            )

            try:
                result = self._run_trial(trial_id)
                self.trials.append(result)

                # Update best
                if result.status == "completed":
                    if self._is_better(result.objective_value):
                        self._best_value = result.objective_value
                        self._best_params = result.params
                        self._logger.info(
                            "new_best_found",
                            trial_id=trial_id,
                            value=result.objective_value,
                        )

                # Update TPE sampler if applicable
                if isinstance(self.sampler, TPESampler):
                    self.sampler.update(result.params, result.objective_value)

            except Exception as e:
                self._logger.error(
                    "trial_failed",
                    trial_id=trial_id,
                    error=str(e),
                )
                self._n_failed += 1

        duration = time.time() - start_time

        self._logger.info(
            "tuning_completed",
            best_value=self._best_value,
            n_trials=len(self.trials),
            duration_seconds=duration,
        )

        return TuningResult(
            best_params=self._best_params or {},
            best_value=self._best_value or float("inf"),
            n_trials_completed=len([t for t in self.trials if t.status == "completed"]),
            n_trials_pruned=self._n_pruned,
            n_trials_failed=self._n_failed,
            trials=self.trials,
            search_space={n: s.model_dump() for n, s in self.config.search_space.items()},
            duration_seconds=duration,
            study_name=self.config.study_name,
        )

    def _run_trial(self, trial_id: int) -> TrialResult:
        """Run a single trial.

        Args:
            trial_id: Trial identifier.

        Returns:
            TrialResult with outcome.

        """
        import time

        start_time = time.time()

        # Sample parameters
        params = self.sampler.sample(self.config.search_space, trial_id)

        # Create scenario config with sampled params
        scenario_config = self.base_scenario_config.copy()
        scenario_config.update(params)

        # Instantiate and run scenario
        scenario = self.scenario_cls(**scenario_config)
        result = scenario.run()

        duration = time.time() - start_time

        # Extract objective value
        objective_value = result.metrics.get(
            self.config.objective_metric,
            float("inf") if self.config.direction == "minimize" else float("-inf"),
        )

        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective_value=objective_value,
            metrics=result.metrics,
            duration_seconds=duration,
            status="completed" if result.passed else "failed",
        )

    def _is_better(self, value: float) -> bool:
        """Check if value is better than current best.

        Args:
            value: Value to compare.

        Returns:
            True if better.

        """
        if self._best_value is None:
            return True

        if self.config.direction == "minimize":
            return value < self._best_value
        else:
            return value > self._best_value

    def save_results(self, path: str | Path) -> None:
        """Save tuning results to JSON file.

        Args:
            path: Output path.

        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "best_params": self._best_params,
            "best_value": self._best_value,
            "config": self.config.model_dump(),
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "params": t.params,
                    "objective_value": t.objective_value,
                    "metrics": t.metrics,
                    "duration_seconds": t.duration_seconds,
                    "status": t.status,
                    "timestamp": t.timestamp,
                }
                for t in self.trials
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._logger.info("results_saved", path=str(path))


def create_tuner(
    scenario_cls: type[BaseScenario],
    search_space: dict[str, Any],
    n_trials: int = 100,
    **kwargs: Any,
) -> HyperparameterTuner:
    """Factory function to create hyperparameter tuner.

    Args:
        scenario_cls: Scenario class to tune.
        search_space: Parameter search spaces.
        n_trials: Number of trials.
        **kwargs: Additional tuning config options.

    Returns:
        Configured HyperparameterTuner instance.

    """
    from src.poc.tuning.config import create_search_space_from_dict

    config = TuningConfig(
        n_trials=n_trials,
        search_space=create_search_space_from_dict(search_space),
        **kwargs,
    )

    return HyperparameterTuner(config, scenario_cls)
