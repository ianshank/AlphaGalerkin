"""Scenario registry for discovering and managing scenarios.

This module provides a decorator-based registration system for scenarios,
enabling automatic discovery and configuration-driven execution.

Usage:
    from src.poc import scenario, BaseScenario

    @scenario("my_scenario")
    class MyScenario(BaseScenario):
        config_class = MyScenarioConfig

        def setup(self) -> None:
            # Initialize resources
            pass

        def execute(self) -> ScenarioResult:
            # Run validation logic
            pass

        def teardown(self) -> None:
            # Cleanup resources
            pass
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

from src.poc.config import BaseScenarioConfig, ScenarioResult, ScenarioStatus

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)

# Type variable for scenario classes
T = TypeVar("T", bound="BaseScenario")


class ScenarioRegistry:
    """Central registry for all scenarios.

    Provides discovery, instantiation, and execution management.
    Thread-safe singleton pattern for global access.
    """

    _instance: "ScenarioRegistry | None" = None
    _scenarios: dict[str, type["BaseScenario"]]

    def __new__(cls) -> "ScenarioRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._scenarios = {}
        return cls._instance

    def register(self, name: str, scenario_cls: type["BaseScenario"]) -> None:
        """Register a scenario class.

        Args:
            name: Unique scenario identifier.
            scenario_cls: Scenario class to register.

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._scenarios:
            raise ValueError(
                f"Scenario '{name}' already registered by {self._scenarios[name]}"
            )

        self._scenarios[name] = scenario_cls
        logger.debug("scenario_registered", name=name, cls=scenario_cls.__name__)

    def get(self, name: str) -> type["BaseScenario"] | None:
        """Get a registered scenario class by name."""
        return self._scenarios.get(name)

    def list_scenarios(self) -> list[str]:
        """List all registered scenario names."""
        return list(self._scenarios.keys())

    def get_all(self) -> dict[str, type["BaseScenario"]]:
        """Get all registered scenarios."""
        return dict(self._scenarios)

    def clear(self) -> None:
        """Clear all registrations (primarily for testing)."""
        self._scenarios.clear()


def scenario(
    name: str,
) -> "Callable[[type[T]], type[T]]":
    """Decorator to register a scenario class.

    Args:
        name: Unique scenario identifier.

    Returns:
        Class decorator.

    Example:
        @scenario("transfer")
        class TransferScenario(BaseScenario):
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        ScenarioRegistry().register(name, cls)
        cls._scenario_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


class BaseScenario(ABC):
    """Abstract base class for all scenarios.

    Provides lifecycle management (setup, execute, teardown) and
    standardized result collection.

    Subclasses must implement:
        - execute(): Core validation logic
        - config_class: Pydantic config class for this scenario

    Optional overrides:
        - setup(): Pre-execution initialization
        - teardown(): Post-execution cleanup
    """

    # Class-level attributes (override in subclasses)
    config_class: type[BaseScenarioConfig] = BaseScenarioConfig
    _scenario_name: str = ""  # Set by @scenario decorator

    def __init__(
        self,
        config: BaseScenarioConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize scenario.

        Args:
            config: Scenario configuration. If None, creates default.
            **kwargs: Override config fields.
        """
        if config is None:
            config = self.config_class(**kwargs)
        elif kwargs:
            # Merge kwargs into config
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = self.config_class(**config_dict)

        self.config = config
        self._start_time: datetime | None = None
        self._metrics: dict[str, float] = {}
        self._artifacts: dict[str, str] = {}
        self._logger = structlog.get_logger(
            __name__,
            scenario=self.config.name,
        )

    @property
    def name(self) -> str:
        """Get scenario name."""
        return self._scenario_name or self.config.name

    def setup(self) -> None:
        """Pre-execution setup (override in subclasses).

        Called before execute(). Use for:
        - Loading models
        - Preparing datasets
        - Allocating resources
        """
        pass

    @abstractmethod
    def execute(self) -> ScenarioResult:
        """Execute the scenario validation logic.

        Must be implemented by subclasses.

        Returns:
            ScenarioResult with metrics, status, and artifacts.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Post-execution cleanup (override in subclasses).

        Called after execute() regardless of success/failure.
        Use for:
        - Releasing resources
        - Saving artifacts
        - Cleanup
        """
        pass

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value.

        Args:
            name: Metric identifier.
            value: Metric value.
        """
        self._metrics[name] = value
        self._logger.debug("metric_recorded", metric=name, value=value)

    def record_artifact(self, name: str, path: str) -> None:
        """Record an artifact path.

        Args:
            name: Artifact identifier.
            path: Path to artifact file.
        """
        self._artifacts[name] = path
        self._logger.debug("artifact_recorded", artifact=name, path=path)

    def run(self) -> ScenarioResult:
        """Execute the full scenario lifecycle.

        Handles setup, execution, teardown, and error handling.

        Returns:
            ScenarioResult capturing the outcome.
        """
        import sys

        import torch

        self._start_time = datetime.now()
        self._metrics = {}
        self._artifacts = {}

        self._logger.info(
            "scenario_starting",
            config_hash=self.config.compute_hash(),
        )

        try:
            # Setup
            self.setup()

            # Execute
            result = self.execute()

            self._logger.info(
                "scenario_completed",
                status=result.status.value,
                passed=result.passed,
                duration=result.duration_seconds,
            )

            return result

        except Exception as e:
            # Handle unexpected errors
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds()

            error_result = ScenarioResult(
                scenario_name=self.name,
                config_hash=self.config.compute_hash(),
                status=ScenarioStatus.ERROR,
                passed=False,
                metrics=self._metrics,
                artifacts={k: str(v) for k, v in self._artifacts.items()},
                start_time=self._start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                python_version=sys.version,
                torch_version=torch.__version__,
            )

            self._logger.error(
                "scenario_error",
                error=str(e),
                duration=duration,
            )

            return error_result

        finally:
            # Always teardown
            try:
                self.teardown()
            except Exception as teardown_error:
                self._logger.warning(
                    "teardown_error",
                    error=str(teardown_error),
                )

    def _evaluate_thresholds(self) -> dict[str, bool]:
        """Evaluate all configured thresholds against recorded metrics.

        Returns:
            Dict mapping threshold names to pass/fail.
        """
        results = {}
        for threshold in self.config.thresholds:
            if threshold.name in self._metrics:
                results[threshold.name] = threshold.evaluate(
                    self._metrics[threshold.name]
                )
            else:
                # Missing metric fails the threshold
                results[threshold.name] = False
                self._logger.warning(
                    "threshold_metric_missing",
                    threshold=threshold.name,
                )
        return results

    def _create_result(
        self,
        status: ScenarioStatus,
        threshold_results: dict[str, bool] | None = None,
    ) -> ScenarioResult:
        """Create a ScenarioResult from current state.

        Args:
            status: Execution status.
            threshold_results: Optional pre-computed threshold results.

        Returns:
            ScenarioResult instance.
        """
        import sys

        import torch

        end_time = datetime.now()
        assert self._start_time is not None
        duration = (end_time - self._start_time).total_seconds()

        if threshold_results is None:
            threshold_results = self._evaluate_thresholds()

        # Passed if all thresholds pass (or no thresholds defined)
        passed = all(threshold_results.values()) if threshold_results else True

        # Override status based on thresholds
        if status == ScenarioStatus.RUNNING:
            status = ScenarioStatus.PASSED if passed else ScenarioStatus.FAILED

        return ScenarioResult(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
            status=status,
            passed=passed,
            metrics=dict(self._metrics),
            threshold_results=threshold_results,
            artifacts={k: str(v) for k, v in self._artifacts.items()},
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=duration,
            device="cuda" if torch.cuda.is_available() else "cpu",
            python_version=sys.version,
            torch_version=torch.__version__,
        )
