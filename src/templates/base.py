"""Base classes for executable components in AlphaGalerkin modules.

This module provides:
- ExecutionStatus enum for tracking execution state
- ExecutionResult dataclass for capturing results
- BaseExecutable abstract class for all runnable components

Example:
    from src.templates.base import BaseExecutable, ExecutionResult, ExecutionStatus

    class MyProcessor(BaseExecutable):
        def execute(self) -> ExecutionResult:
            try:
                result = self._process_data()
                return self._create_result(
                    status=ExecutionStatus.COMPLETED,
                    metrics={"accuracy": result.accuracy},
                )
            except Exception as e:
                return self._create_result(
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                )

"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from src.templates.config import BaseModuleConfig
from src.templates.logging import BaseModuleLogger, create_logger_class

T = TypeVar("T", bound=BaseModuleConfig)


class ExecutionStatus(str, Enum):
    """Status of an execution run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) status."""
        return self in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.SKIPPED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT,
        }

    def is_success(self) -> bool:
        """Check if this represents successful completion."""
        return self == ExecutionStatus.COMPLETED


@dataclass
class ExecutionResult:
    """Result of an execution run.

    Captures all information about an execution including:
    - Status and timing
    - Metrics collected during execution
    - Artifacts produced
    - Error information if failed
    """

    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""

    # Status
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: str | None = None
    error_traceback: str | None = None

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Results
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Configuration hash for reproducibility
    config_hash: str = ""

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status.is_success()

    def is_terminal(self) -> bool:
        """Check if execution is in a terminal state."""
        return self.status.is_terminal()

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a metric value by name.

        Args:
            name: Metric name.
            default: Default value if metric not found.

        Returns:
            Metric value or default.

        """
        return self.metrics.get(name, default)

    def add_metric(self, name: str, value: float) -> None:
        """Add or update a metric.

        Args:
            name: Metric name.
            value: Metric value.

        """
        self.metrics[name] = value

    def add_artifact(self, name: str, value: Any) -> None:
        """Add an artifact.

        Args:
            name: Artifact name.
            value: Artifact value.

        """
        self.artifacts[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.

        """
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status.value,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "metadata": self.metadata,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ExecutionResult instance.

        """
        return cls(
            run_id=data.get("run_id", ""),
            name=data.get("name", ""),
            status=ExecutionStatus(data.get("status", "pending")),
            error=data.get("error"),
            start_time=(
                datetime.fromisoformat(data["start_time"])
                if data.get("start_time")
                else None
            ),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            duration_seconds=data.get("duration_seconds", 0.0),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            metadata=data.get("metadata", {}),
            config_hash=data.get("config_hash", ""),
        )


class BaseExecutable(ABC, Generic[T]):
    """Abstract base class for executable components.

    Provides common infrastructure for:
    - Configuration management
    - Logging
    - Result creation
    - Execution timing

    Subclasses must implement the execute() method.

    Example:
        class MyProcessor(BaseExecutable[MyProcessorConfig]):
            def execute(self) -> ExecutionResult:
                with self.logger.timed("processing"):
                    data = self._load_data()
                    result = self._process(data)

                return self._create_result(
                    status=ExecutionStatus.COMPLETED,
                    metrics={"items_processed": len(data)},
                )

    """

    # Class-level configuration
    _executable_name: str = "base"
    _logger_class: type[BaseModuleLogger] = BaseModuleLogger

    def __init__(
        self,
        config: T,
        run_id: str | None = None,
    ) -> None:
        """Initialize the executable.

        Args:
            config: Configuration for this executable.
            run_id: Optional unique identifier for this run.

        """
        self.config = config
        self.run_id = run_id or str(uuid.uuid4())[:8]

        # Create logger with context
        self.logger = self._logger_class(
            component=self._executable_name,
            run_id=self.run_id,
            config_hash=config.compute_hash(),
        )

        # Internal state
        self._start_time: float | None = None
        self._status = ExecutionStatus.PENDING

    @abstractmethod
    def execute(self) -> ExecutionResult:
        """Execute the component logic.

        Subclasses must implement this method to perform their work.

        Returns:
            ExecutionResult with status, metrics, and artifacts.

        """
        raise NotImplementedError

    def run(self) -> ExecutionResult:
        """Run the executable with timing and error handling.

        This is the main entry point that wraps execute() with:
        - Status tracking
        - Timing
        - Error handling
        - Logging

        Returns:
            ExecutionResult from execute() or error result.

        """
        self._status = ExecutionStatus.RUNNING
        self._start_time = time.perf_counter()
        start_datetime = datetime.now(timezone.utc)

        self.logger.info(
            "execution_started",
            executable=self._executable_name,
            config_hash=self.config.compute_hash(),
        )

        try:
            result = self.execute()
        except Exception as e:
            import traceback

            duration = time.perf_counter() - self._start_time
            self._status = ExecutionStatus.FAILED

            self.logger.exception(
                "execution_failed",
                error=str(e),
                duration_seconds=duration,
            )

            result = ExecutionResult(
                run_id=self.run_id,
                name=self._executable_name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                error_traceback=traceback.format_exc(),
                start_time=start_datetime,
                end_time=datetime.now(timezone.utc),
                duration_seconds=duration,
                config_hash=self.config.compute_hash(),
            )
        else:
            # Update result with timing if not already set
            duration = time.perf_counter() - self._start_time
            self._status = result.status

            if result.start_time is None:
                result.start_time = start_datetime
            if result.end_time is None:
                result.end_time = datetime.now(timezone.utc)
            if result.duration_seconds == 0.0:
                result.duration_seconds = duration
            if not result.config_hash:
                result.config_hash = self.config.compute_hash()
            if not result.run_id:
                result.run_id = self.run_id
            if not result.name:
                result.name = self._executable_name

            self.logger.info(
                "execution_completed",
                status=result.status.value,
                duration_seconds=result.duration_seconds,
                metric_count=len(result.metrics),
            )

        return result

    def _create_result(
        self,
        status: ExecutionStatus,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ExecutionResult:
        """Create an ExecutionResult with common fields filled in.

        Args:
            status: Execution status.
            metrics: Optional metrics dictionary.
            artifacts: Optional artifacts dictionary.
            metadata: Optional metadata dictionary.
            error: Optional error message.

        Returns:
            ExecutionResult with provided and computed fields.

        """
        duration = 0.0
        if self._start_time is not None:
            duration = time.perf_counter() - self._start_time

        return ExecutionResult(
            run_id=self.run_id,
            name=self._executable_name,
            status=status,
            error=error,
            duration_seconds=duration,
            metrics=metrics or {},
            artifacts=artifacts or {},
            metadata=metadata or {},
            config_hash=self.config.compute_hash(),
        )

    @property
    def status(self) -> ExecutionStatus:
        """Current execution status."""
        return self._status

    @property
    def elapsed(self) -> float:
        """Elapsed time since execution started."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def validate_config(self) -> bool:
        """Validate the configuration.

        Override in subclasses for custom validation.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is invalid.

        """
        # Pydantic validation happens at construction time
        # This method is for additional custom validation
        return True


def create_executable_class(
    name: str,
    config_class: type[T],
    module_name: str | None = None,
) -> type[BaseExecutable[T]]:
    """Factory function to create an executable class with proper typing.

    Args:
        name: Name for the executable class.
        config_class: Configuration class to use.
        module_name: Optional module name for logging.

    Returns:
        New executable class.

    Example:
        MyExecutable = create_executable_class(
            "MyExecutable",
            MyConfig,
            module_name="my_module",
        )

        class MyProcessor(MyExecutable):
            def execute(self) -> ExecutionResult:
                ...

    """
    logger_cls = create_logger_class(module_name or name)

    return type(
        name,
        (BaseExecutable,),
        {
            "_executable_name": name.lower(),
            "_logger_class": logger_cls,
            "__annotations__": {"config": config_class},
        },
    )
