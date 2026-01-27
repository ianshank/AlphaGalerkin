"""Structured logging and debugging utilities for validation.

This module provides validation-specific logging configuration
and debugging tools.

Design Principles:
    - Structured: All logs include contextual fields
    - Configurable: Log levels and formats configurable
    - Observable: Metrics and timing automatically captured
    - Debuggable: Rich debugging context for failures
"""

from __future__ import annotations

import functools
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


def configure_validation_logging(
    level: str = "INFO",
    json_output: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structured logging for validation.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_output: Whether to output JSON format.
        include_timestamp: Whether to include timestamps.
    """
    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if include_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level.upper(), structlog.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class ValidationLogger:
    """Logger wrapper for validation operations.

    Provides consistent logging with validation-specific context.

    Example:
        >>> logger = ValidationLogger("gpu_training")
        >>> logger.info("training_started", epoch=1, loss=0.5)
        >>> with logger.timed("forward_pass"):
        ...     model(x)
    """

    def __init__(
        self,
        name: str,
        **context: Any,
    ) -> None:
        """Initialize validation logger.

        Args:
            name: Logger name (typically validation step name).
            **context: Initial context fields.
        """
        self._logger = structlog.get_logger(name)
        self._context = context
        self._start_time = datetime.now()
        self._metrics: dict[str, list[float]] = {}

    def bind(self, **context: Any) -> ValidationLogger:
        """Create a new logger with additional context.

        Args:
            **context: Additional context fields.

        Returns:
            New logger with merged context.
        """
        new_logger = ValidationLogger(
            self._logger._context.get("name", "validation"),
            **{**self._context, **context},
        )
        new_logger._metrics = self._metrics.copy()
        return new_logger

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Internal logging method with context."""
        merged = {**self._context, **kwargs}
        getattr(self._logger, level)(event, **merged)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("error", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log("exception", event, exc_info=True, **kwargs)

    def metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        **tags: Any,
    ) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step/epoch number.
            **tags: Additional tags.
        """
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)

        self.debug(
            "metric",
            metric_name=name,
            value=value,
            step=step,
            **tags,
        )

    def progress(
        self,
        current: int,
        total: int,
        **kwargs: Any,
    ) -> None:
        """Log progress update.

        Args:
            current: Current step.
            total: Total steps.
            **kwargs: Additional fields.
        """
        pct = (current / total * 100) if total > 0 else 0
        self.info(
            "progress",
            current=current,
            total=total,
            percent=f"{pct:.1f}%",
            **kwargs,
        )

    @contextmanager
    def timed(
        self,
        operation: str,
        **context: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed.
            **context: Additional context.

        Yields:
            Dictionary that can be updated with additional timing data.

        Example:
            >>> with logger.timed("model_forward") as t:
            ...     output = model(x)
            ...     t["batch_size"] = x.shape[0]
        """
        start = time.perf_counter()
        timing_data: dict[str, Any] = {"operation": operation, **context}

        try:
            self.debug(f"{operation}_start", **context)
            yield timing_data
            elapsed = time.perf_counter() - start
            timing_data["elapsed_seconds"] = elapsed
            self.info(
                f"{operation}_complete",
                elapsed_seconds=f"{elapsed:.4f}",
                **timing_data,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            timing_data["elapsed_seconds"] = elapsed
            timing_data["error"] = str(e)
            self.error(
                f"{operation}_failed",
                elapsed_seconds=f"{elapsed:.4f}",
                **timing_data,
            )
            raise

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all logged metrics.

        Returns:
            Dictionary mapping metric names to statistics.
        """
        import statistics

        summary = {}
        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "count": len(values),
                }
                if len(values) > 1:
                    summary[name]["std"] = statistics.stdev(values)
        return summary

    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation.

        Returns:
            Elapsed time in seconds.
        """
        return (datetime.now() - self._start_time).total_seconds()


class DebugContext:
    """Context manager for debugging validation failures.

    Captures detailed context when exceptions occur.

    Example:
        >>> with DebugContext("model_inference") as ctx:
        ...     ctx.record("input_shape", input.shape)
        ...     output = model(input)
        ...     ctx.record("output_shape", output.shape)
    """

    def __init__(
        self,
        operation: str,
        logger: ValidationLogger | None = None,
        capture_memory: bool = True,
    ) -> None:
        """Initialize debug context.

        Args:
            operation: Name of the operation.
            logger: Optional logger for output.
            capture_memory: Whether to capture memory stats.
        """
        self.operation = operation
        self.logger = logger or ValidationLogger(operation)
        self.capture_memory = capture_memory
        self._data: dict[str, Any] = {}
        self._checkpoints: list[tuple[str, float, dict[str, Any]]] = []
        self._start_time = 0.0

    def record(self, key: str, value: Any) -> None:
        """Record a debug value.

        Args:
            key: Name of the value.
            value: Value to record.
        """
        self._data[key] = value

    def checkpoint(self, name: str, **data: Any) -> None:
        """Record a checkpoint with timing.

        Args:
            name: Checkpoint name.
            **data: Additional data at this checkpoint.
        """
        elapsed = time.perf_counter() - self._start_time
        self._checkpoints.append((name, elapsed, data))

    def __enter__(self) -> DebugContext:
        """Enter debug context."""
        self._start_time = time.perf_counter()

        if self.capture_memory:
            self._capture_memory_start()

        self.logger.debug(f"{self.operation}_debug_start")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit debug context, logging details on failure."""
        elapsed = time.perf_counter() - self._start_time

        if exc_type is not None:
            # Exception occurred - log detailed debug info
            debug_info = {
                "operation": self.operation,
                "elapsed_seconds": elapsed,
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_val),
                "traceback": traceback.format_exc(),
                "recorded_data": self._data,
                "checkpoints": [
                    {"name": name, "elapsed": t, "data": d}
                    for name, t, d in self._checkpoints
                ],
            }

            if self.capture_memory:
                debug_info["memory"] = self._capture_memory_end()

            self.logger.error(
                f"{self.operation}_debug_failure",
                **debug_info,
            )

            # Don't suppress the exception
            return False

        self.logger.debug(
            f"{self.operation}_debug_complete",
            elapsed_seconds=elapsed,
            data_keys=list(self._data.keys()),
            checkpoints=len(self._checkpoints),
        )

        return False

    def _capture_memory_start(self) -> None:
        """Capture memory state at start."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._data["gpu_memory_start_mb"] = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                )
        except ImportError:
            pass  # PyTorch not installed, skip GPU memory capture

    def _capture_memory_end(self) -> dict[str, float]:
        """Capture memory state at end.

        Returns:
            Dictionary with memory statistics.
        """
        memory = {}
        try:
            import torch

            if torch.cuda.is_available():
                memory["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                memory["gpu_peak_mb"] = (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )
                memory["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass  # PyTorch not installed, skip GPU memory stats

        return memory


def log_timing(
    logger: ValidationLogger | None = None,
) -> Callable[[F], F]:
    """Decorator to log function timing.

    Args:
        logger: Logger to use (creates new if None).

    Returns:
        Decorated function.

    Example:
        >>> @log_timing()
        ... def slow_function():
        ...     time.sleep(1)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or ValidationLogger(func.__name__)
            with log.timed(func.__name__):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def log_call(
    logger: ValidationLogger | None = None,
    log_args: bool = True,
    log_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function calls.

    Args:
        logger: Logger to use.
        log_args: Whether to log arguments.
        log_result: Whether to log return value.

    Returns:
        Decorated function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or ValidationLogger(func.__name__)

            call_info: dict[str, Any] = {"function": func.__name__}
            if log_args:
                call_info["args"] = str(args)[:100]
                call_info["kwargs"] = str(kwargs)[:100]

            log.debug("function_call", **call_info)

            result = func(*args, **kwargs)

            if log_result:
                log.debug("function_return", result=str(result)[:100])

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def create_validation_logger(
    name: str,
    config: Any = None,
    **context: Any,
) -> ValidationLogger:
    """Factory function to create a validation logger.

    Args:
        name: Logger name.
        config: Optional configuration object.
        **context: Initial context.

    Returns:
        Configured validation logger.
    """
    ctx = dict(context)

    if config is not None:
        if hasattr(config, "compute_hash"):
            ctx["config_hash"] = config.compute_hash()
        if hasattr(config, "name"):
            ctx["config_name"] = config.name

    return ValidationLogger(name, **ctx)
