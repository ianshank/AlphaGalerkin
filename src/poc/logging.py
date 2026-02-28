"""Structured logging utilities for PoC scenarios.

This module provides:
    - Scenario-aware logging with correlation IDs
    - Structured log formatting
    - Debug utilities for tracing execution
    - Performance profiling helpers
"""

from __future__ import annotations

import functools
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

# Type variables for decorators
P = ParamSpec("P")
R = TypeVar("R")


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structured logging for the PoC framework.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: Use JSON output format (for production).
        include_timestamp: Include timestamps in output.

    """
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        stream=sys.stderr,
    )


class ScenarioLogger:
    """Logger wrapper with scenario context.

    Provides convenience methods for scenario-specific logging
    with automatic context binding.
    """

    def __init__(
        self,
        scenario_name: str,
        run_id: str | None = None,
        **context: Any,
    ) -> None:
        """Initialize scenario logger.

        Args:
            scenario_name: Name of the scenario.
            run_id: Optional run identifier.
            **context: Additional context to bind.

        """
        self._base_logger = structlog.get_logger(__name__)
        self._context = {
            "scenario": scenario_name,
            **({"run_id": run_id} if run_id else {}),
            **context,
        }
        self._logger = self._base_logger.bind(**self._context)

    def bind(self, **context: Any) -> ScenarioLogger:
        """Create a new logger with additional context.

        Args:
            **context: Additional context to bind.

        Returns:
            New ScenarioLogger with merged context.

        """
        new_context = {**self._context, **context}
        new_logger = ScenarioLogger.__new__(ScenarioLogger)
        new_logger._base_logger = self._base_logger
        new_logger._context = new_context
        new_logger._logger = self._base_logger.bind(**new_context)
        return new_logger

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(event, **kwargs)

    @contextmanager
    def timed(self, operation: str) -> Generator[dict[str, float], None, None]:
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed.

        Yields:
            Dict that will be populated with timing info.

        Example:
            with logger.timed("model_inference") as timing:
                result = model(input)
            # timing["duration_seconds"] is now set

        """
        timing: dict[str, float] = {}
        start_time = time.perf_counter()

        self._logger.debug(f"{operation}_start")

        try:
            yield timing
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            timing["duration_seconds"] = duration

            self._logger.debug(
                f"{operation}_complete",
                duration_seconds=duration,
            )

    def metric(self, name: str, value: float, **tags: Any) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            **tags: Additional tags/dimensions.

        """
        self._logger.info(
            "metric",
            metric_name=name,
            metric_value=value,
            **tags,
        )

    def progress(
        self,
        current: int,
        total: int,
        operation: str = "progress",
    ) -> None:
        """Log progress update.

        Args:
            current: Current step.
            total: Total steps.
            operation: Operation name.

        """
        pct = (current / total * 100) if total > 0 else 0
        self._logger.debug(
            operation,
            current=current,
            total=total,
            percent=f"{pct:.1f}%",
        )


def get_scenario_logger(
    scenario_name: str,
    run_id: str | None = None,
    **context: Any,
) -> ScenarioLogger:
    """Factory function to create a scenario logger.

    Args:
        scenario_name: Name of the scenario.
        run_id: Optional run identifier.
        **context: Additional context to bind.

    Returns:
        Configured ScenarioLogger.

    """
    return ScenarioLogger(scenario_name, run_id, **context)


def log_timing(
    logger: structlog.stdlib.BoundLogger | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function execution time.

    Args:
        logger: Optional logger to use. Uses module logger if None.

    Returns:
        Decorator function.

    Example:
        @log_timing()
        def slow_operation():
            time.sleep(1)

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal logger
        if logger is None:
            logger = structlog.get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                logger.debug(
                    "function_timing",
                    function=func.__name__,
                    duration_seconds=duration,
                )

        return wrapper

    return decorator


def log_call(
    logger: structlog.stdlib.BoundLogger | None = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function calls.

    Args:
        logger: Optional logger to use.
        log_args: Include arguments in log.
        log_result: Include result in log.

    Returns:
        Decorator function.

    Example:
        @log_call(log_args=True)
        def my_function(x, y):
            return x + y

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal logger
        if logger is None:
            logger = structlog.get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_info: dict[str, Any] = {"function": func.__name__}

            if log_args:
                call_info["args"] = str(args)[:200]  # Truncate for safety
                call_info["kwargs"] = str(kwargs)[:200]

            logger.debug("function_call", **call_info)

            result = func(*args, **kwargs)

            if log_result:
                logger.debug(
                    "function_result",
                    function=func.__name__,
                    result=str(result)[:200],
                )

            return result

        return wrapper

    return decorator


class DebugContext:
    """Context manager for debug-level verbose logging.

    Temporarily enables verbose debugging within a code block.
    """

    def __init__(
        self,
        name: str,
        logger: ScenarioLogger | None = None,
        capture_memory: bool = False,
    ) -> None:
        """Initialize debug context.

        Args:
            name: Context name for identification.
            logger: Optional scenario logger.
            capture_memory: Track memory usage (requires torch).

        """
        self.name = name
        self.logger = logger or ScenarioLogger(name)
        self.capture_memory = capture_memory

        self._start_time: float = 0
        self._start_memory: int = 0

    def __enter__(self) -> DebugContext:
        """Enter debug context."""
        self._start_time = time.perf_counter()

        if self.capture_memory:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    self._start_memory = torch.cuda.memory_allocated()
            except ImportError:
                pass

        self.logger.debug(f"{self.name}_enter")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit debug context."""
        duration = time.perf_counter() - self._start_time

        context: dict[str, Any] = {"duration_seconds": duration}

        if self.capture_memory:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    end_memory = torch.cuda.memory_allocated()
                    context["memory_delta_mb"] = (end_memory - self._start_memory) / 1024 / 1024
            except ImportError:
                pass

        if exc_type is not None:
            context["exception"] = str(exc_type.__name__)

        self.logger.debug(f"{self.name}_exit", **context)

    def checkpoint(self, label: str, **data: Any) -> None:
        """Log a checkpoint within the debug context.

        Args:
            label: Checkpoint label.
            **data: Additional data to log.

        """
        elapsed = time.perf_counter() - self._start_time
        self.logger.debug(
            f"{self.name}_checkpoint",
            label=label,
            elapsed_seconds=elapsed,
            **data,
        )
