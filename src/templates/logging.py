"""Structured logging utilities for AlphaGalerkin modules.

This module provides:
- Configurable structured logging with structlog
- Context-aware logger wrapper classes
- Timing decorators and context managers
- Debug context managers with memory tracking

Example:
    from src.templates.logging import (
        configure_module_logging,
        create_logger_class,
        log_timing,
        DebugContext,
    )

    # Configure logging at startup
    configure_module_logging(level="INFO", json_format=False)

    # Create module-specific logger class
    MyLogger = create_logger_class("MyModule")
    logger = MyLogger("my_component", run_id="abc123")

    # Use timing decorator
    @log_timing()
    def expensive_function():
        pass

    # Use debug context
    with DebugContext("processing", capture_memory=True) as ctx:
        result = process_data()
        ctx.checkpoint("halfway", items_processed=50)

"""

from __future__ import annotations

import functools
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

import structlog

P = ParamSpec("P")
R = TypeVar("R")


def configure_module_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_timestamp: bool = True,
    include_caller: bool = False,
) -> None:
    """Configure structured logging for the application.

    Should be called once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output JSON; else colored console.
        include_timestamp: If True, include ISO timestamp.
        include_caller: If True, include caller info (file:line).

    """
    import logging

    # Set stdlib logging level
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        stream=sys.stderr,
    )

    # Build processor chain
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

    if include_caller:
        processors.append(structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ))

    # Add renderer based on format
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


class BaseModuleLogger:
    """Context-aware logger wrapper for module-specific logging.

    Features:
    - Automatic context binding (module name, run ID, etc.)
    - Convenience methods for metrics and timing
    - Child logger creation with additional context

    Subclasses can add module-specific methods.
    """

    _module_name: str = "base"

    def __init__(
        self,
        component: str,
        run_id: str | None = None,
        **context: Any,
    ) -> None:
        """Initialize logger with context.

        Args:
            component: Component name within the module.
            run_id: Optional unique identifier for this run.
            **context: Additional context to bind.

        """
        self._base_logger = structlog.get_logger(self._module_name)
        self._context: dict[str, Any] = {
            "module": self._module_name,
            "component": component,
        }
        if run_id:
            self._context["run_id"] = run_id
        self._context.update(context)
        self._logger = self._base_logger.bind(**self._context)
        self._start_time: float | None = None

    def bind(self, **context: Any) -> BaseModuleLogger:
        """Create a new logger with additional context.

        Args:
            **context: Additional context to bind.

        Returns:
            New logger instance with merged context.

        """
        new_logger = self.__class__.__new__(self.__class__)
        new_logger._base_logger = self._base_logger
        new_logger._context = {**self._context, **context}
        new_logger._logger = self._base_logger.bind(**new_logger._context)
        new_logger._start_time = self._start_time
        return new_logger

    def debug(self, event: str, **kw: Any) -> None:
        """Log at DEBUG level."""
        self._logger.debug(event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        """Log at INFO level."""
        self._logger.info(event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        """Log at WARNING level."""
        self._logger.warning(event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        """Log at ERROR level."""
        self._logger.error(event, **kw)

    def exception(self, event: str, **kw: Any) -> None:
        """Log at ERROR level with exception info."""
        self._logger.exception(event, **kw)

    def critical(self, event: str, **kw: Any) -> None:
        """Log at CRITICAL level."""
        self._logger.critical(event, **kw)

    def metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        **tags: Any,
    ) -> None:
        """Log a metric with optional tags.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step/iteration number.
            **tags: Additional tags for the metric.

        """
        extra: dict[str, Any] = {
            "metric_name": name,
            "metric_value": value,
            **tags,
        }
        if step is not None:
            extra["step"] = step
        self._logger.info("metric", **extra)

    def progress(
        self,
        current: int,
        total: int,
        message: str = "",
        **extra: Any,
    ) -> None:
        """Log progress update.

        Args:
            current: Current progress count.
            total: Total count.
            message: Optional progress message.
            **extra: Additional context.

        """
        percentage = (current / total * 100) if total > 0 else 0.0
        self._logger.info(
            "progress",
            current=current,
            total=total,
            percentage=round(percentage, 1),
            message=message,
            **extra,
        )

    def start_timer(self) -> None:
        """Start an internal timer."""
        self._start_time = time.perf_counter()

    def log_elapsed(self, event: str, **kw: Any) -> float:
        """Log elapsed time since start_timer().

        Args:
            event: Event name.
            **kw: Additional context.

        Returns:
            Elapsed time in seconds.

        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
            elapsed = 0.0
        else:
            elapsed = time.perf_counter() - self._start_time
        self._logger.info(event, elapsed_seconds=elapsed, **kw)
        return elapsed

    @contextmanager
    def timed(
        self,
        operation: str,
        log_start: bool = True,
    ) -> Generator[dict[str, float], None, None]:
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed.
            log_start: If True, log when operation starts.

        Yields:
            Dictionary that will be populated with timing info.

        Example:
            with logger.timed("model_inference") as timing:
                result = model(input)
            print(f"Took {timing['duration_seconds']:.2f}s")

        """
        timing: dict[str, float] = {}
        start_time = time.perf_counter()

        if log_start:
            self._logger.debug(f"{operation}_start")

        try:
            yield timing
        except Exception:
            timing["duration_seconds"] = time.perf_counter() - start_time
            self._logger.error(
                f"{operation}_error",
                duration_seconds=timing["duration_seconds"],
            )
            raise
        else:
            timing["duration_seconds"] = time.perf_counter() - start_time
            self._logger.debug(
                f"{operation}_complete",
                duration_seconds=timing["duration_seconds"],
            )


def create_logger_class(module_name: str) -> type[BaseModuleLogger]:
    """Factory function to create a module-specific logger class.

    Args:
        module_name: Name of the module (used in log context).

    Returns:
        Logger class with the module name bound.

    Example:
        MyLogger = create_logger_class("MyModule")
        logger = MyLogger("component_name", run_id="abc123")
        logger.info("started")

    """
    return type(
        f"{module_name}Logger",
        (BaseModuleLogger,),
        {"_module_name": module_name},
    )


def log_timing(
    logger: structlog.stdlib.BoundLogger | None = None,
    level: str = "debug",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use; if None, creates one from function module.
        level: Log level (debug, info, warning, error).

    Returns:
        Decorator function.

    Example:
        @log_timing()
        def expensive_operation():
            time.sleep(1)

        @log_timing(level="info")
        def important_operation():
            pass

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal logger
        if logger is None:
            logger = structlog.get_logger(func.__module__)

        log_method = getattr(logger, level)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                log_method(
                    "function_timing",
                    function=func.__name__,
                    module=func.__module__,
                    duration_seconds=round(duration, 6),
                )

        return wrapper

    return decorator


def log_call(
    logger: structlog.stdlib.BoundLogger | None = None,
    log_args: bool = False,
    log_result: bool = False,
    max_str_length: int = 200,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function calls with optional args/results.

    Args:
        logger: Logger to use; if None, creates one from function module.
        log_args: If True, log function arguments.
        log_result: If True, log function result.
        max_str_length: Maximum length for stringified args/results.

    Returns:
        Decorator function.

    Example:
        @log_call(log_args=True, log_result=True)
        def process_data(data: list) -> dict:
            return {"count": len(data)}

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal logger
        if logger is None:
            logger = structlog.get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_info: dict[str, Any] = {
                "function": func.__name__,
                "module": func.__module__,
            }

            if log_args:
                args_str = str(args)[:max_str_length]
                kwargs_str = str(kwargs)[:max_str_length]
                call_info["args"] = args_str
                call_info["kwargs"] = kwargs_str

            logger.debug("function_call", **call_info)  # type: ignore[union-attr]

            result = func(*args, **kwargs)

            if log_result:
                result_str = str(result)[:max_str_length]
                logger.debug(  # type: ignore[union-attr]
                    "function_result",
                    function=func.__name__,
                    result=result_str,
                )

            return result

        return wrapper

    return decorator


class DebugContext:
    """Context manager for detailed debug logging with optional memory tracking.

    Features:
    - Automatic entry/exit logging
    - Duration tracking
    - GPU memory delta tracking (if CUDA available)
    - Intermediate checkpoints
    - Exception handling

    Example:
        with DebugContext("training_step", capture_memory=True) as ctx:
            loss = model.train_step(batch)
            ctx.checkpoint("forward_done", loss=loss.item())
            optimizer.step()
            ctx.checkpoint("backward_done")

    """

    def __init__(
        self,
        name: str,
        logger: BaseModuleLogger | None = None,
        capture_memory: bool = False,
        log_level: str = "debug",
    ) -> None:
        """Initialize debug context.

        Args:
            name: Name of the operation being debugged.
            logger: Logger to use; if None, creates a basic one.
            capture_memory: If True, track GPU memory usage.
            log_level: Log level for debug messages.

        """
        self.name = name
        self.capture_memory = capture_memory
        self.log_level = log_level

        if logger is None:
            self._logger = structlog.get_logger(__name__)
        else:
            self._logger = logger._logger

        self._log_method = getattr(self._logger, log_level)
        self._start_time: float = 0.0
        self._start_memory: int = 0
        self._checkpoints: list[dict[str, Any]] = []

    def __enter__(self) -> DebugContext:
        """Enter the debug context."""
        self._start_time = time.perf_counter()
        self._checkpoints = []

        if self.capture_memory:
            self._start_memory = self._get_gpu_memory()

        self._log_method(
            f"{self.name}_enter",
            capture_memory=self.capture_memory,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the debug context."""
        duration = time.perf_counter() - self._start_time

        context: dict[str, Any] = {
            "duration_seconds": round(duration, 6),
            "checkpoint_count": len(self._checkpoints),
        }

        if self.capture_memory:
            end_memory = self._get_gpu_memory()
            memory_delta = end_memory - self._start_memory
            context["memory_delta_mb"] = round(memory_delta / (1024 * 1024), 2)
            context["memory_final_mb"] = round(end_memory / (1024 * 1024), 2)

        if exc_type is not None:
            context["exception_type"] = exc_type.__name__
            context["exception_msg"] = str(exc_val)[:200]
            self._logger.error(f"{self.name}_error", **context)
        else:
            self._log_method(f"{self.name}_exit", **context)

    def checkpoint(self, label: str, **data: Any) -> float:
        """Log an intermediate checkpoint.

        Args:
            label: Label for this checkpoint.
            **data: Additional data to log.

        Returns:
            Elapsed time since context entry.

        """
        elapsed = time.perf_counter() - self._start_time

        checkpoint_info: dict[str, Any] = {
            "label": label,
            "elapsed_seconds": round(elapsed, 6),
            "checkpoint_index": len(self._checkpoints),
            **data,
        }

        if self.capture_memory:
            current_memory = self._get_gpu_memory()
            checkpoint_info["memory_mb"] = round(current_memory / (1024 * 1024), 2)

        self._checkpoints.append(checkpoint_info)
        self._log_method(f"{self.name}_checkpoint", **checkpoint_info)

        return elapsed

    def _get_gpu_memory(self) -> int:
        """Get current GPU memory usage in bytes."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                return torch.cuda.memory_allocated()
        except ImportError:
            pass
        return 0

    @property
    def elapsed(self) -> float:
        """Get elapsed time since context entry."""
        return time.perf_counter() - self._start_time

    @property
    def checkpoints(self) -> list[dict[str, Any]]:
        """Get list of recorded checkpoints."""
        return list(self._checkpoints)
