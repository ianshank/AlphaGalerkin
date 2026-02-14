"""Structured logging utilities for AlphaGalerkin.

Provides:
- configure_logging: One-time structlog configuration.
- get_logger: Named logger with initial context.
- log_context: Temporary context via structlog.contextvars.
- log_duration: Context manager that times a block and logs duration.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog
from structlog.contextvars import bind_contextvars, unbind_contextvars

if TYPE_CHECKING:
    from collections.abc import Generator


def configure_logging(
    level: str = "INFO",
    format: str = "console",  # noqa: A002
) -> None:
    """Configure structured logging for the application.

    Should be called once at application startup before any logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format - ``"console"`` for colored dev output,
            ``"json"`` for machine-readable JSON lines.

    """
    import logging as stdlib_logging
    import sys

    stdlib_logging.basicConfig(
        format="%(message)s",
        level=getattr(stdlib_logging, level.upper()),
        stream=sys.stderr,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True),
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(
    name: str,
    **initial_context: Any,
) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger with optional initial context.

    Args:
        name: Logger name (typically ``__name__`` or a subsystem id).
        **initial_context: Key-value pairs bound to every log entry
            emitted by this logger instance.

    Returns:
        A bound structlog logger ready for use.

    """
    log: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        log = log.bind(**initial_context)
    return log


@contextmanager
def log_context(
    **kwargs: Any,
) -> Generator[None, None, None]:
    """Temporarily bind context variables to all loggers.

    Uses ``structlog.contextvars`` so the context is visible to
    *every* logger within the ``with`` block, including loggers in
    called functions.

    Args:
        **kwargs: Context key-value pairs to bind.

    Yields:
        Nothing. Context is automatically unbound on exit.

    Example::

        with log_context(request_id="abc-123", user="alice"):
            logger.info("handling_request")  # includes request_id

    """
    bind_contextvars(**kwargs)
    try:
        yield
    finally:
        unbind_contextvars(*kwargs.keys())


@contextmanager
def log_duration(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    **extra: Any,
) -> Generator[dict[str, float], None, None]:
    """Time a block and log the duration on exit.

    Args:
        logger: Logger instance to emit the timing event on.
        event: Event name written to the log (e.g. ``"train_step"``).
        **extra: Additional key-value pairs included in the log entry.

    Yields:
        A mutable dict that will contain ``duration_seconds`` after
        the block completes.

    Example::

        with log_duration(logger, "forward_pass", batch=42) as t:
            output = model(x)
        print(t["duration_seconds"])

    """
    timing: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed = time.perf_counter() - start
        timing["duration_seconds"] = elapsed
        logger.info(
            event,
            duration_seconds=round(elapsed, 6),
            **extra,
        )
