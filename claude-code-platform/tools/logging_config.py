"""Shared structlog configuration for the dev-side CLI tools.

One configuration path for ``tools.validate``, ``tools.sync_runtime``,
and ``tools.sync_catalog`` instead of three copies. The stderr stream is
resolved AT LOG-CALL TIME via a proxy: capturing ``sys.stderr`` at
configure time (as ``PrintLoggerFactory(sys.stderr)`` does) pins the
logger to whatever stream happened to be installed then — under pytest
that is a per-test capture object that gets closed, turning later log
calls into ``ValueError: I/O operation on closed file``.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO, cast

import structlog


class _StderrAtCallTime:
    """File-like proxy that always writes to the CURRENT ``sys.stderr``."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def configure_tool_logging() -> None:
    """Configure structlog for JSON-lines-to-stderr tool output."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        # print() only needs write/flush; the cast satisfies the TextIO
        # annotation without implementing the full unused interface.
        logger_factory=structlog.PrintLoggerFactory(cast(TextIO, _StderrAtCallTime())),
    )


def get_tool_logger(name: str) -> Any:
    """A structlog logger for a tool module (config applied at call time)."""
    return structlog.get_logger(name)
