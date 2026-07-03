"""JSON-lines logging to stderr for hook scripts (stdlib-only).

Every hook emits structured JSON log lines to stderr and honors
``CCP_LOG_LEVEL`` / ``CCP_DEBUG`` without code changes, per the plugin
contract. stdout is left untouched — Claude Code interprets hook stdout.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from os import environ
from typing import Any, TextIO

from . import constants

#: Attributes present on every LogRecord; anything else was passed via
#: ``extra=`` and is emitted as a structured field.
_BASE_RECORD_KEYS: frozenset[str] = frozenset(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonLineFormatter(logging.Formatter):
    """Format records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        document: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _BASE_RECORD_KEYS and not key.startswith("_"):
                document[key] = value
        if record.exc_info:
            document["exception"] = self.formatException(record.exc_info)
        return json.dumps(document, default=str, ensure_ascii=False)


def resolve_level(env: Mapping[str, str] | None = None) -> int:
    """Resolve the log level: CCP_DEBUG (truthy) wins, then CCP_LOG_LEVEL."""
    source = environ if env is None else env
    if source.get(constants.ENV_DEBUG, "").strip().lower() in constants.TRUTHY_VALUES:
        return logging.DEBUG
    name = source.get(constants.ENV_LOG_LEVEL, constants.DEFAULT_LOG_LEVEL)
    mapping = logging.getLevelNamesMapping()
    return mapping.get(name.strip().upper(), mapping[constants.DEFAULT_LOG_LEVEL])


def configure_logging(
    component: str,
    *,
    stream: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> logging.Logger:
    """Return a logger writing JSON lines to ``stream`` (default stderr).

    Reconfiguration is idempotent: handlers are replaced, not appended, so
    repeated calls (e.g. in tests) never duplicate output.
    """
    logger = logging.getLogger(component)
    logger.setLevel(resolve_level(env))
    logger.propagate = False
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    logger.handlers = [handler]
    return logger
