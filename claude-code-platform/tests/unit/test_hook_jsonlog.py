"""Unit tests for JSON-lines stderr logging (tools.hook_runtime.jsonlog)."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest

from tools.hook_runtime import constants
from tools.hook_runtime.jsonlog import (
    JsonLineFormatter,
    configure_logging,
    resolve_level,
)


def emit_and_parse(logger: logging.Logger, stream: io.StringIO) -> list[dict[str, Any]]:
    stream.seek(0)
    return [json.loads(line) for line in stream.read().splitlines()]


class TestResolveLevel:
    def test_default_is_info(self) -> None:
        assert resolve_level(env={}) == logging.INFO

    def test_log_level_env(self) -> None:
        level = resolve_level(env={constants.ENV_LOG_LEVEL: "warning"})
        assert level == logging.WARNING

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_debug_env_wins(self, value: str) -> None:
        env = {constants.ENV_DEBUG: value, constants.ENV_LOG_LEVEL: "ERROR"}
        assert resolve_level(env=env) == logging.DEBUG

    def test_falsy_debug_ignored(self) -> None:
        env = {constants.ENV_DEBUG: "0", constants.ENV_LOG_LEVEL: "ERROR"}
        assert resolve_level(env=env) == logging.ERROR

    def test_invalid_level_falls_back_to_default(self) -> None:
        assert resolve_level(env={constants.ENV_LOG_LEVEL: "bogus"}) == logging.INFO


class TestJsonOutput:
    def test_lines_are_json_with_required_keys(self) -> None:
        stream = io.StringIO()
        logger = configure_logging("test.component", stream=stream, env={})
        logger.info("thing_happened", extra={"count": 3})
        (document,) = emit_and_parse(logger, stream)
        assert document["event"] == "thing_happened"
        assert document["level"] == "INFO"
        assert document["component"] == "test.component"
        assert document["count"] == 3
        assert "ts" in document

    def test_exception_serialized(self) -> None:
        stream = io.StringIO()
        logger = configure_logging("test.exc", stream=stream, env={})
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
        (document,) = emit_and_parse(logger, stream)
        assert "ValueError: boom" in document["exception"]

    def test_non_serializable_extra_stringified(self) -> None:
        stream = io.StringIO()
        logger = configure_logging("test.obj", stream=stream, env={})
        logger.info("evt", extra={"path": object()})
        (document,) = emit_and_parse(logger, stream)
        assert isinstance(document["path"], str)

    def test_reconfiguration_is_idempotent(self) -> None:
        stream = io.StringIO()
        configure_logging("test.idem", stream=stream, env={})
        logger = configure_logging("test.idem", stream=stream, env={})
        logger.info("once")
        assert len(emit_and_parse(logger, stream)) == 1

    def test_level_filtering_applies(self) -> None:
        stream = io.StringIO()
        logger = configure_logging(
            "test.filter", stream=stream, env={constants.ENV_LOG_LEVEL: "WARNING"}
        )
        logger.info("hidden")
        logger.warning("shown")
        documents = emit_and_parse(logger, stream)
        assert [d["event"] for d in documents] == ["shown"]

    def test_formatter_direct(self) -> None:
        record = logging.makeLogRecord(
            {"name": "x", "levelname": "INFO", "msg": "m", "created": 0.0}
        )
        document = json.loads(JsonLineFormatter().format(record))
        assert document["event"] == "m"
        assert document["ts"].startswith("1970-01-01")
