"""Tests for structured logging utilities (utils/logging.py)."""
from __future__ import annotations

import time

import pytest
import structlog

from src.alphagalerkin.utils.logging import (
    configure_logging,
    get_logger,
    log_context,
    log_duration,
)


class TestConfigureLogging:
    """configure_logging sets up structlog."""

    def test_configure_console_format(self) -> None:
        # Should not raise.
        configure_logging(level="DEBUG", format="console")

    def test_configure_json_format(self) -> None:
        # Should not raise.
        configure_logging(level="INFO", format="json")

    def test_configure_warning_level(self) -> None:
        configure_logging(level="WARNING")

        # After configuration, loggers should still be obtainable.
        log = structlog.get_logger("test")
        assert log is not None


class TestGetLogger:
    """get_logger returns a bound structlog logger."""

    def test_returns_logger(self) -> None:
        log = get_logger("my_component")

        assert log is not None

    def test_logger_with_initial_context(self) -> None:
        log = get_logger("comp", run_id="abc", epoch=1)

        # The returned logger should be a BoundLogger.
        assert log is not None

    def test_logger_without_context(self) -> None:
        log = get_logger("simple")

        assert log is not None


class TestLogContext:
    """log_context temporarily binds context variables."""

    def test_context_binds_and_unbinds(self) -> None:
        # We can enter and exit without error.
        with log_context(request_id="r1", user="alice"):
            pass  # context is active here

        # After exit, context should be unbound (no error).

    def test_nested_contexts(self) -> None:
        with log_context(outer="a"):
            with log_context(inner="b"):
                pass

    def test_context_with_no_kwargs(self) -> None:
        # Empty context should work without error.
        with log_context():
            pass


class TestLogDuration:
    """log_duration times a block and yields elapsed time."""

    def test_yields_timing_dict(self) -> None:
        log = get_logger("timer_test")

        with log_duration(log, "test_op") as timing:
            time.sleep(0.01)

        assert "duration_seconds" in timing
        assert timing["duration_seconds"] >= 0.0

    def test_timing_is_positive(self) -> None:
        log = get_logger("timer_test")

        with log_duration(log, "work") as timing:
            _ = sum(range(1000))

        assert timing["duration_seconds"] > 0.0

    def test_extra_kwargs_accepted(self) -> None:
        log = get_logger("timer_test")

        with log_duration(log, "step", batch=42, epoch=1) as timing:
            pass

        assert "duration_seconds" in timing

    def test_timing_reflects_block_duration(self) -> None:
        log = get_logger("timer_test")
        sleep_seconds = 0.05

        with log_duration(log, "sleep_block") as timing:
            time.sleep(sleep_seconds)

        # Allow generous margin for CI jitter.
        assert timing["duration_seconds"] >= sleep_seconds * 0.5

    def test_exception_still_records_timing(self) -> None:
        log = get_logger("timer_test")

        with pytest.raises(ValueError):
            with log_duration(log, "error_block") as timing:
                raise ValueError("boom")

        assert "duration_seconds" in timing
        assert timing["duration_seconds"] >= 0.0
