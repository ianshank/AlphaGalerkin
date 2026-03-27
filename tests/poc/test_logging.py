"""Tests for PoC structured logging utilities.

Validates:
    - configure_logging with different levels and formats
    - ScenarioLogger context binding and log methods
    - log_timing decorator measures execution time
    - Metric logging with tags
    - DebugContext timing and checkpoints
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import structlog

from src.poc.logging import (
    DebugContext,
    ScenarioLogger,
    configure_logging,
    get_scenario_logger,
    log_call,
    log_timing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["DEBUG", "INFO", "WARNING", "ERROR"])
def log_level(request: pytest.FixtureRequest) -> str:
    """Parametrize over all supported log levels."""
    return request.param


@pytest.fixture()
def scenario_name() -> str:
    return "test_scenario"


@pytest.fixture()
def run_id() -> str:
    return "run_abc123"


@pytest.fixture()
def scenario_logger(scenario_name: str, run_id: str) -> ScenarioLogger:
    """Create a ScenarioLogger with deterministic identifiers."""
    return ScenarioLogger(scenario_name, run_id=run_id)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_level(self, log_level: str) -> None:
        """configure_logging should not raise for any valid level."""
        configure_logging(level=log_level)

    def test_configure_json_format(self) -> None:
        """JSON format flag should configure without error."""
        configure_logging(level="INFO", json_format=True)

    def test_configure_without_timestamp(self) -> None:
        """Disabling timestamp should configure without error."""
        configure_logging(level="INFO", include_timestamp=False)

    def test_configure_all_options(self) -> None:
        """All options together should configure without error."""
        configure_logging(level="DEBUG", json_format=True, include_timestamp=False)


# ---------------------------------------------------------------------------
# ScenarioLogger
# ---------------------------------------------------------------------------


class TestScenarioLogger:
    """Tests for ScenarioLogger context binding and log methods."""

    def test_init_stores_context(self, scenario_name: str, run_id: str) -> None:
        """Logger should store scenario name and run_id in context."""
        logger = ScenarioLogger(scenario_name, run_id=run_id)
        assert logger._context["scenario"] == scenario_name
        assert logger._context["run_id"] == run_id

    def test_init_without_run_id(self, scenario_name: str) -> None:
        """Logger should work without run_id."""
        logger = ScenarioLogger(scenario_name)
        assert logger._context["scenario"] == scenario_name
        assert "run_id" not in logger._context

    def test_init_extra_context(self, scenario_name: str) -> None:
        """Extra kwargs should appear in context."""
        logger = ScenarioLogger(scenario_name, phase="training", epoch=10)
        assert logger._context["phase"] == "training"
        assert logger._context["epoch"] == 10

    def test_bind_creates_new_logger(self, scenario_logger: ScenarioLogger) -> None:
        """bind() should return a new logger with merged context."""
        child = scenario_logger.bind(step=5)
        assert child is not scenario_logger
        assert child._context["step"] == 5
        # Original context preserved
        assert child._context["scenario"] == scenario_logger._context["scenario"]

    def test_bind_does_not_mutate_parent(self, scenario_logger: ScenarioLogger) -> None:
        """bind() must not alter the parent logger's context."""
        original_context = dict(scenario_logger._context)
        scenario_logger.bind(extra_key="extra_value")
        assert scenario_logger._context == original_context

    @pytest.mark.parametrize("method", ["debug", "info", "warning", "error"])
    def test_log_methods_callable(
        self, scenario_logger: ScenarioLogger, method: str
    ) -> None:
        """All standard log methods should be callable without error."""
        log_fn = getattr(scenario_logger, method)
        log_fn("test_event", key="value")

    def test_exception_method_callable(self, scenario_logger: ScenarioLogger) -> None:
        """exception() should be callable without raising."""
        try:
            raise ValueError("deliberate")
        except ValueError:
            scenario_logger.exception("caught_error")


# ---------------------------------------------------------------------------
# ScenarioLogger.timed
# ---------------------------------------------------------------------------


class TestScenarioLoggerTimed:
    """Tests for ScenarioLogger.timed context manager."""

    def test_timed_populates_duration(self, scenario_logger: ScenarioLogger) -> None:
        """timed() should populate timing dict with duration_seconds."""
        with scenario_logger.timed("operation") as timing:
            time.sleep(0.05)

        assert "duration_seconds" in timing
        assert timing["duration_seconds"] >= 0.04  # allow small jitter

    def test_timed_duration_is_positive(self, scenario_logger: ScenarioLogger) -> None:
        """duration_seconds should always be > 0."""
        with scenario_logger.timed("fast") as timing:
            pass  # nearly instant
        assert timing["duration_seconds"] > 0

    def test_timed_on_exception(self, scenario_logger: ScenarioLogger) -> None:
        """duration should still be recorded when body raises."""
        timing: dict[str, float] = {}
        with pytest.raises(RuntimeError):
            with scenario_logger.timed("failing") as timing:
                raise RuntimeError("boom")

        assert "duration_seconds" in timing
        assert timing["duration_seconds"] > 0


# ---------------------------------------------------------------------------
# ScenarioLogger.metric & progress
# ---------------------------------------------------------------------------


class TestScenarioLoggerMetric:
    """Tests for metric and progress logging."""

    def test_metric_logs_without_error(self, scenario_logger: ScenarioLogger) -> None:
        """metric() should accept name, value, and tags."""
        scenario_logger.metric("loss", 0.042, epoch=5, split="val")

    @pytest.mark.parametrize(
        "current,total",
        [(0, 100), (50, 100), (100, 100), (0, 0)],
    )
    def test_progress_logs_without_error(
        self, scenario_logger: ScenarioLogger, current: int, total: int
    ) -> None:
        """progress() should handle boundary values."""
        scenario_logger.progress(current, total, operation="training")


# ---------------------------------------------------------------------------
# get_scenario_logger factory
# ---------------------------------------------------------------------------


class TestGetScenarioLogger:
    """Tests for the factory function."""

    def test_returns_scenario_logger(self, scenario_name: str, run_id: str) -> None:
        logger = get_scenario_logger(scenario_name, run_id=run_id)
        assert isinstance(logger, ScenarioLogger)
        assert logger._context["scenario"] == scenario_name

    def test_extra_context_forwarded(self) -> None:
        logger = get_scenario_logger("s", run_id="r", custom="val")
        assert logger._context["custom"] == "val"


# ---------------------------------------------------------------------------
# log_timing decorator
# ---------------------------------------------------------------------------


class TestLogTimingDecorator:
    """Tests for the @log_timing decorator."""

    def test_preserves_return_value(self) -> None:
        """Decorated function should return the original value."""
        sentinel = object()

        @log_timing()
        def fn() -> object:
            return sentinel

        assert fn() is sentinel

    def test_preserves_function_name(self) -> None:
        """functools.wraps should preserve __name__."""

        @log_timing()
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_measures_time(self) -> None:
        """Decorator should call logger with duration."""
        mock_logger = MagicMock()

        @log_timing(logger=mock_logger)
        def slow() -> int:
            time.sleep(0.05)
            return 42

        result = slow()
        assert result == 42
        mock_logger.debug.assert_called_once()
        call_kwargs = mock_logger.debug.call_args
        assert call_kwargs[1]["function"] == "slow"
        assert call_kwargs[1]["duration_seconds"] >= 0.04

    def test_timing_on_exception(self) -> None:
        """Duration should still be logged even if function raises."""
        mock_logger = MagicMock()

        @log_timing(logger=mock_logger)
        def failing() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            failing()

        mock_logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# log_call decorator
# ---------------------------------------------------------------------------


class TestLogCallDecorator:
    """Tests for the @log_call decorator."""

    def test_preserves_return(self) -> None:
        @log_call()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_log_args_flag(self) -> None:
        mock_logger = MagicMock()

        @log_call(logger=mock_logger, log_args=True)
        def fn(x: int) -> int:
            return x

        fn(99)
        call_kwargs = mock_logger.debug.call_args
        assert "args" in call_kwargs[1]

    def test_log_result_flag(self) -> None:
        mock_logger = MagicMock()

        @log_call(logger=mock_logger, log_result=True)
        def fn() -> str:
            return "hello"

        fn()
        # Two calls: function_call and function_result
        assert mock_logger.debug.call_count == 2


# ---------------------------------------------------------------------------
# DebugContext
# ---------------------------------------------------------------------------


class TestDebugContext:
    """Tests for DebugContext context manager."""

    def test_records_duration(self) -> None:
        """DebugContext should track elapsed time."""
        ctx = DebugContext("test_ctx")
        with ctx:
            time.sleep(0.05)
        # No explicit assertion on internal state; ensures no error raised.

    def test_checkpoint_records_elapsed(self) -> None:
        """checkpoint() should log with elapsed time."""
        ctx = DebugContext("ckpt_ctx")
        with ctx:
            time.sleep(0.02)
            ctx.checkpoint("mid", info="halfway")

    def test_exception_captured(self) -> None:
        """DebugContext should still exit cleanly on exception."""
        ctx = DebugContext("err_ctx")
        with pytest.raises(RuntimeError):
            with ctx:
                raise RuntimeError("deliberate")
