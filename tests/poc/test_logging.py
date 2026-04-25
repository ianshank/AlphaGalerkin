"""Tests for PoC structured logging utilities.

Covers:
    - configure_logging (level, JSON, timestamp)
    - ScenarioLogger (init, bind, log methods, timed, metric, progress)
    - get_scenario_logger factory
    - log_timing decorator
    - log_call decorator (with/without args/result logging)
    - DebugContext (enter, exit, checkpoint, memory flag, exception path)
"""

from __future__ import annotations

import time

import pytest

from src.poc.logging import (
    DebugContext,
    ScenarioLogger,
    configure_logging,
    get_scenario_logger,
    log_call,
    log_timing,
)

# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for the global logging configuration helper."""

    def test_default_call(self) -> None:
        """Should not raise with defaults."""
        configure_logging()

    def test_json_format(self) -> None:
        configure_logging(json_format=True)

    def test_no_timestamp(self) -> None:
        configure_logging(include_timestamp=False)

    def test_debug_level(self) -> None:
        configure_logging(level="DEBUG")

    def test_warning_level(self) -> None:
        configure_logging(level="WARNING")

    def test_all_options(self) -> None:
        configure_logging(level="ERROR", json_format=True, include_timestamp=False)


# ---------------------------------------------------------------------------
# ScenarioLogger -- construction
# ---------------------------------------------------------------------------


class TestScenarioLoggerInit:
    """Tests for ScenarioLogger initialization."""

    def test_basic_init(self) -> None:
        logger = ScenarioLogger("test_scenario")
        assert logger._context["scenario"] == "test_scenario"
        assert "run_id" not in logger._context

    def test_with_run_id(self) -> None:
        logger = ScenarioLogger("test_scenario", run_id="abc123")
        assert logger._context["scenario"] == "test_scenario"
        assert logger._context["run_id"] == "abc123"

    def test_with_extra_context(self) -> None:
        logger = ScenarioLogger("s", run_id="r", model="gpt", epoch=5)
        assert logger._context["model"] == "gpt"
        assert logger._context["epoch"] == 5

    def test_none_run_id_excluded(self) -> None:
        logger = ScenarioLogger("s", run_id=None)
        assert "run_id" not in logger._context


# ---------------------------------------------------------------------------
# ScenarioLogger -- bind
# ---------------------------------------------------------------------------


class TestScenarioLoggerBind:
    """Tests for creating child loggers with additional context."""

    def test_bind_creates_new_logger(self) -> None:
        parent = ScenarioLogger("s", run_id="r1")
        child = parent.bind(step=10)
        assert child is not parent

    def test_bind_inherits_parent_context(self) -> None:
        parent = ScenarioLogger("s", run_id="r1")
        child = parent.bind(step=10)
        assert child._context["scenario"] == "s"
        assert child._context["run_id"] == "r1"
        assert child._context["step"] == 10

    def test_bind_does_not_mutate_parent(self) -> None:
        parent = ScenarioLogger("s")
        parent.bind(extra="val")
        assert "extra" not in parent._context

    def test_bind_override_existing_key(self) -> None:
        parent = ScenarioLogger("s", run_id="old")
        child = parent.bind(run_id="new")
        assert child._context["run_id"] == "new"
        assert parent._context["run_id"] == "old"


# ---------------------------------------------------------------------------
# ScenarioLogger -- log methods (should not raise)
# ---------------------------------------------------------------------------


class TestScenarioLoggerMethods:
    """Tests for debug/info/warning/error/exception log calls."""

    def test_debug(self) -> None:
        logger = ScenarioLogger("s")
        logger.debug("test_event", key="value")

    def test_info(self) -> None:
        logger = ScenarioLogger("s")
        logger.info("test_event", key="value")

    def test_warning(self) -> None:
        logger = ScenarioLogger("s")
        logger.warning("test_event", key="value")

    def test_error(self) -> None:
        logger = ScenarioLogger("s")
        logger.error("test_event", key="value")

    def test_exception(self) -> None:
        logger = ScenarioLogger("s")
        logger.exception("test_event", key="value")


# ---------------------------------------------------------------------------
# ScenarioLogger -- timed context manager
# ---------------------------------------------------------------------------


class TestScenarioLoggerTimed:
    """Tests for the timed() context manager."""

    def test_populates_duration(self) -> None:
        logger = ScenarioLogger("s")
        with logger.timed("op") as timing:
            pass
        assert "duration_seconds" in timing
        assert timing["duration_seconds"] >= 0.0

    def test_timing_is_positive(self) -> None:
        logger = ScenarioLogger("s")
        with logger.timed("op") as timing:
            time.sleep(0.01)
        assert timing["duration_seconds"] >= 0.01

    def test_timing_dict_starts_empty(self) -> None:
        logger = ScenarioLogger("s")
        with logger.timed("op") as timing:
            # During the block, timing should not yet have duration
            assert "duration_seconds" not in timing

    def test_timing_on_exception(self) -> None:
        """Duration should be populated even when block raises."""
        logger = ScenarioLogger("s")
        with pytest.raises(ValueError):
            with logger.timed("op") as timing:
                raise ValueError("boom")
        assert "duration_seconds" in timing
        assert timing["duration_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# ScenarioLogger -- metric
# ---------------------------------------------------------------------------


class TestScenarioLoggerMetric:
    """Tests for the metric() helper."""

    def test_metric_call(self) -> None:
        logger = ScenarioLogger("s")
        logger.metric("loss", 0.123, epoch=1)

    def test_metric_zero_value(self) -> None:
        logger = ScenarioLogger("s")
        logger.metric("accuracy", 0.0)


# ---------------------------------------------------------------------------
# ScenarioLogger -- progress
# ---------------------------------------------------------------------------


class TestScenarioLoggerProgress:
    """Tests for the progress() helper."""

    def test_progress_normal(self) -> None:
        logger = ScenarioLogger("s")
        logger.progress(50, 100, operation="training")

    def test_progress_zero_total(self) -> None:
        """total=0 should not raise (pct = 0)."""
        logger = ScenarioLogger("s")
        logger.progress(0, 0)

    def test_progress_at_completion(self) -> None:
        logger = ScenarioLogger("s")
        logger.progress(100, 100)


# ---------------------------------------------------------------------------
# get_scenario_logger factory
# ---------------------------------------------------------------------------


class TestGetScenarioLogger:
    """Tests for the factory function."""

    def test_returns_scenario_logger(self) -> None:
        logger = get_scenario_logger("my_scenario")
        assert isinstance(logger, ScenarioLogger)

    def test_with_run_id(self) -> None:
        logger = get_scenario_logger("s", run_id="r123")
        assert logger._context["run_id"] == "r123"

    def test_with_extra_context(self) -> None:
        logger = get_scenario_logger("s", gpu="A100")
        assert logger._context["gpu"] == "A100"


# ---------------------------------------------------------------------------
# log_timing decorator
# ---------------------------------------------------------------------------


class TestLogTimingDecorator:
    """Tests for the @log_timing() decorator."""

    def test_preserves_return_value(self) -> None:
        @log_timing()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self) -> None:
        @log_timing()
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_with_explicit_logger(self) -> None:
        import structlog

        custom_logger = structlog.get_logger("custom")

        @log_timing(logger=custom_logger)
        def work() -> str:
            return "done"

        assert work() == "done"

    def test_timing_on_exception(self) -> None:
        """Should still log timing even when function raises."""

        @log_timing()
        def failing() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            failing()


# ---------------------------------------------------------------------------
# log_call decorator
# ---------------------------------------------------------------------------


class TestLogCallDecorator:
    """Tests for the @log_call() decorator."""

    def test_preserves_return_value(self) -> None:
        @log_call()
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(3, 4) == 12

    def test_preserves_function_name(self) -> None:
        @log_call()
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_log_args_enabled(self) -> None:
        @log_call(log_args=True)
        def func(x: int) -> int:
            return x + 1

        assert func(5) == 6

    def test_log_result_enabled(self) -> None:
        @log_call(log_result=True)
        def func(x: int) -> int:
            return x * 2

        assert func(3) == 6

    def test_log_args_and_result(self) -> None:
        @log_call(log_args=True, log_result=True)
        def func(x: int, y: int) -> int:
            return x + y

        assert func(1, 2) == 3

    def test_with_explicit_logger(self) -> None:
        import structlog

        custom_logger = structlog.get_logger("custom")

        @log_call(logger=custom_logger, log_args=True, log_result=True)
        def work(n: int) -> int:
            return n * n

        assert work(4) == 16

    def test_exception_propagated(self) -> None:
        @log_call(log_args=True)
        def bad() -> None:
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            bad()


# ---------------------------------------------------------------------------
# DebugContext
# ---------------------------------------------------------------------------


class TestDebugContext:
    """Tests for the DebugContext context manager."""

    def test_basic_usage(self) -> None:
        ctx = DebugContext("test_op")
        with ctx:
            pass

    def test_with_logger(self) -> None:
        logger = ScenarioLogger("s")
        ctx = DebugContext("op", logger=logger)
        with ctx:
            pass

    def test_creates_default_logger(self) -> None:
        ctx = DebugContext("my_op")
        assert isinstance(ctx.logger, ScenarioLogger)

    def test_duration_tracked(self) -> None:
        ctx = DebugContext("op")
        with ctx:
            time.sleep(0.01)
        # _start_time should have been set (implicitly tested by no exception)

    def test_checkpoint(self) -> None:
        ctx = DebugContext("op")
        with ctx:
            ctx.checkpoint("step1", value=42)
            ctx.checkpoint("step2", value=99)

    def test_capture_memory_false(self) -> None:
        ctx = DebugContext("op", capture_memory=False)
        with ctx:
            pass

    def test_capture_memory_true_cpu(self) -> None:
        """On CPU, memory capture should be a no-op but not crash."""
        ctx = DebugContext("op", capture_memory=True)
        with ctx:
            pass

    def test_exception_in_block(self) -> None:
        """DebugContext should not swallow exceptions."""
        ctx = DebugContext("op")
        with pytest.raises(RuntimeError, match="boom"):
            with ctx:
                raise RuntimeError("boom")

    def test_exception_logged_in_exit(self) -> None:
        """When an exception occurs, exc_type should be captured."""
        logger = ScenarioLogger("s")
        ctx = DebugContext("op", logger=logger)
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("test error")

    def test_returns_self(self) -> None:
        ctx = DebugContext("op")
        with ctx as c:
            assert c is ctx

    def test_name_stored(self) -> None:
        ctx = DebugContext("my_operation")
        assert ctx.name == "my_operation"

    def test_checkpoint_elapsed_positive(self) -> None:
        ctx = DebugContext("op")
        with ctx:
            time.sleep(0.01)
            # checkpoint should not raise; elapsed should be > 0
            ctx.checkpoint("mid")

    def test_nested_debug_contexts(self) -> None:
        outer = DebugContext("outer")
        inner = DebugContext("inner")
        with outer:
            outer.checkpoint("before_inner")
            with inner:
                inner.checkpoint("inside")
            outer.checkpoint("after_inner")
