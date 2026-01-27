"""Tests for the templates.logging module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.templates.logging import (
    BaseModuleLogger,
    DebugContext,
    configure_module_logging,
    create_logger_class,
    log_call,
    log_timing,
)


class TestConfigureModuleLogging:
    """Tests for configure_module_logging."""

    def test_configure_default(self) -> None:
        """Test default configuration doesn't raise."""
        # Should not raise
        configure_module_logging()

    def test_configure_json_format(self) -> None:
        """Test JSON format configuration."""
        configure_module_logging(json_format=True)

    def test_configure_all_options(self) -> None:
        """Test configuration with all options."""
        configure_module_logging(
            level="DEBUG",
            json_format=False,
            include_timestamp=True,
            include_caller=True,
        )


class TestBaseModuleLogger:
    """Tests for BaseModuleLogger."""

    def test_init_with_minimal_args(self) -> None:
        """Test initialization with minimal arguments."""
        logger = BaseModuleLogger("test_component")
        assert logger._context["component"] == "test_component"
        assert "run_id" not in logger._context

    def test_init_with_run_id(self) -> None:
        """Test initialization with run_id."""
        logger = BaseModuleLogger("test_component", run_id="abc123")
        assert logger._context["run_id"] == "abc123"

    def test_init_with_extra_context(self) -> None:
        """Test initialization with extra context."""
        logger = BaseModuleLogger(
            "test_component",
            run_id="abc123",
            experiment="test",
            version=1,
        )
        assert logger._context["experiment"] == "test"
        assert logger._context["version"] == 1

    def test_bind_creates_new_logger(self) -> None:
        """Test that bind creates a new logger with merged context."""
        logger = BaseModuleLogger("test_component", run_id="abc123")
        new_logger = logger.bind(step=10)

        assert new_logger is not logger
        assert new_logger._context["step"] == 10
        assert new_logger._context["run_id"] == "abc123"
        # Original unchanged
        assert "step" not in logger._context

    def test_log_methods_exist(self) -> None:
        """Test that all log methods exist."""
        logger = BaseModuleLogger("test")

        # These should all be callable
        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.exception)
        assert callable(logger.critical)

    def test_metric_logging(self) -> None:
        """Test metric logging includes required fields."""
        logger = BaseModuleLogger("test")

        with patch.object(logger._logger, "info") as mock_info:
            logger.metric("loss", 0.5, step=10, epoch=1)

            mock_info.assert_called_once()
            call_kwargs = mock_info.call_args[1]
            assert call_kwargs["metric_name"] == "loss"
            assert call_kwargs["metric_value"] == 0.5
            assert call_kwargs["step"] == 10
            assert call_kwargs["epoch"] == 1

    def test_progress_logging(self) -> None:
        """Test progress logging."""
        logger = BaseModuleLogger("test")

        with patch.object(logger._logger, "info") as mock_info:
            logger.progress(50, 100, message="Processing")

            mock_info.assert_called_once()
            call_kwargs = mock_info.call_args[1]
            assert call_kwargs["current"] == 50
            assert call_kwargs["total"] == 100
            assert call_kwargs["percentage"] == 50.0
            assert call_kwargs["message"] == "Processing"

    def test_progress_with_zero_total(self) -> None:
        """Test progress logging with zero total."""
        logger = BaseModuleLogger("test")

        with patch.object(logger._logger, "info") as mock_info:
            logger.progress(0, 0)

            call_kwargs = mock_info.call_args[1]
            assert call_kwargs["percentage"] == 0.0

    def test_timer_methods(self) -> None:
        """Test start_timer and log_elapsed."""
        logger = BaseModuleLogger("test")

        logger.start_timer()
        time.sleep(0.01)
        elapsed = logger.log_elapsed("test_event")

        assert elapsed >= 0.01

    def test_timed_context_manager(self) -> None:
        """Test timed context manager."""
        logger = BaseModuleLogger("test")

        with logger.timed("test_operation") as timing:
            time.sleep(0.01)

        assert "duration_seconds" in timing
        assert timing["duration_seconds"] >= 0.01

    def test_timed_context_manager_on_exception(self) -> None:
        """Test timed context manager logs on exception."""
        logger = BaseModuleLogger("test")

        with pytest.raises(ValueError):
            with logger.timed("failing_operation") as timing:
                time.sleep(0.01)
                raise ValueError("Test error")

        # Timing should still be recorded
        assert "duration_seconds" in timing


class TestCreateLoggerClass:
    """Tests for create_logger_class factory."""

    def test_creates_logger_class(self) -> None:
        """Test that factory creates a logger class."""
        MyLogger = create_logger_class("MyModule")

        assert MyLogger._module_name == "MyModule"
        assert issubclass(MyLogger, BaseModuleLogger)

    def test_created_logger_works(self) -> None:
        """Test that created logger class works."""
        MyLogger = create_logger_class("MyModule")
        logger = MyLogger("component", run_id="test123")

        assert logger._context["module"] == "MyModule"
        assert logger._context["component"] == "component"
        assert logger._context["run_id"] == "test123"

    def test_created_logger_has_all_methods(self) -> None:
        """Test that created logger has all methods."""
        MyLogger = create_logger_class("MyModule")
        logger = MyLogger("test")

        assert callable(logger.info)
        assert callable(logger.metric)
        assert callable(logger.timed)


class TestLogTimingDecorator:
    """Tests for log_timing decorator."""

    def test_logs_timing(self) -> None:
        """Test that decorator logs timing."""
        mock_logger = MagicMock()

        @log_timing(logger=mock_logger)
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()

        assert result == "result"
        mock_logger.debug.assert_called_once()

        call_kwargs = mock_logger.debug.call_args[1]
        assert call_kwargs["function"] == "slow_function"
        assert call_kwargs["duration_seconds"] >= 0.01

    def test_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function metadata."""

        @log_timing()
        def documented_function():
            """This is a docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert "docstring" in documented_function.__doc__

    def test_custom_log_level(self) -> None:
        """Test using custom log level."""
        mock_logger = MagicMock()

        @log_timing(logger=mock_logger, level="info")
        def my_function():
            pass

        my_function()
        mock_logger.info.assert_called_once()


class TestLogCallDecorator:
    """Tests for log_call decorator."""

    def test_logs_call(self) -> None:
        """Test that decorator logs function call."""
        mock_logger = MagicMock()

        @log_call(logger=mock_logger)
        def my_function(x: int, y: int) -> int:
            return x + y

        result = my_function(1, 2)

        assert result == 3
        mock_logger.debug.assert_called()

    def test_logs_args_when_enabled(self) -> None:
        """Test that args are logged when enabled."""
        mock_logger = MagicMock()

        @log_call(logger=mock_logger, log_args=True)
        def my_function(x: int, y: int) -> int:
            return x + y

        my_function(1, 2)

        call_kwargs = mock_logger.debug.call_args_list[0][1]
        assert "args" in call_kwargs
        assert "kwargs" in call_kwargs

    def test_logs_result_when_enabled(self) -> None:
        """Test that result is logged when enabled."""
        mock_logger = MagicMock()

        @log_call(logger=mock_logger, log_result=True)
        def my_function() -> str:
            return "hello"

        my_function()

        # Should have two calls: one for call, one for result
        assert mock_logger.debug.call_count == 2

    def test_truncates_long_values(self) -> None:
        """Test that long values are truncated."""
        mock_logger = MagicMock()

        @log_call(logger=mock_logger, log_args=True, max_str_length=10)
        def my_function(x: str) -> str:
            return x

        my_function("a" * 100)

        call_kwargs = mock_logger.debug.call_args_list[0][1]
        assert len(call_kwargs["args"]) <= 10


class TestDebugContext:
    """Tests for DebugContext."""

    def test_basic_usage(self) -> None:
        """Test basic debug context usage."""
        with DebugContext("test_op") as ctx:
            time.sleep(0.01)

        assert ctx.elapsed >= 0.01

    def test_checkpoints(self) -> None:
        """Test checkpoint functionality."""
        with DebugContext("test_op") as ctx:
            time.sleep(0.01)
            elapsed1 = ctx.checkpoint("first", value=1)
            time.sleep(0.01)
            elapsed2 = ctx.checkpoint("second", value=2)

        assert elapsed1 >= 0.01
        assert elapsed2 >= 0.02
        assert len(ctx.checkpoints) == 2
        assert ctx.checkpoints[0]["label"] == "first"
        assert ctx.checkpoints[0]["value"] == 1
        assert ctx.checkpoints[1]["label"] == "second"

    def test_exception_handling(self) -> None:
        """Test that exceptions are handled properly."""
        with pytest.raises(ValueError):
            with DebugContext("failing_op") as ctx:
                ctx.checkpoint("before_error")
                raise ValueError("Test error")

        # Checkpoints should be preserved
        assert len(ctx.checkpoints) == 1

    def test_elapsed_property(self) -> None:
        """Test elapsed property during execution."""
        with DebugContext("test_op") as ctx:
            time.sleep(0.01)
            elapsed = ctx.elapsed
            assert elapsed >= 0.01

    def test_memory_capture_disabled_by_default(self) -> None:
        """Test that memory capture is disabled by default."""
        with DebugContext("test_op") as ctx:
            ctx.checkpoint("test")

        # Memory should not be in checkpoints
        assert "memory_mb" not in ctx.checkpoints[0]

    def test_custom_logger(self) -> None:
        """Test using custom logger."""
        custom_logger = BaseModuleLogger("custom")

        with DebugContext("test_op", logger=custom_logger) as ctx:
            ctx.checkpoint("test")

        assert len(ctx.checkpoints) == 1

    def test_custom_log_level(self) -> None:
        """Test using custom log level."""
        with DebugContext("test_op", log_level="info") as ctx:
            pass

        assert ctx.log_level == "info"
