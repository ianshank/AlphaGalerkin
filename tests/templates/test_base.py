"""Tests for the templates.base module."""

from __future__ import annotations

import time
from datetime import datetime

import pytest
from pydantic import Field

from src.templates.base import (
    BaseExecutable,
    ExecutionResult,
    ExecutionStatus,
    create_executable_class,
)
from src.templates.config import BaseModuleConfig


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_terminal_states(self) -> None:
        """Test is_terminal method."""
        terminal = [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.SKIPPED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT,
        ]
        non_terminal = [
            ExecutionStatus.PENDING,
            ExecutionStatus.RUNNING,
        ]

        for status in terminal:
            assert status.is_terminal(), f"{status} should be terminal"

        for status in non_terminal:
            assert not status.is_terminal(), f"{status} should not be terminal"

    def test_success_state(self) -> None:
        """Test is_success method."""
        assert ExecutionStatus.COMPLETED.is_success()
        assert not ExecutionStatus.FAILED.is_success()
        assert not ExecutionStatus.RUNNING.is_success()

    def test_string_value(self) -> None:
        """Test that enum has string values."""
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert str(ExecutionStatus.COMPLETED) == "ExecutionStatus.COMPLETED"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = ExecutionResult()

        assert result.status == ExecutionStatus.PENDING
        assert result.error is None
        assert result.metrics == {}
        assert result.artifacts == {}
        assert result.run_id  # Should have auto-generated ID

    def test_is_success(self) -> None:
        """Test is_success method."""
        success = ExecutionResult(status=ExecutionStatus.COMPLETED)
        failure = ExecutionResult(status=ExecutionStatus.FAILED)

        assert success.is_success()
        assert not failure.is_success()

    def test_is_terminal(self) -> None:
        """Test is_terminal method."""
        running = ExecutionResult(status=ExecutionStatus.RUNNING)
        completed = ExecutionResult(status=ExecutionStatus.COMPLETED)

        assert not running.is_terminal()
        assert completed.is_terminal()

    def test_get_metric(self) -> None:
        """Test get_metric with default."""
        result = ExecutionResult(metrics={"loss": 0.5})

        assert result.get_metric("loss") == 0.5
        assert result.get_metric("accuracy") == 0.0
        assert result.get_metric("accuracy", 1.0) == 1.0

    def test_add_metric(self) -> None:
        """Test add_metric method."""
        result = ExecutionResult()
        result.add_metric("loss", 0.5)

        assert result.metrics["loss"] == 0.5

    def test_add_artifact(self) -> None:
        """Test add_artifact method."""
        result = ExecutionResult()
        result.add_artifact("model_path", "/path/to/model.pt")

        assert result.artifacts["model_path"] == "/path/to/model.pt"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        now = datetime.utcnow()
        result = ExecutionResult(
            run_id="test123",
            name="test",
            status=ExecutionStatus.COMPLETED,
            start_time=now,
            end_time=now,
            duration_seconds=1.5,
            metrics={"loss": 0.5},
            config_hash="abc123",
        )

        data = result.to_dict()

        assert data["run_id"] == "test123"
        assert data["status"] == "completed"
        assert data["metrics"]["loss"] == 0.5
        assert isinstance(data["start_time"], str)  # ISO format

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "run_id": "test123",
            "name": "test",
            "status": "completed",
            "error": None,
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:01:00",
            "duration_seconds": 60.0,
            "metrics": {"loss": 0.5},
            "artifacts": {},
            "metadata": {},
            "config_hash": "abc123",
        }

        result = ExecutionResult.from_dict(data)

        assert result.run_id == "test123"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.metrics["loss"] == 0.5
        assert result.start_time == datetime(2024, 1, 1, 0, 0, 0)

    def test_roundtrip_serialization(self) -> None:
        """Test serialization roundtrip."""
        original = ExecutionResult(
            run_id="test",
            status=ExecutionStatus.COMPLETED,
            metrics={"loss": 0.5},
        )

        data = original.to_dict()
        restored = ExecutionResult.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert restored.metrics == original.metrics


class TestBaseExecutable:
    """Tests for BaseExecutable abstract class."""

    def test_requires_execute_implementation(self) -> None:
        """Test that execute must be implemented."""

        class IncompleteExecutable(BaseExecutable):
            pass

        config = BaseModuleConfig(name="test")

        with pytest.raises(TypeError, match="abstract"):
            IncompleteExecutable(config)  # type: ignore[abstract]

    def test_init_creates_logger(self, sample_executable) -> None:
        """Test that initialization creates a logger."""
        assert sample_executable.logger is not None
        assert sample_executable.run_id

    def test_execute_returns_result(self, sample_executable) -> None:
        """Test that execute returns a result."""
        result = sample_executable.execute()

        assert isinstance(result, ExecutionResult)
        assert result.status == ExecutionStatus.COMPLETED

    def test_run_wraps_execute(self, sample_executable) -> None:
        """Test that run() wraps execute() with timing."""
        result = sample_executable.run()

        assert result.status == ExecutionStatus.COMPLETED
        assert result.duration_seconds > 0
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.run_id == sample_executable.run_id

    def test_run_handles_exceptions(self, sample_config) -> None:
        """Test that run() handles exceptions."""

        class FailingExecutable(BaseExecutable):
            _executable_name = "failing"

            def execute(self) -> ExecutionResult:
                raise ValueError("Test error")

        executable = FailingExecutable(sample_config)
        result = executable.run()

        assert result.status == ExecutionStatus.FAILED
        assert "Test error" in result.error
        assert result.error_traceback is not None

    def test_status_property(self, sample_executable) -> None:
        """Test status property."""
        assert sample_executable.status == ExecutionStatus.PENDING

        sample_executable.run()
        assert sample_executable.status == ExecutionStatus.COMPLETED

    def test_elapsed_property(self, sample_config) -> None:
        """Test elapsed property."""

        class SlowExecutable(BaseExecutable):
            _executable_name = "slow"

            def execute(self) -> ExecutionResult:
                time.sleep(0.01)
                return self._create_result(ExecutionStatus.COMPLETED)

        executable = SlowExecutable(sample_config)
        assert executable.elapsed == 0.0  # Not started

        result = executable.run()
        assert result.duration_seconds >= 0.01

    def test_create_result_helper(self, sample_executable) -> None:
        """Test _create_result helper method."""
        # Simulate execution
        sample_executable._start_time = time.perf_counter()
        time.sleep(0.01)

        result = sample_executable._create_result(
            status=ExecutionStatus.COMPLETED,
            metrics={"accuracy": 0.95},
            artifacts={"model": "path/to/model"},
            metadata={"version": "1.0"},
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.metrics["accuracy"] == 0.95
        assert result.artifacts["model"] == "path/to/model"
        assert result.metadata["version"] == "1.0"
        assert result.duration_seconds >= 0.01
        assert result.config_hash == sample_executable.config.compute_hash()

    def test_validate_config(self, sample_executable) -> None:
        """Test validate_config method."""
        # Default implementation returns True
        assert sample_executable.validate_config()

    def test_custom_run_id(self, sample_config) -> None:
        """Test providing custom run_id."""
        from tests.templates.conftest import SampleExecutable

        executable = SampleExecutable(sample_config, run_id="custom_run_123")
        assert executable.run_id == "custom_run_123"


class TestCreateExecutableClass:
    """Tests for create_executable_class factory."""

    def test_creates_executable_class(self) -> None:
        """Test that factory creates an executable class."""

        class MyConfig(BaseModuleConfig):
            my_param: int = Field(default=10)

        MyExecutable = create_executable_class("MyExecutable", MyConfig)

        assert issubclass(MyExecutable, BaseExecutable)
        assert MyExecutable._executable_name == "myexecutable"

    def test_created_class_has_logger(self) -> None:
        """Test that created class has proper logger."""

        class MyConfig(BaseModuleConfig):
            pass

        MyExecutable = create_executable_class(
            "MyExecutable", MyConfig, module_name="my_module"
        )

        assert MyExecutable._logger_class._module_name == "my_module"

    def test_created_class_can_be_subclassed(self) -> None:
        """Test that created class can be subclassed and used."""

        class MyConfig(BaseModuleConfig):
            value: int = Field(default=42)

        MyExecutable = create_executable_class("MyExecutable", MyConfig)

        class ConcreteExecutable(MyExecutable):
            def execute(self) -> ExecutionResult:
                return self._create_result(
                    status=ExecutionStatus.COMPLETED,
                    metrics={"value": float(self.config.value)},
                )

        config = MyConfig(name="test")
        executable = ConcreteExecutable(config)
        result = executable.run()

        assert result.status == ExecutionStatus.COMPLETED
        assert result.metrics["value"] == 42.0
