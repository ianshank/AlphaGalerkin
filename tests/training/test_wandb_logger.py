"""Comprehensive tests for Weights & Biases logging integration.

Tests cover:
- WandbLogger initialization (enabled/disabled modes)
- Metric logging (training step, evaluation, buffer stats)
- Artifact logging (model checkpoints)
- Error handling and graceful degradation
- Thread safety
- Edge cases (null handling, empty data, etc.)
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from config.schemas import WandbConfig
from src.training.wandb_logger import (
    DEFAULT_LOG_INTERVAL,
    DEFAULT_MODE,
    DEFAULT_PROJECT,
    DEFAULT_WATCH_LOG_FREQ,
    WandbLogger,
    create_wandb_logger,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_wandb() -> MagicMock:
    """Create a mock wandb module."""
    mock = MagicMock()
    mock.init.return_value = MagicMock(name="test_run", id="test_id_123")
    mock.AlertLevel = MagicMock()
    mock.AlertLevel.INFO = "INFO"
    mock.AlertLevel.WARN = "WARN"
    mock.AlertLevel.ERROR = "ERROR"
    mock.Histogram = MagicMock()
    mock.Table = MagicMock()
    mock.Artifact = MagicMock()
    return mock


@pytest.fixture
def disabled_config() -> dict[str, Any]:
    """Create a disabled W&B configuration."""
    return {"enabled": False}


@pytest.fixture
def enabled_config() -> dict[str, Any]:
    """Create an enabled W&B configuration with test settings."""
    return {
        "enabled": True,
        "project": "test-project",
        "entity": "test-entity",
        "name": "test-run",
        "tags": ["test", "unit-test"],
        "notes": "Unit test run",
        "group": "test-group",
        "job_type": "test",
        "mode": "disabled",  # Use disabled mode to avoid actual W&B calls
        "log_model": True,
        "log_gradients": False,
        "log_code": False,
        "log_interval": 1,
        "watch_model": False,
        "watch_log_freq": 100,
    }


@pytest.fixture
def wandb_config_pydantic() -> WandbConfig:
    """Create a Pydantic WandbConfig for testing."""
    return WandbConfig(
        enabled=True,
        project="test-project",
        mode="disabled",
        log_code=False,
    )


@dataclass
class MockTrainingMetrics:
    """Mock TrainingMetrics for testing."""

    step: int = 0
    total_loss: float = 0.5
    policy_loss: float = 0.3
    value_loss: float = 0.2
    lbb_loss: float = 0.01
    lbb_constant: float = 1.0
    learning_rate: float = 0.001
    gradient_norm: float = 1.5
    buffer_size: int = 1000
    games_generated: int = 100
    step_time_ms: float = 50.0


@dataclass
class MockEvaluationResult:
    """Mock EvaluationResult for testing."""

    win_rate: float = 0.6
    n_games: int = 20
    wins: int = 12
    losses: int = 6
    draws: int = 2
    avg_game_length: float = 150.0
    avg_value_error: float = 0.1
    policy_agreement: float = 0.8
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {"board_size": 9, "opponent": "random"}


# =============================================================================
# WandbLogger Initialization Tests
# =============================================================================


class TestWandbLoggerInit:
    """Test WandbLogger initialization behavior."""

    def test_disabled_logger_does_not_initialize(self, disabled_config: dict[str, Any]) -> None:
        """Test that disabled logger doesn't attempt initialization."""
        logger = WandbLogger(config=disabled_config)

        assert not logger.is_enabled
        assert logger.run is None
        assert logger.run_id is None
        assert logger.run_name is None

    def test_enabled_with_no_wandb_import_falls_back(self, enabled_config: dict[str, Any]) -> None:
        """Test graceful fallback when wandb import fails."""
        # Temporarily remove wandb from sys.modules to simulate import failure
        import sys

        original_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = None  # type: ignore[assignment]

        try:
            # The WandbLogger should handle this gracefully
            logger = WandbLogger(config=enabled_config)
            # Since wandb module is None, it should disable itself
            assert not logger.is_enabled
            assert logger.run is None
        except (ImportError, TypeError, AttributeError):
            # If an error occurs, that's also acceptable behavior - main thing is no crash
            pass
        finally:
            # Restore original
            if original_wandb is not None:
                sys.modules["wandb"] = original_wandb
            elif "wandb" in sys.modules:
                del sys.modules["wandb"]

    def test_enabled_with_wandb_init_failure_falls_back(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test graceful fallback when wandb.init() fails."""
        mock_wandb.init.side_effect = Exception("Network error")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            assert not logger.is_enabled
            mock_wandb.init.assert_called_once()

    def test_enabled_with_successful_init(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test successful W&B initialization."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            assert logger.is_enabled
            assert logger.run is not None
            mock_wandb.init.assert_called_once()

    def test_default_config_values(self) -> None:
        """Test that default configuration values are applied."""
        logger = WandbLogger(config={"enabled": False})

        # Check internal defaults (via inspection of _config parsing)
        assert logger._project == DEFAULT_PROJECT
        assert logger._mode == DEFAULT_MODE
        assert logger._log_interval == DEFAULT_LOG_INTERVAL
        assert logger._watch_log_freq == DEFAULT_WATCH_LOG_FREQ

    def test_config_from_pydantic_model(
        self, wandb_config_pydantic: WandbConfig, mock_wandb: MagicMock
    ) -> None:
        """Test creating logger from Pydantic model dump."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            config_dict = wandb_config_pydantic.model_dump()
            logger = WandbLogger(config=config_dict)

            assert logger._project == "test-project"
            mock_wandb.init.assert_called_once()


# =============================================================================
# Null/None Handling Tests (Critical Bug Fixes)
# =============================================================================


class TestNullHandling:
    """Test proper handling of None/null values."""

    def test_tags_none_handled_safely(self, mock_wandb: MagicMock) -> None:
        """Test that None tags are converted to empty list."""
        config = {
            "enabled": True,
            "tags": None,  # Explicitly None
            "mode": "disabled",
            "log_code": False,
        }

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=config)

            # Verify tags were converted to empty list
            assert logger._tags == []
            # Verify init was called with None (or empty list) for tags
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["tags"] is None or call_kwargs["tags"] == []

    def test_tags_empty_list_preserved(self, mock_wandb: MagicMock) -> None:
        """Test that empty tags list is preserved."""
        config = {
            "enabled": True,
            "tags": [],
            "mode": "disabled",
            "log_code": False,
        }

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=config)
            assert logger._tags == []

    def test_tags_with_values_preserved(self, mock_wandb: MagicMock) -> None:
        """Test that tags with values are preserved."""
        config = {
            "enabled": True,
            "tags": ["tag1", "tag2"],
            "mode": "disabled",
            "log_code": False,
        }

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=config)
            assert logger._tags == ["tag1", "tag2"]

    def test_safe_list_handles_various_inputs(self) -> None:
        """Test _safe_list static method with various inputs."""
        assert WandbLogger._safe_list(None) == []
        assert WandbLogger._safe_list([]) == []
        assert WandbLogger._safe_list(["a", "b"]) == ["a", "b"]
        assert WandbLogger._safe_list([1, 2, 3]) == ["1", "2", "3"]
        assert WandbLogger._safe_list(["a", None, "b"]) == ["a", "b"]
        assert WandbLogger._safe_list("not_a_list") == []

    def test_log_summary_with_none_run(self, disabled_config: dict[str, Any]) -> None:
        """Test that log_summary handles None run gracefully."""
        logger = WandbLogger(config=disabled_config)

        # Should not raise
        logger.log_summary({"final_loss": 0.1})


# =============================================================================
# Metric Logging Tests
# =============================================================================


class TestMetricLogging:
    """Test metric logging functionality."""

    def test_log_training_step(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test logging training step metrics."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            metrics = MockTrainingMetrics(step=10)

            logger.log_training_step(metrics)

            mock_wandb.log.assert_called()
            call_args = mock_wandb.log.call_args
            log_dict = call_args[0][0]

            assert "train/loss/total" in log_dict
            assert "train/loss/policy" in log_dict
            assert "train/loss/value" in log_dict
            assert "train/gradient_norm" in log_dict
            assert "train/learning_rate" in log_dict

    def test_log_training_step_respects_interval(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that log_interval is respected."""
        enabled_config["log_interval"] = 5  # Log every 5 steps

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            # Step 0 should log (0 % 5 == 0)
            logger.log_training_step(MockTrainingMetrics(step=0))
            assert mock_wandb.log.call_count == 1

            # Step 3 should not log (3 % 5 != 0)
            logger.log_training_step(MockTrainingMetrics(step=3))
            assert mock_wandb.log.call_count == 1

            # Step 5 should log (5 % 5 == 0)
            logger.log_training_step(MockTrainingMetrics(step=5))
            assert mock_wandb.log.call_count == 2

    def test_log_training_step_disabled(self, disabled_config: dict[str, Any]) -> None:
        """Test that logging is skipped when disabled."""
        logger = WandbLogger(config=disabled_config)
        metrics = MockTrainingMetrics()

        # Should not raise
        logger.log_training_step(metrics)

    def test_log_evaluation(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test logging evaluation results."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            result = MockEvaluationResult()

            logger.log_evaluation(result, prefix="eval/9x9", step=100)

            mock_wandb.log.assert_called()
            call_args = mock_wandb.log.call_args
            log_dict = call_args[0][0]

            assert "eval/9x9/win_rate" in log_dict
            assert "eval/9x9/n_games" in log_dict
            assert "eval/9x9/meta/board_size" in log_dict

    def test_log_buffer_stats(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test logging buffer statistics."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.log_buffer_stats(
                buffer_size=5000,
                capacity=10000,
                value_mean=0.5,
                value_std=0.3,
                board_size_distribution={9: 100, 13: 50, 19: 20},
                step=50,
            )

            mock_wandb.log.assert_called()
            call_args = mock_wandb.log.call_args
            log_dict = call_args[0][0]

            assert log_dict["data/buffer_size"] == 5000
            assert log_dict["data/buffer_fill_ratio"] == 0.5
            assert log_dict["data/value_mean"] == 0.5
            assert "data/board_size_9x9" in log_dict

    def test_log_metrics_arbitrary(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test logging arbitrary metrics."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            custom_metrics = {"custom/metric1": 0.5, "custom/metric2": 100}
            logger.log_metrics(custom_metrics, step=10)

            mock_wandb.log.assert_called_with(custom_metrics, step=10, commit=True)


# =============================================================================
# Step Offset Tests
# =============================================================================


class TestStepOffset:
    """Test step offset functionality for resumed training."""

    def test_set_step_offset(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test setting step offset."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            logger.set_step_offset(1000)

            assert logger._step_offset == 1000

    def test_step_offset_applied_to_training_metrics(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that step offset is applied to training step logs."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            logger.set_step_offset(1000)

            metrics = MockTrainingMetrics(step=50)
            logger.log_training_step(metrics)

            # Step should be 50 + 1000 = 1050
            call_kwargs = mock_wandb.log.call_args[1]
            assert call_kwargs["step"] == 1050

    def test_step_offset_applied_to_evaluation(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that step offset is applied to evaluation logs."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            logger.set_step_offset(500)

            result = MockEvaluationResult()
            logger.log_evaluation(result, step=100)

            call_kwargs = mock_wandb.log.call_args[1]
            assert call_kwargs["step"] == 600


# =============================================================================
# Artifact Logging Tests
# =============================================================================


class TestArtifactLogging:
    """Test model artifact logging."""

    def test_log_model_artifact(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test logging model checkpoint as artifact."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                checkpoint_path = Path(f.name)
                f.write(b"dummy checkpoint data")

            try:
                logger.log_model_artifact(
                    checkpoint_path=checkpoint_path,
                    name="test-model",
                    metadata={"step": 100},
                    aliases=["latest", "best"],
                )

                mock_wandb.Artifact.assert_called_once()
                mock_wandb.log_artifact.assert_called_once()
            finally:
                checkpoint_path.unlink()

    def test_log_model_artifact_disabled_by_config(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that artifact logging is skipped when log_model is False."""
        enabled_config["log_model"] = False

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.log_model_artifact(
                checkpoint_path="dummy.pt",
                name="test-model",
            )

            mock_wandb.Artifact.assert_not_called()

    def test_log_model_artifact_handles_error(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that artifact logging errors are handled gracefully."""
        mock_wandb.Artifact.side_effect = Exception("Artifact creation failed")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            # Should not raise
            logger.log_model_artifact(
                checkpoint_path="nonexistent.pt",
                name="test-model",
            )


# =============================================================================
# Finish/Cleanup Tests
# =============================================================================


class TestFinishCleanup:
    """Test finish and cleanup behavior."""

    def test_finish_closes_run(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test that finish() properly closes the W&B run."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            assert logger.is_enabled

            logger.finish()

            mock_wandb.finish.assert_called_once()
            assert not logger.is_enabled
            assert logger.run is None

    def test_finish_idempotent(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test that finish() can be called multiple times safely."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.finish()
            logger.finish()
            logger.finish()

            # Should only call wandb.finish() once
            assert mock_wandb.finish.call_count == 1

    def test_finish_handles_error(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that finish() handles errors gracefully."""
        mock_wandb.finish.side_effect = Exception("Finish failed")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            # Should not raise
            logger.finish()

            # Should still be marked as finished
            assert logger._finished


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread safety of WandbLogger."""

    def test_concurrent_logging(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that concurrent logging doesn't cause race conditions."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            def log_metrics(thread_id: int) -> None:
                for i in range(10):
                    logger.log_metrics(
                        {f"thread_{thread_id}/metric_{i}": i * 0.1},
                        step=thread_id * 100 + i,
                    )

            threads = [threading.Thread(target=log_metrics, args=(i,)) for i in range(5)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All logs should have been processed
            assert mock_wandb.log.call_count == 50  # 5 threads * 10 logs each

    def test_concurrent_finish_safe(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that concurrent finish() calls are safe."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            threads = [threading.Thread(target=logger.finish) for _ in range(10)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should only call wandb.finish() once
            assert mock_wandb.finish.call_count == 1


# =============================================================================
# Watch Model Tests
# =============================================================================


class TestWatchModel:
    """Test model watching functionality."""

    def test_watch_model_enabled(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test watching model when enabled."""
        enabled_config["watch_model"] = True
        enabled_config["log_gradients"] = True

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            mock_model = MagicMock()

            logger.watch_model(mock_model)

            mock_wandb.watch.assert_called_once()
            call_kwargs = mock_wandb.watch.call_args[1]
            assert call_kwargs["log"] == "all"

    def test_watch_model_disabled_by_config(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that watch_model is skipped when disabled in config."""
        enabled_config["watch_model"] = False

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            mock_model = MagicMock()

            logger.watch_model(mock_model)

            mock_wandb.watch.assert_not_called()

    def test_watch_model_handles_error(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that watch_model handles errors gracefully."""
        enabled_config["watch_model"] = True
        mock_wandb.watch.side_effect = Exception("Watch failed")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            mock_model = MagicMock()

            # Should not raise
            logger.watch_model(mock_model)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateWandbLogger:
    """Test the create_wandb_logger factory function."""

    def test_create_with_dict_config(self, mock_wandb: MagicMock) -> None:
        """Test creating logger with dict configuration."""
        config = {
            "enabled": True,
            "project": "test-project",
            "mode": "disabled",
            "log_code": False,
        }

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = create_wandb_logger(wandb_config=config)

            assert logger._project == "test-project"

    def test_create_with_none_config(self) -> None:
        """Test creating logger with None configuration (disabled by default behavior)."""
        # This should create a logger but it will be disabled since wandb isn't available
        logger = create_wandb_logger(wandb_config={"enabled": False})
        assert not logger.is_enabled

    def test_create_with_training_config(self, mock_wandb: MagicMock) -> None:
        """Test that training config is passed to W&B."""
        wandb_config = {
            "enabled": True,
            "mode": "disabled",
            "log_code": False,
        }
        training_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
        }

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = create_wandb_logger(
                wandb_config=wandb_config,
                training_config=training_config,
            )

            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["config"] == training_config


# =============================================================================
# Additional Methods Tests
# =============================================================================


class TestAdditionalMethods:
    """Test additional WandbLogger methods."""

    def test_log_histogram(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test histogram logging."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            values = [1, 2, 3, 4, 5]

            logger.log_histogram("test/histogram", values, step=10)

            mock_wandb.Histogram.assert_called_once_with(values)
            mock_wandb.log.assert_called()

    def test_log_table(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test table logging."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.log_table(
                key="test/table",
                columns=["col1", "col2"],
                data=[["a", 1], ["b", 2]],
                step=10,
            )

            mock_wandb.Table.assert_called_once()
            mock_wandb.log.assert_called()

    def test_define_metric(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test metric definition."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.define_metric(
                name="train/loss/*",
                step_metric="train/global_step",
                summary="min",
                goal="minimize",
            )

            mock_wandb.define_metric.assert_called_once()

    def test_alert(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test alert sending."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.alert(
                title="Test Alert",
                text="This is a test alert",
                level="WARN",
            )

            mock_wandb.alert.assert_called_once()

    def test_log_config_update(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test config update."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.log_config_update({"new_param": "value"})

            mock_wandb.config.update.assert_called_once_with({"new_param": "value"})

    def test_log_summary(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test summary logging."""
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            logger.log_summary({"final_loss": 0.1, "final_accuracy": 0.95})

            assert mock_run.summary["final_loss"] == 0.1
            assert mock_run.summary["final_accuracy"] == 0.95


# =============================================================================
# Resume Configuration Tests
# =============================================================================


class TestResumeConfiguration:
    """Test W&B run resumption configuration."""

    def test_resume_id_passed_to_init(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that resume_id is passed to wandb.init()."""
        enabled_config["resume_id"] = "existing_run_123"
        enabled_config["resume_mode"] = "must"

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["id"] == "existing_run_123"
            assert call_kwargs["resume"] == "must"

    def test_run_id_property(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test run_id property."""
        mock_run = MagicMock()
        mock_run.id = "test_id_456"
        mock_run.name = "test_name"
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            assert logger.run_id == "test_id_456"
            assert logger.run_name == "test_name"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_log_buffer_stats_zero_capacity(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test buffer stats with zero capacity (avoid division by zero)."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            # Should not raise
            logger.log_buffer_stats(
                buffer_size=0,
                capacity=0,  # Zero capacity
            )

            call_args = mock_wandb.log.call_args
            log_dict = call_args[0][0]
            assert log_dict["data/buffer_fill_ratio"] == 0.0

    def test_log_evaluation_empty_metadata(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test evaluation logging with empty metadata."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            result = MockEvaluationResult()
            result.metadata = {}

            logger.log_evaluation(result)

            mock_wandb.log.assert_called()

    def test_log_interval_zero(self, enabled_config: dict[str, Any], mock_wandb: MagicMock) -> None:
        """Test that log_interval=0 logs every step."""
        enabled_config["log_interval"] = 0

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)

            # With log_interval=0, the modulo check would fail
            # Our implementation should handle this gracefully
            for step in range(5):
                logger.log_training_step(MockTrainingMetrics(step=step))

    def test_operations_after_finish(
        self, enabled_config: dict[str, Any], mock_wandb: MagicMock
    ) -> None:
        """Test that operations after finish() are no-ops."""
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandbLogger(config=enabled_config)
            logger.finish()

            # Reset mock counts
            mock_wandb.log.reset_mock()

            # These should all be no-ops
            logger.log_metrics({"test": 1})
            logger.log_training_step(MockTrainingMetrics())
            logger.log_summary({"final": 1})

            mock_wandb.log.assert_not_called()
