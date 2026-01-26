"""Weights & Biases logging integration for AlphaGalerkin.

Provides comprehensive experiment tracking including:
- Training metrics (losses, learning rate, gradients)
- Buffer statistics (size, distribution)
- Evaluation metrics (win rates, policy agreement)
- System metrics (device, model parameters)
- Model checkpoints and artifacts

This module provides thread-safe logging with graceful degradation
when W&B is unavailable or initialization fails.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from src.training.evaluation import EvaluationResult
    from src.training.trainer import TrainingMetrics

logger = structlog.get_logger(__name__)


# Valid W&B modes for type safety
WandbMode = Literal["online", "offline", "disabled"]

# Default configuration values (centralized, no hardcoding)
DEFAULT_PROJECT = "alphagalerkin"
DEFAULT_JOB_TYPE = "train"
DEFAULT_MODE: WandbMode = "online"
DEFAULT_LOG_INTERVAL = 1
DEFAULT_WATCH_LOG_FREQ = 100


class LogLevel(str, Enum):
    """Alert levels for W&B alerts."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class WandbLogger:
    """Weights & Biases logger for AlphaGalerkin training.

    Handles all W&B interactions including initialization,
    metric logging, artifact management, and cleanup.

    Thread-safe implementation with graceful degradation when
    W&B is unavailable or initialization fails.

    Attributes:
        config: W&B configuration dictionary.
        is_enabled: Whether W&B logging is active.
        run: The W&B run object (if initialized).

    Example:
        >>> from config.schemas import WandbConfig
        >>> config = WandbConfig(enabled=True, project="my-project")
        >>> wandb_logger = create_wandb_logger(config.model_dump())
        >>> wandb_logger.log_metrics({"loss": 0.5}, step=1)
        >>> wandb_logger.finish()
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize W&B logger.

        Args:
            config: W&B configuration dictionary with fields matching WandbConfig.
            training_config: Full training configuration to log.
        """
        self._config = config or {}
        self._training_config = training_config or {}
        self._run: Any = None
        self._wandb: Any = None
        self._initialized = False
        self._step_offset = 0
        self._finished = False
        self._atexit_registered = False

        # Thread safety lock for all mutable state
        self._lock = threading.RLock()

        # Extract config values with defaults
        self._enabled = self._config.get("enabled", True)
        self._project = self._config.get("project", DEFAULT_PROJECT)
        self._entity = self._config.get("entity")
        self._name = self._config.get("name")
        self._tags = self._safe_list(self._config.get("tags"))
        self._notes = self._config.get("notes")
        self._group = self._config.get("group")
        self._job_type = self._config.get("job_type", DEFAULT_JOB_TYPE)
        self._mode = self._config.get("mode", DEFAULT_MODE)
        self._log_model = self._config.get("log_model", True)
        self._log_gradients = self._config.get("log_gradients", False)
        self._log_code = self._config.get("log_code", True)
        self._log_interval = self._config.get("log_interval", DEFAULT_LOG_INTERVAL)
        self._watch_model_enabled = self._config.get("watch_model", False)
        self._watch_log_freq = self._config.get("watch_log_freq", DEFAULT_WATCH_LOG_FREQ)

        # Resume configuration for W&B run resumption
        self._resume_id = self._config.get("resume_id")
        self._resume_mode = self._config.get("resume_mode", "allow")

        if not self._enabled:
            logger.info("wandb_disabled")
            return

        self._initialize()

    @staticmethod
    def _safe_list(value: Any) -> list[str]:
        """Safely convert a value to a list of strings.

        Handles None, empty lists, and other edge cases.

        Args:
            value: Value to convert (may be None, list, or other).

        Returns:
            A list of strings (empty list if value is None or invalid).
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return []

    def _initialize(self) -> None:
        """Initialize W&B run with comprehensive error handling."""
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            logger.warning(
                "wandb_import_failed",
                message="Install wandb with: pip install wandb",
            )
            self._enabled = False
            return

        # Initialize run with comprehensive error handling
        try:
            init_kwargs: dict[str, Any] = {
                "project": self._project,
                "entity": self._entity,
                "name": self._name,
                "tags": self._tags if self._tags else None,
                "notes": self._notes,
                "group": self._group,
                "job_type": self._job_type,
                "mode": self._mode,
                "config": self._training_config,
                "reinit": True,
            }

            # Add resume configuration if provided
            if self._resume_id:
                init_kwargs["id"] = self._resume_id
                init_kwargs["resume"] = self._resume_mode
                logger.info(
                    "wandb_resuming_run",
                    run_id=self._resume_id,
                    resume_mode=self._resume_mode,
                )

            self._run = self._wandb.init(**init_kwargs)

        except Exception as e:
            logger.warning(
                "wandb_init_failed",
                error=str(e),
                error_type=type(e).__name__,
                message="W&B logging disabled, training will continue without it",
            )
            self._enabled = False
            return

        self._initialized = True
        logger.info(
            "wandb_initialized",
            project=self._project,
            run_name=self._run.name if self._run else None,
            run_id=self._run.id if self._run else None,
        )

        # Log code if enabled (best-effort with logging)
        if self._log_code:
            try:
                if self._run is not None:
                    self._run.log_code(".")
            except Exception as e:
                logger.debug(
                    "wandb_log_code_failed",
                    error=str(e),
                    message="Code logging skipped, continuing without it",
                )

        # Register cleanup (only once)
        with self._lock:
            if not self._atexit_registered:
                atexit.register(self._atexit_finish)
                self._atexit_registered = True

    def _atexit_finish(self) -> None:
        """Atexit handler for cleanup - calls finish safely."""
        self.finish()

    @property
    def is_enabled(self) -> bool:
        """Check if W&B logging is enabled and initialized.

        Thread-safe property access.

        Returns:
            True if W&B is enabled and successfully initialized.
        """
        with self._lock:
            return self._enabled and self._initialized and not self._finished

    @property
    def run(self) -> Any:
        """Get the W&B run object.

        Returns:
            The W&B run object, or None if not initialized.
        """
        with self._lock:
            return self._run

    @property
    def run_id(self) -> str | None:
        """Get the W&B run ID for potential resumption.

        Returns:
            The run ID string, or None if not initialized.
        """
        with self._lock:
            return self._run.id if self._run else None

    @property
    def run_name(self) -> str | None:
        """Get the W&B run name.

        Returns:
            The run name string, or None if not initialized.
        """
        with self._lock:
            return self._run.name if self._run else None

    def set_step_offset(self, offset: int) -> None:
        """Set step offset for resumed training.

        Args:
            offset: Step number to start from.
        """
        with self._lock:
            self._step_offset = offset
            logger.debug("wandb_step_offset_set", offset=offset)

    def watch_model(self, model: Any) -> None:
        """Enable gradient and parameter logging for model.

        Args:
            model: PyTorch model to watch.
        """
        if not self.is_enabled or not self._watch_model_enabled:
            return

        try:
            log_type = "all" if self._log_gradients else "parameters"
            self._wandb.watch(
                model,
                log=log_type,
                log_freq=self._watch_log_freq,
            )
            logger.info("wandb_watching_model", log_type=log_type)
        except Exception as e:
            logger.warning(
                "wandb_watch_model_failed",
                error=str(e),
                message="Model watching disabled, continuing without it",
            )

    def _apply_step_offset(self, step: int | None) -> int | None:
        """Apply step offset to a step value.

        Args:
            step: Original step value (may be None).

        Returns:
            Step with offset applied, or None if input is None.
        """
        if step is None:
            return None
        return step + self._step_offset

    def log_training_step(
        self,
        metrics: TrainingMetrics,
        commit: bool = True,
    ) -> None:
        """Log training step metrics.

        Args:
            metrics: Training metrics dataclass.
            commit: Whether to commit the log (advance step).
        """
        if not self.is_enabled:
            return

        step = metrics.step + self._step_offset

        # Only log at configured interval
        if self._log_interval > 0 and step % self._log_interval != 0:
            return

        log_dict = {
            # Loss components
            "train/loss/total": metrics.total_loss,
            "train/loss/policy": metrics.policy_loss,
            "train/loss/value": metrics.value_loss,
            "train/loss/lbb": metrics.lbb_loss,
            # Stability metrics
            "train/lbb_constant": metrics.lbb_constant,
            "train/gradient_norm": metrics.gradient_norm,
            # Learning dynamics
            "train/learning_rate": metrics.learning_rate,
            # Buffer statistics
            "data/buffer_size": metrics.buffer_size,
            "data/games_generated": metrics.games_generated,
            # Performance
            "perf/step_time_ms": metrics.step_time_ms,
        }

        self._safe_log(log_dict, step=step, commit=commit)

    def log_evaluation(
        self,
        result: EvaluationResult,
        prefix: str = "eval",
        step: int | None = None,
    ) -> None:
        """Log evaluation results.

        Args:
            result: Evaluation result dataclass.
            prefix: Metric prefix (e.g., 'eval', 'eval/9x9').
            step: Training step (None for current).
        """
        if not self.is_enabled:
            return

        log_dict = {
            f"{prefix}/win_rate": result.win_rate,
            f"{prefix}/n_games": result.n_games,
            f"{prefix}/wins": result.wins,
            f"{prefix}/losses": result.losses,
            f"{prefix}/draws": result.draws,
            f"{prefix}/avg_game_length": result.avg_game_length,
            f"{prefix}/avg_value_error": result.avg_value_error,
            f"{prefix}/policy_agreement": result.policy_agreement,
        }

        # Add metadata (filter to loggable types)
        for key, value in result.metadata.items():
            if isinstance(value, (int, float, str, bool)):
                log_dict[f"{prefix}/meta/{key}"] = value

        adjusted_step = self._apply_step_offset(step)
        self._safe_log(log_dict, step=adjusted_step)

    def log_buffer_stats(
        self,
        buffer_size: int,
        capacity: int,
        value_mean: float | None = None,
        value_std: float | None = None,
        board_size_distribution: dict[int, int] | None = None,
        step: int | None = None,
    ) -> None:
        """Log replay buffer statistics.

        Args:
            buffer_size: Current buffer size.
            capacity: Buffer capacity.
            value_mean: Mean of target values in buffer.
            value_std: Std of target values in buffer.
            board_size_distribution: Distribution of board sizes.
            step: Training step.
        """
        if not self.is_enabled:
            return

        log_dict: dict[str, Any] = {
            "data/buffer_size": buffer_size,
            "data/buffer_fill_ratio": buffer_size / capacity if capacity > 0 else 0.0,
        }

        if value_mean is not None:
            log_dict["data/value_mean"] = value_mean
        if value_std is not None:
            log_dict["data/value_std"] = value_std

        if board_size_distribution:
            for size, count in board_size_distribution.items():
                log_dict[f"data/board_size_{size}x{size}"] = count

        adjusted_step = self._apply_step_offset(step)
        self._safe_log(log_dict, step=adjusted_step)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log arbitrary metrics.

        Args:
            metrics: Dictionary of metrics.
            step: Training step.
            commit: Whether to commit the log.
        """
        if not self.is_enabled:
            return

        adjusted_step = self._apply_step_offset(step)
        self._safe_log(metrics, step=adjusted_step, commit=commit)

    def _safe_log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Safely log data with error handling.

        Args:
            data: Data dictionary to log.
            step: Training step.
            commit: Whether to commit the log.
        """
        if not self.is_enabled:
            return

        try:
            self._wandb.log(data, step=step, commit=commit)
        except Exception as e:
            logger.debug("wandb_log_failed", error=str(e), data_keys=list(data.keys()))

    def log_histogram(
        self,
        key: str,
        values: Any,
        step: int | None = None,
    ) -> None:
        """Log histogram of values.

        Args:
            key: Metric key.
            values: Array of values.
            step: Training step.
        """
        if not self.is_enabled:
            return

        try:
            adjusted_step = self._apply_step_offset(step)
            self._wandb.log({key: self._wandb.Histogram(values)}, step=adjusted_step)
        except Exception as e:
            logger.debug("wandb_histogram_failed", error=str(e), key=key)

    def log_model_artifact(
        self,
        checkpoint_path: Path | str,
        name: str = "model",
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Log model checkpoint as W&B artifact.

        Args:
            checkpoint_path: Path to checkpoint file.
            name: Artifact name.
            metadata: Additional metadata.
            aliases: Artifact aliases (e.g., ['best', 'latest']).
        """
        if not self.is_enabled or not self._log_model:
            return

        try:
            artifact = self._wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {},
            )
            artifact.add_file(str(checkpoint_path))
            self._wandb.log_artifact(artifact, aliases=aliases or [])

            logger.info(
                "wandb_artifact_logged",
                name=name,
                path=str(checkpoint_path),
                aliases=aliases,
            )
        except Exception as e:
            logger.warning("wandb_artifact_failed", error=str(e))

    def log_config_update(self, config_updates: dict[str, Any]) -> None:
        """Update run configuration.

        Args:
            config_updates: Configuration updates.
        """
        if not self.is_enabled:
            return

        try:
            self._wandb.config.update(config_updates)
        except Exception as e:
            logger.debug("wandb_config_update_failed", error=str(e))

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary metrics (final values).

        Args:
            summary: Summary metrics.
        """
        if not self.is_enabled:
            return

        with self._lock:
            if self._run is None:
                logger.debug("wandb_log_summary_skipped", reason="run is None")
                return

            try:
                for key, value in summary.items():
                    self._run.summary[key] = value
            except Exception as e:
                logger.debug("wandb_summary_failed", error=str(e))

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        """Log a table of data.

        Args:
            key: Table key.
            columns: Column names.
            data: Table data (list of rows).
            step: Training step.
        """
        if not self.is_enabled:
            return

        try:
            table = self._wandb.Table(columns=columns, data=data)
            adjusted_step = self._apply_step_offset(step)
            self._wandb.log({key: table}, step=adjusted_step)
        except Exception as e:
            logger.debug("wandb_table_failed", error=str(e), key=key)

    def define_metric(
        self,
        name: str,
        step_metric: str = "trainer/global_step",
        summary: str | None = None,
        goal: str | None = None,
    ) -> None:
        """Define a metric with custom step and summary.

        Args:
            name: Metric name (supports wildcards).
            step_metric: Step metric to use.
            summary: Summary type ('min', 'max', 'mean', 'last', 'none').
            goal: Goal for optimization ('minimize', 'maximize').
        """
        if not self.is_enabled:
            return

        try:
            kwargs: dict[str, Any] = {"step_metric": step_metric}
            if summary is not None:
                kwargs["summary"] = summary
            if goal is not None:
                kwargs["goal"] = goal

            self._wandb.define_metric(name, **kwargs)
        except Exception as e:
            logger.debug("wandb_define_metric_failed", error=str(e), name=name)

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ) -> None:
        """Send an alert.

        Args:
            title: Alert title.
            text: Alert text.
            level: Alert level ('INFO', 'WARN', 'ERROR').
        """
        if not self.is_enabled:
            return

        try:
            alert_level = getattr(
                self._wandb.AlertLevel, level, self._wandb.AlertLevel.INFO
            )
            self._wandb.alert(title=title, text=text, level=alert_level)
        except Exception as e:
            logger.debug("wandb_alert_failed", error=str(e), title=title)

    def finish(self) -> None:
        """Finish W&B run and cleanup.

        Thread-safe and idempotent - can be called multiple times safely.
        """
        with self._lock:
            if self._finished:
                return

            if self._run is not None:
                try:
                    self._wandb.finish()
                except Exception as e:
                    logger.debug("wandb_finish_error", error=str(e))

                self._run = None
                self._initialized = False
                logger.info("wandb_finished")

            self._finished = True


def create_wandb_logger(
    wandb_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> WandbLogger:
    """Create and configure a W&B logger instance.

    Factory function that creates a WandbLogger with proper configuration.
    Supports both raw dictionaries and Pydantic model dumps.

    Args:
        wandb_config: W&B configuration dictionary.
            Expected keys: enabled, project, entity, name, tags, notes,
            group, job_type, mode, log_model, log_gradients, log_code,
            log_interval, watch_model, watch_log_freq.
        training_config: Full training configuration to log.

    Returns:
        Configured WandbLogger instance.

    Example:
        >>> from config.schemas import WandbConfig
        >>> config = WandbConfig(enabled=True, project="my-project")
        >>> wandb_logger = create_wandb_logger(config.model_dump())
    """
    return WandbLogger(config=wandb_config, training_config=training_config)
