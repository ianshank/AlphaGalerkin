"""Weights & Biases logging integration for AlphaGalerkin.

Provides comprehensive experiment tracking including:
- Training metrics (losses, learning rate, gradients)
- Buffer statistics (size, distribution)
- Evaluation metrics (win rates, policy agreement)
- System metrics (device, model parameters)
- Model checkpoints and artifacts
"""

from __future__ import annotations

import atexit
import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.training.evaluation import EvaluationResult
    from src.training.trainer import TrainingMetrics

logger = structlog.get_logger(__name__)


@dataclass
class WandbConfig:
    """Configuration for W&B logging.

    Attributes:
        enabled: Whether W&B logging is enabled.
        project: W&B project name.
        entity: W&B team/user entity (None for default).
        name: Run name (None for auto-generated).
        tags: Tags for the run.
        notes: Notes for the run.
        group: Group name for organizing runs.
        job_type: Type of job (train, eval, etc.).
        mode: W&B mode ('online', 'offline', 'disabled').
        log_model: Whether to log model checkpoints as artifacts.
        log_gradients: Whether to log gradient histograms.
        log_code: Whether to log source code.
        log_interval: Steps between metric logging (separate from trainer log_interval).
        watch_model: Whether to use wandb.watch() on model.
        watch_log_freq: Frequency for gradient logging when watching.

    """

    enabled: bool = True
    project: str = "alphagalerkin"
    entity: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    group: str | None = None
    job_type: str = "train"
    mode: str = "online"
    log_model: bool = True
    log_gradients: bool = False
    log_code: bool = True
    log_interval: int = 1  # Log every step by default
    watch_model: bool = False
    watch_log_freq: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class WandbLogger:
    """Weights & Biases logger for AlphaGalerkin training.

    Handles all W&B interactions including initialization,
    metric logging, artifact management, and cleanup.
    """

    def __init__(
        self,
        config: WandbConfig | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize W&B logger.

        Args:
            config: W&B configuration.
            training_config: Full training configuration to log.

        """
        self.config = config or WandbConfig()
        self._training_config = training_config or {}
        self._run: Any = None
        self._initialized = False
        self._step_offset = 0

        if not self.config.enabled:
            logger.info("wandb_disabled")
            return

        self._initialize()

    def _initialize(self) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            logger.warning(
                "wandb_import_failed",
                message="Install wandb with: pip install wandb",
            )
            self.config.enabled = False
            return

        # Initialize run
        self._run = self._wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.name,
            tags=self.config.tags,
            notes=self.config.notes,
            group=self.config.group,
            job_type=self.config.job_type,
            mode=self.config.mode,
            config=self._training_config,
            reinit=True,
        )

        self._initialized = True
        logger.info(
            "wandb_initialized",
            project=self.config.project,
            run_name=self._run.name if self._run else None,
            run_id=self._run.id if self._run else None,
        )

        # Log code if enabled (best-effort, may fail in some environments)
        if self.config.log_code:
            with contextlib.suppress(Exception):
                self._wandb.run.log_code(".")

        # Register cleanup
        atexit.register(self.finish)

    @property
    def is_enabled(self) -> bool:
        """Check if W&B logging is enabled and initialized."""
        return self.config.enabled and self._initialized

    @property
    def run(self) -> Any:
        """Get the W&B run object."""
        return self._run

    def set_step_offset(self, offset: int) -> None:
        """Set step offset for resumed training.

        Args:
            offset: Step number to start from.

        """
        self._step_offset = offset

    def watch_model(self, model: Any) -> None:
        """Enable gradient and parameter logging for model.

        Args:
            model: PyTorch model to watch.

        """
        if not self.is_enabled or not self.config.watch_model:
            return

        log_type = "all" if self.config.log_gradients else "parameters"
        self._wandb.watch(
            model,
            log=log_type,
            log_freq=self.config.watch_log_freq,
        )
        logger.info("wandb_watching_model", log_type=log_type)

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
        if step % self.config.log_interval != 0:
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

        self._wandb.log(log_dict, step=step, commit=commit)

    def log_evaluation(
        self,
        result: EvaluationResult,
        prefix: str = "eval",
        step: int | None = None,
    ) -> None:
        """Log evaluation results.

        Args:
            result: Evaluation result dataclass.
            prefix: Metric prefix (e.g., 'eval', 'eval_9x9').
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

        # Add metadata
        for key, value in result.metadata.items():
            if isinstance(value, (int, float, str, bool)):
                log_dict[f"{prefix}/meta/{key}"] = value

        if step is not None:
            step += self._step_offset

        self._wandb.log(log_dict, step=step)

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

        log_dict = {
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

        if step is not None:
            step += self._step_offset

        self._wandb.log(log_dict, step=step)

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

        if step is not None:
            step += self._step_offset

        self._wandb.log(metrics, step=step, commit=commit)

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

        if step is not None:
            step += self._step_offset

        self._wandb.log({key: self._wandb.Histogram(values)}, step=step)

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
        if not self.is_enabled or not self.config.log_model:
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

        self._wandb.config.update(config_updates)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary metrics (final values).

        Args:
            summary: Summary metrics.

        """
        if not self.is_enabled:
            return

        for key, value in summary.items():
            self._wandb.run.summary[key] = value

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

        table = self._wandb.Table(columns=columns, data=data)

        if step is not None:
            step += self._step_offset

        self._wandb.log({key: table}, step=step)

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

        kwargs: dict[str, Any] = {"step_metric": step_metric}
        if summary is not None:
            kwargs["summary"] = summary
        if goal is not None:
            kwargs["goal"] = goal

        self._wandb.define_metric(name, **kwargs)

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

        alert_level = getattr(self._wandb.AlertLevel, level, self._wandb.AlertLevel.INFO)
        self._wandb.alert(title=title, text=text, level=alert_level)

    def finish(self) -> None:
        """Finish W&B run and cleanup."""
        if self._run is not None:
            self._wandb.finish()
            self._run = None
            self._initialized = False
            logger.info("wandb_finished")


def create_wandb_logger(
    wandb_config: dict[str, Any] | WandbConfig | None = None,
    training_config: dict[str, Any] | None = None,
) -> WandbLogger:
    """Create and configure a W&B logger instance.

    Args:
        wandb_config: W&B configuration (dict or WandbConfig).
        training_config: Full training configuration.

    Returns:
        Configured WandbLogger.

    """
    if wandb_config is None:
        config = WandbConfig()
    elif isinstance(wandb_config, dict):
        config = WandbConfig(**wandb_config)
    else:
        config = wandb_config

    return WandbLogger(config=config, training_config=training_config)
