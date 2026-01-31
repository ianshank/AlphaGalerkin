"""Vertex AI-aware trainer wrapper.

This module provides a trainer wrapper that integrates with the existing
AlphaGalerkin training infrastructure while adding Vertex AI-specific
features like GCS checkpointing, preemption handling, and cost tracking.

Example:
    from src.vertex.trainer import VertexTrainer

    trainer = VertexTrainer(
        model=model,
        config=training_config,
        vertex_config=vertex_config,
    )

    results = trainer.train(resume_from="gs://bucket/checkpoint.pt")

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.vertex.config import VertexTrainingConfig
from src.vertex.cost import CostTracker
from src.vertex.multi_node import DistributedContext, setup_distributed_training
from src.vertex.preemption import PreemptionHandler, create_preemption_handler
from src.vertex.storage import GCSCheckpointManager

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


def _get_torch() -> Any:
    """Lazily import torch to avoid import errors when not installed."""
    import torch

    return torch


def _get_dist() -> Any:
    """Lazily import torch.distributed."""
    import torch.distributed as dist

    return dist

logger = structlog.get_logger(__name__)


@dataclass
class VertexTrainingResult:
    """Result of Vertex AI training run.

    Attributes:
        status: Training status (completed, preempted, failed).
        final_step: Last completed training step.
        final_checkpoint: Path to final checkpoint.
        metrics: Final training metrics.
        cost_estimate: Estimated training cost.
        preemption_event: Preemption details if preempted.

    """

    status: str
    final_step: int
    final_checkpoint: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    cost_estimate: dict[str, Any] | None = None
    preemption_event: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "final_step": self.final_step,
            "final_checkpoint": self.final_checkpoint,
            "metrics": self.metrics,
            "cost_estimate": self.cost_estimate,
            "preemption_event": self.preemption_event,
        }


class VertexTrainer:
    """Vertex AI-aware training wrapper.

    This class wraps the training process with Vertex AI-specific
    functionality:
    - GCS checkpoint management with local caching
    - Preemption handling for spot instances
    - Cost tracking and estimation
    - Distributed training setup

    The trainer is designed to work with any PyTorch model and
    integrates with the existing AlphaGalerkin training infrastructure.

    Example:
        trainer = VertexTrainer(
            model=model,
            config=training_config,
            vertex_config=vertex_config,
        )

        # Resume from checkpoint if available
        resume_path = "gs://bucket/checkpoints/checkpoint_00010000.pt"
        results = trainer.train(resume_from=resume_path)

        print(f"Training {results.status} at step {results.final_step}")

    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        vertex_config: VertexTrainingConfig,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        train_step_fn: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        """Initialize Vertex AI trainer.

        Args:
            model: PyTorch model to train.
            config: Training configuration dictionary.
            vertex_config: Vertex AI configuration.
            optimizer: Optimizer (created if None).
            scheduler: LR scheduler (optional).
            train_step_fn: Custom training step function.

        """
        self.model = model
        self.config = config
        self.vertex_config = vertex_config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._train_step_fn = train_step_fn

        # Initialize components
        self._distributed_ctx: DistributedContext | None = None
        self._checkpoint_manager: GCSCheckpointManager | None = None
        self._preemption_handler: PreemptionHandler | None = None
        self._cost_tracker: CostTracker | None = None

        # Training state
        self._current_step = 0
        self._best_metric: float | None = None
        self._metrics_history: list[dict[str, float]] = []

        logger.info(
            "vertex_trainer_initialized",
            model_class=model.__class__.__name__,
            project=vertex_config.project_id,
        )

    def setup(self) -> None:
        """Set up training components.

        This method initializes:
        - Distributed training context
        - GCS checkpoint manager
        - Preemption handler
        - Cost tracker

        Should be called before training starts.
        """
        # Setup distributed training
        self._distributed_ctx = setup_distributed_training()

        # Setup GCS checkpoint manager
        self._checkpoint_manager = GCSCheckpointManager(
            bucket_name=self.vertex_config.storage.bucket_name,
            checkpoint_prefix=self.vertex_config.storage.checkpoint_prefix,
            local_cache_dir=Path(self.vertex_config.storage.local_cache_dir),
            max_checkpoints=self.vertex_config.storage.max_checkpoints,
        )

        # Setup preemption handler
        checkpoint_interval = self.vertex_config.get_effective_checkpoint_interval() * 60  # Convert to steps
        self._preemption_handler = create_preemption_handler(
            checkpoint_callback=self._emergency_checkpoint,
            enable_spot=self.vertex_config.enable_spot,
            checkpoint_interval=checkpoint_interval,
        )

        # Setup cost tracker
        self._cost_tracker = CostTracker()
        self._cost_tracker.start(
            machine_type=self.vertex_config.resources.machine_type,
            accelerator_type=self.vertex_config.resources.accelerator_type,
            accelerator_count=self.vertex_config.resources.accelerator_count,
            replica_count=self.vertex_config.resources.replica_count,
            is_spot=self.vertex_config.enable_spot,
        )

        # Move model to device
        torch = _get_torch()
        if torch.cuda.is_available() and self._distributed_ctx.local_rank < torch.cuda.device_count():
            device = torch.device(f"cuda:{self._distributed_ctx.local_rank}")
            self.model = self.model.to(device)

        # Wrap with DDP if distributed
        if self._distributed_ctx.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[self._distributed_ctx.local_rank],
            )

        logger.info(
            "vertex_trainer_setup_complete",
            rank=self._distributed_ctx.rank,
            world_size=self._distributed_ctx.world_size,
            device=str(next(self.model.parameters()).device),
        )

    def train(
        self,
        total_steps: int | None = None,
        resume_from: str | None = None,
    ) -> VertexTrainingResult:
        """Run training loop.

        Args:
            total_steps: Total steps to train (uses config if None).
            resume_from: GCS or local path to resume from.

        Returns:
            VertexTrainingResult with training outcomes.

        """
        # Setup if not already done
        if self._checkpoint_manager is None:
            self.setup()

        assert self._checkpoint_manager is not None
        assert self._preemption_handler is not None
        assert self._cost_tracker is not None
        assert self._distributed_ctx is not None

        # Get total steps from config
        if total_steps is None:
            total_steps = self.config.get("training", {}).get("total_steps", 10000)

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        elif self._checkpoint_manager.get_latest_step() is not None:
            # Auto-resume from latest checkpoint
            logger.info("auto_resuming_from_latest_checkpoint")
            try:
                state = self._checkpoint_manager.load_latest()
                self._load_state(state)
            except Exception as e:
                logger.warning("auto_resume_failed", error=str(e))

        # Log training start
        logger.info(
            "training_started",
            current_step=self._current_step,
            total_steps=total_steps,
            is_main=self._distributed_ctx.is_main_process(),
        )

        # Training loop
        status = "completed"
        try:
            while self._current_step < total_steps:
                # Check for preemption
                if self._preemption_handler.is_preempted:
                    status = "preempted"
                    logger.warning("training_preempted", step=self._current_step)
                    break

                # Execute training step
                metrics = self._train_step()
                self._current_step += 1

                # Update preemption handler
                self._preemption_handler.update_step(self._current_step)

                # Log metrics periodically
                if self._current_step % 100 == 0 and self._distributed_ctx.is_main_process():
                    self._log_metrics(metrics)

                # Checkpoint based on preemption handler recommendation
                if (self._preemption_handler.should_save_checkpoint(self._current_step)
                        and self._distributed_ctx.is_main_process()):
                    self._save_checkpoint(metrics)

        except Exception as e:
            status = "failed"
            logger.exception("training_failed", error=str(e))
            raise

        finally:
            # Save final checkpoint
            if self._distributed_ctx.is_main_process():
                final_path = self._save_checkpoint(
                    metrics={"final": True},
                    force=True,
                )
            else:
                final_path = None

            # Stop cost tracking
            self._cost_tracker.stop()

            # Cleanup preemption handler
            self._preemption_handler.cleanup()

        # Build result
        cost = self._cost_tracker.get_current_cost()
        preemption = None
        if self._preemption_handler.preemption_event:
            preemption = self._preemption_handler.preemption_event.to_dict()

        return VertexTrainingResult(
            status=status,
            final_step=self._current_step,
            final_checkpoint=final_path,
            metrics=self._get_latest_metrics(),
            cost_estimate=cost.to_dict() if cost else None,
            preemption_event=preemption,
        )

    def _train_step(self) -> dict[str, float]:
        """Execute a single training step.

        Returns:
            Dictionary of metrics from the step.

        """
        if self._train_step_fn is not None:
            return self._train_step_fn()

        # Default placeholder training step
        # In practice, this would be provided by the user
        self.model.train()
        return {"loss": 0.0, "step": self._current_step}

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint.

        Args:
            checkpoint_path: GCS or local checkpoint path.

        """
        assert self._checkpoint_manager is not None

        logger.info("resuming_from_checkpoint", path=checkpoint_path)

        try:
            state = self._checkpoint_manager.load(gcs_path=checkpoint_path)
            self._load_state(state)
        except FileNotFoundError:
            logger.warning("checkpoint_not_found", path=checkpoint_path)
            raise

    def _load_state(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: Checkpoint state dictionary.

        """
        # Load model state
        model_state = state["model_state_dict"]

        # Handle DDP wrapped model
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)

        # Load optimizer state
        if self.optimizer and state.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        # Restore training step
        self._current_step = state.get("step", 0)

        logger.info(
            "state_loaded",
            step=self._current_step,
            has_optimizer=state.get("optimizer_state_dict") is not None,
        )

    def _save_checkpoint(
        self,
        metrics: dict[str, float] | None = None,
        force: bool = False,
    ) -> str | None:
        """Save checkpoint to GCS.

        Args:
            metrics: Training metrics to include.
            force: Force save even if not main process.

        Returns:
            GCS path of saved checkpoint.

        """
        assert self._checkpoint_manager is not None
        assert self._distributed_ctx is not None

        if not force and not self._distributed_ctx.is_main_process():
            return None

        metrics = metrics or {}

        # Get model state (handle DDP)
        model = self.model.module if hasattr(self.model, "module") else self.model

        gcs_path = self._checkpoint_manager.save(
            step=self._current_step,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            metrics=metrics,
        )

        logger.info(
            "checkpoint_saved",
            step=self._current_step,
            path=gcs_path,
        )

        return gcs_path

    def _emergency_checkpoint(self) -> None:
        """Save emergency checkpoint on preemption."""
        try:
            self._save_checkpoint(
                metrics={"emergency": True, "preempted": True},
                force=True,
            )
        except Exception as e:
            logger.error("emergency_checkpoint_failed", error=str(e))

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log training metrics.

        Args:
            metrics: Metrics to log.

        """
        assert self._cost_tracker is not None

        # Add cost estimate
        cost = self._cost_tracker.get_current_cost()
        if cost:
            metrics["cost_usd"] = cost.estimated_total_cost

        logger.info(
            "training_metrics",
            step=self._current_step,
            **metrics,
        )

        self._metrics_history.append(metrics.copy())

    def _get_latest_metrics(self) -> dict[str, float]:
        """Get most recent metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return {}

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self._distributed_ctx is None:
            return True
        return self._distributed_ctx.is_main_process()

    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self._current_step

    @property
    def distributed_context(self) -> DistributedContext | None:
        """Get distributed training context."""
        return self._distributed_ctx


def create_vertex_trainer(
    model: nn.Module,
    config: dict[str, Any],
    vertex_config: VertexTrainingConfig,
    **kwargs: Any,
) -> VertexTrainer:
    """Factory function to create Vertex AI trainer.

    Args:
        model: PyTorch model to train.
        config: Training configuration.
        vertex_config: Vertex AI configuration.
        **kwargs: Additional trainer arguments.

    Returns:
        Configured VertexTrainer instance.

    """
    return VertexTrainer(
        model=model,
        config=config,
        vertex_config=vertex_config,
        **kwargs,
    )
