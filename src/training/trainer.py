"""Main training loop for AlphaGalerkin.

Orchestrates the complete training pipeline:
- Self-play game generation
- Replay buffer management
- Model training with mixed precision
- Checkpoint management
- Metrics logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler

from src.data.collate import TrainingBatch, VariableSizeCollator
from src.training.checkpoint import CheckpointManager
from src.training.evaluation import Evaluator
from src.training.loss import AlphaGalerkinLoss, LossOutput
from src.training.replay_buffer import create_replay_buffer
from src.training.self_play import SelfPlayWorker
from src.training.wandb_logger import WandbLogger

if TYPE_CHECKING:
    from config.schemas import AlphaGalerkinConfig
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    step: int = 0
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    lbb_loss: float = 0.0
    lbb_constant: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    buffer_size: int = 0
    games_generated: int = 0
    step_time_ms: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "total_loss": self.total_loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "lbb_loss": self.lbb_loss,
            "lbb_constant": self.lbb_constant,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "buffer_size": self.buffer_size,
            "games_generated": self.games_generated,
            "step_time_ms": self.step_time_ms,
        }


class Trainer:
    """Main trainer for AlphaGalerkin.

    Coordinates self-play, training, and checkpoint management.
    Supports mixed precision training and gradient accumulation.
    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        config: AlphaGalerkinConfig,
        device: torch.device | str = "auto",
        checkpoint_dir: Path | str | None = None,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: AlphaGalerkin model to train.
            config: Complete configuration.
            device: Training device ("auto" for automatic selection).
            checkpoint_dir: Directory for checkpoints.
            wandb_logger: Optional W&B logger for experiment tracking.

        """
        self.config = config
        self.training_config = config.training
        self.mcts_config = config.mcts
        self.wandb_logger = wandb_logger

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("trainer_initialized", device=str(self.device))

        # Model setup
        self.model = model.to(self.device)

        # Loss function
        self.loss_fn = AlphaGalerkinLoss(
            policy_weight=self.training_config.policy_loss_weight,
            value_weight=self.training_config.value_loss_weight,
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = self.training_config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Replay buffer
        self.buffer = create_replay_buffer(
            capacity=self.training_config.replay_buffer_size,
            prioritized=False,  # Use uniform for simplicity
        )

        # Self-play worker
        self.self_play_worker = SelfPlayWorker(
            model=self.model,
            mcts_config=self.mcts_config,
            device=self.device,
            board_sizes=getattr(config, "board_sizes", [9, 13, 19]),
        )

        # Collator for batching
        self.collator = VariableSizeCollator()

        # Checkpoint manager
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints") / config.experiment_name
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,
            keep_best=True,
            best_metric="total_loss",
            best_mode="min",
        )

        # Training state
        self.global_step = 0
        self.total_games_generated = 0

        # Metrics history
        self._metrics_history: list[TrainingMetrics] = []

        # Evaluator for periodic evaluation
        self.evaluator = Evaluator(
            model=self.model,
            mcts_config=self.mcts_config,
            device=self.device,
            board_sizes=getattr(config, "board_sizes", [9, 13, 19]),
        )

        # Watch model with W&B if enabled
        if self.wandb_logger is not None:
            self.wandb_logger.watch_model(self.model)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        return AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

    def _create_scheduler(self) -> LRScheduler:
        """Create learning rate scheduler from config."""
        scheduler_type = self.training_config.lr_scheduler
        warmup_steps = self.training_config.warmup_steps
        total_steps = self.training_config.total_steps

        if scheduler_type == "cosine":
            # Cosine annealing after warmup
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
            )
        elif scheduler_type == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - warmup_steps,
            )
        else:  # constant
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_steps,
            )

        # Add warmup
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )

        return main_scheduler

    def _fill_buffer(self, min_size: int) -> None:
        """Fill replay buffer to minimum size.

        Args:
            min_size: Minimum number of experiences needed.

        """
        while len(self.buffer) < min_size:
            n_games = self.training_config.n_self_play_games
            logger.info(
                "generating_self_play_games",
                n_games=n_games,
                buffer_size=len(self.buffer),
                target_size=min_size,
            )

            # Generate games
            self.model.eval()
            experiences = self.self_play_worker.generate_experiences(n_games)
            self.model.train()

            # Add to buffer
            self.buffer.add_batch(experiences)
            self.total_games_generated += n_games

    def _sample_batch(self) -> TrainingBatch:
        """Sample and collate a training batch.

        Returns:
            Collated training batch.

        """
        experiences = self.buffer.sample(self.training_config.batch_size)
        batch = self.collator(experiences)
        return batch.to(self.device)

    def _training_step(
        self, batch: TrainingBatch
    ) -> tuple[LossOutput, float | None, float]:
        """Execute single training step.

        Args:
            batch: Training batch.

        Returns:
            Tuple of (loss output, LBB constant, gradient norm).

        """
        self.optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast(device_type=self.device.type):
                output = self.model(batch.board_states, return_lbb=True)
                loss_output = self.loss_fn(
                    policy_logits=output.policy_logits,
                    value=output.value,
                    target_policy=batch.target_policies,
                    target_value=batch.target_values,
                    lbb_constant=output.lbb_constant,
                    action_mask=batch.action_mask.float(),
                )
        else:
            output = self.model(batch.board_states, return_lbb=True)
            loss_output = self.loss_fn(
                policy_logits=output.policy_logits,
                value=output.value,
                target_policy=batch.target_policies,
                target_value=batch.target_values,
                lbb_constant=output.lbb_constant,
                action_mask=batch.action_mask.float(),
            )

        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss_output.total).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_output.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip,
            )
            self.optimizer.step()

        # Update scheduler
        self.scheduler.step()

        # Get LBB constant
        lbb_constant = None
        if output.lbb_constant is not None:
            lbb_constant = output.lbb_constant.mean().item()

        # Convert grad_norm to float
        grad_norm_float = float(grad_norm) if hasattr(grad_norm, "item") else grad_norm

        return loss_output, lbb_constant, grad_norm_float

    def train(
        self,
        n_steps: int | None = None,
        log_interval: int = 100,
        checkpoint_interval: int | None = None,
        eval_interval: int | None = None,
    ) -> None:
        """Run training loop.

        Args:
            n_steps: Number of training steps (None for config default).
            log_interval: Steps between logging.
            checkpoint_interval: Steps between checkpoints.
            eval_interval: Steps between evaluation.

        """
        n_steps = n_steps or self.training_config.total_steps
        checkpoint_interval = (
            checkpoint_interval or self.training_config.checkpoint_interval
        )
        eval_interval = eval_interval or getattr(
            self.training_config, "eval_interval", None
        )

        # Minimum buffer size before training
        min_buffer_size = min(
            self.training_config.batch_size * 10,
            self.training_config.replay_buffer_size // 10,
        )

        logger.info(
            "training_started",
            n_steps=n_steps,
            batch_size=self.training_config.batch_size,
            min_buffer_size=min_buffer_size,
        )

        # Initial buffer fill
        self._fill_buffer(min_buffer_size)

        self.model.train()
        start_step = self.global_step

        for step in range(start_step, start_step + n_steps):
            step_start = time.time()

            # Periodically add more games
            if step > 0 and step % (checkpoint_interval // 2) == 0:
                self.model.eval()
                new_experiences = self.self_play_worker.generate_experiences(
                    self.training_config.n_self_play_games // 2
                )
                self.buffer.add_batch(new_experiences)
                self.total_games_generated += self.training_config.n_self_play_games // 2
                self.model.train()

            # Sample batch and train
            batch = self._sample_batch()
            loss_output, lbb_constant, grad_norm = self._training_step(batch)

            step_time = (time.time() - step_start) * 1000

            # Record metrics
            metrics = TrainingMetrics(
                step=step,
                total_loss=loss_output.total.item(),
                policy_loss=loss_output.policy.item(),
                value_loss=loss_output.value.item(),
                lbb_loss=loss_output.lbb.item(),
                lbb_constant=lbb_constant or 0.0,
                learning_rate=self.scheduler.get_last_lr()[0],
                gradient_norm=grad_norm,
                buffer_size=len(self.buffer),
                games_generated=self.total_games_generated,
                step_time_ms=step_time,
            )
            self._metrics_history.append(metrics)

            # W&B logging (every step by default, configurable via wandb.log_interval)
            if self.wandb_logger is not None:
                self.wandb_logger.log_training_step(metrics)

            # Console logging
            if step % log_interval == 0:
                logger.info(
                    "training_step",
                    step=step,
                    loss=f"{loss_output.total.item():.4f}",
                    policy_loss=f"{loss_output.policy.item():.4f}",
                    value_loss=f"{loss_output.value.item():.4f}",
                    lbb_loss=f"{loss_output.lbb.item():.4f}",
                    lr=f"{metrics.learning_rate:.2e}",
                    grad_norm=f"{grad_norm:.4f}",
                    buffer_size=len(self.buffer),
                    step_time_ms=f"{step_time:.1f}",
                )

            # Periodic evaluation
            if eval_interval and step > 0 and step % eval_interval == 0:
                self._run_evaluation(step)

            # Checkpointing
            if step > 0 and step % checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint(metrics=metrics.to_dict())

                # Log checkpoint as W&B artifact
                if self.wandb_logger is not None:
                    self.wandb_logger.log_model_artifact(
                        checkpoint_path=checkpoint_path,
                        name=f"checkpoint-{step}",
                        metadata=metrics.to_dict(),
                        aliases=["latest"],
                    )

            self.global_step = step + 1

        logger.info(
            "training_completed",
            total_steps=n_steps,
            final_loss=self._metrics_history[-1].total_loss if self._metrics_history else 0,
        )

        # Final checkpoint
        final_checkpoint_path = self.save_checkpoint(
            metrics=self._metrics_history[-1].to_dict() if self._metrics_history else {}
        )

        # Log final summary to W&B
        if self.wandb_logger is not None and self._metrics_history:
            final_metrics = self._metrics_history[-1]
            self.wandb_logger.log_summary({
                "final/total_loss": final_metrics.total_loss,
                "final/policy_loss": final_metrics.policy_loss,
                "final/value_loss": final_metrics.value_loss,
                "final/lbb_loss": final_metrics.lbb_loss,
                "final/total_steps": n_steps,
                "final/total_games": self.total_games_generated,
            })

            # Log final checkpoint as best model artifact
            self.wandb_logger.log_model_artifact(
                checkpoint_path=final_checkpoint_path,
                name="model-final",
                metadata=final_metrics.to_dict(),
                aliases=["final"],
            )

    def _run_evaluation(self, step: int) -> None:
        """Run evaluation and log results.

        Args:
            step: Current training step.

        """
        logger.info("evaluation_starting", step=step)
        self.model.eval()

        # Evaluate against random player
        n_games = getattr(self.training_config, "eval_games", 20)

        # Evaluate on each board size
        for board_size in getattr(self.config, "board_sizes", [9]):
            result = self.evaluator.evaluate_vs_random(
                n_games=n_games,
                board_size=board_size,
            )

            # Log to W&B
            if self.wandb_logger is not None:
                self.wandb_logger.log_evaluation(
                    result=result,
                    prefix=f"eval/{board_size}x{board_size}",
                    step=step,
                )

        # Measure policy agreement
        policy_agreement = self.evaluator.measure_policy_agreement(
            n_positions=100,
            board_size=9,
        )

        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(
                {"eval/policy_agreement": policy_agreement},
                step=step,
            )

        self.model.train()
        logger.info("evaluation_completed", step=step)

    def save_checkpoint(
        self,
        path: Path | str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Save training checkpoint.

        Args:
            path: Optional specific path.
            metrics: Current metrics.

        Returns:
            Path to saved checkpoint.

        """
        return self.checkpoint_manager.save(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            metrics=metrics or {},
        )

    def load_checkpoint(
        self,
        path: Path | str | None = None,
        load_best: bool = False,
    ) -> int:
        """Load training checkpoint.

        Args:
            path: Specific checkpoint path.
            load_best: Whether to load best checkpoint.

        Returns:
            Training step from checkpoint.

        """
        step = self.checkpoint_manager.restore(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=path,
            load_best=load_best,
        )
        self.global_step = step
        return step

    def get_metrics_history(self) -> list[dict[str, Any]]:
        """Get training metrics history.

        Returns:
            List of metrics dictionaries.

        """
        return [m.to_dict() for m in self._metrics_history]

    def get_current_lr(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate.

        """
        return self.scheduler.get_last_lr()[0]


def create_trainer(
    model: AlphaGalerkinModel,
    config: AlphaGalerkinConfig,
    checkpoint_dir: Path | str | None = None,
    resume_from: Path | str | None = None,
    device: str = "auto",
    wandb_logger: WandbLogger | None = None,
) -> Trainer:
    """Create and optionally resume a trainer instance.

    Args:
        model: Model to train.
        config: Training configuration.
        checkpoint_dir: Checkpoint directory.
        resume_from: Path to checkpoint to resume from.
        device: Training device.
        wandb_logger: Optional W&B logger for experiment tracking.

    Returns:
        Configured trainer.

    """
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        wandb_logger=wandb_logger,
    )

    if resume_from is not None:
        trainer.load_checkpoint(path=resume_from)
        logger.info("training_resumed", from_step=trainer.global_step)

        # Update W&B step offset for resumed training
        if wandb_logger is not None:
            wandb_logger.set_step_offset(trainer.global_step)

    return trainer
