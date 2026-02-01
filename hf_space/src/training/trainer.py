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
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler

from src.data.collate import TrainingBatch, VariableSizeCollator
from src.training.checkpoint import CheckpointManager
from src.training.curriculum import BoardSizeCurriculum
from src.training.distributed_context import DistributedContext
from src.training.eval_utils import EloTracker
from src.training.evaluation import Evaluator
from src.training.loss import AlphaGalerkinLoss, LossOutput
from src.training.loss_balancing import (
    BalancingStrategy,
    LossBalancer,
    LossBalancingConfig,
    create_loss_balancer,
)
from src.training.replay_buffer import create_replay_buffer
from src.training.self_play import SelfPlayWorker
from src.training.stability import (
    EarlyStopping,
    EarlyStoppingConfig,
    PlateauConfig,
    PlateauDetector,
    TrainingStabilityMonitor,
)
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
    # Loss balancing weights
    policy_weight: float = 1.0
    value_weight: float = 1.0
    lbb_weight: float = 1.0

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
            "policy_weight": self.policy_weight,
            "value_weight": self.value_weight,
            "lbb_weight": self.lbb_weight,
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
        distributed_context: DistributedContext | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: AlphaGalerkin model to train.
            config: Complete configuration.
            device: Training device ("auto" for automatic selection).
            checkpoint_dir: Directory for checkpoints.
            wandb_logger: Optional W&B logger for experiment tracking.
            distributed_context: Optional distributed context (auto-detected if None).

        """
        self.config = config
        self.training_config = config.training
        self.mcts_config = config.mcts

        # Setup distributed context (auto-detect from environment if not provided)
        distributed_config = getattr(config, "distributed", None)
        if distributed_context is not None:
            self.dist_ctx = distributed_context
        else:
            self.dist_ctx = DistributedContext.from_environment(distributed_config)

        # Initialize process group if distributed
        if self.dist_ctx.is_distributed and distributed_config is not None:
            self.dist_ctx.initialize_process_group(
                backend=distributed_config.backend,
                timeout_seconds=distributed_config.timeout_seconds,
            )

        # Device selection (use distributed context device if available)
        if device == "auto":
            self.device = self.dist_ctx.device
        else:
            self.device = torch.device(device)

        # W&B logging only on rank 0
        if wandb_logger is not None and not self.dist_ctx.is_main_process:
            self.wandb_logger = None  # Disable W&B on non-main ranks
            logger.info("wandb_disabled_non_main_rank", rank=self.dist_ctx.rank)
        else:
            self.wandb_logger = wandb_logger

        logger.info(
            "trainer_initialized",
            device=str(self.device),
            rank=self.dist_ctx.rank,
            world_size=self.dist_ctx.world_size,
            is_distributed=self.dist_ctx.is_distributed,
        )

        # Model setup - keep reference to raw model before DDP wrapping
        self._raw_model = model.to(self.device)

        # Get DDP options from config
        find_unused = False
        broadcast_bufs = True
        if distributed_config is not None:
            find_unused = getattr(distributed_config, "find_unused_parameters", False)
            broadcast_bufs = getattr(distributed_config, "broadcast_buffers", True)

        self.model = self.dist_ctx.wrap_model(
            self._raw_model,
            find_unused_parameters=find_unused,
            broadcast_buffers=broadcast_bufs,
        )

        # Loss function
        self.loss_fn = AlphaGalerkinLoss(
            policy_weight=self.training_config.policy_loss_weight,
            value_weight=self.training_config.value_loss_weight,
        )

        # Loss balancer for adaptive weighting
        self.loss_balancer = self._create_loss_balancer()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = self.training_config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Replay buffer (prioritized or uniform based on config)
        self.use_prioritized_replay = getattr(
            self.training_config, "use_prioritized_replay", False
        )
        self.buffer = create_replay_buffer(
            capacity=self.training_config.replay_buffer_size,
            prioritized=self.use_prioritized_replay,
            alpha=getattr(self.training_config, "per_alpha", 0.6),
            beta=getattr(self.training_config, "per_beta", 0.4),
        )

        # Board size curriculum (optional)
        self.curriculum = self._create_curriculum() if getattr(
            self.training_config, "curriculum_enabled", False
        ) else None

        # Self-play worker (use raw model, not DDP-wrapped)
        self.self_play_worker = SelfPlayWorker(
            model=self._raw_model,
            mcts_config=self.mcts_config,
            device=self.device,
            board_sizes=config.board_sizes,
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

        # Evaluator for periodic evaluation (use raw model, not DDP-wrapped)
        self.evaluator = Evaluator(
            model=self._raw_model,
            mcts_config=self.mcts_config,
            device=self.device,
            board_sizes=config.board_sizes,
        )

        # Elo tracker for checkpoint evaluation (optional)
        self.elo_tracker: EloTracker | None = None
        if getattr(self.training_config, "eval_vs_checkpoints", False):
            k_factor = getattr(self.training_config, "elo_k_factor", 32.0)
            self.elo_tracker = EloTracker(k_factor=k_factor)
            logger.info("elo_tracker_enabled", k_factor=k_factor)

        # Training stability monitor (optional)
        self.stability_monitor = self._create_stability_monitor()

        # Track warmup completion for plateau gating
        # Plateau detection should not trigger during warmup (unstable losses)
        self._warmup_completed = self.training_config.warmup_steps == 0
        self._warmup_steps = self.training_config.warmup_steps

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

    def _create_loss_balancer(self) -> LossBalancer:
        """Create loss balancer from config.

        Returns:
            Configured loss balancer.

        """
        # Map strategy name to enum
        strategy_map = {
            "static": BalancingStrategy.STATIC,
            "relobralo": BalancingStrategy.RELOBRALO,
            "gradnorm": BalancingStrategy.GRADNORM,
            "uncertainty": BalancingStrategy.UNCERTAINTY,
            "softadapt": BalancingStrategy.SOFTADAPT,
        }

        strategy = strategy_map.get(
            self.training_config.loss_balancing_strategy,
            BalancingStrategy.RELOBRALO,
        )

        config = LossBalancingConfig(
            name="training_loss_balancer",
            strategy=strategy,
            beta=self.training_config.loss_balancing_beta,
            tau=self.training_config.loss_balancing_tau,
            warmup_steps=self.training_config.loss_balancing_warmup,
        )

        logger.info(
            "loss_balancer_created",
            strategy=strategy.value,
            beta=config.beta,
            tau=config.tau,
            warmup_steps=config.warmup_steps,
        )

        return create_loss_balancer(
            config=config,
            loss_names=["policy", "value", "lbb"],
            model=self.model,
        )

    def _create_curriculum(self) -> BoardSizeCurriculum | None:
        """Create board size curriculum from config.

        Returns:
            Configured curriculum or None if not enabled.

        """
        # Default curriculum schedule
        default_schedule = {
            0: [9],
            10000: [9, 13],
            50000: [9, 13, 19],
        }

        # Get schedule from config if available
        schedule = getattr(self.training_config, "curriculum_schedule", None)
        if schedule is None:
            schedule = default_schedule

        curriculum = BoardSizeCurriculum.from_config(schedule)

        logger.info(
            "curriculum_created",
            schedule=curriculum.get_schedule_info(),
        )

        return curriculum

    def _create_stability_monitor(self) -> TrainingStabilityMonitor | None:
        """Create training stability monitor from config.

        Returns:
            Configured stability monitor or None if no features enabled.

        """
        early_stopping: EarlyStopping | None = None
        plateau_detector: PlateauDetector | None = None

        # Early stopping
        if getattr(self.training_config, "early_stopping_enabled", False):
            patience = getattr(self.training_config, "early_stopping_patience", 10)
            min_delta = getattr(self.training_config, "early_stopping_min_delta", 0.01)
            es_config = EarlyStoppingConfig(
                patience=patience,
                min_delta=min_delta,
                metric="eval/win_rate",
                mode="max",  # Higher win rate is better
            )
            early_stopping = EarlyStopping(es_config)
            logger.info(
                "early_stopping_enabled",
                patience=patience,
                min_delta=min_delta,
            )

        # Plateau detection (LR reduction)
        if getattr(self.training_config, "plateau_detection_enabled", False):
            patience = getattr(self.training_config, "plateau_patience", 5)
            factor = getattr(self.training_config, "plateau_factor", 0.5)
            min_lr = getattr(self.training_config, "plateau_min_lr", 1e-6)
            pd_config = PlateauConfig(
                patience=patience,
                factor=factor,
                min_lr=min_lr,
                metric="train/loss/total",
                mode="min",  # Lower loss is better
            )
            plateau_detector = PlateauDetector(pd_config, self.optimizer)
            logger.info(
                "plateau_detection_enabled",
                patience=patience,
                factor=factor,
                min_lr=min_lr,
            )

        # Return monitor if any feature is enabled
        if early_stopping is not None or plateau_detector is not None:
            return TrainingStabilityMonitor(
                early_stopping=early_stopping,
                plateau_detector=plateau_detector,
            )

        return None

    def _fill_buffer(self, min_size: int) -> None:
        """Fill replay buffer to minimum size.

        Args:
            min_size: Minimum number of experiences needed.

        """
        fill_start = time.time()
        initial_size = len(self.buffer)

        while len(self.buffer) < min_size:
            n_games = self.training_config.n_self_play_games
            logger.info(
                "generating_self_play_games",
                n_games=n_games,
                buffer_size=len(self.buffer),
                target_size=min_size,
            )

            # Generate games (use curriculum board size if enabled)
            self.model.eval()
            board_size = None
            if self.curriculum is not None:
                board_size = self.curriculum.sample_board_size(self.global_step)
            experiences = self.self_play_worker.generate_experiences(
                n_games, board_size=board_size
            )
            self.model.train()

            # Add to buffer
            self.buffer.add_batch(experiences)
            self.total_games_generated += n_games

            # Log self-play progress to W&B
            if self.wandb_logger is not None:
                stats = self.self_play_worker.get_stats()
                self.wandb_logger.log_metrics(
                    {
                        "self_play/games_completed": stats["games_played"],
                        "self_play/avg_game_length": stats["avg_game_length"],
                        "self_play/buffer_size": len(self.buffer),
                        "self_play/black_wins": stats["outcomes"]["black"],
                        "self_play/white_wins": stats["outcomes"]["white"],
                        "self_play/draws": stats["outcomes"]["draw"],
                    },
                    step=self.global_step,
                )

        # Log buffer fill statistics
        fill_time = time.time() - fill_start
        experiences_added = len(self.buffer) - initial_size
        fill_rate = experiences_added / max(fill_time, 0.001)
        logger.info(
            "buffer_filled",
            initial_size=initial_size,
            final_size=len(self.buffer),
            target_size=min_size,
            experiences_added=experiences_added,
            fill_time_seconds=round(fill_time, 2),
            fill_rate_per_second=round(fill_rate, 1),
        )

        # Log buffer fill summary to W&B
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(
                {
                    "self_play/fill_time_seconds": round(fill_time, 2),
                    "self_play/experiences_added": experiences_added,
                    "self_play/fill_rate_per_second": round(fill_rate, 1),
                    "self_play/total_games_generated": self.total_games_generated,
                },
                step=self.global_step,
            )

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
    ) -> tuple[LossOutput, float | None, float, dict[str, float]]:
        """Execute single training step with adaptive loss balancing.

        Args:
            batch: Training batch.

        Returns:
            Tuple of (loss output, LBB constant, gradient norm, loss weights).

        """
        self.optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast(device_type=self.device.type):
                output = self.model(batch.board_states, return_lbb=True)
                # Compute individual losses
                policy_loss = self.loss_fn.compute_policy_loss(
                    policy_logits=output.policy_logits,
                    target_policy=batch.target_policies,
                    mask=batch.action_mask.float(),
                )
                value_loss = self.loss_fn.compute_value_loss(
                    value=output.value,
                    target_value=batch.target_values,
                )
                lbb_loss = self.loss_fn.compute_lbb_loss(
                    lbb_constant=output.lbb_constant,
                )
        else:
            output = self.model(batch.board_states, return_lbb=True)
            # Compute individual losses
            policy_loss = self.loss_fn.compute_policy_loss(
                policy_logits=output.policy_logits,
                target_policy=batch.target_policies,
                mask=batch.action_mask.float(),
            )
            value_loss = self.loss_fn.compute_value_loss(
                value=output.value,
                target_value=batch.target_values,
            )
            lbb_loss = self.loss_fn.compute_lbb_loss(
                lbb_constant=output.lbb_constant,
            )

        # Apply adaptive loss balancing
        losses = {
            "policy": policy_loss,
            "value": value_loss,
            "lbb": lbb_loss,
        }
        loss_terms = self.loss_balancer.compute_weighted_loss(losses)
        total_loss = loss_terms.weighted_sum
        weights = loss_terms.weights

        # Create LossOutput for compatibility
        loss_output = LossOutput(
            total=total_loss,
            policy=policy_loss,
            value=value_loss,
            lbb=lbb_loss,
        )

        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
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
        grad_norm_float = grad_norm.item()

        # Log if gradient norm is near clipping threshold (debugging aid)
        clip_threshold = self.training_config.gradient_clip
        if grad_norm_float > clip_threshold * 0.9:
            logger.debug(
                "gradient_near_clip_threshold",
                grad_norm=f"{grad_norm_float:.4f}",
                clip_threshold=clip_threshold,
                ratio=f"{grad_norm_float / clip_threshold:.2f}",
            )

        return loss_output, lbb_constant, grad_norm_float, weights

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

            # Periodically add more games (at half the checkpoint interval)
            self_play_interval = max(checkpoint_interval // 2, 1)
            if step > 0 and step % self_play_interval == 0:
                self.model.eval()
                # Use curriculum board size if enabled
                board_size = None
                if self.curriculum is not None:
                    board_size = self.curriculum.sample_board_size(step)
                new_experiences = self.self_play_worker.generate_experiences(
                    self.training_config.n_self_play_games // 2,
                    board_size=board_size,
                )
                self.buffer.add_batch(new_experiences)
                self.total_games_generated += self.training_config.n_self_play_games // 2
                self.model.train()

            # Sample batch and train
            batch = self._sample_batch()
            loss_output, lbb_constant, grad_norm, loss_weights = self._training_step(batch)

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
                policy_weight=loss_weights.get("policy", 1.0),
                value_weight=loss_weights.get("value", 1.0),
                lbb_weight=loss_weights.get("lbb", 1.0),
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
                avg_win_rate = self._run_evaluation(step)

                # Check early stopping
                if self.stability_monitor is not None:
                    if self.stability_monitor.check_early_stopping(avg_win_rate):
                        logger.info(
                            "early_stopping_triggered",
                            step=step,
                            win_rate=avg_win_rate,
                        )
                        break

            # Check if warmup just completed (log transition)
            if not self._warmup_completed and step >= self._warmup_steps:
                self._warmup_completed = True
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "warmup_completed",
                    step=step,
                    warmup_steps=self._warmup_steps,
                    current_lr=f"{current_lr:.2e}",
                )

            # Check plateau detection (LR reduction) - ONLY after warmup completes
            # During warmup, losses are naturally unstable and shouldn't trigger LR reduction
            if self.stability_monitor is not None and self._warmup_completed:
                lr_reduced = self.stability_monitor.check_plateau(loss_output.total.item())
                if lr_reduced:
                    new_lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        "plateau_lr_reduced",
                        step=step,
                        new_lr=f"{new_lr:.2e}",
                        loss=f"{loss_output.total.item():.4f}",
                    )

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

    def _run_evaluation(self, step: int) -> float:
        """Run evaluation and log results.

        Args:
            step: Current training step.

        Returns:
            Average win rate across board sizes (for early stopping).

        """
        logger.info("evaluation_starting", step=step)
        self.model.eval()

        n_games = getattr(self.training_config, "eval_games", 20)
        use_multi_res = getattr(self.training_config, "multi_resolution_eval", True)

        win_rates: list[float] = []

        if use_multi_res and hasattr(self.evaluator, "evaluate_multi_resolution"):
            # Use multi-resolution evaluation
            results = self.evaluator.evaluate_multi_resolution(
                n_games_per_size=n_games
            )
            for board_size, result in results.items():
                win_rates.append(result.win_rate)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_evaluation(
                        result=result,
                        prefix=f"eval/{board_size}x{board_size}",
                        step=step,
                    )
        else:
            # Evaluate on each board size individually
            for board_size in self.config.board_sizes:
                result = self.evaluator.evaluate_vs_random(
                    n_games=n_games,
                    board_size=board_size,
                )
                win_rates.append(result.win_rate)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_evaluation(
                        result=result,
                        prefix=f"eval/{board_size}x{board_size}",
                        step=step,
                    )

        # Checkpoint tournament evaluation (Elo tracking)
        if self.elo_tracker is not None:
            self._run_checkpoint_tournament(step, n_games)

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

        # Return average win rate for early stopping
        return sum(win_rates) / len(win_rates) if win_rates else 0.0

    def _run_checkpoint_tournament(self, step: int, n_games: int) -> None:
        """Run tournament against previous checkpoints for Elo tracking.

        Args:
            step: Current training step.
            n_games: Number of games per opponent.

        """
        if self.elo_tracker is None:
            return

        # Get list of available checkpoints
        checkpoint_paths = self.checkpoint_manager.get_all_checkpoints()
        n_opponents = min(
            len(checkpoint_paths),
            getattr(self.training_config, "n_tournament_opponents", 5),
        )

        if n_opponents == 0:
            return

        logger.info(
            "checkpoint_tournament_starting",
            step=step,
            n_opponents=n_opponents,
        )

        # Select recent checkpoints as opponents
        opponent_paths = checkpoint_paths[-n_opponents:]

        for opponent_path in opponent_paths:
            try:
                result = self.evaluator.evaluate_vs_checkpoint(
                    checkpoint_path=opponent_path,
                    n_games=n_games,
                )

                # Extract opponent step from checkpoint filename
                opponent_step = self._extract_step_from_checkpoint(opponent_path)

                # Determine score: 1.0=win, 0.5=draw, 0.0=loss
                if result.win_rate > 0.55:
                    score = 1.0
                elif result.win_rate < 0.45:
                    score = 0.0
                else:
                    score = 0.5

                # Update Elo ratings
                self.elo_tracker.update_ratings(step, opponent_step, score)

                # Log to W&B
                if self.wandb_logger is not None:
                    current_rating = self.elo_tracker.get_rating(step)
                    self.wandb_logger.log_metrics(
                        {
                            f"elo/vs_step_{opponent_step}": result.win_rate,
                            "elo/current_rating": current_rating,
                        },
                        step=step,
                    )

                logger.debug(
                    "checkpoint_match_completed",
                    opponent_step=opponent_step,
                    win_rate=result.win_rate,
                    score=score,
                )

            except Exception as e:
                logger.warning(
                    "checkpoint_match_failed",
                    opponent_path=str(opponent_path),
                    error=str(e),
                )

    def _extract_step_from_checkpoint(self, checkpoint_path: Path) -> int:
        """Extract training step from checkpoint filename.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Training step number.

        """
        # Filename format: checkpoint_00010000.pt
        import re

        match = re.search(r"checkpoint_(\d+)", checkpoint_path.stem)
        if match:
            return int(match.group(1))
        return 0

    def save_checkpoint(
        self,
        path: Path | str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save training checkpoint.

        Only rank 0 saves checkpoints in distributed mode.

        Args:
            path: Optional specific path.
            metrics: Current metrics.

        Returns:
            Path to saved checkpoint, or None on non-main ranks.

        """
        # Only save on main process in distributed mode
        if not self.dist_ctx.is_main_process:
            self.dist_ctx.barrier()  # Wait for main to save
            return None

        # Use raw model state dict (not DDP wrapper)
        checkpoint_path = self.checkpoint_manager.save(
            step=self.global_step,
            model=self._raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            metrics=metrics or {},
        )

        # Barrier to ensure checkpoint is saved before other ranks proceed
        self.dist_ctx.barrier()

        return checkpoint_path

    def load_checkpoint(
        self,
        path: Path | str | None = None,
        load_best: bool = False,
    ) -> int:
        """Load training checkpoint.

        All ranks load from the same checkpoint, then synchronize to
        ensure consistency.

        Args:
            path: Specific checkpoint path.
            load_best: Whether to load best checkpoint.

        Returns:
            Training step from checkpoint.

        """
        # Load into raw model (not DDP wrapper)
        step = self.checkpoint_manager.restore(
            model=self._raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=path,
            load_best=load_best,
        )
        self.global_step = step

        # Synchronize across ranks
        self.dist_ctx.barrier()

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
    distributed_context: DistributedContext | None = None,
) -> Trainer:
    """Create and optionally resume a trainer instance.

    Args:
        model: Model to train.
        config: Training configuration.
        checkpoint_dir: Checkpoint directory.
        resume_from: Path to checkpoint to resume from.
        device: Training device.
        wandb_logger: Optional W&B logger for experiment tracking.
        distributed_context: Optional distributed context (auto-detected if None).

    Returns:
        Configured trainer.

    """
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        wandb_logger=wandb_logger,
        distributed_context=distributed_context,
    )

    if resume_from is not None:
        trainer.load_checkpoint(path=resume_from)
        logger.info("training_resumed", from_step=trainer.global_step)

        # Update W&B step offset for resumed training
        if wandb_logger is not None:
            wandb_logger.set_step_offset(trainer.global_step)

    return trainer
