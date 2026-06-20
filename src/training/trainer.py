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
from typing import TYPE_CHECKING, Any, cast

import structlog
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.constants import (
    DEFAULT_CURRICULUM_SCHEDULE,
    DEFAULT_PER_ALPHA,
    DEFAULT_PER_BETA,
    WIN_RATE_ACCEPT_THRESHOLD,
    WIN_RATE_REJECT_THRESHOLD,
)
from src.data.collate import TrainingBatch, VariableSizeCollator
from src.training.base_trainer import BaseTrainer
from src.training.callbacks import (
    Callback,
    build_callbacks_from_specs,
)
from src.training.checkpoint import CheckpointManager
from src.training.curriculum import BoardSizeCurriculum
from src.training.distributed_context import DistributedContext
from src.training.eval_utils import EloTracker
from src.training.evaluation import Evaluator
from src.training.langfuse_tracker import LangfuseTracker
from src.training.loss_balancing import (
    BalancingStrategy,
    LossBalancer,
    LossBalancingConfig,
    create_loss_balancer,
)
from src.training.losses import AlphaGalerkinLoss, LossOutput
from src.training.physics_loss import (
    PhysicsInformedLoss,
    PhysicsLossConfig,
    PhysicsLossOutput,
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

if TYPE_CHECKING:
    from config.schemas import AlphaGalerkinConfig
    from src.games.interface import GameInterface
    from src.modeling.model import AlphaGalerkinModel
    from src.training.losses.physics import CombinedAlphaGalerkinPhysicsLoss

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
    # Physics-informed loss metrics (optional)
    physics_loss: float = 0.0
    physics_residual_loss: float = 0.0
    physics_boundary_loss: float = 0.0
    physics_weight: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        result = {
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
        # Only include physics metrics if physics training is enabled
        if self.physics_weight > 0:
            result["physics_loss"] = self.physics_loss
            result["physics_residual_loss"] = self.physics_residual_loss
            result["physics_boundary_loss"] = self.physics_boundary_loss
            result["physics_weight"] = self.physics_weight
        return result


class Trainer(BaseTrainer):
    """Main trainer for AlphaGalerkin.

    Coordinates self-play, training, and checkpoint management.
    Supports mixed precision training and gradient accumulation.

    Inherits shared AMP, gradient-clipping, LR-scheduling, and
    checkpoint helpers from :class:`BaseTrainer`.  The ``__init__``
    does **not** call ``super().__init__()`` because the AlphaGalerkin
    trainer has a substantially different setup flow; instead it
    sets the attributes that ``BaseTrainer`` helpers rely on directly.
    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        config: AlphaGalerkinConfig,
        device: torch.device | str = "auto",
        checkpoint_dir: Path | str | None = None,
        tracker: LangfuseTracker | None = None,
        distributed_context: DistributedContext | None = None,
        game: GameInterface | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: AlphaGalerkin model to train.
            config: Complete configuration.
            device: Training device ("auto" for automatic selection).
            checkpoint_dir: Directory for checkpoints.
            tracker: Optional Langfuse experiment tracker.
            distributed_context: Optional distributed context (auto-detected if None).
            game: Optional GameInterface for non-Go games (e.g. chess).
                  When provided, self-play uses this game instead of SimpleGoGame.
            callbacks: Optional list of :class:`Callback` instances dispatched
                at lifecycle events. Programmatic alternative to
                ``training_config.callbacks`` (specs). When both are provided,
                programmatic callbacks are appended after spec-resolved ones.

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

        # Experiment tracking only on rank 0
        if tracker is not None and not self.dist_ctx.is_main_process:
            self.tracker = None  # Disable tracking on non-main ranks
            logger.info("tracking_disabled_non_main_rank", rank=self.dist_ctx.rank)
        else:
            self.tracker = tracker

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

        # Combined physics loss (wraps policy/value/LBB + physics into one module)
        # Stored separately from loss_fn to preserve _training_step compatibility.
        self.combined_physics_loss_fn: CombinedAlphaGalerkinPhysicsLoss | None = None
        _physics_loss_type = self.training_config.physics_loss_type
        if _physics_loss_type != "none":
            from src.training.losses.physics import CombinedAlphaGalerkinPhysicsLoss

            _physics_weight = self.training_config.physics_weight
            self.combined_physics_loss_fn = CombinedAlphaGalerkinPhysicsLoss(
                physics_weight=_physics_weight,
            )
            logger.info(
                "combined_physics_loss_enabled",
                physics_loss_type=_physics_loss_type,
                physics_weight=_physics_weight,
            )

        # Physics-informed loss (optional)
        self.physics_loss_fn: PhysicsInformedLoss | None = None
        self.use_physics_loss = self.training_config.physics_informed
        self.physics_loss_weight = self.training_config.physics_loss_weight
        if self.use_physics_loss:
            self.physics_loss_fn = self._create_physics_loss()
            logger.info(
                "physics_loss_enabled",
                weight=self.physics_loss_weight,
                n_collocation=self.training_config.physics_n_collocation_points,
                n_boundary=self.training_config.physics_n_boundary_points,
            )

        # Loss balancer for adaptive weighting
        self.loss_balancer = self._create_loss_balancer()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision (uses BaseTrainer helper)
        self.use_amp, self.scaler, self._amp_dtype = self._setup_amp(
            use_amp=self.training_config.use_amp,
            device=self.device,
        )

        # Replay buffer (prioritized or uniform based on config)
        self.use_prioritized_replay = self.training_config.use_prioritized_replay
        self.buffer = create_replay_buffer(
            capacity=self.training_config.replay_buffer_size,
            prioritized=self.use_prioritized_replay,
            alpha=getattr(self.training_config, "per_alpha", DEFAULT_PER_ALPHA),
            beta=getattr(self.training_config, "per_beta", DEFAULT_PER_BETA),
        )

        # Board size curriculum (optional)
        self.curriculum = (
            self._create_curriculum() if self.training_config.curriculum_enabled else None
        )

        # Self-play worker (use raw model, not DDP-wrapped)
        self.self_play_worker = SelfPlayWorker(
            model=self._raw_model,
            mcts_config=self.mcts_config,
            device=self.device,
            board_sizes=config.board_sizes,
            game=game,
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
        if self.training_config.eval_vs_checkpoints:
            self.elo_tracker = EloTracker(k_factor=self.training_config.elo_k_factor)
            logger.info("elo_tracker_enabled", k_factor=self.training_config.elo_k_factor)

        # Training stability monitor (optional)
        self.stability_monitor = self._create_stability_monitor()

        # Track warmup completion for plateau gating
        # Plateau detection should not trigger during warmup (unstable losses)
        self._warmup_completed = self.training_config.warmup_steps == 0
        self._warmup_steps = self.training_config.warmup_steps

        # Watch model with the tracker if enabled (no-op for Langfuse)
        if self.tracker is not None:
            self.tracker.watch_model(self.model)

        # Lifecycle callbacks: resolve specs from config, then append any
        # explicit callbacks the caller passed.  This keeps user-facing
        # YAML config in charge while letting programmatic harnesses
        # (tests, demos) inject extra callbacks at construction time.
        spec_callbacks: list[Callback] = []
        config_specs = getattr(self.training_config, "callbacks", None) or []
        if config_specs:
            spec_callbacks = build_callbacks_from_specs(list(config_specs))
        self.callbacks: list[Callback] = spec_callbacks + list(callbacks or [])
        # Bound logger required by BaseTrainer dispatch helpers.
        self._log = logger.bind(
            trainer=type(self).__name__,
            device=str(self.device),
        )
        if self.callbacks:
            logger.info(
                "trainer_callbacks_registered",
                count=len(self.callbacks),
                callbacks=[type(cb).__name__ for cb in self.callbacks],
            )

    def _create_optimizer(self) -> Optimizer:  # type: ignore[override]
        """Create optimizer from config.

        Delegates to :meth:`BaseTrainer._create_optimizer` static helper.
        """
        return BaseTrainer._create_optimizer(
            self.model,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

    def _create_scheduler(self) -> LRScheduler:  # type: ignore[override]
        """Create learning rate scheduler from config.

        Delegates to :meth:`BaseTrainer._create_scheduler` static helper.
        The ``"constant"`` type maps to ``"none"`` in the base helper
        for backwards compatibility.
        """
        scheduler_type: str = self.training_config.lr_scheduler
        # Map legacy "constant" to "none" (BaseTrainer recognises both)
        if scheduler_type == "constant":
            scheduler_type = "none"

        return BaseTrainer._create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=self.training_config.warmup_steps,
            total_steps=self.training_config.total_steps,
            min_lr_ratio=0.1,
            warmup_start_factor=0.1,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations (required by BaseTrainer ABC)
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Not used directly -- Trainer uses _training_step instead."""
        raise NotImplementedError(
            "Trainer uses _training_step(); use that method or the train() loop."
        )

    def generate_data(self) -> Any:
        """Not used directly -- Trainer uses _sample_batch instead."""
        raise NotImplementedError(
            "Trainer uses _sample_batch(); use that method or the train() loop."
        )

    def evaluate(self) -> dict[str, float]:
        """Not used directly -- Trainer uses _run_evaluation instead."""
        raise NotImplementedError(
            "Trainer uses _run_evaluation(); use that method or the train() loop."
        )

    # ------------------------------------------------------------------
    # Physics loss, loss balancer, curriculum, stability
    # ------------------------------------------------------------------

    def _create_physics_loss(self) -> PhysicsInformedLoss | None:
        """Create physics-informed loss from config.

        Returns:
            Configured physics loss or None if PDE operator unavailable.

        """
        # Try to import and create a default PDE operator
        try:
            from src.pde.config import PDEConfig, PDEType
            from src.pde.operators import PoissonOperator

            # Create Poisson operator as default (can be overridden via config)
            pde_config = PDEConfig(
                name="training_pde",
                pde_type=PDEType.POISSON,
            )
            pde_operator = PoissonOperator(pde_config)

            # Create physics loss config from training config
            physics_config = PhysicsLossConfig(
                name="training_physics_loss",
                residual_weight=self.training_config.physics_residual_weight,
                boundary_weight=self.training_config.physics_boundary_weight,
                initial_weight=self.training_config.physics_initial_weight,
                conservation_weight=self.training_config.physics_conservation_weight,
                n_collocation_points=self.training_config.physics_n_collocation_points,
                n_boundary_points=self.training_config.physics_n_boundary_points,
                use_adaptive_weights=self.training_config.physics_use_adaptive_weights,
            )

            return PhysicsInformedLoss(pde_operator, physics_config)

        except ImportError as e:
            logger.warning(
                "physics_loss_unavailable",
                reason="PDE module not available",
                error=str(e),
            )
            return None
        except Exception as e:
            logger.warning(
                "physics_loss_creation_failed",
                error=str(e),
            )
            return None

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

        # Include physics loss in balancing if enabled
        loss_names = ["policy", "value", "lbb"]
        if self.use_physics_loss and self.physics_loss_fn is not None:
            loss_names.append("physics")

        logger.info(
            "loss_balancer_created",
            strategy=strategy.value,
            beta=config.beta,
            tau=config.tau,
            warmup_steps=config.warmup_steps,
            loss_names=loss_names,
        )

        return create_loss_balancer(
            config=config,
            loss_names=loss_names,
            model=self.model,
        )

    def _create_curriculum(self) -> BoardSizeCurriculum | None:
        """Create board size curriculum from config.

        Returns:
            Configured curriculum or None if not enabled.

        """
        # Get schedule from config if available, else use constant default
        schedule = getattr(self.training_config, "curriculum_schedule", None)
        if schedule is None:
            # Fallback to BoardSizeCurriculum's built-in default
            schedule = dict(DEFAULT_CURRICULUM_SCHEDULE)

        curriculum = BoardSizeCurriculum.from_config(cast(dict[str, Any], schedule))

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
        if self.training_config.early_stopping_enabled:
            es_config = EarlyStoppingConfig(
                patience=self.training_config.early_stopping_patience,
                min_delta=self.training_config.early_stopping_min_delta,
                metric="eval/win_rate",
                mode="max",  # Higher win rate is better
            )
            early_stopping = EarlyStopping(es_config)
            logger.info(
                "early_stopping_enabled",
                patience=es_config.patience,
                min_delta=es_config.min_delta,
            )

        # Plateau detection (LR reduction)
        if self.training_config.plateau_detection_enabled:
            pd_config = PlateauConfig(
                patience=self.training_config.plateau_patience,
                factor=self.training_config.plateau_factor,
                min_lr=self.training_config.plateau_min_lr,
                metric="train/loss/total",
                mode="min",  # Lower loss is better
            )
            plateau_detector = PlateauDetector(pd_config, self.optimizer)
            logger.info(
                "plateau_detection_enabled",
                patience=pd_config.patience,
                factor=pd_config.factor,
                min_lr=pd_config.min_lr,
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
            experiences = self.self_play_worker.generate_experiences(n_games, board_size=board_size)
            self.model.train()

            # Add to buffer
            self.buffer.add_batch(experiences)
            self.total_games_generated += n_games

            # Log self-play progress to W&B
            if self.tracker is not None:
                stats = self.self_play_worker.get_stats()
                self.tracker.log_metrics(
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
        if self.tracker is not None:
            self.tracker.log_metrics(
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
        sample_result = self.buffer.sample(self.training_config.batch_size)
        # UniformReplayBuffer returns list[Experience]; PrioritizedReplayBuffer returns tuple
        experiences: list[Any] = (
            sample_result[0] if isinstance(sample_result, tuple) else sample_result
        )
        batch = self.collator(experiences)
        return batch.to(self.device)

    def _training_step(
        self, batch: TrainingBatch
    ) -> tuple[LossOutput, float | None, float, dict[str, float], PhysicsLossOutput | None]:
        """Execute single training step with adaptive loss balancing.

        Args:
            batch: Training batch.

        Returns:
            Tuple of (loss output, LBB constant, gradient norm, loss weights, physics output).

        """
        self.optimizer.zero_grad()

        # Forward pass and loss computation (shared between AMP and non-AMP paths)
        def _forward_and_losses() -> tuple[Any, Tensor, Tensor, Tensor]:
            out = self.model(batch.board_states, return_lbb=True)
            p_loss = self.loss_fn.compute_policy_loss(
                policy_logits=out.policy_logits,
                target_policy=batch.target_policies,
                mask=batch.action_mask.float(),
            )
            v_loss = self.loss_fn.compute_value_loss(
                value=out.value,
                target_value=batch.target_values,
            )
            l_loss = self.loss_fn.compute_lbb_loss(
                lbb_constant=out.lbb_constant,
            )
            return out, p_loss, v_loss, l_loss

        if self.use_amp:
            with autocast():
                output, policy_loss, value_loss, lbb_loss = _forward_and_losses()
        else:
            output, policy_loss, value_loss, lbb_loss = _forward_and_losses()

        # Apply adaptive loss balancing
        losses = {
            "policy": policy_loss,
            "value": value_loss,
            "lbb": lbb_loss,
        }

        # Compute physics loss if enabled
        physics_output: PhysicsLossOutput | None = None
        if self.use_physics_loss and self.physics_loss_fn is not None:
            try:
                # Use the raw model for physics loss (not DDP wrapped)
                physics_output = self.physics_loss_fn(self._raw_model)
                if physics_output is not None:
                    losses["physics"] = physics_output.total * self.physics_loss_weight
                else:
                    losses["physics"] = torch.tensor(0.0, device=self.device)
            except Exception as e:
                logger.warning(
                    "physics_loss_computation_failed",
                    error=str(e),
                    step=self.global_step,
                )
                # Fallback to zero physics loss
                losses["physics"] = torch.tensor(0.0, device=self.device)

        loss_terms = self.loss_balancer.compute_weighted_loss(losses)
        total_loss = loss_terms.weighted_sum
        weights = loss_terms.weights

        # Add combined physics loss contribution if enabled
        if self.combined_physics_loss_fn is not None:
            try:
                combined_physics_result = self.combined_physics_loss_fn(
                    policy_logits=output.policy_logits,
                    value=output.value,
                    target_policy=batch.target_policies,
                    target_value=batch.target_values,
                    lbb_constant=output.lbb_constant,
                    action_mask=(
                        batch.action_mask.float() if batch.action_mask is not None else None
                    ),
                    model=self._raw_model,
                )
                physics_total = combined_physics_result.get(
                    "total", torch.tensor(0.0, device=self.device)
                )
                total_loss = total_loss + self.training_config.physics_weight * physics_total
            except Exception as e:
                logger.warning(
                    "combined_physics_loss_computation_failed",
                    error=str(e),
                    step=self.global_step,
                )

        # Create LossOutput for compatibility
        loss_output = LossOutput(
            total=total_loss,
            policy=policy_loss,
            value=value_loss,
            lbb=lbb_loss,
        )

        # Backward pass (uses BaseTrainer helpers for AMP and gradient clipping)
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
            grad_norm = self._clip_gradients(self.model, self.training_config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()  # type: ignore[no-untyped-call]
            grad_norm = self._clip_gradients(self.model, self.training_config.gradient_clip)
            self.optimizer.step()

        # Update scheduler
        self.scheduler.step()

        # Get LBB constant
        lbb_constant = (
            output.lbb_constant.mean().item() if output.lbb_constant is not None else None
        )

        # grad_norm is already a float from _clip_gradients
        grad_norm_float = grad_norm

        # Log if gradient norm is near clipping threshold (debugging aid)
        clip_threshold = self.training_config.gradient_clip
        if grad_norm_float > clip_threshold * 0.9:
            logger.debug(
                "gradient_near_clip_threshold",
                grad_norm=f"{grad_norm_float:.4f}",
                clip_threshold=clip_threshold,
                ratio=f"{grad_norm_float / clip_threshold:.2f}",
            )

        return loss_output, lbb_constant, grad_norm_float, weights, physics_output

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
        checkpoint_interval = checkpoint_interval or self.training_config.checkpoint_interval
        eval_interval = eval_interval or self.training_config.eval_interval

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

        # Lifecycle: on_train_start (dispatched once before the loop)
        self._dispatch_callback(
            "on_train_start",
            self._build_callback_context(
                step=start_step,
                metrics={},
                extras={"n_steps": n_steps, "start_step": start_step},
            ),
        )

        for step in range(start_step, start_step + n_steps):
            step_start = time.time()

            # Log curriculum stage transitions
            if self.curriculum is not None and self.curriculum.is_transition_step(step):
                stage = self.curriculum.get_current_stage(step)
                logger.info(
                    "curriculum_stage_transition",
                    step=step,
                    board_sizes=stage.board_sizes,
                    weights=stage.size_weights,
                )
                if self.tracker is not None:
                    self.tracker.log_metrics(
                        {
                            "curriculum/n_board_sizes": len(stage.board_sizes),
                            "curriculum/max_board_size": max(stage.board_sizes),
                        },
                        step=step,
                    )

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
            loss_output, lbb_constant, grad_norm, loss_weights, physics_output = (
                self._training_step(batch)
            )

            step_time = (time.time() - step_start) * 1000

            # Extract physics metrics if available
            physics_loss = 0.0
            physics_residual_loss = 0.0
            physics_boundary_loss = 0.0
            physics_weight = loss_weights.get("physics", 0.0)
            if physics_output is not None:
                physics_loss = physics_output.total.item()
                physics_residual_loss = physics_output.residual.item()
                physics_boundary_loss = physics_output.boundary.item()

            # Record metrics
            metrics = TrainingMetrics(
                step=step,
                total_loss=loss_output.total.item(),
                policy_loss=loss_output.policy.item(),
                value_loss=loss_output.value.item(),
                lbb_loss=loss_output.lbb.item(),
                lbb_constant=lbb_constant or 0.0,
                learning_rate=float(self.scheduler.get_last_lr()[0]),
                gradient_norm=grad_norm,
                buffer_size=len(self.buffer),
                games_generated=self.total_games_generated,
                step_time_ms=step_time,
                policy_weight=loss_weights.get("policy", 1.0),
                value_weight=loss_weights.get("value", 1.0),
                lbb_weight=loss_weights.get("lbb", 1.0),
                physics_loss=physics_loss,
                physics_residual_loss=physics_residual_loss,
                physics_boundary_loss=physics_boundary_loss,
                physics_weight=physics_weight,
            )
            self._metrics_history.append(metrics)

            # Lifecycle: on_step_end (dispatched after every training step)
            self._dispatch_callback(
                "on_step_end",
                self._build_callback_context(
                    step=step,
                    metrics=metrics.to_dict(),
                ),
            )

            # Tracker logging (every step by default, configurable via langfuse.log_interval)
            if self.tracker is not None:
                self.tracker.log_training_step(metrics)

            # Console logging
            if step % log_interval == 0:
                log_kwargs: dict[str, Any] = {
                    "step": step,
                    "loss": f"{loss_output.total.item():.4f}",
                    "policy_loss": f"{loss_output.policy.item():.4f}",
                    "value_loss": f"{loss_output.value.item():.4f}",
                    "lbb_loss": f"{loss_output.lbb.item():.4f}",
                    "lr": f"{metrics.learning_rate:.2e}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "buffer_size": len(self.buffer),
                    "step_time_ms": f"{step_time:.1f}",
                }
                # Add physics loss to log if enabled
                if self.use_physics_loss and physics_output is not None:
                    log_kwargs["physics_loss"] = f"{physics_loss:.4f}"
                logger.info("training_step", **log_kwargs)

            # Periodic evaluation
            if eval_interval and step > 0 and step % eval_interval == 0:
                avg_win_rate = self._run_evaluation(step)

                # Lifecycle: on_evaluation
                self._dispatch_callback(
                    "on_evaluation",
                    self._build_callback_context(
                        step=step,
                        metrics={"avg_win_rate": float(avg_win_rate)},
                    ),
                )

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

                # Lifecycle: on_checkpoint
                self._dispatch_callback(
                    "on_checkpoint",
                    self._build_callback_context(
                        step=step,
                        metrics=metrics.to_dict(),
                        extras={"path": str(checkpoint_path) if checkpoint_path else ""},
                    ),
                )

                # Log checkpoint as W&B artifact
                if self.tracker is not None and checkpoint_path is not None:
                    self.tracker.log_model_artifact(
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

        # Lifecycle: on_train_end (after final checkpoint, before W&B summary)
        _final_metrics = self._metrics_history[-1].to_dict() if self._metrics_history else {}
        self._dispatch_callback(
            "on_train_end",
            self._build_callback_context(
                step=self.global_step,
                metrics=_final_metrics,
                extras={
                    "final_checkpoint_path": (
                        str(final_checkpoint_path) if final_checkpoint_path else ""
                    ),
                    "n_steps": n_steps,
                },
            ),
        )

        # Log final summary to W&B
        if self.tracker is not None and self._metrics_history:
            final_metrics = self._metrics_history[-1]
            self.tracker.log_summary(
                {
                    "final/total_loss": final_metrics.total_loss,
                    "final/policy_loss": final_metrics.policy_loss,
                    "final/value_loss": final_metrics.value_loss,
                    "final/lbb_loss": final_metrics.lbb_loss,
                    "final/total_steps": n_steps,
                    "final/total_games": self.total_games_generated,
                }
            )

            # Log final checkpoint as best model artifact
            if final_checkpoint_path is not None:
                self.tracker.log_model_artifact(
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

        n_games = self.training_config.eval_games
        use_multi_res = self.training_config.multi_resolution_eval

        win_rates: list[float] = []

        if use_multi_res and hasattr(self.evaluator, "evaluate_multi_resolution"):
            # Use multi-resolution evaluation
            results = self.evaluator.evaluate_multi_resolution(n_games_per_size=n_games)
            for board_size, result in results.items():
                win_rates.append(result.win_rate)
                if self.tracker is not None:
                    self.tracker.log_evaluation(
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
                if self.tracker is not None:
                    self.tracker.log_evaluation(
                        result=result,
                        prefix=f"eval/{board_size}x{board_size}",
                        step=step,
                    )

        # Checkpoint tournament evaluation (Elo tracking)
        if self.elo_tracker is not None:
            self._run_checkpoint_tournament(step, n_games)

        # Engine evaluation (Stockfish benchmark)
        if (
            self.training_config.engine_eval_enabled
            and self.training_config.engine_eval_path is not None
            and self.evaluator.game is not None
        ):
            self._run_engine_evaluation(step)

        # Measure policy agreement
        policy_agreement = self.evaluator.measure_policy_agreement(
            n_positions=100,
            board_size=9,
        )

        if self.tracker is not None:
            self.tracker.log_metrics(
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
            self.training_config.n_tournament_opponents,
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
                if result.win_rate > WIN_RATE_ACCEPT_THRESHOLD:
                    score = 1.0
                elif result.win_rate < WIN_RATE_REJECT_THRESHOLD:
                    score = 0.0
                else:
                    score = 0.5

                # Update Elo ratings
                self.elo_tracker.update_ratings(step, opponent_step, score)

                # Log to W&B
                if self.tracker is not None:
                    current_rating = self.elo_tracker.get_rating(step)
                    self.tracker.log_metrics(
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

    def _run_engine_evaluation(self, step: int) -> None:
        """Run evaluation against external UCI engine (e.g., Stockfish).

        Creates engine and match configs from training config values,
        plays games, and logs Elo metrics to W&B.

        Args:
            step: Current training step.

        """
        if self.training_config.engine_eval_path is None:
            return

        from pathlib import Path

        from src.engines.config import MatchConfig, UCIConfig

        engine_path = self.training_config.engine_eval_path
        depth = self.training_config.engine_eval_depth
        n_games = self.training_config.engine_eval_games
        movetime = self.training_config.engine_eval_movetime_ms

        try:
            engine_config = UCIConfig(
                name="stockfish_eval",
                engine_path=Path(engine_path),
                depth_limit=depth if movetime is None else None,
                movetime_ms=movetime,
            )
            match_config = MatchConfig(
                name="engine_eval_match",
                n_games=n_games,
            )

            logger.info(
                "engine_evaluation_starting",
                step=step,
                engine_path=engine_path,
                depth=depth,
                n_games=n_games,
            )

            result = self.evaluator.evaluate_vs_engine(
                engine_config=engine_config,
                match_config=match_config,
            )

            # Log Elo metrics to W&B
            elo_metrics: dict[str, float | int] = {
                "eval/engine/win_rate": result.win_rate,
                "eval/engine/wins": result.wins,
                "eval/engine/losses": result.losses,
                "eval/engine/draws": result.draws,
                "eval/engine/n_games": result.n_games,
                "eval/engine/avg_game_length": result.avg_game_length,
            }

            # Extract Elo estimate from metadata if available
            if "elo_difference" in result.metadata:
                elo_metrics["eval/engine/elo_diff"] = result.metadata["elo_difference"]
            if "los" in result.metadata:
                elo_metrics["eval/engine/los"] = result.metadata["los"]

            if self.tracker is not None:
                self.tracker.log_metrics(
                    elo_metrics,
                    step=step,
                )

            logger.info(
                "engine_evaluation_completed",
                step=step,
                win_rate=f"{result.win_rate:.2%}",
                elo_diff=result.metadata.get("elo_difference", "N/A"),
            )

        except Exception as e:
            logger.warning(
                "engine_evaluation_failed",
                step=step,
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

    def load_checkpoint(  # type: ignore[override]
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
        return float(self.scheduler.get_last_lr()[0])


def create_trainer(
    model: AlphaGalerkinModel,
    config: AlphaGalerkinConfig,
    checkpoint_dir: Path | str | None = None,
    resume_from: Path | str | None = None,
    device: str = "auto",
    tracker: LangfuseTracker | None = None,
    distributed_context: DistributedContext | None = None,
    game: GameInterface | None = None,
) -> Trainer:
    """Create and optionally resume a trainer instance.

    Args:
        model: Model to train.
        config: Training configuration.
        checkpoint_dir: Checkpoint directory.
        resume_from: Path to checkpoint to resume from.
        device: Training device.
        tracker: Optional Langfuse experiment tracker.
        distributed_context: Optional distributed context (auto-detected if None).
        game: Optional GameInterface for non-Go games (e.g. PDE, chess).

    Returns:
        Configured trainer.

    """
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        tracker=tracker,
        distributed_context=distributed_context,
        game=game,
    )

    if resume_from is not None:
        trainer.load_checkpoint(path=resume_from)
        logger.info("training_resumed", from_step=trainer.global_step)

        # Update tracker step offset for resumed training
        if tracker is not None:
            tracker.set_step_offset(trainer.global_step)

    return trainer
