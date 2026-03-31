"""Base trainer class for AlphaGalerkin training infrastructure.

Provides shared machinery that all concrete trainers can inherit:
- Device selection (auto or explicit)
- Mixed precision (AMP) with GradScaler
- Gradient clipping
- Optimizer creation (AdamW)
- LR scheduling (cosine / linear warmup / none)
- Checkpoint save/load interface
- Structured logging with step timing
- Abstract hooks: compute_loss, generate_data, evaluate

Concrete trainers override the abstract methods and call super().__init__()
to receive the shared setup.

Usage::

    class MyTrainer(BaseTrainer[MyConfig]):
        def compute_loss(self, batch):
            return my_loss_fn(batch)

        def generate_data(self):
            return my_data_generator()

        def evaluate(self):
            return my_eval_fn()

Migration note: The existing ``Trainer``, ``OperatorTrainer``,
``VertexTrainer``, and video-compression ``Trainer`` will be migrated
to extend this base in a future PR.  They currently operate independently
for backwards-compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import structlog
import torch
from pydantic import Field
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from src.templates.config import BaseModuleConfig

logger = structlog.get_logger(__name__)

ConfigT = TypeVar("ConfigT", bound="BaseTrainerConfig")


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------


class BaseTrainerConfig(BaseModuleConfig):
    """Shared configuration for all AlphaGalerkin trainers.

    Concrete trainers should subclass this and add domain-specific fields.
    All hyperparameters must be declared here rather than hardcoded, so
    that every config is reproducible from YAML.
    """

    # Optimizer
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Peak learning rate for AdamW.",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        description="L2 weight decay coefficient.",
    )
    gradient_clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for clipping.",
    )

    # Scheduler
    lr_scheduler: str = Field(
        default="cosine",
        description="LR scheduler type: 'cosine', 'linear', 'none'.",
    )
    warmup_steps: int = Field(
        default=1000,
        ge=0,
        description="Number of linear warmup steps.",
    )
    warmup_start_factor: float = Field(
        default=1e-6,
        gt=0.0,
        le=1.0,
        description="Starting LR factor for warmup (lr * factor at step 0).",
    )
    total_steps: int = Field(
        default=100_000,
        ge=1,
        description="Total training steps (used for cosine annealing).",
    )
    min_lr_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Ratio of min_lr to peak lr for cosine schedule.",
    )

    # Mixed precision
    use_amp: bool = Field(
        default=False,
        description="Enable automatic mixed precision (AMP) training.",
    )

    # Checkpointing
    save_every: int = Field(
        default=1000,
        ge=1,
        description="Save checkpoint every N steps.",
    )
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for saving checkpoints.",
    )

    # Logging
    log_every: int = Field(
        default=100,
        ge=1,
        description="Log metrics every N steps.",
    )


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result from a single training step.

    Attributes:
        loss: Scalar loss value for this step.
        metrics: Extra scalar metrics to log (e.g. per-component losses).
        grad_norm: Gradient norm after clipping, or None if not computed.

    """

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    grad_norm: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Serialize to flat dict for logging."""
        result: dict[str, float] = {"loss": self.loss}
        result.update(self.metrics)
        if self.grad_norm is not None:
            result["grad_norm"] = self.grad_norm
        return result


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------


class BaseTrainer(ABC, Generic[ConfigT]):
    """Abstract base class for AlphaGalerkin trainers.

    Provides shared infrastructure (AMP, gradient clipping, LR scheduling,
    checkpointing) that concrete trainers build on top of.

    Generic type parameter ``ConfigT`` binds to the trainer's specific
    ``BaseTrainerConfig`` subclass, enabling typed access to ``self.config``
    in concrete implementations.
    """

    def __init__(
        self,
        model: Module,
        config: ConfigT,
        device: torch.device | str = "auto",
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        """Initialise shared training infrastructure.

        Args:
            model: Model to train.
            config: Trainer configuration (subclass of BaseTrainerConfig).
            device: Training device. ``"auto"`` selects CUDA if available.
            checkpoint_dir: Override for checkpoint directory. Falls back to
                ``config.checkpoint_dir``.

        """
        self.config = config

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Structured logger (must be set before _setup_scheduler which may log warnings)
        self._log = logger.bind(
            trainer=type(self).__name__,
            device=str(self.device),
        )

        # Optimizer and scheduler
        self.optimizer: Optimizer = self._setup_optimizer()
        self.scheduler: LRScheduler = self._setup_scheduler()

        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler: GradScaler | None = GradScaler() if self.use_amp else None

        # Checkpoint directory
        _ckpt_dir = checkpoint_dir if checkpoint_dir is not None else config.checkpoint_dir
        self.checkpoint_dir = Path(_ckpt_dir)

        # Step counter
        self.global_step: int = 0

        self._log.info(
            "base_trainer_initialized",
            use_amp=self.use_amp,
            lr=config.learning_rate,
            scheduler=config.lr_scheduler,
            checkpoint_dir=str(self.checkpoint_dir),
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_loss(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        """Compute loss for a training batch.

        Args:
            batch: A batch of training data (type depends on concrete trainer).

        Returns:
            Tuple of (loss_tensor, metrics_dict) where metrics_dict contains
            any extra per-component losses or metrics to log.

        """
        ...

    @abstractmethod
    def generate_data(self) -> Any:
        """Generate or fetch a training batch.

        Returns:
            A batch suitable for ``compute_loss``.

        """
        ...

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation and return metrics.

        Returns:
            Dictionary of evaluation metric name -> value.

        """
        ...

    # ------------------------------------------------------------------
    # Training step (shared)
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        """Execute one training step.

        1. Generates data via ``generate_data()``.
        2. Computes loss via ``compute_loss(batch)``.
        3. Runs backward pass with optional AMP.
        4. Clips gradients and steps optimizer.
        5. Steps LR scheduler.

        Returns:
            StepResult with loss, metrics, and grad norm.

        """
        self.model.train()
        batch = self.generate_data()

        self.optimizer.zero_grad()

        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, metrics = self.compute_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, metrics = self.compute_loss(batch)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            )
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        result = StepResult(
            loss=float(loss),
            metrics=metrics,
            grad_norm=grad_norm,
        )

        if self.global_step % self.config.log_every == 0:
            self._log.info(
                "training_step",
                step=self.global_step,
                **result.to_dict(),
                lr=self.get_current_lr(),
            )

        return result

    # ------------------------------------------------------------------
    # Optimizer and scheduler helpers
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> Optimizer:
        """Create AdamW optimizer from config.

        Override to use a different optimizer.
        """
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self) -> LRScheduler:
        """Create LR scheduler from config.

        Supports:
        - ``"cosine"``: linear warmup then cosine annealing.
        - ``"linear"``: linear warmup then constant.
        - ``"none"``: constant LR (no scheduling).
        """
        scheduler_type = self.config.lr_scheduler.lower()
        warmup_steps = self.config.warmup_steps
        total_steps = self.config.total_steps
        min_lr = self.config.learning_rate * self.config.min_lr_ratio

        if scheduler_type == "none" or total_steps <= 0:
            # Constant LR - use ConstantLR (factor=1 = no change)
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=0)

        main_steps = max(1, total_steps - warmup_steps)

        if scheduler_type == "cosine":
            main_sched: LRScheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=main_steps,
                eta_min=min_lr,
            )
        elif scheduler_type == "linear":
            main_sched = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=main_steps,
            )
        else:
            self._log.warning(
                "unknown_scheduler_type",
                scheduler=scheduler_type,
                fallback="cosine",
            )
            main_sched = CosineAnnealingLR(
                self.optimizer,
                T_max=main_steps,
                eta_min=min_lr,
            )

        if warmup_steps > 0:
            warmup_sched = LinearLR(
                self.optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_steps],
            )

        return main_sched

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str | None = None) -> Path:
        """Save model and optimizer state to disk.

        Args:
            filename: Filename for the checkpoint. Defaults to
                ``checkpoint_{global_step:08d}.pt``.

        Returns:
            Path to the saved checkpoint file.

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"checkpoint_{self.global_step:08d}.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": (
                    self.scaler.state_dict() if self.scaler is not None else None
                ),
                "config": self.config.model_dump(),
                "trainer_class": type(self).__name__,
            },
            path,
        )
        self._log.info("checkpoint_saved", path=str(path), step=self.global_step)
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model and optimizer state from a checkpoint.

        Args:
            path: Path to the checkpoint file.

        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self.scaler is not None and state.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.global_step = state.get("global_step", 0)
        self._log.info(
            "checkpoint_loaded",
            path=str(path),
            step=self.global_step,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_current_lr(self) -> float:
        """Return current learning rate from optimizer."""
        return float(self.optimizer.param_groups[0]["lr"])

    def set_training(self, mode: bool = True) -> None:
        """Set model training mode."""
        self.model.train(mode)
