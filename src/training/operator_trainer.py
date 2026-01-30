"""Training infrastructure for neural operators.

Provides a complete training loop with:
- Configurable loss functions
- Learning rate scheduling
- Gradient clipping
- Checkpoint saving/loading
- Structured logging
"""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import structlog
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from src.training.losses import get_loss

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for operator training."""

    # Optimization
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Loss
    loss_type: str = "l2_relative"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)

    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'onecycle', 'none'

    # Checkpointing
    save_every: int = 10
    checkpoint_dir: Path | str = "checkpoints"

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-5

    # Device
    device: str = "auto"

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class OperatorTrainer:
    """Training loop for neural operators.
    
    Example:
        >>> from src.modeling.operator import NeuralOperator
        >>> from src.data.physics_dataset import PhysicsDataset
        >>> 
        >>> model = NeuralOperator(in_channels=1, out_channels=1)
        >>> trainer = OperatorTrainer(model, config=TrainingConfig())
        >>> trainer.fit(train_loader, val_loader)

    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Neural operator model.
            config: Training configuration.

        """
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)

        # Loss function
        self.criterion = get_loss(self.config.loss_type, **self.config.loss_kwargs)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler (set in fit())
        self.scheduler: Any = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        logger.info(
            "trainer_initialized",
            model_params=sum(p.numel() for p in model.parameters()),
            device=self.config.device,
            loss=self.config.loss_type,
        )

    def _prepare_batch(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Move batch to device and reshape for model."""
        device = self.config.device

        # Get input/output
        x = batch["input"].to(device)
        y = batch["output"].to(device)
        grid_size = int(batch["grid_size"][0].item())

        # Reshape to (batch, 1, h, w)
        x = x.view(-1, 1, grid_size, grid_size)
        y = y.view(-1, 1, grid_size, grid_size)

        return x, y

    def train_epoch(self, train_loader: DataLoader[Any]) -> float:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Average training loss.

        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y = self._prepare_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            # Step scheduler if it's OneCycleLR (batch-wise)
            if self.config.scheduler == "onecycle" and self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[Any]) -> float:
        """Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Average validation loss.

        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            x, y = self._prepare_batch(batch)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_loader: Training data.
            val_loader: Optional validation data.

        Returns:
            Training history.

        """
        # Setup scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        elif self.config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                epochs=self.config.epochs,
                steps_per_epoch=len(train_loader),
            )

        logger.info(
            "training_start",
            epochs=self.config.epochs,
            train_samples=len(cast(Sized, train_loader.dataset)),
            val_samples=len(cast(Sized, val_loader.dataset)) if val_loader else 0,
        )

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)

            # Log current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Step scheduler
            if self.scheduler is not None:
                if self.config.scheduler != "onecycle":
                    self.scheduler.step()

            # Logging
            logger.info(
                "epoch_complete",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}" if val_loss else "N/A",
                lr=f"{current_lr:.2e}",
            )

            # Early stopping check
            if val_loss is not None:
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        logger.info(
                            "early_stopping",
                            best_val_loss=self.best_val_loss,
                            epoch=epoch + 1,
                        )
                        break

            # Periodic checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        logger.info(
            "training_complete",
            final_train_loss=self.history["train_loss"][-1],
            best_val_loss=self.best_val_loss,
        )

        return self.history

    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.

        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.debug("checkpoint_saved", path=str(path))

        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        logger.info("checkpoint_loaded", path=str(path), epoch=self.current_epoch)
