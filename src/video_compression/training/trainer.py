"""Trainer for video compression models.

Provides a complete training loop with:
- Multi-lambda training for full R-D curve
- Mixed precision training
- Gradient clipping
- Learning rate scheduling
- Checkpointing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from src.video_compression.config import CodecConfig, TrainingConfig
from src.video_compression.codec.codec import VideoCodec
from src.video_compression.training.loss import CompressionLoss
from src.video_compression.metrics.quality import compute_psnr, compute_ms_ssim


logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Mutable training state."""

    step: int = 0
    epoch: int = 0
    best_rd_loss: float = float("inf")
    lambda_idx: int = 0


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    loss: float
    rate: float
    distortion: float
    psnr: float
    ms_ssim: float | None = None
    lr: float | None = None


class VideoCompressionTrainer:
    """Trainer for video compression models."""

    def __init__(
        self,
        codec: VideoCodec,
        config: TrainingConfig,
        output_dir: Path | str = "outputs/compression",
    ) -> None:
        """Initialize trainer.

        Args:
            codec: Video codec model.
            config: Training configuration.
            output_dir: Output directory for checkpoints.
        """
        self.codec = codec
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.codec.to(self.device)

        # Loss function
        self.loss_fn = CompressionLoss(
            lambda_rd=config.lambda_rd,
            distortion_metric=config.distortion_metric,
            ms_ssim_weight=config.ms_ssim_weight,
            use_perceptual=config.use_perceptual_loss,
            perceptual_weight=config.perceptual_weight,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.codec.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps - config.warmup_steps,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # State
        self.state = TrainingState()

    def train_step(
        self,
        batch: torch.Tensor,
    ) -> TrainingMetrics:
        """Execute a single training step.

        Args:
            batch: Input images (B, 3, H, W) in [0, 1].

        Returns:
            Training metrics.
        """
        self.codec.train()
        batch = batch.to(self.device)

        # Get current lambda
        lambda_rd = self.config.lambda_values[self.state.lambda_idx]
        self.loss_fn.rd_loss.lambda_rd = lambda_rd

        # Forward pass with optional mixed precision
        with autocast(enabled=self.config.use_amp):
            x_hat, rate, distortion = self.codec(batch)
            losses = self.loss_fn(x_hat, batch, rate)
            loss = losses["total"]

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.codec.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.codec.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        # LR scheduling (after warmup)
        if self.state.step >= self.config.warmup_steps:
            self.scheduler.step()
        else:
            # Linear warmup
            warmup_factor = self.state.step / self.config.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.learning_rate * warmup_factor

        self.state.step += 1

        # Cycle through lambda values
        if self.state.step % 100 == 0:
            self.state.lambda_idx = (self.state.lambda_idx + 1) % len(self.config.lambda_values)

        return TrainingMetrics(
            loss=loss.item(),
            rate=losses["rate"].item(),
            distortion=losses["distortion"].item(),
            psnr=losses["psnr"].item(),
            ms_ssim=losses.get("ms_ssim_loss", torch.tensor(0.0)).item() if "ms_ssim_loss" in losses else None,
            lr=self.optimizer.param_groups[0]["lr"],
        )

    @torch.no_grad()
    def eval_step(
        self,
        batch: torch.Tensor,
    ) -> TrainingMetrics:
        """Execute evaluation step.

        Args:
            batch: Input images (B, 3, H, W) in [0, 1].

        Returns:
            Evaluation metrics.
        """
        self.codec.eval()
        batch = batch.to(self.device)

        x_hat, rate, distortion = self.codec(batch)
        losses = self.loss_fn(x_hat, batch, rate)

        return TrainingMetrics(
            loss=losses["total"].item(),
            rate=losses["rate"].item(),
            distortion=losses["distortion"].item(),
            psnr=losses["psnr"].item(),
            ms_ssim=losses.get("ms_ssim_loss", torch.tensor(0.0)).item() if "ms_ssim_loss" in losses else None,
        )

    def train(
        self,
        train_loader: Iterator[torch.Tensor],
        val_loader: Iterator[torch.Tensor] | None = None,
    ) -> None:
        """Run full training loop.

        Args:
            train_loader: Training data iterator.
            val_loader: Optional validation data iterator.
        """
        logger.info(f"Starting training for {self.config.total_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Lambda values: {self.config.lambda_values}")

        for batch in train_loader:
            if self.state.step >= self.config.total_steps:
                break

            metrics = self.train_step(batch)

            # Logging
            if self.state.step % 100 == 0:
                logger.info(
                    f"Step {self.state.step}: loss={metrics.loss:.4f}, "
                    f"rate={metrics.rate:.4f}bpp, PSNR={metrics.psnr:.2f}dB, "
                    f"lr={metrics.lr:.2e}"
                )

            # Evaluation
            if val_loader is not None and self.state.step % self.config.eval_interval == 0:
                val_metrics = self._evaluate(val_loader)
                logger.info(
                    f"Validation: loss={val_metrics.loss:.4f}, "
                    f"PSNR={val_metrics.psnr:.2f}dB"
                )

                # Save best model
                if val_metrics.loss < self.state.best_rd_loss:
                    self.state.best_rd_loss = val_metrics.loss
                    self.save_checkpoint("best.pt")

            # Checkpoint
            if self.state.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.state.step:08d}.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")
        logger.info("Training complete")

    def _evaluate(
        self,
        val_loader: Iterator[torch.Tensor],
        max_batches: int = 10,
    ) -> TrainingMetrics:
        """Run evaluation on validation set.

        Args:
            val_loader: Validation data iterator.
            max_batches: Maximum batches to evaluate.

        Returns:
            Averaged metrics.
        """
        total_metrics = {
            "loss": 0.0,
            "rate": 0.0,
            "distortion": 0.0,
            "psnr": 0.0,
        }
        count = 0

        for batch in val_loader:
            if count >= max_batches:
                break

            metrics = self.eval_step(batch)
            total_metrics["loss"] += metrics.loss
            total_metrics["rate"] += metrics.rate
            total_metrics["distortion"] += metrics.distortion
            total_metrics["psnr"] += metrics.psnr
            count += 1

        return TrainingMetrics(
            loss=total_metrics["loss"] / count,
            rate=total_metrics["rate"] / count,
            distortion=total_metrics["distortion"] / count,
            psnr=total_metrics["psnr"] / count,
        )

    def save_checkpoint(self, filename: str) -> Path:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        path = self.output_dir / filename

        checkpoint = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "best_rd_loss": self.state.best_rd_loss,
            "model_state": self.codec.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.model_dump(),
        }

        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.state.step = checkpoint["step"]
        self.state.epoch = checkpoint["epoch"]
        self.state.best_rd_loss = checkpoint["best_rd_loss"]

        self.codec.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if self.scaler is not None and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        logger.info(f"Loaded checkpoint from {path} at step {self.state.step}")
