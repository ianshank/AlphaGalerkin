"""Distributed trainer for AlphaGalerkin.

This module provides the main distributed training coordinator that
handles multi-node, multi-GPU training with gradient synchronization.

Features:
    - Automatic DDP wrapping
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint management with rank awareness
    - Metric aggregation across processes
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from src.distributed.config import DistributedInfraConfig, _get_env_rank_info
from src.distributed.gradient_sync import GradientAccumulator, GradientSynchronizer
from src.training.base_trainer import BaseTrainer

if TYPE_CHECKING:
    from config.schemas import AlphaGalerkinConfig
    from src.data.collate import TrainingBatch
    from src.modeling.model import AlphaGalerkinModel
    from src.training.losses import AlphaGalerkinLoss

logger = structlog.get_logger(__name__)


@dataclass
class DistributedMetrics:
    """Aggregated metrics from distributed training step."""

    step: int = 0
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    lbb_loss: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    throughput_samples_per_sec: float = 0.0
    sync_time_ms: float = 0.0
    step_time_ms: float = 0.0

    # Distributed-specific
    world_size: int = 1
    rank: int = 0
    global_batch_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "total_loss": self.total_loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "lbb_loss": self.lbb_loss,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "sync_time_ms": self.sync_time_ms,
            "step_time_ms": self.step_time_ms,
            "world_size": self.world_size,
            "rank": self.rank,
            "global_batch_size": self.global_batch_size,
        }


class DistributedTrainer(BaseTrainer):  # type: ignore[type-arg]
    """Distributed trainer for AlphaGalerkin.

    Coordinates distributed training across multiple nodes/GPUs using
    PyTorch's DistributedDataParallel.

    Inherits shared AMP, gradient-clipping, LR-scheduling, and
    checkpoint helpers from :class:`BaseTrainer`.  The ``__init__``
    does **not** call ``super().__init__()`` because the distributed
    trainer has a substantially different setup flow; instead it
    sets the attributes that ``BaseTrainer`` helpers rely on directly.

    Attributes:
        model: The model being trained (unwrapped).
        ddp_model: DDP-wrapped model for distributed training.
        config: Distributed training configuration.
        rank: Global rank of this process.
        local_rank: Local rank within the node.
        world_size: Total number of processes.

    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        config: AlphaGalerkinConfig,
        distributed_config: DistributedInfraConfig,
        loss_fn: AlphaGalerkinLoss,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Initialize distributed trainer.

        Args:
            model: AlphaGalerkin model to train.
            config: Full training configuration.
            distributed_config: Distributed-specific configuration.
            loss_fn: Loss function.
            optimizer: Optional optimizer (created if not provided).
            scheduler: Optional LR scheduler.

        """
        self.config = config
        self.distributed_config = distributed_config
        self.loss_fn = loss_fn

        # Get distributed info from environment
        self.rank, self.local_rank, self.world_size = _get_env_rank_info()
        self._is_main_process = self.rank == 0

        # Device setup
        self.device = self._setup_device()

        # Model setup
        self.model = model.to(self.device)
        self.ddp_model: DDP | None = None

        # Optimizer and scheduler
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler

        # Mixed precision (uses BaseTrainer helper)
        _amp_dtype = getattr(torch, distributed_config.amp_dtype, torch.float16)
        self.use_amp, self.scaler, self.amp_dtype = self._setup_amp(
            use_amp=distributed_config.use_amp,
            device=self.device,
            amp_dtype=_amp_dtype,
        )

        # Gradient synchronization
        self.grad_sync: GradientSynchronizer | None = None
        self.grad_accumulator = GradientAccumulator(
            accumulation_steps=distributed_config.gradient_accumulation_steps,
        )

        # Training state
        self.global_step = 0
        self._is_initialized = False

        self._logger = structlog.get_logger(__name__).bind(
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

    def _setup_device(self) -> torch.device:
        """Setup device based on local rank.

        Returns:
            Device for this process.

        """
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def _create_optimizer(self) -> Optimizer:  # type: ignore[override]
        """Create optimizer with learning rate scaling.

        Uses :meth:`BaseTrainer._create_optimizer` static helper with
        linear scaling rule applied to the learning rate.

        Returns:
            Configured optimizer.

        """
        # Scale learning rate by world size (linear scaling rule)
        base_lr = self.config.training.learning_rate
        scaled_lr = base_lr * self.world_size

        return BaseTrainer._create_optimizer(
            self.model,
            lr=scaled_lr,
            weight_decay=self.config.training.weight_decay,
        )

    def setup(self) -> None:
        """Initialize distributed process group and wrap model.

        Must be called before training starts.
        """
        if self._is_initialized:
            return

        self._logger.info("initializing_distributed_training")

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.distributed_config.backend.value,
                init_method=self.distributed_config.init_method,
                world_size=self.world_size,
                rank=self.rank,
            )

        # Convert BatchNorm to SyncBatchNorm if requested
        if self.distributed_config.sync_batch_norm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Wrap model in DDP
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            output_device=self.local_rank if self.device.type == "cuda" else None,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
            broadcast_buffers=self.distributed_config.broadcast_buffers,
            bucket_cap_mb=self.distributed_config.bucket_cap_mb,
            gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
        )

        # Setup gradient synchronizer
        self.grad_sync = GradientSynchronizer(
            model=self.ddp_model.module,
            config=self.distributed_config,
        )

        self._is_initialized = True
        self._logger.info(
            "distributed_training_initialized",
            device=str(self.device),
            ddp_enabled=True,
        )

    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        self._is_initialized = False
        self._logger.info("distributed_training_cleanup_complete")

    def train_step(self, batch: TrainingBatch) -> DistributedMetrics:
        """Execute single distributed training step.

        Handles gradient accumulation and synchronization.

        Args:
            batch: Training batch (already on device).

        Returns:
            Aggregated training metrics.

        """
        if not self._is_initialized:
            raise RuntimeError("Distributed trainer not initialized. Call setup() first.")

        step_start = time.perf_counter()
        sync_time = 0.0

        # Determine if this is an accumulation step
        is_sync_step = self.grad_accumulator.should_step()

        # Forward pass with optional AMP and no_sync context
        context = (
            self.ddp_model.no_sync()
            if not is_sync_step and hasattr(self.ddp_model, "no_sync")
            else nullcontext()
        )

        with context:
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    output = self.ddp_model(batch.board_states, return_lbb=True)
                    loss_output = self.loss_fn(
                        policy_logits=output.policy_logits,
                        value=output.value,
                        target_policy=batch.target_policies,
                        target_value=batch.target_values,
                        lbb_constant=output.lbb_constant,
                        action_mask=batch.action_mask.float(),
                    )

                # Scale loss for accumulation
                scaled_loss = self.grad_accumulator.scale(loss_output.total)

                # Backward with scaler
                self.scaler.scale(scaled_loss).backward()
            else:
                output = self.ddp_model(batch.board_states, return_lbb=True)
                loss_output = self.loss_fn(
                    policy_logits=output.policy_logits,
                    value=output.value,
                    target_policy=batch.target_policies,
                    target_value=batch.target_values,
                    lbb_constant=output.lbb_constant,
                    action_mask=batch.action_mask.float(),
                )

                scaled_loss = self.grad_accumulator.scale(loss_output.total)
                scaled_loss.backward()

        # Accumulate loss
        self.grad_accumulator.accumulate(loss_output.total.item())

        # Optimizer step if accumulation complete
        grad_norm = 0.0
        if is_sync_step:
            # Use BaseTrainer._clip_gradients for consistent clipping
            grad_norm = self._clip_gradients(self.ddp_model, self.config.training.gradient_clip)
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            self.grad_accumulator.reset()

        # Calculate timing
        step_time = (time.perf_counter() - step_start) * 1000

        # Get effective batch size
        local_batch_size = batch.board_states.shape[0]
        global_batch_size = local_batch_size * self.world_size

        # Calculate throughput
        throughput = global_batch_size / (step_time / 1000) if step_time > 0 else 0

        metrics = DistributedMetrics(
            step=self.global_step,
            total_loss=loss_output.total.item(),
            policy_loss=loss_output.policy.item(),
            value_loss=loss_output.value.item(),
            lbb_loss=loss_output.lbb.item(),
            gradient_norm=grad_norm,
            learning_rate=self._get_lr(),
            throughput_samples_per_sec=throughput,
            sync_time_ms=sync_time,
            step_time_ms=step_time,
            world_size=self.world_size,
            rank=self.rank,
            global_batch_size=global_batch_size,
        )

        return metrics

    # ------------------------------------------------------------------
    # Abstract method implementations (required by BaseTrainer ABC)
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Not used directly -- DistributedTrainer uses train_step instead."""
        raise NotImplementedError("DistributedTrainer uses train_step(); use that method instead.")

    def generate_data(self) -> Any:
        """Not used directly -- DistributedTrainer manages data externally."""
        raise NotImplementedError("DistributedTrainer receives batches via train_step().")

    def evaluate(self) -> dict[str, float]:
        """Not used directly -- evaluation is handled externally."""
        raise NotImplementedError("DistributedTrainer evaluation is managed externally.")

    def _get_lr(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate.

        """
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]

    def aggregate_metrics(self, local_metrics: DistributedMetrics) -> DistributedMetrics:
        """Aggregate metrics across all processes.

        Args:
            local_metrics: Metrics from local process.

        Returns:
            Aggregated metrics (averaged across processes).

        """
        if not dist.is_initialized() or self.world_size == 1:
            return local_metrics

        # Create tensor of metrics to reduce
        metrics_tensor = torch.tensor(
            [
                local_metrics.total_loss,
                local_metrics.policy_loss,
                local_metrics.value_loss,
                local_metrics.lbb_loss,
                local_metrics.gradient_norm,
                local_metrics.throughput_samples_per_sec,
                local_metrics.step_time_ms,
            ],
            device=self.device,
        )

        # All-reduce (sum then divide by world_size)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= self.world_size

        # Update metrics with averaged values
        return DistributedMetrics(
            step=local_metrics.step,
            total_loss=metrics_tensor[0].item(),
            policy_loss=metrics_tensor[1].item(),
            value_loss=metrics_tensor[2].item(),
            lbb_loss=metrics_tensor[3].item(),
            gradient_norm=metrics_tensor[4].item(),
            learning_rate=local_metrics.learning_rate,
            throughput_samples_per_sec=metrics_tensor[5].item() * self.world_size,
            sync_time_ms=local_metrics.sync_time_ms,
            step_time_ms=metrics_tensor[6].item(),
            world_size=self.world_size,
            rank=self.rank,
            global_batch_size=local_metrics.global_batch_size,
        )

    def save_checkpoint(  # type: ignore[override]
        self,
        path: Path | str,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save checkpoint (only on rank 0 by default).

        Args:
            path: Checkpoint save path.
            metrics: Optional metrics to include.

        Returns:
            Path to saved checkpoint, or None if not main process.

        """
        if not self._is_main_process and self.distributed_config.save_on_rank_0_only:
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else {},
            "distributed_config": self.distributed_config.model_dump(),
            "metrics": metrics or {},
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

        self._logger.info(
            "checkpoint_saved",
            path=str(path),
            step=self.global_step,
        )

        return path

    def load_checkpoint(self, path: Path | str) -> int:  # type: ignore[override]
        """Load checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Training step from checkpoint.

        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint.get("step", 0)

        self._logger.info(
            "checkpoint_loaded",
            path=str(path),
            step=self.global_step,
        )

        return self.global_step

    def create_distributed_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a DataLoader with distributed sampling.

        Args:
            dataset: Dataset to load from.
            batch_size: Per-process batch size.
            shuffle: Whether to shuffle data.
            num_workers: Number of data loading workers.
            **kwargs: Additional DataLoader arguments.

        Returns:
            Configured DataLoader with DistributedSampler.

        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.distributed_config.pin_memory,
            prefetch_factor=self.distributed_config.prefetch_factor,
            **kwargs,
        )

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0).

        Returns:
            True if main process.

        """
        return self._is_main_process


# nullcontext imported at module level from contextlib


def create_distributed_trainer(
    model: AlphaGalerkinModel,
    config: AlphaGalerkinConfig,
    distributed_config: DistributedInfraConfig,
    loss_fn: AlphaGalerkinLoss,
    **kwargs: Any,
) -> DistributedTrainer:
    """Factory function to create distributed trainer.

    Args:
        model: Model to train.
        config: Training configuration.
        distributed_config: Distributed configuration.
        loss_fn: Loss function.
        **kwargs: Additional trainer arguments.

    Returns:
        Configured DistributedTrainer instance.

    """
    return DistributedTrainer(
        model=model,
        config=config,
        distributed_config=distributed_config,
        loss_fn=loss_fn,
        **kwargs,
    )
