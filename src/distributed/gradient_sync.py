"""Gradient synchronization utilities for distributed training.

This module provides utilities for synchronizing gradients across
distributed processes using NCCL or Gloo backends.

Features:
    - All-reduce gradient synchronization
    - Gradient compression for bandwidth reduction
    - Gradient accumulation support
    - Scaling for effective batch size
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
import torch
import torch.distributed as dist
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.distributed.config import DistributedInfraConfig

logger = structlog.get_logger(__name__)


@dataclass
class SyncMetrics:
    """Metrics from gradient synchronization."""

    sync_time_ms: float = 0.0
    gradient_norm_before: float = 0.0
    gradient_norm_after: float = 0.0
    num_parameters: int = 0
    compression_ratio: float = 1.0


class GradientSynchronizer:
    """Synchronizes gradients across distributed processes.

    Supports various synchronization strategies including:
    - Standard all-reduce
    - Gradient accumulation
    - Optional gradient compression

    Attributes:
        config: Distributed training configuration.
        model: The model being trained.
        accumulation_counter: Counter for gradient accumulation.

    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedInfraConfig,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initialize gradient synchronizer.

        Args:
            model: Model whose gradients will be synchronized.
            config: Distributed training configuration.
            process_group: Optional process group for communication.

        """
        self.model = model
        self.config = config
        self.process_group = process_group
        self.accumulation_counter = 0

        self._logger = structlog.get_logger(__name__).bind(
            world_size=config.world_size,
            backend=config.backend.value,
        )

        # Cache for gradient shapes (used in compression)
        self._grad_shapes: dict[str, torch.Size] = {}
        self._grad_numel: dict[str, int] = {}

        # Initialize gradient buckets for efficient all-reduce
        self._initialize_buckets()

    def _initialize_buckets(self) -> None:
        """Initialize gradient buckets for bucketed all-reduce."""
        self._buckets: list[list[Tensor]] = []
        self._bucket_indices: dict[str, tuple[int, int]] = {}

        current_bucket: list[Tensor] = []
        current_bucket_size = 0
        bucket_cap_bytes = int(self.config.bucket_cap_mb * 1024 * 1024)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            param_size = param.numel() * param.element_size()
            self._grad_shapes[name] = param.shape
            self._grad_numel[name] = param.numel()

            if current_bucket_size + param_size > bucket_cap_bytes and current_bucket:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0

            bucket_idx = len(self._buckets)
            param_idx_in_bucket = len(current_bucket)
            self._bucket_indices[name] = (bucket_idx, param_idx_in_bucket)

            current_bucket.append(param)
            current_bucket_size += param_size

        if current_bucket:
            self._buckets.append(current_bucket)

        self._logger.debug(
            "buckets_initialized",
            num_buckets=len(self._buckets),
            bucket_cap_mb=self.config.bucket_cap_mb,
        )

    def should_sync(self) -> bool:
        """Check if gradients should be synchronized this step.

        Returns:
            True if accumulation count reached, False otherwise.

        """
        return (
            self.accumulation_counter + 1
        ) % self.config.gradient_accumulation_steps == 0

    def step(self) -> None:
        """Increment accumulation counter."""
        self.accumulation_counter += 1

    def reset(self) -> None:
        """Reset accumulation counter."""
        self.accumulation_counter = 0

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Context manager to disable gradient synchronization.

        Useful for gradient accumulation steps where you don't
        want to synchronize until the final step.

        Yields:
            None

        """
        # If model is wrapped in DDP, use its no_sync context
        if hasattr(self.model, "no_sync"):
            with self.model.no_sync():
                yield
        else:
            yield

    def synchronize(self) -> SyncMetrics:
        """Synchronize gradients across all processes.

        Performs all-reduce on model gradients with optional compression.

        Returns:
            Metrics about the synchronization operation.

        """
        if not dist.is_initialized():
            return SyncMetrics()

        # Use CUDA timing only if available and on CUDA device
        use_cuda_timing = torch.cuda.is_available() and self._get_device().type == "cuda"

        if use_cuda_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            import time as time_module
            start_time = time_module.perf_counter()

        # Collect gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad)

        if not grads:
            return SyncMetrics()

        # Compute gradient norm before sync
        grad_norm_before = self._compute_grad_norm(grads)

        # Flatten for efficient communication
        flat_grads = torch.cat([g.view(-1) for g in grads])

        # All-reduce
        if self.config.gradient_compression:
            compressed, compression_ratio = self._compress_gradients(flat_grads)
            dist.all_reduce(compressed, op=dist.ReduceOp.SUM, group=self.process_group)
            flat_grads = self._decompress_gradients(compressed, flat_grads.numel())
        else:
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, group=self.process_group)
            compression_ratio = 1.0

        # Average gradients
        flat_grads.div_(self.config.world_size)

        # Unflatten back to original shapes
        offset = 0
        for param in self.model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grads[offset : offset + numel].view_as(param.grad))
                offset += numel

        # Compute gradient norm after sync
        grads_after = [p.grad for p in self.model.parameters() if p.grad is not None]
        grad_norm_after = self._compute_grad_norm(grads_after)

        # Calculate sync time based on available timing method
        if use_cuda_timing:
            end_event.record()
            torch.cuda.synchronize()
            sync_time = start_event.elapsed_time(end_event)
        else:
            sync_time = (time_module.perf_counter() - start_time) * 1000  # Convert to ms

        metrics = SyncMetrics(
            sync_time_ms=sync_time,
            gradient_norm_before=grad_norm_before,
            gradient_norm_after=grad_norm_after,
            num_parameters=flat_grads.numel(),
            compression_ratio=compression_ratio,
        )

        self._logger.debug(
            "gradients_synchronized",
            sync_time_ms=f"{sync_time:.2f}",
            grad_norm_before=f"{grad_norm_before:.4f}",
            grad_norm_after=f"{grad_norm_after:.4f}",
        )

        return metrics

    def _compute_grad_norm(self, grads: list[Tensor]) -> float:
        """Compute total gradient norm.

        Args:
            grads: List of gradient tensors.

        Returns:
            L2 norm of all gradients.

        """
        if not grads:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(g, 2) for g in grads]), 2
        )
        return total_norm.item()

    def _compress_gradients(
        self, flat_grads: Tensor
    ) -> tuple[Tensor, float]:
        """Compress gradients for bandwidth reduction.

        Uses top-k sparsification by default.

        Args:
            flat_grads: Flattened gradient tensor.

        Returns:
            Tuple of (compressed tensor, compression ratio).

        """
        # Top-k sparsification (keep top 10% by magnitude)
        k = max(1, int(0.1 * flat_grads.numel()))
        _, indices = torch.topk(flat_grads.abs(), k)

        # Create sparse representation
        values = flat_grads[indices]
        compressed = torch.zeros(k * 2, device=flat_grads.device)
        compressed[:k] = values
        compressed[k:] = indices.float()

        compression_ratio = flat_grads.numel() / compressed.numel()
        return compressed, compression_ratio

    def _decompress_gradients(
        self, compressed: Tensor, original_numel: int
    ) -> Tensor:
        """Decompress sparse gradient representation.

        Args:
            compressed: Compressed gradient tensor.
            original_numel: Original number of elements.

        Returns:
            Decompressed gradient tensor.

        """
        k = compressed.numel() // 2
        values = compressed[:k]
        indices = compressed[k:].long()

        decompressed = torch.zeros(original_numel, device=compressed.device)
        decompressed[indices] = values

        return decompressed

    def all_reduce_scalar(
        self,
        value: float | Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> float:
        """All-reduce a scalar value across processes.

        Args:
            value: Scalar value to reduce.
            op: Reduction operation.

        Returns:
            Reduced scalar value.

        """
        if not dist.is_initialized():
            return float(value)

        if isinstance(value, float):
            tensor = torch.tensor(value, device=self._get_device())
        else:
            tensor = value.clone()

        dist.all_reduce(tensor, op=op, group=self.process_group)

        return tensor.item()

    def broadcast_model(self, src: int = 0) -> None:
        """Broadcast model parameters from source rank.

        Args:
            src: Source rank for broadcasting.

        """
        if not dist.is_initialized():
            return

        for param in self.model.parameters():
            dist.broadcast(param.data, src=src, group=self.process_group)

        self._logger.debug("model_broadcasted", src_rank=src)

    def _get_device(self) -> torch.device:
        """Get the device for the model.

        Returns:
            Device where model resides.

        """
        return next(self.model.parameters()).device


@dataclass
class GradientAccumulator:
    """Helper for gradient accumulation with proper scaling.

    Attributes:
        accumulation_steps: Number of steps to accumulate.
        scale_loss: Whether to scale loss by accumulation steps.
        accumulated_loss: Running sum of losses.

    """

    accumulation_steps: int = 1
    scale_loss: bool = True
    accumulated_loss: float = field(default=0.0, init=False)
    _step_count: int = field(default=0, init=False)

    def scale(self, loss: Tensor) -> Tensor:
        """Scale loss for gradient accumulation.

        Args:
            loss: Original loss tensor.

        Returns:
            Scaled loss tensor.

        """
        if self.scale_loss and self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss

    def accumulate(self, loss: float) -> None:
        """Accumulate loss value.

        Args:
            loss: Loss value to accumulate.

        """
        self.accumulated_loss += loss
        self._step_count += 1

    def should_step(self) -> bool:
        """Check if optimizer step should occur.

        Returns:
            True if accumulation count reached.

        """
        return self._step_count >= self.accumulation_steps

    def get_average_loss(self) -> float:
        """Get average loss over accumulated steps.

        Returns:
            Average loss value.

        """
        if self._step_count == 0:
            return 0.0
        return self.accumulated_loss / self._step_count

    def reset(self) -> None:
        """Reset accumulation state."""
        self.accumulated_loss = 0.0
        self._step_count = 0


def create_gradient_synchronizer(
    model: nn.Module,
    config: DistributedInfraConfig,
) -> GradientSynchronizer:
    """Factory function to create gradient synchronizer.

    Args:
        model: Model to synchronize.
        config: Distributed configuration.

    Returns:
        Configured GradientSynchronizer instance.

    """
    return GradientSynchronizer(model=model, config=config)
