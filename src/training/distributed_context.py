"""Distributed training context utilities.

Provides a unified interface for distributed training that gracefully
handles both single-GPU and multi-GPU scenarios.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ContextManager

import structlog
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from config.schemas import DistributedConfig  # User-facing config from schemas

logger = structlog.get_logger(__name__)


@dataclass
class DistributedContext:
    """Context for distributed training.

    Encapsulates rank, world_size, and provides utility methods for
    distributed-aware operations.

    Attributes:
        rank: Global rank (0 for single-GPU)
        local_rank: Local rank within node (0 for single-GPU)
        world_size: Total number of processes (1 for single-GPU)
        is_distributed: Whether running in distributed mode
        device: Device for this process

    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    device: torch.device = torch.device("cpu")
    _process_group_initialized: bool = False

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    @classmethod
    def from_environment(
        cls,
        config: DistributedConfig | None = None,
    ) -> DistributedContext:
        """Create context from environment variables.

        Automatically detects distributed environment via WORLD_SIZE, RANK, LOCAL_RANK.
        Falls back to single-GPU if not in distributed environment.

        Args:
            config: Optional distributed config (for explicit settings)

        Returns:
            Configured DistributedContext

        """
        # Check environment for distributed setup
        # Only consider distributed if WORLD_SIZE env var is actually set
        env_world_size = os.environ.get("WORLD_SIZE")
        env_rank = os.environ.get("RANK")

        if env_world_size is not None and env_rank is not None:
            # Distributed environment is properly set up (via torchrun)
            world_size = int(env_world_size)
            rank = int(env_rank)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Not in distributed environment
            world_size = 1
            rank = 0
            local_rank = 0

            # Warn if config enables distributed but env isn't set up
            if config is not None and config.enabled and config.world_size > 1:
                logger.warning(
                    "distributed_config_enabled_but_env_not_set",
                    hint="Use torchrun to launch distributed training",
                )

        is_distributed = world_size > 1

        # Device selection
        if torch.cuda.is_available():
            if is_distributed:
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(local_rank)
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        ctx = cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=is_distributed,
            device=device,
        )

        logger.info(
            "distributed_context_created",
            rank=ctx.rank,
            local_rank=ctx.local_rank,
            world_size=ctx.world_size,
            is_distributed=ctx.is_distributed,
            device=str(ctx.device),
        )

        return ctx

    def initialize_process_group(
        self,
        backend: str = "nccl",
        timeout_seconds: int = 1800,
    ) -> None:
        """Initialize distributed process group if not already done.

        Args:
            backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
            timeout_seconds: Timeout for collective operations

        """
        if not self.is_distributed:
            return

        if dist.is_initialized():
            logger.debug("process_group_already_initialized")
            return

        # Use gloo for CPU, nccl for GPU
        if self.device.type == "cpu" and backend == "nccl":
            backend = "gloo"
            logger.info("switching_to_gloo_backend", reason="CPU device")

        import datetime

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        self._process_group_initialized = True
        logger.info("process_group_initialized", backend=backend)

    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if self._process_group_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._process_group_initialized = False
            logger.info("process_group_destroyed")

    def wrap_model(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
    ) -> torch.nn.Module:
        """Wrap model with DDP if in distributed mode.

        Args:
            model: Model to wrap (should already be on device)
            find_unused_parameters: Allow unused params in backward
            broadcast_buffers: Sync buffers each forward

        Returns:
            Original model (single-GPU) or DDP-wrapped model (distributed)

        """
        if not self.is_distributed:
            return model

        return DDP(
            model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            output_device=self.local_rank if self.device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers,
        )

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap model from DDP if wrapped.

        Args:
            model: Potentially DDP-wrapped model

        Returns:
            Unwrapped model

        """
        if isinstance(model, DDP):
            return model.module
        return model

    def no_sync(self, model: torch.nn.Module) -> ContextManager[None]:
        """Get no_sync context for gradient accumulation.

        Args:
            model: DDP-wrapped model

        Returns:
            Context manager that disables gradient sync

        """
        if isinstance(model, DDP) and hasattr(model, "no_sync"):
            return model.no_sync()
        return nullcontext()

    def barrier(self) -> None:
        """Synchronization barrier across all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def all_reduce_scalar(
        self,
        value: float,
        op: str = "sum",
    ) -> float:
        """All-reduce a scalar value across processes.

        Args:
            value: Scalar to reduce
            op: Reduction operation ('sum', 'mean', 'max', 'min')

        Returns:
            Reduced value

        """
        if not self.is_distributed:
            return value

        tensor = torch.tensor(value, device=self.device)

        reduce_op = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Divide after
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }.get(op, dist.ReduceOp.SUM)

        dist.all_reduce(tensor, op=reduce_op)

        if op == "mean":
            tensor = tensor / self.world_size

        return tensor.item()

    def broadcast_tensor(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast tensor from source rank.

        Args:
            tensor: Tensor to broadcast (modified in-place)
            src: Source rank

        Returns:
            Broadcasted tensor

        """
        if not self.is_distributed:
            return tensor

        dist.broadcast(tensor, src=src)
        return tensor

    def broadcast_object(
        self,
        obj: Any,
        src: int = 0,
    ) -> Any:
        """Broadcast arbitrary Python object from source rank.

        Args:
            obj: Object to broadcast (only used on src rank)
            src: Source rank

        Returns:
            Broadcasted object

        """
        if not self.is_distributed:
            return obj

        objects = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]

    def get_effective_batch_size(
        self,
        per_gpu_batch_size: int,
        gradient_accumulation_steps: int = 1,
    ) -> int:
        """Calculate effective global batch size.

        Args:
            per_gpu_batch_size: Batch size per GPU
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Effective global batch size

        """
        return per_gpu_batch_size * self.world_size * gradient_accumulation_steps
