"""Configuration schemas for distributed training.

This module defines Pydantic models for distributed training configuration,
ensuring type safety, validation, and serialization.

Design Principles:
    - No hardcoded values: All constants are configurable with sensible defaults
    - Backwards compatible: New fields have defaults, removed fields are deprecated
    - Validated: Pydantic enforces types and constraints at runtime
    - Serializable: All configs can be saved/loaded from YAML/JSON
"""

from __future__ import annotations

import math
import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DistributedBackend(str, Enum):
    """Backend options for distributed training."""

    NCCL = "nccl"  # NVIDIA Collective Communications Library (GPU)
    GLOO = "gloo"  # Facebook's Gloo (CPU/GPU)
    MPI = "mpi"  # Message Passing Interface


class DistributedInfraConfig(BaseModel):
    """Configuration for distributed training infrastructure.

    This is the full-featured infrastructure config for advanced multi-node
    setups. For basic single-node DDP training, use DistributedConfig from
    config/schemas.py instead.

    Supports multi-node, multi-GPU training with gradient synchronization
    via NCCL or Gloo backends.

    Attributes:
        enabled: Whether distributed training is enabled.
        world_size: Total number of processes across all nodes.
        backend: Communication backend (nccl for GPU, gloo for CPU).
        gradient_accumulation_steps: Steps before gradient sync.
        sync_batch_norm: Convert BatchNorm to SyncBatchNorm.
        find_unused_parameters: DDP parameter for unused params.
        broadcast_buffers: Whether to broadcast model buffers.
        bucket_cap_mb: Max bucket size for gradient reduction.
        gradient_as_bucket_view: Memory optimization for gradients.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Core settings
    enabled: bool = Field(
        default=False,
        description="Enable distributed training",
    )
    launcher: LauncherConfig | None = Field(
        default=None,
        description="Launcher configuration for distributed training",
    )
    learning_rate_scaling: Literal["none", "linear", "sqrt"] = Field(
        default="none",
        description="How to scale learning rate with world size",
    )
    gradient_compression_bits: int = Field(
        default=32,
        ge=1,
        le=32,
        description="Number of bits for gradient compression",
    )
    world_size: int = Field(
        default=1,
        ge=1,
        description="Total number of processes (typically = num_gpus)",
    )
    backend: DistributedBackend = Field(
        default=DistributedBackend.NCCL,
        description="Distributed backend (nccl for GPU, gloo for CPU/GPU)",
    )

    # Process group settings
    init_method: str = Field(
        default="env://",
        description="URL specifying how to initialize process group",
    )
    timeout_seconds: int = Field(
        default=1800,
        ge=60,
        description="Timeout for collective operations",
    )

    # Gradient synchronization
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of steps to accumulate gradients before sync",
    )
    gradient_compression: bool = Field(
        default=False,
        description="Enable gradient compression for bandwidth reduction",
    )
    all_reduce_algorithm: Literal["ring", "tree", "auto"] = Field(
        default="auto",
        description="Algorithm for all-reduce operations",
    )

    # DDP settings
    sync_batch_norm: bool = Field(
        default=True,
        description="Convert BatchNorm layers to SyncBatchNorm",
    )
    find_unused_parameters: bool = Field(
        default=False,
        description="Enable for models with unused parameters (slower)",
    )
    broadcast_buffers: bool = Field(
        default=True,
        description="Broadcast model buffers on each forward pass",
    )
    bucket_cap_mb: float = Field(
        default=25.0,
        gt=0,
        description="Maximum bucket size in MB for gradient reduction",
    )
    gradient_as_bucket_view: bool = Field(
        default=True,
        description="Use gradient tensor as bucket view (memory optimization)",
    )

    # Mixed precision distributed
    use_amp: bool = Field(
        default=True,
        description="Use automatic mixed precision (AMP)",
    )
    amp_dtype: Literal["float16", "bfloat16"] = Field(
        default="float16",
        description="Data type for mixed precision training",
    )

    # Checkpointing
    checkpoint_strategy: Literal["rank_0", "all_ranks", "fsdp"] = Field(
        default="rank_0",
        description="Which ranks save checkpoints",
    )
    save_on_rank_0_only: bool = Field(
        default=True,
        description="Only rank 0 saves model checkpoints",
    )

    # Performance tuning
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description="Number of batches to prefetch per worker",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer",
    )
    non_blocking: bool = Field(
        default=True,
        description="Use non-blocking data transfers",
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: DistributedBackend) -> DistributedBackend:
        """Warn if NCCL is selected but CUDA unavailable."""
        if v == DistributedBackend.NCCL:
            import torch

            if not torch.cuda.is_available():
                import warnings

                warnings.warn(
                    "NCCL backend selected but CUDA not available. "
                    "Consider using 'gloo' backend for CPU training.",
                    UserWarning,
                    stacklevel=2,
                )
        return v

    @model_validator(mode="after")
    def validate_config(self) -> DistributedInfraConfig:
        """Validate configuration consistency."""
        if not self.enabled:
            return self

        # Warn if gradient accumulation with small world size
        if self.gradient_accumulation_steps > 1 and self.world_size < 2:
            import warnings

            warnings.warn(
                f"Gradient accumulation ({self.gradient_accumulation_steps}) "
                f"with world_size={self.world_size} provides no distribution benefit.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def get_effective_batch_size(self, per_gpu_batch_size: int) -> int:
        """Calculate effective batch size across all processes.

        Args:
            per_gpu_batch_size: Batch size per GPU.

        Returns:
            Total effective batch size.

        """
        return per_gpu_batch_size * self.world_size * self.gradient_accumulation_steps

    def should_sync_at_step(self, step: int) -> bool:
        """Check if gradients should be synchronized at this step.

        Args:
            step: Current training step (1-indexed).

        Returns:
            True if gradients should be synced at this step.

        """
        return step % self.gradient_accumulation_steps == 0

    def should_save_checkpoint(self, rank: int) -> bool:
        """Check if this rank should save a checkpoint.

        Args:
            rank: Process rank.

        Returns:
            True if this rank should save checkpoints.

        """
        if self.checkpoint_strategy == "all_ranks":
            return True
        return rank == 0

    def requires_barrier_before_checkpoint(self) -> bool:
        """Check if a barrier is needed before checkpointing.

        Returns:
            True if a barrier synchronization is needed.

        """
        return self.enabled and self.world_size > 1

    def scale_learning_rate(self, base_lr: float) -> float:
        """Scale learning rate based on world size and scaling strategy.

        Args:
            base_lr: Base learning rate.

        Returns:
            Scaled learning rate.

        """
        if self.learning_rate_scaling == "linear":
            return base_lr * self.world_size
        if self.learning_rate_scaling == "sqrt":
            return base_lr * math.sqrt(self.world_size)
        return base_lr


class LauncherConfig(BaseModel):
    """Configuration for distributed training launcher.

    Supports various launch methods including torchrun, mpirun, and custom.

    Attributes:
        method: Launch method (torchrun, slurm, custom).
        nnodes: Number of nodes in the cluster.
        nproc_per_node: Processes per node (typically = GPUs per node).
        master_addr: Address of the master node.
        master_port: Port for master node communication.
        node_rank: Rank of the current node.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Launch method
    method: Literal["torchrun", "slurm", "custom"] = Field(
        default="torchrun",
        description="Method for launching distributed processes",
    )

    # Node configuration
    nnodes: int = Field(
        default=1,
        ge=1,
        description="Number of nodes",
    )
    nproc_per_node: int = Field(
        default=1,
        ge=1,
        description="Number of processes per node (typically = GPUs)",
    )

    # Master node settings (auto-detected from environment if not set)
    master_addr: str = Field(
        default="localhost",
        description="Master node address",
    )
    master_port: int = Field(
        default=29500,
        ge=1024,
        le=65535,
        description="Master node port",
    )
    node_rank: int = Field(
        default=0,
        ge=0,
        description="Rank of current node (0-indexed)",
    )

    # Resource allocation
    rdzv_backend: Literal["static", "c10d", "etcd", "etcd-v2"] = Field(
        default="c10d",
        description="Rendezvous backend for elastic training",
    )
    rdzv_endpoint: str | None = Field(
        default=None,
        description="Rendezvous endpoint (auto-set if None)",
    )

    # Fault tolerance
    max_restarts: int = Field(
        default=0,
        ge=0,
        description="Maximum number of worker restarts (0 = no restart)",
    )
    monitor_interval: float = Field(
        default=5.0,
        gt=0,
        description="Interval for health monitoring in seconds",
    )

    @model_validator(mode="after")
    def validate_and_populate(self) -> LauncherConfig:
        """Validate and populate from environment variables."""
        # Auto-detect from SLURM environment
        if self.method == "slurm":
            if "SLURM_NNODES" in os.environ:
                self.nnodes = int(os.environ["SLURM_NNODES"])
            if "SLURM_NTASKS_PER_NODE" in os.environ:
                self.nproc_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
            if "SLURM_NODEID" in os.environ:
                self.node_rank = int(os.environ["SLURM_NODEID"])
            if "MASTER_ADDR" in os.environ:
                self.master_addr = os.environ["MASTER_ADDR"]
            if "MASTER_PORT" in os.environ:
                self.master_port = int(os.environ["MASTER_PORT"])

        # Set rendezvous endpoint if not provided
        if self.rdzv_endpoint is None:
            self.rdzv_endpoint = f"{self.master_addr}:{self.master_port}"

        return self

    def get_world_size(self) -> int:
        """Calculate total world size.

        Returns:
            Total number of processes across all nodes.

        """
        return self.nnodes * self.nproc_per_node

    def get_local_rank(self, global_rank: int) -> int:
        """Get local rank from global rank.

        Args:
            global_rank: Global process rank.

        Returns:
            Local rank within the node.

        """
        return global_rank % self.nproc_per_node

    def get_node_rank(self, global_rank: int) -> int:
        """Get node rank from global rank.

        Args:
            global_rank: Global process rank.

        Returns:
            Node rank (0-indexed).

        """
        return global_rank // self.nproc_per_node


class SelfPlayDistributedConfig(BaseModel):
    """Configuration for distributed self-play generation.

    Coordinates self-play workers across multiple nodes with
    efficient experience sharing.

    Attributes:
        workers_per_node: Number of self-play workers per node.
        games_per_worker: Games each worker generates per iteration.
        experience_sharing: How experiences are shared across nodes.
        buffer_sync_interval: Steps between buffer synchronization.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Worker configuration
    num_workers: int = Field(
        default=2,
        ge=1,
        description="Total number of self-play workers",
    )
    workers_per_node: int = Field(
        default=2,
        ge=1,
        description="Number of self-play workers per node",
    )
    games_per_worker: int = Field(
        default=50,
        ge=1,
        description="Games generated per worker per iteration",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for self-play",
    )

    # Experience sharing
    experience_sharing: Literal["local", "global", "hierarchical"] = Field(
        default="global",
        description=(
            "local: No sharing, "
            "global: All experiences shared, "
            "hierarchical: Intra-node then inter-node"
        ),
    )
    buffer_sync_interval: int = Field(
        default=100,
        ge=1,
        description="Training steps between experience buffer syncs",
    )

    # Model update propagation
    model_broadcast_interval: int = Field(
        default=100,
        ge=1,
        description="Steps between broadcasting updated model to workers",
    )
    async_model_updates: bool = Field(
        default=True,
        description="Allow workers to use slightly stale models",
    )
    max_model_staleness_steps: int = Field(
        default=500,
        ge=0,
        description="Maximum allowed model staleness (0 = sync always)",
    )

    # Resource management
    cpu_workers: bool = Field(
        default=True,
        description="Run self-play workers on CPU (GPU for training only)",
    )
    inference_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for worker inference",
    )

    @property
    def total_games(self) -> int:
        """Total games across all workers."""
        return self.num_workers * self.games_per_worker

    def get_games_for_worker(self, worker_id: int) -> tuple[int, int]:
        """Get game index range for a specific worker.

        Args:
            worker_id: Worker ID (0-indexed).

        Returns:
            Tuple of (start_index, end_index) for this worker's games.

        """
        start = worker_id * self.games_per_worker
        end = start + self.games_per_worker
        return (start, end)


def create_distributed_config(
    world_size: int = 1,
    backend: str = "nccl",
    **kwargs: Any,
) -> DistributedInfraConfig:
    """Factory function to create distributed config.

    Args:
        world_size: Number of processes.
        backend: Communication backend.
        **kwargs: Additional configuration options.

    Returns:
        Configured DistributedInfraConfig instance.

    """
    backend_enum = DistributedBackend(backend)

    # Derive world_size from launcher if provided and not explicitly set
    launcher = kwargs.get("launcher")
    if launcher is not None and world_size == 1:
        world_size = launcher.get_world_size()

    # Allow caller to override enabled; default to world_size > 1
    if "enabled" not in kwargs:
        kwargs["enabled"] = world_size > 1

    return DistributedInfraConfig(
        world_size=world_size,
        backend=backend_enum,
        **kwargs,
    )


def from_environment() -> DistributedInfraConfig | None:
    """Create distributed config from environment variables.

    Returns:
        DistributedInfraConfig if distributed env is detected, None otherwise.

    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1 and "RANK" not in os.environ:
        return None
    backend_str = os.environ.get("DISTRIBUTED_BACKEND", "nccl")
    try:
        backend = DistributedBackend(backend_str)
    except ValueError:
        backend = DistributedBackend.GLOO
    return DistributedInfraConfig(
        enabled=world_size > 1,
        world_size=world_size,
        backend=backend,
    )


# Backward-compatible alias for tests and external usage
# DistributedInfraConfig is the full-featured class; this alias maintains API compatibility
DistributedConfig = DistributedInfraConfig
