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

import os
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


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

    # Gradient compression precision
    gradient_compression_bits: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Bit precision for gradient compression",
    )

    # Learning rate scaling strategy
    learning_rate_scaling: Literal["linear", "sqrt", "none"] = Field(
        default="none",
        description=(
            "How to scale LR with world_size: linear=lr*ws, sqrt=lr*sqrt(ws), none=no scaling"
        ),
    )

    # Launcher config reference
    launcher: LauncherConfig | None = Field(
        default=None,
        description="Launcher configuration for deriving world_size etc.",
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

    # Per-rank batch size override.  Either an int (uniform per rank,
    # equivalent to the legacy behaviour where every rank gets the
    # caller-supplied per-GPU batch size) or a list of ints — one entry
    # per rank — for asymmetric multi-GPU rigs where the cards have
    # different VRAM (e.g. the user's RTX 5060 Ti + 5060 setup, where
    # the smaller card needs a smaller batch to avoid OOM).  When ``None``
    # the existing :meth:`get_effective_batch_size` helper is used as
    # before; setting a list activates the cross-field validator that
    # checks ``len(list) == world_size``.
    per_rank_batch_size: int | list[int] | None = Field(
        default=None,
        description=(
            "Optional per-rank batch size override.  None = uniform "
            "(legacy behaviour).  int = uniform but explicit.  list[int] "
            "= per-rank values; len must equal world_size."
        ),
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
    def validate_per_rank_batch_size(self) -> DistributedInfraConfig:
        """Cross-field validation for per-rank batch override.

        When ``per_rank_batch_size`` is a list, its length must match
        ``world_size``.  Each entry must be strictly positive.  An int
        override or ``None`` is always accepted.
        """
        prbs = self.per_rank_batch_size
        if isinstance(prbs, list):
            if len(prbs) != self.world_size:
                msg = (
                    f"per_rank_batch_size has {len(prbs)} entries but "
                    f"world_size={self.world_size}.  Provide one entry per rank."
                )
                raise ValueError(msg)
            for i, b in enumerate(prbs):
                if b <= 0:
                    msg = f"per_rank_batch_size[{i}]={b} must be strictly " f"positive."
                    raise ValueError(msg)
        elif isinstance(prbs, int) and prbs <= 0:
            msg = f"per_rank_batch_size={prbs} must be strictly positive."
            raise ValueError(msg)
        return self

    def get_rank_batch_size(self, rank: int, default: int) -> int:
        """Resolve the batch size for ``rank``.

        ``per_rank_batch_size`` takes precedence over ``default`` when
        set.  This is the canonical way to read the rank's batch size
        from the trainer.

        Args:
            rank: Rank id in ``[0, world_size)``.
            default: Fallback batch size when ``per_rank_batch_size`` is None.

        Returns:
            The batch size to use on this rank.

        """
        if not 0 <= rank < self.world_size:
            msg = f"rank={rank} out of range for world_size={self.world_size}"
            raise ValueError(msg)
        prbs = self.per_rank_batch_size
        if prbs is None:
            return default
        if isinstance(prbs, int):
            return prbs
        return prbs[rank]

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
        """Return True if gradients should be synced at this step.

        Args:
            step: Current training step (1-indexed).

        Returns:
            True if step is a multiple of gradient_accumulation_steps.

        """
        return step % self.gradient_accumulation_steps == 0

    def should_save_checkpoint(self, rank: int) -> bool:
        """Return True if this rank should save a checkpoint.

        Args:
            rank: Current process rank.

        Returns:
            True if rank 0 (when save_on_rank_0_only) or always True.

        """
        if self.save_on_rank_0_only:
            return rank == 0
        return True

    def requires_barrier_before_checkpoint(self) -> bool:
        """Return True if all ranks must sync before checkpointing.

        Returns:
            True when distributed training is enabled.

        """
        return self.enabled

    def scale_learning_rate(self, base_lr: float) -> float:
        """Scale learning rate based on world_size.

        Args:
            base_lr: Base learning rate for single-process training.

        Returns:
            Scaled learning rate.

        """
        import math

        if self.learning_rate_scaling == "linear":
            return base_lr * self.world_size
        if self.learning_rate_scaling == "sqrt":
            return base_lr * math.sqrt(self.world_size)
        return base_lr

    @model_validator(mode="after")
    def sync_world_size_from_launcher(self) -> DistributedInfraConfig:
        """Derive world_size from launcher if provided and world_size is default."""
        if self.launcher is not None and self.world_size == 1:
            # Derive world_size from launcher topology
            self.world_size = self.launcher.get_world_size()
        return self


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
            Node rank (which node this process is on).

        """
        return global_rank // self.nproc_per_node


class SelfPlayDistributedConfig(BaseModel):
    """Configuration for distributed self-play generation.

    Coordinates self-play workers across multiple nodes with
    efficient experience sharing.

    Attributes:
        num_workers: Number of self-play workers per node.
        games_per_worker: Games each worker generates per iteration.
        experience_sharing: How experiences are shared across nodes.
        buffer_sync_interval: Steps between buffer synchronization.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    # Worker configuration
    num_workers: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices("num_workers", "workers_per_node"),
        description="Number of self-play workers per node",
    )
    games_per_worker: int = Field(
        default=50,
        ge=1,
        description="Games generated per worker per iteration",
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
    batch_size: int = Field(
        default=32,
        ge=1,
        validation_alias=AliasChoices("batch_size", "inference_batch_size"),
        description="Batch size for worker inference",
    )

    @property
    def workers_per_node(self) -> int:
        """Backward-compatible alias for num_workers."""
        return self.num_workers

    @property
    def inference_batch_size(self) -> int:
        """Backward-compatible alias for batch_size."""
        return self.batch_size

    @property
    def total_games(self) -> int:
        """Total games across all workers per iteration.

        Returns:
            num_workers * games_per_worker.

        """
        return self.num_workers * self.games_per_worker

    def get_games_for_worker(self, worker_id: int) -> tuple[int, int]:
        """Get game range for a specific worker.

        Args:
            worker_id: Worker index (0-indexed, < num_workers).

        Returns:
            Tuple (start, end) representing range(start, end).

        """
        start = worker_id * self.games_per_worker
        end = start + self.games_per_worker
        return start, end


def create_distributed_config(
    world_size: int = 1,
    backend: str = "nccl",
    enabled: bool | None = None,
    launcher: LauncherConfig | None = None,
    **kwargs: Any,
) -> DistributedInfraConfig:
    """Factory function to create distributed config.

    Args:
        world_size: Number of processes (ignored when launcher provided).
        backend: Communication backend.
        enabled: Override whether distributed is enabled. If None, derived
            from world_size (enabled when > 1).
        launcher: Optional launcher config; world_size is derived from it.
        **kwargs: Additional configuration options passed to DistributedInfraConfig.

    Returns:
        Configured DistributedInfraConfig instance.

    """
    backend_enum = DistributedBackend(backend)

    # Derive world_size from launcher if provided
    effective_world_size = world_size
    if launcher is not None:
        effective_world_size = launcher.get_world_size()

    # Derive enabled flag
    if enabled is None:
        effective_enabled = effective_world_size > 1
    else:
        effective_enabled = enabled

    return DistributedInfraConfig(
        enabled=effective_enabled,
        world_size=effective_world_size,
        backend=backend_enum,
        launcher=launcher,
        **kwargs,
    )


def _get_env_rank_info() -> tuple[int, int, int]:
    """Extract rank info from environment variables.

    Internal helper for backward-compatible usage by trainer and worker.

    Returns:
        Tuple of (rank, local_rank, world_size).

    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def config_from_environment() -> DistributedInfraConfig | None:
    """Build :class:`DistributedInfraConfig` from environment variables.

    Returns:
        ``DistributedInfraConfig`` if running in distributed mode
        (``WORLD_SIZE`` > 1), or ``None`` if single-process.

    """
    _rank, _local_rank, world_size = _get_env_rank_info()
    if world_size <= 1:
        return None
    return DistributedInfraConfig(
        enabled=True,
        world_size=world_size,
    )


def from_environment() -> tuple[int, int, int]:
    """Return ``(rank, local_rank, world_size)`` from environment variables.

    .. deprecated::
        This function originally returned a 3-tuple. PR #53 briefly
        repurposed it to return :class:`DistributedInfraConfig`; that
        renamed helper is now :func:`config_from_environment`. This
        function preserves the historical tuple contract for any
        downstream callers that imported it directly.
    """
    return _get_env_rank_info()


# Rebuild DistributedInfraConfig to resolve forward references to LauncherConfig
DistributedInfraConfig.model_rebuild()

# Backward-compatible alias for tests and external usage
# DistributedInfraConfig is the full-featured class; this alias maintains API compatibility
DistributedConfig = DistributedInfraConfig
