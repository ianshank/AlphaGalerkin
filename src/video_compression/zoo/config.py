"""Configuration schemas for the video-codec model zoo (Phase 2).

Every measurement-affecting knob is a Pydantic field with a validated
default. There are no hardcoded numbers in zoo training, sweep, or
validation code paths — they all flow through these schemas.

Schema versioning is explicit so older manifest JSON documents remain
loadable as new fields are added (forward-compat: ``extra="ignore"``,
unversioned-to-v1 migration in :mod:`src.video_compression.zoo.manifest`).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from src.templates.config import BaseModuleConfig

# Schema-version constants are surfaced (not magic literals) so the
# migrator and consumers compare against a single source of truth.
PERF_ZOO_MANIFEST_SCHEMA_VERSION: int = 1
PERF_ZOO_ENTRY_SCHEMA_VERSION: int = 1


class StorageBackend(str, Enum):
    """Where checkpoint payloads live."""

    FILESYSTEM = "filesystem"
    GCS = "gcs"


class DeviceAssignmentStrategy(str, Enum):
    """How zoo entries are assigned to physical accelerators.

    - ``manual``: every entry must declare its own ``device`` field; no
      auto-assignment is performed.
    - ``round_robin``: entries are dealt across visible CUDA devices in
      manifest order. Cheapest to reason about; ignores VRAM differences.
    - ``vram_aware`` (default): entries are sorted by their declared VRAM
      requirement (descending) and packed onto the device with the most
      free VRAM at assignment time. Matches the dual-GPU reference rig
      (RTX 5060 Ti 16 GB at ``cuda:0`` + RTX 5060 8 GB at ``cuda:1``).
    - ``single_device``: every entry uses the run-level
      ``device_preference``. Useful for CPU smoke tests.
    """

    MANUAL = "manual"
    ROUND_ROBIN = "round_robin"
    VRAM_AWARE = "vram_aware"
    SINGLE_DEVICE = "single_device"


class EntryStatus(str, Enum):
    """Lifecycle status of an entry in the manifest."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizerConfig(BaseModuleConfig):
    """Optimizer hyperparameters for a single zoo entry."""

    optimizer_type: Literal["adamw", "adam", "sgd"] = Field(
        default="adamw",
        description="Optimizer family.",
    )
    learning_rate: float = Field(
        default=1.0e-4,
        gt=0.0,
        le=1.0,
        description="Initial learning rate.",
    )
    weight_decay: float = Field(
        default=1.0e-5,
        ge=0.0,
        le=1.0,
        description="L2 weight decay coefficient.",
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Adam(W) momentum betas.",
    )
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="SGD momentum (only used when optimizer_type='sgd').",
    )

    @model_validator(mode="after")
    def _validate_betas(self) -> OptimizerConfig:
        b1, b2 = self.betas
        if not (0.0 < b1 < 1.0 and 0.0 < b2 < 1.0):
            raise ValueError(f"betas must be in (0, 1); got {self.betas!r}")
        return self


class SchedulerConfig(BaseModuleConfig):
    """LR scheduler hyperparameters for a single zoo entry."""

    scheduler_type: Literal["cosine", "linear", "constant", "warmup_cosine"] = Field(
        default="warmup_cosine",
        description="Schedule shape.",
    )
    warmup_steps: int = Field(
        default=500,
        ge=0,
        le=1_000_000,
        description="Linear-warmup step count (0 disables warmup).",
    )
    min_lr_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Floor for cosine schedule, as a ratio of base LR.",
    )


class ModelZooEntryConfig(BaseModuleConfig):
    """Single entry in the lambda grid.

    Each entry trains independently. The set of entries forms the
    rate-distortion curve published by the zoo.

    The ``device`` field is **optional**: when ``None`` (default), the
    sweep planner assigns a device per the manifest's
    :class:`DeviceAssignmentStrategy`. Setting ``device`` explicitly
    forces an entry onto that accelerator (useful for reproducing a
    previously-published checkpoint).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = Field(
        default=PERF_ZOO_ENTRY_SCHEMA_VERSION,
        ge=1,
        description="Entry schema version for migration.",
    )

    # ``name`` is inherited as required from BaseModuleConfig; we relax
    # it to a placeholder so the entry_id can be promoted to name in
    # the post-validator below.
    name: str = Field(
        default="entry",
        min_length=1,
        description="Inherited identifier; auto-set from entry_id when default.",
    )

    # Identity
    entry_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Unique identifier within the manifest.",
    )

    # Rate-distortion point
    lambda_rd: float = Field(
        ...,
        gt=0.0,
        le=10.0,
        description=(
            "Rate-distortion tradeoff (higher = more compression). The "
            "set of lambda_rd values across the manifest defines the R-D "
            "curve."
        ),
    )
    distortion_metric: Literal["mse", "ms_ssim", "mixed"] = Field(
        default="mixed",
        description="Distortion metric (matches video_compression.training.loss).",
    )
    ms_ssim_weight: float = Field(
        default=0.84,
        ge=0.0,
        le=1.0,
        description="MS-SSIM weight for 'mixed' distortion.",
    )

    # R-D acceptance targets (Phase F)
    target_bpp: float = Field(
        ...,
        gt=0.0,
        le=24.0,
        description="Target bits-per-pixel after training.",
    )
    target_psnr_db: float = Field(
        ...,
        gt=0.0,
        le=80.0,
        description="Target PSNR in dB after training.",
    )
    bpp_tolerance: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="Allowed |observed - target| / target for bpp.",
    )
    psnr_tolerance_db: float = Field(
        default=1.0,
        gt=0.0,
        le=20.0,
        description="Allowed |observed - target| in dB for PSNR.",
    )

    # Training schedule
    train_steps: int = Field(
        ...,
        ge=1,
        le=100_000_000,
        description="Number of optimizer steps for this entry.",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=4096,
        description="Per-step batch size.",
    )
    seed: int = Field(
        default=20260501,
        ge=0,
        description="Deterministic seed for this entry.",
    )

    # Optimization
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(name="optimizer"),
        description="Optimizer hyperparameters.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=lambda: SchedulerConfig(name="scheduler"),
        description="LR scheduler hyperparameters.",
    )
    grad_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Max gradient L2 norm.",
    )
    use_amp: bool = Field(
        default=True,
        description="Enable AMP mixed-precision (auto-disabled on CPU).",
    )

    # Resource hints (used by vram_aware planner)
    estimated_vram_mib: float = Field(
        default=4096.0,
        gt=0.0,
        le=131_072.0,
        description=(
            "Estimated peak VRAM for this entry in MiB. Used by "
            "DeviceAssignmentStrategy.VRAM_AWARE to pack entries onto "
            "the largest-headroom GPU first."
        ),
    )

    # Pinning / warm-start
    device: str | None = Field(
        default=None,
        description=(
            "Optional explicit device override ('cuda:0', 'cuda:1', "
            "'cpu', ...). When set, the planner skips this entry."
        ),
    )
    parent_entry_id: str | None = Field(
        default=None,
        description=(
            "If set, warm-start from this entry's checkpoint before "
            "training. Cuts wall-clock for adjacent lambda points."
        ),
    )

    # External references (no payloads inlined)
    codec_config_ref: str | None = Field(
        default=None,
        description=(
            "Path to the YAML CodecConfig used for this entry. None "
            "means the manifest-level default applies."
        ),
    )
    train_dataset_ref: str | None = Field(
        default=None,
        description=(
            "Path / URI of the training dataset. None means the "
            "manifest-level default applies. Synthetic data is used "
            "when both are None (CI smoke)."
        ),
    )

    @model_validator(mode="after")
    def _validate_warmup_against_steps(self) -> ModelZooEntryConfig:
        if self.scheduler.warmup_steps > self.train_steps:
            raise ValueError(
                f"scheduler.warmup_steps ({self.scheduler.warmup_steps}) "
                f"must be <= train_steps ({self.train_steps}) for entry "
                f"{self.entry_id!r}",
            )
        # Promote entry_id to ``name`` when the user didn't override it.
        if self.name == "entry":
            object.__setattr__(self, "name", self.entry_id)
        return self


class ModelZooManifestConfig(BaseModuleConfig):
    """Full manifest covering N rate-distortion entries.

    A manifest is a versioned JSON document persisted via
    :mod:`src.video_compression.zoo.manifest`. The Phase 0 pattern
    (``BaselineDocument``) is mirrored: forward-compat ``extra="ignore"``
    and a single-anchor schema version with a documented migration table.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    schema_version: int = Field(
        default=PERF_ZOO_MANIFEST_SCHEMA_VERSION,
        ge=1,
        description="Manifest schema version for migration.",
    )

    # Storage
    storage_backend: StorageBackend = Field(
        default=StorageBackend.FILESYSTEM,
        description="Where checkpoint payloads live.",
    )
    storage_root: str = Field(
        ...,
        min_length=1,
        description=(
            "Root path/URI for checkpoint storage. Filesystem: a local "
            "directory. GCS: a 'gs://bucket/prefix' URI."
        ),
    )

    # Entries
    entries: list[ModelZooEntryConfig] = Field(
        ...,
        min_length=1,
        description="Lambda grid points to train.",
    )

    # Defaults inherited by entries that do not override them
    default_codec_config_ref: str | None = Field(
        default=None,
        description="Default YAML CodecConfig path for entries with no override.",
    )
    default_train_dataset_ref: str | None = Field(
        default=None,
        description="Default training dataset path/URI for entries with no override.",
    )

    # Sweep behavior
    device_assignment_strategy: DeviceAssignmentStrategy = Field(
        default=DeviceAssignmentStrategy.VRAM_AWARE,
        description="How entries are mapped to physical devices.",
    )
    device_preference: str = Field(
        default="cuda",
        description=(
            "Run-level device preference. Accepts 'cuda', 'cuda:N', "
            "'cpu', or 'auto'. Used as fallback / for SINGLE_DEVICE."
        ),
    )
    parallel_workers_per_device: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "How many entries may train concurrently on a single GPU. "
            "Default 1 (one entry per card). >1 only safe for small "
            "models."
        ),
    )
    fail_fast: bool = Field(
        default=False,
        description=(
            "If True, abort the sweep on first entry failure. If False "
            "(default), record failures in the manifest and continue."
        ),
    )

    @model_validator(mode="after")
    def _validate_entry_ids(self) -> ModelZooManifestConfig:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.entry_id in seen:
                raise ValueError(
                    f"duplicate entry_id {entry.entry_id!r} in manifest",
                )
            seen.add(entry.entry_id)
        # Check parent_entry_id references resolve.
        for entry in self.entries:
            if entry.parent_entry_id is not None and entry.parent_entry_id not in seen:
                raise ValueError(
                    f"entry {entry.entry_id!r} declares parent_entry_id="
                    f"{entry.parent_entry_id!r} which is not present in the "
                    f"manifest",
                )
        return self
