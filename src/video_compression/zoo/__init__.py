"""Pretrained model zoo for the AlphaGalerkin video codec.

Phase 2 of the self-hosted transcoder roadmap (see ``docs/NEXT_STEPS_PLAN.md``
Milestone 11). The zoo trains and ships one checkpoint per declared
rate-distortion lambda point, then exposes each checkpoint as a selectable
``RuntimeProfile`` in the Phase 0 perf benchmark harness.

Public surface:

- :class:`ModelZooEntryConfig` — one entry in the lambda grid.
- :class:`ModelZooManifestConfig` — full manifest with schema-versioned
  forward-compat migration.
- :class:`DeviceAssignmentStrategy` — how zoo entries are mapped to GPUs.
- :class:`VideoCodecZoo` — filesystem-backed checkpoint and artifact
  registry for codec zoo entries. Intentionally orthogonal to
  :class:`src.distributed.model_zoo.ModelZoo`, which carries AlphaZero
  curriculum semantics that do not apply to a static R-D grid.
- :func:`scan_devices` / :func:`assign_devices` — dual-GPU planner.
"""

from __future__ import annotations

from src.video_compression.zoo.config import (
    PERF_ZOO_MANIFEST_SCHEMA_VERSION,
    DeviceAssignmentStrategy,
    EntryStatus,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
    OptimizerConfig,
    SchedulerConfig,
    StorageBackend,
)
from src.video_compression.zoo.device_planner import (
    DeviceCapability,
    DevicePlan,
    EntryAssignment,
    assign_devices,
    scan_devices,
)
from src.video_compression.zoo.manifest import (
    ManifestMigrationError,
    load_manifest,
    save_manifest,
)
from src.video_compression.zoo.storage import (
    EntryArtifacts,
    VideoCodecZoo,
)

__all__ = [
    "PERF_ZOO_MANIFEST_SCHEMA_VERSION",
    "DeviceAssignmentStrategy",
    "DeviceCapability",
    "DevicePlan",
    "EntryArtifacts",
    "EntryAssignment",
    "EntryStatus",
    "ManifestMigrationError",
    "ModelZooEntryConfig",
    "ModelZooManifestConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "StorageBackend",
    "VideoCodecZoo",
    "assign_devices",
    "load_manifest",
    "save_manifest",
    "scan_devices",
]
