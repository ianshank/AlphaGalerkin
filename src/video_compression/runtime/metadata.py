"""Persisted metadata for compiled / exported decoder artifacts.

This schema is the *persistence boundary* for runtime-package state.
Anything that lands on disk (compiled-graph cache entries, ONNX
sidecar JSON, TensorRT engine manifests) goes through here.

Forward-compat rules carried over from Phase 0:

* Override ``BaseModuleConfig``'s ``extra="forbid"`` default with
  ``extra="ignore"`` so future fields don't break old binaries.
* Surface schema version as a module-level constant, not a literal,
  so callers and migrators compare against the same number.
* Provide a ``_migrate_compiled_artifact_metadata`` hook with an
  explicit migration table. Unversioned (legacy) input is migrated
  to v1 on read.
* Provide a ``classmethod from_dict`` so call sites use a public API
  (the lesson from PR #75: ``BenchmarkReport.from_dict`` over an
  imported private function).
"""

from __future__ import annotations

import copy
from typing import Any

import structlog
from pydantic import ConfigDict, Field

from src.templates.config import BaseModuleConfig

# Schema version surfaced as a constant. Bump on every schema change
# and add a migration row below.
COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION: int = 1

logger = structlog.get_logger(__name__)


class CompiledArtifactMetadata(BaseModuleConfig):
    """Provenance + perf data for a runtime-built artifact.

    A "runtime artifact" is a backend-specific cached object: a
    ``torch.compile`` AOT graph, an ONNX file, a TensorRT engine, or
    just a wrapper around the eager PyTorch model. Every runtime
    populates one of these so reports / caches / equivalence checks
    have a uniform shape to read.

    Persisted to disk as JSON sidecar next to the artifact itself.
    """

    # Persisted artifact: must tolerate unknown fields written by a
    # future binary. Phase 0 lesson — BaseModuleConfig is strict by
    # default; we override here. ``protected_namespaces=()`` clears
    # the ``model_*`` prefix warning so ``model_hash`` doesn't trip
    # Pydantic's reserved-namespace check (the field genuinely names
    # the codec model state hash, not a Pydantic accessor).
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    schema_version: int = Field(
        default=COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION,
        ge=1,
        description="Schema version for migration.",
    )

    # Identity
    runtime_name: str = Field(
        ...,
        min_length=1,
        description="Stable runtime identifier, e.g. 'pytorch-eager'.",
    )
    backend: str = Field(
        ...,
        min_length=1,
        description="Backend tag mirroring RuntimeBackend enum value.",
    )
    precision: str = Field(
        ...,
        min_length=1,
        description="Precision tag mirroring Precision enum value.",
    )

    # Provenance
    model_hash: str = Field(
        ...,
        min_length=1,
        description="Hash of the model state used to build this artifact.",
    )
    device_label: str = Field(
        ...,
        min_length=1,
        description=(
            "Device label as produced by "
            "``src/video_compression/perf/device.py::device_label``. "
            "Includes hardware name for cuda devices."
        ),
    )

    # Input shape committed to (some backends are shape-locked)
    batch_size: int = Field(..., ge=1)
    latent_channels: int = Field(..., ge=1)
    latent_height: int = Field(..., ge=1)
    latent_width: int = Field(..., ge=1)

    # Build-time observations
    build_time_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock seconds spent building this artifact. ``0.0`` "
            "for runtimes (e.g. eager) that have no build step."
        ),
    )
    artifact_path: str | None = Field(
        default=None,
        description=(
            "Filesystem path to the persisted artifact, if any. ``None`` "
            "for in-memory-only runtimes (e.g. eager)."
        ),
    )
    artifact_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Size of the on-disk artifact, if any.",
    )

    # Free-form key/value tags so individual runtimes can attach
    # backend-specific provenance (compile_mode, ONNX opset, TRT
    # workspace size) without us touching this schema for every new
    # field. Read by humans; never gates behaviour.
    extra_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Backend-specific provenance tags.",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompiledArtifactMetadata:
        """Rehydrate from a JSON dict, applying schema migrations."""
        migrated = _migrate_compiled_artifact_metadata(data)
        return cls.model_validate(migrated)


def _migrate_compiled_artifact_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate a raw metadata dict to the current schema version.

    Migration table:

    +----------------+----------------+--------------------------------+
    | from           | to             | change                         |
    +================+================+================================+
    | (unversioned)  | 1              | add ``schema_version`` field   |
    +----------------+----------------+--------------------------------+

    New schemas are added by appending here; old code remains able to
    load every metadata file ever written.
    """
    # Deep copy so future migrations that mutate nested values
    # (e.g. extra_tags, or a nested dict added in a v2 schema) do
    # not leak side effects back into the caller's dict.
    raw = copy.deepcopy(raw)

    schema_version = raw.get("schema_version")
    if schema_version is None:
        logger.info(
            "compiled_artifact_metadata.migration.unversioned_to_v1",
            keys=sorted(raw.keys()),
        )
        raw["schema_version"] = 1
        schema_version = 1

    if schema_version > COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"compiled artifact metadata schema_version={schema_version} "
            f"is newer than this binary "
            f"({COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION}); upgrade the "
            f"package or pin a compatible artifact.",
        )
    return raw
