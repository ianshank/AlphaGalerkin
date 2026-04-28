"""Tests for the persisted CompiledArtifactMetadata schema.

Mirrors the Phase 0 forward-compat / migration test pattern from
``tests/video_compression/perf/test_baseline.py``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.video_compression.runtime.metadata import (
    COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION,
    CompiledArtifactMetadata,
    _migrate_compiled_artifact_metadata,
)


def _baseline_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "meta",
        "runtime_name": "pytorch-eager",
        "backend": "pytorch",
        "precision": "float32",
        "model_hash": "abc123",
        "device_label": "cpu",
        "batch_size": 1,
        "latent_channels": 32,
        "latent_height": 4,
        "latent_width": 4,
    }
    base.update(overrides)
    return base


class TestConstruction:
    def test_minimal(self) -> None:
        meta = CompiledArtifactMetadata(**_baseline_kwargs())
        assert meta.schema_version == COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION
        assert meta.build_time_s == 0.0
        assert meta.artifact_path is None
        assert meta.artifact_size_bytes is None
        assert meta.extra_tags == {}

    def test_extra_tags_round_trip(self) -> None:
        meta = CompiledArtifactMetadata(
            **_baseline_kwargs(extra_tags={"compile_mode": "max-autotune"}),
        )
        assert meta.extra_tags == {"compile_mode": "max-autotune"}

    @pytest.mark.parametrize(
        "field",
        ["runtime_name", "backend", "precision", "model_hash", "device_label"],
    )
    def test_required_string_fields(self, field: str) -> None:
        kwargs = _baseline_kwargs(**{field: ""})
        with pytest.raises(ValidationError):
            CompiledArtifactMetadata(**kwargs)


class TestForwardCompatibility:
    def test_unknown_field_silently_dropped(self) -> None:
        # Persisted schema must tolerate fields written by a future
        # binary. ``extra="ignore"`` keeps the load path working.
        kwargs = _baseline_kwargs(future_only_field="should be dropped")
        meta = CompiledArtifactMetadata(**kwargs)
        assert not hasattr(meta, "future_only_field")

    def test_round_trip_through_json(self) -> None:
        meta = CompiledArtifactMetadata(**_baseline_kwargs())
        as_json = json.dumps(meta.model_dump(mode="json"))
        rehydrated = CompiledArtifactMetadata.from_dict(json.loads(as_json))
        assert rehydrated.schema_version == meta.schema_version
        assert rehydrated.runtime_name == meta.runtime_name


class TestMigration:
    def test_unversioned_to_v1(self) -> None:
        raw = {"runtime_name": "x"}
        migrated = _migrate_compiled_artifact_metadata(raw)
        assert migrated["schema_version"] == 1
        # Defensive copy: original untouched.
        assert "schema_version" not in raw

    def test_nested_mutation_does_not_leak_to_caller(self) -> None:
        # Guards the deep-copy contract: a future migration that mutates
        # nested data (e.g. extra_tags) must not bleed back into the
        # caller's dict. Shallow ``dict(raw)`` would fail this test.
        raw = {"runtime_name": "x", "extra_tags": {"compile_mode": "default"}}
        migrated = _migrate_compiled_artifact_metadata(raw)
        migrated["extra_tags"]["compile_mode"] = "max-autotune"
        assert raw["extra_tags"] == {"compile_mode": "default"}

    def test_current_version_passes_through(self) -> None:
        raw = {
            "schema_version": COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION,
            "runtime_name": "x",
        }
        migrated = _migrate_compiled_artifact_metadata(raw)
        assert migrated["schema_version"] == COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION

    def test_future_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="newer"):
            _migrate_compiled_artifact_metadata(
                {"schema_version": 999, "runtime_name": "x"},
            )

    def test_from_dict_invokes_migration(self) -> None:
        # An unversioned dict must round-trip through from_dict.
        raw = _baseline_kwargs()
        meta = CompiledArtifactMetadata.from_dict(raw)
        assert meta.schema_version == 1
