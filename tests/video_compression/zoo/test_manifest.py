"""Manifest persistence + migration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.video_compression.zoo.config import (
    PERF_ZOO_MANIFEST_SCHEMA_VERSION,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.manifest import (
    ManifestMigrationError,
    _migrate_manifest_document,
    load_manifest,
    save_manifest,
)


def _entry(entry_id: str = "e1", lambda_rd: float = 0.01) -> ModelZooEntryConfig:
    return ModelZooEntryConfig(
        entry_id=entry_id,
        lambda_rd=lambda_rd,
        target_bpp=0.5,
        target_psnr_db=33.0,
        train_steps=1000,
    )


def _manifest(tmp_path: Path) -> ModelZooManifestConfig:
    return ModelZooManifestConfig(
        name="m",
        storage_root=str(tmp_path / "zoo"),
        entries=[_entry("a", 0.01), _entry("b", 0.02)],
    )


class TestMigration:
    def test_unversioned_promoted_to_v1(self) -> None:
        raw = {"name": "m", "storage_root": "./zoo", "entries": []}
        out = _migrate_manifest_document(raw)
        assert out["schema_version"] == 1

    def test_current_version_passes_through(self) -> None:
        raw = {"schema_version": PERF_ZOO_MANIFEST_SCHEMA_VERSION, "x": 1}
        out = _migrate_manifest_document(raw)
        assert out["schema_version"] == PERF_ZOO_MANIFEST_SCHEMA_VERSION

    def test_newer_version_fails_loud(self) -> None:
        raw = {"schema_version": PERF_ZOO_MANIFEST_SCHEMA_VERSION + 5}
        with pytest.raises(ManifestMigrationError, match="newer than this binary"):
            _migrate_manifest_document(raw)

    def test_non_int_version_rejected(self) -> None:
        raw = {"schema_version": "1"}
        with pytest.raises(ManifestMigrationError, match="must be int"):
            _migrate_manifest_document(raw)


class TestRoundTrip:
    def test_save_then_load(self, tmp_path: Path) -> None:
        m = _manifest(tmp_path)
        path = save_manifest(m, tmp_path / "manifest.json")
        loaded = load_manifest(path)
        assert loaded.name == m.name
        assert loaded.storage_root == m.storage_root
        assert [e.entry_id for e in loaded.entries] == ["a", "b"]
        assert loaded.entries[0].lambda_rd == pytest.approx(0.01)

    def test_load_unversioned_file(self, tmp_path: Path) -> None:
        path = tmp_path / "old.json"
        legacy = {
            "name": "old",
            "storage_root": "./zoo",
            "entries": [_entry("a").to_yaml_dict()],
        }
        with path.open("w") as fh:
            json.dump(legacy, fh, default=str)
        loaded = load_manifest(path)
        assert loaded.schema_version == PERF_ZOO_MANIFEST_SCHEMA_VERSION

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.json")

    def test_load_non_object_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        with path.open("w") as fh:
            json.dump([1, 2, 3], fh)
        with pytest.raises(ManifestMigrationError):
            load_manifest(path)

    def test_save_then_load_yaml(self, tmp_path: Path) -> None:
        # YAML round-trip: shipped configs (e.g. lambda_grid.yaml) use
        # YAML, so save_manifest/load_manifest must dispatch on suffix.
        m = _manifest(tmp_path)
        path = save_manifest(m, tmp_path / "manifest.yaml")
        loaded = load_manifest(path)
        assert [e.entry_id for e in loaded.entries] == ["a", "b"]
        assert loaded.entries[0].lambda_rd == pytest.approx(0.01)

    def test_load_shipped_lambda_grid(self) -> None:
        # E2E smoke: the shipped 8-point R-D grid must load cleanly.
        path = Path("config/video_compression/zoo/lambda_grid.yaml")
        if not path.exists():
            pytest.skip("shipped grid not present")
        loaded = load_manifest(path)
        assert len(loaded.entries) == 8
        assert all(e.lambda_rd > 0 for e in loaded.entries)


class TestPropertyBased:
    @given(
        pairs=st.lists(
            st.tuples(
                st.text(
                    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
                    min_size=1,
                    max_size=20,
                ),
                st.floats(min_value=1e-4, max_value=1.0, allow_nan=False),
            ),
            min_size=1,
            max_size=4,
            unique_by=lambda pair: pair[0],
        ),
    )
    def test_round_trip_property(
        self,
        pairs: list[tuple[str, float]],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        tmp_path = tmp_path_factory.mktemp("zoo")
        manifest = ModelZooManifestConfig(
            name="prop",
            storage_root=str(tmp_path / "zoo"),
            entries=[_entry(eid, lr) for eid, lr in pairs],
        )
        path = save_manifest(manifest, tmp_path / "m.json")
        loaded = load_manifest(path)
        assert [e.entry_id for e in loaded.entries] == [p[0] for p in pairs]
        for original, recovered in zip(pairs, loaded.entries):
            assert recovered.lambda_rd == pytest.approx(original[1])
