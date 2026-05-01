"""Direct unit tests for ``src.video_compression.zoo.cli_helpers``.

The script-level tests in ``tests/scripts/`` exercise the helpers
indirectly through the multi-entry CLI; this file pins the public
contract — error paths, precedence rules, and the no-op early return
— so coverage on the helpers stays at parity with the rest of the zoo
subpackage (≥85%).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.video_compression.config import CodecConfig
from src.video_compression.zoo.cli_helpers import (
    load_codec_config,
    load_dict,
    override_entry,
    resolve_codec_config_for_entry,
    resolve_device,
    resolve_entry,
    resolve_path,
)
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, payload: Any, *, suffix: str) -> Path:
    """Write ``payload`` as YAML or JSON to ``path`` based on ``suffix``."""
    target = path.with_suffix(suffix)
    if suffix in {".yaml", ".yml"}:
        target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    elif suffix == ".json":
        target.write_text(json.dumps(payload), encoding="utf-8")
    else:  # pragma: no cover — guard against typos in callers
        raise AssertionError(f"unexpected suffix in test helper: {suffix}")
    return target


def _make_codec_payload() -> dict[str, Any]:
    """Round-trip a default ``CodecConfig`` to a plain dict for fixtures."""
    return CodecConfig(name="codec").model_dump(mode="json")


def _make_entry(entry_id: str = "e0", **overrides: Any) -> ModelZooEntryConfig:
    """Construct a minimal valid entry; tests override one field at a time."""
    payload: dict[str, Any] = {
        "entry_id": entry_id,
        "lambda_rd": 0.01,
        "target_bpp": 0.25,
        "target_psnr_db": 35.0,
        "train_steps": 1,
        "batch_size": 2,
        "scheduler": {"name": "scheduler", "warmup_steps": 1},
    }
    payload.update(overrides)
    return ModelZooEntryConfig.model_validate(payload)


def _make_manifest(
    entries: list[ModelZooEntryConfig],
    **overrides: Any,
) -> ModelZooManifestConfig:
    payload: dict[str, Any] = {
        "name": "m",
        "storage_root": "./zoo",
        "device_assignment_strategy": DeviceAssignmentStrategy.SINGLE_DEVICE.value,
        "device_preference": "cpu",
        "entries": [e.model_dump() for e in entries],
    }
    payload.update(overrides)
    return ModelZooManifestConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# load_dict
# ---------------------------------------------------------------------------


class TestLoadDict:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cfg", {"k": 1, "n": "x"}, suffix=".yaml")
        assert load_dict(path) == {"k": 1, "n": "x"}

    def test_loads_yml(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cfg", {"k": 1}, suffix=".yml")
        assert load_dict(path) == {"k": 1}

    def test_loads_json(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cfg", {"k": 2}, suffix=".json")
        assert load_dict(path) == {"k": 2}

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.toml"
        path.write_text("k = 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported config suffix"):
            load_dict(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="config file is empty"):
            load_dict(path)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_dict(path)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_returned_unchanged(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "m.yaml"
        target = (tmp_path / "abs.txt").resolve()
        assert resolve_path(str(target), manifest_path=manifest_path) == target

    def test_relative_existing_resolved_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "here.txt").write_text("x", encoding="utf-8")
        manifest_path = tmp_path / "sub" / "m.yaml"
        manifest_path.parent.mkdir()
        result = resolve_path("here.txt", manifest_path=manifest_path)
        assert result == (tmp_path / "here.txt").resolve()

    def test_relative_falls_back_to_manifest_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        manifest_dir = tmp_path / "sub"
        manifest_dir.mkdir()
        (manifest_dir / "ref.yaml").write_text("a: 1", encoding="utf-8")
        manifest_path = manifest_dir / "m.yaml"
        result = resolve_path("ref.yaml", manifest_path=manifest_path)
        assert result == (manifest_dir / "ref.yaml").resolve()


# ---------------------------------------------------------------------------
# load_codec_config
# ---------------------------------------------------------------------------


def test_load_codec_config_round_trip(tmp_path: Path) -> None:
    payload = _make_codec_payload()
    path = _write(tmp_path / "codec", payload, suffix=".yaml")
    cfg = load_codec_config(path)
    assert isinstance(cfg, CodecConfig)


# ---------------------------------------------------------------------------
# resolve_entry
# ---------------------------------------------------------------------------


class TestResolveEntry:
    def test_returns_matching_entry(self) -> None:
        e0 = _make_entry("alpha")
        e1 = _make_entry("beta")
        manifest = _make_manifest([e0, e1])
        assert resolve_entry(manifest, "beta").entry_id == "beta"

    def test_missing_entry_raises_keyerror(self) -> None:
        manifest = _make_manifest([_make_entry("alpha")])
        with pytest.raises(KeyError, match="not found in manifest"):
            resolve_entry(manifest, "missing")


# ---------------------------------------------------------------------------
# resolve_codec_config_for_entry
# ---------------------------------------------------------------------------


class TestResolveCodecConfigForEntry:
    def test_uses_entry_ref_first(self, tmp_path: Path) -> None:
        codec_path = _write(tmp_path / "codec", _make_codec_payload(), suffix=".yaml")
        entry = _make_entry(codec_config_ref=str(codec_path))
        manifest = _make_manifest([entry])
        manifest_path = tmp_path / "m.yaml"
        cfg = resolve_codec_config_for_entry(
            manifest, entry, manifest_path=manifest_path
        )
        assert isinstance(cfg, CodecConfig)

    def test_falls_back_to_manifest_default(self, tmp_path: Path) -> None:
        codec_path = _write(tmp_path / "codec", _make_codec_payload(), suffix=".yaml")
        entry = _make_entry()
        manifest = _make_manifest(
            [entry], default_codec_config_ref=str(codec_path)
        )
        manifest_path = tmp_path / "m.yaml"
        cfg = resolve_codec_config_for_entry(
            manifest, entry, manifest_path=manifest_path
        )
        assert isinstance(cfg, CodecConfig)

    def test_no_ref_anywhere_raises(self, tmp_path: Path) -> None:
        entry = _make_entry()
        manifest = _make_manifest([entry])
        manifest_path = tmp_path / "m.yaml"
        with pytest.raises(ValueError, match="does not declare codec_config_ref"):
            resolve_codec_config_for_entry(
                manifest, entry, manifest_path=manifest_path
            )


# ---------------------------------------------------------------------------
# override_entry
# ---------------------------------------------------------------------------


class TestOverrideEntry:
    def test_no_overrides_returns_same_instance(self) -> None:
        entry = _make_entry()
        result = override_entry(entry)
        # Identity short-circuit avoids a needless re-validation round trip.
        assert result is entry

    def test_max_steps_override(self) -> None:
        entry = _make_entry(train_steps=10)
        result = override_entry(entry, max_steps=42)
        assert result.train_steps == 42
        assert entry.train_steps == 10  # original untouched

    def test_device_override(self) -> None:
        entry = _make_entry()
        result = override_entry(entry, device="cuda:1")
        assert result.device == "cuda:1"

    def test_both_overrides(self) -> None:
        entry = _make_entry(train_steps=5)
        result = override_entry(entry, max_steps=99, device="cpu")
        assert result.train_steps == 99
        assert result.device == "cpu"


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_explicit_override_wins(self) -> None:
        entry = _make_entry(device="cuda:0")
        manifest = _make_manifest([entry], device_preference="cpu")
        assert resolve_device(manifest, entry, device_override="cuda:1") == "cuda:1"

    def test_entry_device_used_when_no_override(self) -> None:
        entry = _make_entry(device="cuda:0")
        manifest = _make_manifest([entry], device_preference="cpu")
        assert resolve_device(manifest, entry, device_override=None) == "cuda:0"

    def test_falls_back_to_manifest_preference(self) -> None:
        entry = _make_entry()
        manifest = _make_manifest([entry], device_preference="cpu")
        assert resolve_device(manifest, entry, device_override=None) == "cpu"
