"""Shared CLI helpers for the video-compression zoo trainers.

These primitives (config loading, path resolution, codec-config
resolution, entry overrides, device resolution) are factored out of
:mod:`scripts.train_compression_zoo_entry` so that the multi-entry
sweep CLI in :mod:`scripts.train_compression_zoo` can reuse the exact
same parsing semantics. Keeping a single source of truth here avoids
drift between the two CLIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.video_compression.config import CodecConfig
from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)


def load_dict(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON config file into a plain dict."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"unsupported config suffix {suffix!r}; expected .yaml/.yml/.json",
        )
    if data is None:
        raise ValueError(f"config file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping; got {type(data).__name__}")
    return data


def resolve_path(path_str: str, *, manifest_path: Path) -> Path:
    """Resolve ``path_str`` against the manifest's directory if needed.

    Absolute paths are returned untouched. Relative paths are tried
    first against ``cwd``; if they don't exist there, fall back to the
    manifest's parent directory.
    """
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def load_codec_config(path: Path) -> CodecConfig:
    """Load + validate a codec config from a YAML/JSON file."""
    return CodecConfig.model_validate(load_dict(path))


def resolve_entry(
    manifest: ModelZooManifestConfig,
    entry_id: str,
) -> ModelZooEntryConfig:
    """Look up ``entry_id`` in the manifest; raise on miss."""
    for entry in manifest.entries:
        if entry.entry_id == entry_id:
            return entry
    raise KeyError(f"entry_id {entry_id!r} not found in manifest {manifest.name!r}")


def resolve_codec_config_for_entry(
    manifest: ModelZooManifestConfig,
    entry: ModelZooEntryConfig,
    *,
    manifest_path: Path,
) -> CodecConfig:
    """Resolve the codec config for one entry.

    Precedence: ``entry.codec_config_ref`` > ``manifest.default_codec_config_ref``.
    Raises ``ValueError`` if neither is set.
    """
    ref = entry.codec_config_ref or manifest.default_codec_config_ref
    if ref is None:
        raise ValueError(
            f"entry {entry.entry_id!r} does not declare codec_config_ref and the "
            "manifest has no default_codec_config_ref",
        )
    return load_codec_config(resolve_path(ref, manifest_path=manifest_path))


def override_entry(
    entry: ModelZooEntryConfig,
    *,
    max_steps: int | None = None,
    device: str | None = None,
) -> ModelZooEntryConfig:
    """Return a copy of ``entry`` with optional CLI overrides applied."""
    overrides: dict[str, Any] = {}
    if max_steps is not None:
        overrides["train_steps"] = max_steps
    if device is not None:
        overrides["device"] = device
    if not overrides:
        return entry
    payload = entry.model_dump()
    payload.update(overrides)
    return ModelZooEntryConfig.model_validate(payload)


def resolve_device(
    manifest: ModelZooManifestConfig,
    entry: ModelZooEntryConfig,
    *,
    device_override: str | None,
) -> str:
    """Pick the runtime device.

    Precedence: explicit CLI ``--device`` > entry's pinned device >
    manifest-level ``device_preference``.
    """
    return device_override or entry.device or manifest.device_preference


__all__ = [
    "load_dict",
    "resolve_path",
    "load_codec_config",
    "resolve_entry",
    "resolve_codec_config_for_entry",
    "override_entry",
    "resolve_device",
]
