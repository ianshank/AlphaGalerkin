"""Manifest persistence + forward-compat migration.

Mirrors the Phase 0 ``BaselineDocument`` migration pattern: load is
permissive (unknown fields ignored, unversioned manifests promoted to
v1), save is strict (current schema only). Newer-than-binary manifests
fail loud.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.video_compression.zoo.config import (
    PERF_ZOO_MANIFEST_SCHEMA_VERSION,
    ModelZooManifestConfig,
)

logger = structlog.get_logger(__name__)

#: File extensions that should be parsed/serialized as YAML rather than JSON.
YAML_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml"})


class ManifestMigrationError(ValueError):
    """Raised when a manifest cannot be migrated to the current schema."""


def _migrate_manifest_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate a raw manifest dict to the current schema.

    Migration table:

    +----------------+----------------+--------------------------------+
    | from           | to             | change                         |
    +================+================+================================+
    | (unversioned)  | 1              | add ``schema_version`` field   |
    +----------------+----------------+--------------------------------+

    New schemas are appended here. Older manifests remain loadable.
    """
    raw = dict(raw)  # defensive copy

    schema_version = raw.get("schema_version")
    if schema_version is None:
        logger.info(
            "zoo.manifest.migration.unversioned_to_v1",
            keys=sorted(raw.keys()),
        )
        raw["schema_version"] = 1
        schema_version = 1

    if not isinstance(schema_version, int):
        raise ManifestMigrationError(
            f"schema_version must be int; got {type(schema_version).__name__}",
        )

    if schema_version > PERF_ZOO_MANIFEST_SCHEMA_VERSION:
        raise ManifestMigrationError(
            f"manifest schema_version={schema_version} is newer than this "
            f"binary ({PERF_ZOO_MANIFEST_SCHEMA_VERSION}); upgrade the "
            f"package or pin a compatible manifest.",
        )
    return raw


def load_manifest(path: str | Path) -> ModelZooManifestConfig:
    """Load a manifest from JSON or YAML, applying migrations as needed.

    The serialization format is dispatched by ``path.suffix``: ``.yaml``
    and ``.yml`` use ``yaml.safe_load``; everything else is parsed as
    JSON.

    Args:
        path: Path to a manifest JSON or YAML file.

    Returns:
        Validated :class:`ModelZooManifestConfig`.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ManifestMigrationError: When the manifest schema is newer than
            the running binary or otherwise unmigratable.
        pydantic.ValidationError: When the migrated payload fails schema
            validation.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        if p.suffix.lower() in YAML_SUFFIXES:
            raw = yaml.safe_load(fh)
        else:
            raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ManifestMigrationError(
            f"manifest root must be a mapping; got {type(raw).__name__}",
        )

    migrated = _migrate_manifest_document(raw)
    # ``name`` is required by BaseModuleConfig; older manifests may not
    # have one. Provide a stable default derived from the file path.
    migrated.setdefault("name", p.stem)
    return ModelZooManifestConfig.model_validate(migrated)


def save_manifest(
    manifest: ModelZooManifestConfig,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Persist a manifest as JSON or YAML.

    The serialization format is dispatched by ``path.suffix``: ``.yaml``
    and ``.yml`` use ``yaml.safe_dump``; everything else is JSON.

    Args:
        manifest: Validated manifest.
        path: Destination path. Parent directories are created.
        indent: JSON indent width for human-readability (ignored for YAML).

    Returns:
        The :class:`pathlib.Path` written to.

    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_yaml_dict()
    with p.open("w", encoding="utf-8") as fh:
        if p.suffix.lower() in YAML_SUFFIXES:
            yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)
        else:
            json.dump(payload, fh, indent=indent, sort_keys=True, default=str)
    logger.info(
        "zoo.manifest.saved",
        path=str(p),
        n_entries=len(manifest.entries),
        schema_version=manifest.schema_version,
    )
    return p
