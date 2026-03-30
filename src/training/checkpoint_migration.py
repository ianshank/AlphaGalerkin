"""Checkpoint version migration for backwards compatibility.

Provides a registry of migration functions that upgrade checkpoint data
from one version to the next. Migrations are applied sequentially to
bring old checkpoints up to the current version.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Type alias for migration functions
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

# Registry of version migrations: (from_version, to_version) -> migration_fn
_MIGRATIONS: dict[tuple[str, str], MigrationFn] = {}


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a semver string into a comparable tuple.

    Args:
        version: Semver string (e.g. "1.2.3").

    Returns:
        Tuple of integer components.

    """
    return tuple(int(x) for x in version.split("."))


def register_migration(from_version: str, to_version: str) -> Callable[[MigrationFn], MigrationFn]:
    """Register a checkpoint migration function.

    Args:
        from_version: Source checkpoint version (semver string).
        to_version: Target checkpoint version (semver string).

    Returns:
        Decorator that registers the migration function.

    Example:
        @register_migration("1.0.0", "1.1.0")
        def migrate_1_0_to_1_1(data: dict[str, Any]) -> dict[str, Any]:
            data["new_field"] = "default_value"
            data["version"] = "1.1.0"
            return data

    """
    def decorator(fn: MigrationFn) -> MigrationFn:
        _MIGRATIONS[(from_version, to_version)] = fn
        return fn
    return decorator


def get_migration_path(from_version: str, target_version: str) -> list[tuple[str, str]]:
    """Compute the sequence of migrations needed to reach target version.

    Uses a simple greedy approach: at each step, find the migration
    that starts from the current version and has the highest target version
    that doesn't exceed the target.

    Args:
        from_version: Current checkpoint version.
        target_version: Desired checkpoint version.

    Returns:
        Ordered list of (from, to) version pairs representing the migration path.

    Raises:
        ValueError: If no migration path exists.

    """
    if from_version == target_version:
        return []

    current = from_version
    path: list[tuple[str, str]] = []

    while _parse_version(current) < _parse_version(target_version):
        # Find all migrations from current version
        candidates = [
            (f, t) for (f, t) in _MIGRATIONS
            if f == current and _parse_version(t) <= _parse_version(target_version)
        ]

        if not candidates:
            msg = (
                f"No migration path from {current} to {target_version}. "
                f"Available migrations: {list(_MIGRATIONS.keys())}"
            )
            raise ValueError(msg)

        # Pick the migration with the highest target version
        best = max(candidates, key=lambda x: _parse_version(x[1]))
        path.append(best)
        current = best[1]

    return path


def migrate_checkpoint(
    data: dict[str, Any],
    target_version: str,
) -> dict[str, Any]:
    """Migrate checkpoint data to the target version.

    Applies migrations sequentially from the checkpoint's current version
    to the target version. If the checkpoint is already at or above the
    target version, returns the data unchanged.

    Args:
        data: Checkpoint data dictionary.
        target_version: Desired checkpoint version.

    Returns:
        Migrated checkpoint data with updated version field.

    """
    current_version = data.get("version", "0.0.0")

    if _parse_version(current_version) >= _parse_version(target_version):
        logger.debug(
            "checkpoint_already_current",
            current_version=current_version,
            target_version=target_version,
        )
        return data

    path = get_migration_path(current_version, target_version)

    logger.info(
        "migrating_checkpoint",
        from_version=current_version,
        to_version=target_version,
        n_migrations=len(path),
        path=[f"{f}->{t}" for f, t in path],
    )

    for from_ver, to_ver in path:
        migration_fn = _MIGRATIONS[(from_ver, to_ver)]
        logger.debug("applying_migration", from_version=from_ver, to_version=to_ver)
        data = migration_fn(data)

        # Verify version was updated
        if data.get("version") != to_ver:
            data["version"] = to_ver

    return data


# ============================================================
# Built-in Migrations
# ============================================================

@register_migration("0.0.0", "1.0.0")
def _migrate_0_0_to_1_0(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy checkpoints (no version field) to v1.0.0.

    Legacy checkpoints may lack version, timestamp, or metrics fields.
    """
    if "version" not in data:
        data["version"] = "1.0.0"
    if "timestamp" not in data:
        data["timestamp"] = ""
    if "metrics" not in data:
        data["metrics"] = {}
    return data


@register_migration("1.0.0", "1.1.0")
def _migrate_1_0_to_1_1(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate v1.0.0 to v1.1.0.

    Adds log_barrier_weight to loss config for backwards compatibility
    with new configurable LBB loss parameters.
    """
    config = data.get("config")
    if isinstance(config, dict):
        training = config.get("training", {})
        if isinstance(training, dict):
            training.setdefault("lbb_loss_weight", 0.01)
            training.setdefault("lbb_target", 0.1)
            training.setdefault("log_barrier_weight", 0.1)
            training.setdefault("label_smoothing", 0.0)
    data["version"] = "1.1.0"
    return data
