"""Tests for checkpoint version migration.

Covers the migration registry, path computation, and built-in migration
functions that upgrade checkpoint data between versions.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.training.checkpoint_migration import (
    _MIGRATIONS,
    _parse_version,
    get_migration_path,
    migrate_checkpoint,
)

# --- _parse_version Tests ---


class TestParseVersion:
    """Tests for _parse_version helper."""

    def test_simple_version(self) -> None:
        """Parses standard semver."""
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_zero_version(self) -> None:
        """Parses zero version."""
        assert _parse_version("0.0.0") == (0, 0, 0)

    def test_large_numbers(self) -> None:
        """Parses large version numbers."""
        assert _parse_version("10.20.30") == (10, 20, 30)

    def test_comparison(self) -> None:
        """Parsed versions compare correctly."""
        assert _parse_version("1.0.0") < _parse_version("1.1.0")
        assert _parse_version("1.1.0") < _parse_version("2.0.0")
        assert _parse_version("0.0.0") < _parse_version("1.0.0")
        assert _parse_version("1.0.0") == _parse_version("1.0.0")


# --- Built-in Migrations Tests ---


class TestBuiltInMigrations:
    """Tests for built-in migration functions."""

    def test_0_0_to_1_0_adds_version(self) -> None:
        """Migration 0.0.0 -> 1.0.0 adds version field."""
        data: dict[str, Any] = {"model_state_dict": {}}
        migration = _MIGRATIONS[("0.0.0", "1.0.0")]
        result = migration(data)
        assert result["version"] == "1.0.0"

    def test_0_0_to_1_0_adds_timestamp(self) -> None:
        """Migration 0.0.0 -> 1.0.0 adds timestamp field."""
        data: dict[str, Any] = {}
        migration = _MIGRATIONS[("0.0.0", "1.0.0")]
        result = migration(data)
        assert "timestamp" in result

    def test_0_0_to_1_0_adds_metrics(self) -> None:
        """Migration 0.0.0 -> 1.0.0 adds metrics field."""
        data: dict[str, Any] = {}
        migration = _MIGRATIONS[("0.0.0", "1.0.0")]
        result = migration(data)
        assert "metrics" in result
        assert result["metrics"] == {}

    def test_0_0_to_1_0_preserves_existing(self) -> None:
        """Migration 0.0.0 -> 1.0.0 preserves existing fields."""
        data: dict[str, Any] = {
            "version": "already",
            "timestamp": "2024-01-01",
            "metrics": {"loss": 0.1},
        }
        migration = _MIGRATIONS[("0.0.0", "1.0.0")]
        result = migration(data)
        assert result["version"] == "already"
        assert result["timestamp"] == "2024-01-01"
        assert result["metrics"] == {"loss": 0.1}

    def test_1_0_to_1_1_adds_loss_fields(self) -> None:
        """Migration 1.0.0 -> 1.1.0 adds LBB loss config fields."""
        data: dict[str, Any] = {
            "version": "1.0.0",
            "config": {
                "training": {},
            },
        }
        migration = _MIGRATIONS[("1.0.0", "1.1.0")]
        result = migration(data)
        assert result["version"] == "1.1.0"
        training = result["config"]["training"]
        assert "lbb_loss_weight" in training
        assert "lbb_target" in training
        assert "log_barrier_weight" in training
        assert "label_smoothing" in training

    def test_1_0_to_1_1_preserves_existing_config(self) -> None:
        """Migration 1.0.0 -> 1.1.0 doesn't overwrite existing values."""
        data: dict[str, Any] = {
            "version": "1.0.0",
            "config": {
                "training": {
                    "lbb_loss_weight": 0.5,
                    "log_barrier_weight": 0.2,
                },
            },
        }
        migration = _MIGRATIONS[("1.0.0", "1.1.0")]
        result = migration(data)
        assert result["config"]["training"]["lbb_loss_weight"] == 0.5
        assert result["config"]["training"]["log_barrier_weight"] == 0.2

    def test_1_0_to_1_1_no_config(self) -> None:
        """Migration 1.0.0 -> 1.1.0 handles missing config gracefully."""
        data: dict[str, Any] = {"version": "1.0.0"}
        migration = _MIGRATIONS[("1.0.0", "1.1.0")]
        result = migration(data)
        assert result["version"] == "1.1.0"

    def test_1_0_to_1_1_config_not_dict(self) -> None:
        """Migration 1.0.0 -> 1.1.0 handles non-dict config."""
        data: dict[str, Any] = {"version": "1.0.0", "config": "not a dict"}
        migration = _MIGRATIONS[("1.0.0", "1.1.0")]
        result = migration(data)
        assert result["version"] == "1.1.0"


# --- get_migration_path Tests ---


class TestGetMigrationPath:
    """Tests for get_migration_path."""

    def test_same_version_returns_empty(self) -> None:
        """Same source and target returns empty path."""
        path = get_migration_path("1.0.0", "1.0.0")
        assert path == []

    def test_0_0_to_1_0(self) -> None:
        """Finds path from 0.0.0 to 1.0.0."""
        path = get_migration_path("0.0.0", "1.0.0")
        assert ("0.0.0", "1.0.0") in path

    def test_0_0_to_1_1(self) -> None:
        """Finds path from 0.0.0 to 1.1.0 (through 1.0.0)."""
        path = get_migration_path("0.0.0", "1.1.0")
        assert len(path) == 2
        assert path[0] == ("0.0.0", "1.0.0")
        assert path[1] == ("1.0.0", "1.1.0")

    def test_no_path_raises(self) -> None:
        """Raises ValueError when no migration path exists."""
        with pytest.raises(ValueError, match="No migration path"):
            get_migration_path("99.0.0", "100.0.0")


# --- migrate_checkpoint Tests ---


class TestMigrateCheckpoint:
    """Tests for migrate_checkpoint."""

    def test_already_current_returns_unchanged(self) -> None:
        """Checkpoint at target version is returned unchanged."""
        data = {"version": "1.1.0", "model": "data"}
        result = migrate_checkpoint(data, "1.1.0")
        assert result is data

    def test_above_target_returns_unchanged(self) -> None:
        """Checkpoint above target version is returned unchanged."""
        data = {"version": "2.0.0", "model": "data"}
        result = migrate_checkpoint(data, "1.1.0")
        assert result is data

    def test_full_migration_0_0_to_1_1(self) -> None:
        """Full migration from 0.0.0 to 1.1.0."""
        data: dict[str, Any] = {
            "model_state_dict": {"weight": [1, 2, 3]},
            "config": {"training": {}},
        }
        result = migrate_checkpoint(data, "1.1.0")
        assert result["version"] == "1.1.0"
        assert "timestamp" in result
        assert "metrics" in result
        assert "lbb_loss_weight" in result["config"]["training"]

    def test_migration_from_1_0_to_1_1(self) -> None:
        """Migration from 1.0.0 to 1.1.0."""
        data: dict[str, Any] = {
            "version": "1.0.0",
            "config": {"training": {}},
        }
        result = migrate_checkpoint(data, "1.1.0")
        assert result["version"] == "1.1.0"

    def test_missing_version_defaults_to_0_0_0(self) -> None:
        """Checkpoint without version is treated as 0.0.0."""
        data: dict[str, Any] = {"model_state_dict": {}}
        result = migrate_checkpoint(data, "1.0.0")
        assert result["version"] == "1.0.0"

    def test_version_field_force_set(self) -> None:
        """Migration forces version field even if migration didn't set it."""
        # Test that the version is correctly set after migration
        data: dict[str, Any] = {}
        result = migrate_checkpoint(data, "1.1.0")
        assert result["version"] == "1.1.0"
