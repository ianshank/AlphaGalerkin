"""Tests for checkpoint version migration.

Covers the migration registry, path computation, and built-in migration
functions that upgrade checkpoint data between versions.
"""

from __future__ import annotations

import textwrap
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


class TestMigrationDefaultFreeze:
    """Drift alarm for the intentionally frozen v1.1.0 migration defaults.

    The 1.0.0 -> 1.1.0 migration injects the training defaults that v1.1.0
    shipped with, as frozen literals (see the freeze comment in
    src/training/checkpoint_migration.py). This test asserts those literals
    still equal the live defaults in config.schemas.TrainingConfig. If a live
    default is ever retuned, this fails on purpose: decide explicitly whether
    the retune needs a new migration step (old checkpoints keep the old
    value) or the freeze comment should be updated (old checkpoints adopt
    the new value). Do not silently re-point the migration at the live
    constants — that would rewrite what historical checkpoints migrate to.
    """

    def test_migration_defaults_match_v1_1_shipped_values(self) -> None:
        """Frozen migration literals equal today's TrainingConfig defaults."""
        from config.schemas import TrainingConfig

        data: dict[str, Any] = {"version": "1.0.0", "config": {"training": {}}}
        migrated = migrate_checkpoint(data, "1.1.0")
        injected = migrated["config"]["training"]

        live = TrainingConfig()
        assert injected["lbb_loss_weight"] == live.lbb_loss_weight
        assert injected["lbb_target"] == live.lbb_target
        assert injected["log_barrier_weight"] == live.log_barrier_weight
        assert injected["label_smoothing"] == live.label_smoothing

    # -- Additions: the drift alarm above compares the migration against the
    # -- *live* defaults, so it cannot detect (a) both sides being retuned in
    # -- one commit, or (b) the literals being re-pointed at the live config,
    # -- which is precisely what the freeze comment forbids. The two tests
    # -- below close those holes.

    V1_1_SHIPPED_DEFAULTS = {
        "lbb_loss_weight": 0.01,
        "lbb_target": 0.1,
        "log_barrier_weight": 0.1,
        "label_smoothing": 0.0,
    }

    def test_migration_injects_the_v1_1_shipped_literals(self) -> None:
        """Pin the frozen values themselves, independent of any live config.

        Changing a migration literal *and* the matching live default in one
        commit keeps ``test_migration_defaults_match_v1_1_shipped_values``
        green while silently rewriting what historical checkpoints migrate
        to. This test is the actual freeze.
        """
        data: dict[str, Any] = {"version": "1.0.0", "config": {"training": {}}}
        migrated = migrate_checkpoint(data, "1.1.0")

        assert migrated["config"]["training"] == self.V1_1_SHIPPED_DEFAULTS

    def test_migration_defaults_are_source_literals_not_live_references(self) -> None:
        """The setdefault values must be inline constants, per the freeze comment.

        A static check, because re-pointing the literals at
        ``config.schemas.TrainingConfig`` / ``src.constants`` is behaviourally
        invisible today (the values agree) yet breaks the freeze.
        """
        import ast
        import inspect

        import src.training.checkpoint_migration as migration_module

        source = inspect.getsource(migration_module._migrate_1_0_to_1_1)
        tree = ast.parse(textwrap.dedent(source))

        setdefault_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
        ]
        assert len(setdefault_calls) == len(self.V1_1_SHIPPED_DEFAULTS)

        for call in setdefault_calls:
            key, value = call.args
            assert isinstance(key, ast.Constant), ast.dump(key)
            assert isinstance(value, ast.Constant), (
                f"{key.value}'s default must stay an inline literal, not {ast.dump(value)}"
            )
            assert value.value == self.V1_1_SHIPPED_DEFAULTS[key.value]


class TestMigrationTrainingKeyAbsent:
    """The 1.0.0 -> 1.1.0 migration when ``config`` has no ``training`` section.

    Contract: the section is created and the frozen v1.1.0 defaults are
    injected, so a migrated checkpoint always carries the LBB parameters its
    version claims. A non-dict ``training`` value is the one exception — it is
    left untouched rather than overwritten, since replacing unrecognised data
    is not this migration's job.

    Both cases were previously broken by ``training = config.get("training", {})``,
    which handed back an orphan dict that the ``setdefault`` calls populated and
    then discarded while the version was still stamped 1.1.0.
    """

    def test_defaults_are_injected_when_training_key_is_absent(self) -> None:
        """A config without a 'training' key still receives the v1.1.0 defaults.

        Regression test: the migration previously read the section via
        ``config.get("training", {})``, so an absent key produced an orphan dict
        that the setdefault calls populated and then discarded — the checkpoint
        was stamped 1.1.0 while carrying none of the LBB parameters the
        migration exists to add.

        The defect is the false version stamp, not a crash: a consumer loading
        through ``AlphaGalerkinConfig`` gets a ``TrainingConfig`` from
        ``default_factory`` regardless, so the missing section is silently
        replaced by *current* defaults rather than the v1.1.0 ones the stamp
        promises — which is precisely the drift the freeze exists to prevent.
        """
        data: dict[str, Any] = {"version": "1.0.0", "config": {}}
        migrated = migrate_checkpoint(data, "1.1.0")

        assert migrated["version"] == "1.1.0"
        assert migrated["config"]["training"] == {
            "lbb_loss_weight": 0.01,
            "lbb_target": 0.1,
            "log_barrier_weight": 0.1,
            "label_smoothing": 0.0,
        }

    def test_defaults_are_dropped_when_training_is_not_a_dict(self) -> None:
        """A non-dict 'training' value is left untouched (no crash)."""
        data: dict[str, Any] = {"version": "1.0.0", "config": {"training": None}}
        migrated = migrate_checkpoint(data, "1.1.0")

        assert migrated["version"] == "1.1.0"
        assert migrated["config"]["training"] is None
