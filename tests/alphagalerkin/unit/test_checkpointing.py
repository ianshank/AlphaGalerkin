"""Comprehensive tests for the CheckpointManager in training/checkpointing.py.

Covers:
- Save / load roundtrip (with and without optional fields)
- Best-model tracking in both "min" and "max" modes
- _is_improvement logic (first value, better, worse, equal)
- Checkpoint rotation via keep_last_n
- Latest checkpoint discovery
- list_checkpoints ordering
- Empty directory handling (load raises FileNotFoundError)
- Migration system (_apply_migrations with missing migration)
- Checkpoint filename pattern matching (ignores non-matching files)
- CheckpointConfig validation (preserved from original tests)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.alphagalerkin.core.config import CheckpointConfig
from src.alphagalerkin.core.exceptions import CheckpointError
from src.alphagalerkin.training.checkpointing import (
    _BEST_FILENAME,
    _CHECKPOINT_PATTERN,
    CURRENT_VERSION,
    MIGRATIONS,
    CheckpointManager,
    _apply_migrations,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for checkpoints."""
    return tmp_path / "checkpoints"


@pytest.fixture()
def min_config(ckpt_dir: Path) -> CheckpointConfig:
    """Config that tracks the best model via *min* mode."""
    return CheckpointConfig(
        checkpoint_dir=str(ckpt_dir),
        keep_last_n=3,
        save_best=True,
        best_metric="loss",
        best_metric_mode="min",
    )


@pytest.fixture()
def max_config(ckpt_dir: Path) -> CheckpointConfig:
    """Config that tracks the best model via *max* mode."""
    return CheckpointConfig(
        checkpoint_dir=str(ckpt_dir),
        keep_last_n=3,
        save_best=True,
        best_metric="accuracy",
        best_metric_mode="max",
    )


@pytest.fixture()
def no_best_config(ckpt_dir: Path) -> CheckpointConfig:
    """Config with save_best disabled."""
    return CheckpointConfig(
        checkpoint_dir=str(ckpt_dir),
        keep_last_n=5,
        save_best=False,
    )


@pytest.fixture()
def manager(min_config: CheckpointConfig) -> CheckpointManager:
    return CheckpointManager(min_config)


@pytest.fixture()
def sample_network_state() -> dict[str, Any]:
    return {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}


@pytest.fixture()
def sample_optimizer_state() -> dict[str, Any]:
    return {"lr": 0.001, "step": 100}


# -------------------------------------------------------------------
# CheckpointConfig validation (kept from original tests)
# -------------------------------------------------------------------


class TestCheckpointConfig:
    """Tests for CheckpointConfig validation."""

    def test_default_config_valid(self) -> None:
        config = CheckpointConfig()
        assert config.keep_last_n >= 1
        assert config.save_interval_steps >= 1
        assert config.best_metric_mode in ("min", "max")

    def test_keep_last_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            CheckpointConfig(keep_last_n=0)

    def test_save_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            CheckpointConfig(save_interval_steps=0)

    def test_resume_from_none_by_default(self) -> None:
        config = CheckpointConfig()
        assert config.resume_from is None

    def test_resume_from_path(self) -> None:
        config = CheckpointConfig(resume_from="/path/to/ckpt.pt")
        assert config.resume_from == "/path/to/ckpt.pt"


# -------------------------------------------------------------------
# CheckpointError (kept from original tests)
# -------------------------------------------------------------------


class TestCheckpointError:
    """Tests for the CheckpointError exception."""

    def test_checkpoint_error_has_path(self) -> None:
        err = CheckpointError(
            "load failed",
            checkpoint_path="/tmp/ckpt.pt",
        )
        assert err.checkpoint_path == "/tmp/ckpt.pt"
        assert "checkpoint_path" in err.context

    def test_checkpoint_error_without_path(self) -> None:
        err = CheckpointError("generic failure")
        assert err.checkpoint_path is None


# -------------------------------------------------------------------
# Init
# -------------------------------------------------------------------


class TestCheckpointManagerInit:
    """Tests for CheckpointManager.__init__."""

    def test_creates_checkpoint_directory(
        self, min_config: CheckpointConfig, ckpt_dir: Path,
    ) -> None:
        """__init__ should create the checkpoint dir if it doesn't exist."""
        assert not ckpt_dir.exists()
        CheckpointManager(min_config)
        assert ckpt_dir.is_dir()

    def test_existing_directory_ok(
        self, min_config: CheckpointConfig, ckpt_dir: Path,
    ) -> None:
        """__init__ should not raise if the dir already exists."""
        ckpt_dir.mkdir(parents=True)
        mgr = CheckpointManager(min_config)
        assert mgr is not None

    def test_nested_directory_creation(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        config = CheckpointConfig(checkpoint_dir=str(deep))
        CheckpointManager(config)
        assert deep.is_dir()


# -------------------------------------------------------------------
# Save / Load roundtrip
# -------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """Verify save() -> load() preserves all fields."""

    def test_basic_roundtrip(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        path = manager.save(
            iteration=100,
            network_state=sample_network_state,
            optimizer_state=sample_optimizer_state,
        )
        assert path.exists()
        loaded = manager.load(path)
        assert loaded["iteration"] == 100
        assert loaded["version"] == CURRENT_VERSION
        assert "network_state_dict" in loaded
        assert "optimizer_state_dict" in loaded

    def test_roundtrip_with_optional_fields(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        replay = {"positions": [1, 2, 3]}
        metrics = {"loss": 0.5, "accuracy": 0.9}
        extra = {"custom_key": "custom_value"}

        path = manager.save(
            iteration=200,
            network_state=sample_network_state,
            optimizer_state=sample_optimizer_state,
            replay_buffer_state=replay,
            training_metrics=metrics,
            extra=extra,
        )
        loaded = manager.load(path)

        assert loaded["replay_buffer_state"] == replay
        assert loaded["training_metrics"] == metrics
        assert loaded["extra"] == extra

    def test_optional_fields_absent_when_not_provided(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        path = manager.save(
            iteration=300,
            network_state=sample_network_state,
            optimizer_state=sample_optimizer_state,
        )
        loaded = manager.load(path)
        assert "replay_buffer_state" not in loaded
        assert "training_metrics" not in loaded
        assert "extra" not in loaded

    def test_filename_format(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        path = manager.save(
            iteration=42,
            network_state=sample_network_state,
            optimizer_state=sample_optimizer_state,
        )
        assert path.name == "checkpoint_00000042.pt"

    @pytest.mark.parametrize("iteration", [0, 1, 999, 10_000, 99_999_999])
    def test_filename_for_various_iterations(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
        iteration: int,
    ) -> None:
        path = manager.save(
            iteration=iteration,
            network_state=sample_network_state,
            optimizer_state=sample_optimizer_state,
        )
        expected = f"checkpoint_{iteration:08d}.pt"
        assert path.name == expected

    def test_network_state_tensor_values_preserved(
        self,
        manager: CheckpointManager,
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        path = manager.save(
            iteration=1,
            network_state={"w": t},
            optimizer_state=sample_optimizer_state,
        )
        loaded = manager.load(path)
        assert torch.equal(loaded["network_state_dict"]["w"], t)


# -------------------------------------------------------------------
# Best model tracking
# -------------------------------------------------------------------


class TestBestModelTracking:
    """Tests for save_best and _is_improvement."""

    def test_first_save_always_improvement_min_mode(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        path = mgr.save_best(
            metric_value=1.0,
            network_state={"w": 1},
            iteration=1,
        )
        assert path is not None
        assert path.name == _BEST_FILENAME

    def test_first_save_always_improvement_max_mode(
        self, max_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(max_config)
        path = mgr.save_best(
            metric_value=0.5,
            network_state={"w": 1},
            iteration=1,
        )
        assert path is not None

    @pytest.mark.parametrize(
        ("mode", "first", "second", "should_save"),
        [
            ("min", 1.0, 0.5, True),   # lower is better -> improvement
            ("min", 0.5, 1.0, False),   # higher is worse -> no save
            ("min", 0.5, 0.5, False),   # equal -> no improvement
            ("max", 0.5, 1.0, True),    # higher is better -> improvement
            ("max", 1.0, 0.5, False),   # lower is worse -> no save
            ("max", 1.0, 1.0, False),   # equal -> no improvement
        ],
    )
    def test_improvement_detection(
        self,
        ckpt_dir: Path,
        mode: str,
        first: float,
        second: float,
        should_save: bool,
    ) -> None:
        config = CheckpointConfig(
            checkpoint_dir=str(ckpt_dir),
            save_best=True,
            best_metric_mode=mode,
        )
        mgr = CheckpointManager(config)
        # First save always succeeds
        mgr.save_best(metric_value=first, network_state={"w": 1}, iteration=1)
        # Second save depends on improvement
        result = mgr.save_best(
            metric_value=second, network_state={"w": 2}, iteration=2,
        )
        if should_save:
            assert result is not None
        else:
            assert result is None

    def test_save_best_disabled(
        self, no_best_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(no_best_config)
        result = mgr.save_best(
            metric_value=0.1,
            network_state={"w": 1},
            iteration=1,
        )
        assert result is None

    def test_best_model_contains_metric_info(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        path = mgr.save_best(
            metric_value=0.42,
            network_state={"w": 1},
            iteration=10,
        )
        assert path is not None
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["best_metric"] == "loss"
        assert loaded["best_metric_value"] == 0.42
        assert loaded["iteration"] == 10
        assert loaded["version"] == CURRENT_VERSION

    def test_best_model_overwrites_previous(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        mgr.save_best(metric_value=1.0, network_state={"v": 1}, iteration=1)
        mgr.save_best(metric_value=0.5, network_state={"v": 2}, iteration=2)
        path = Path(min_config.checkpoint_dir) / _BEST_FILENAME
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["best_metric_value"] == 0.5
        assert loaded["iteration"] == 2

    def test_successive_improvements_min(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        values = [1.0, 0.8, 0.9, 0.6, 0.7]
        saved_iterations = []
        for i, v in enumerate(values):
            result = mgr.save_best(
                metric_value=v, network_state={"v": i}, iteration=i,
            )
            if result is not None:
                saved_iterations.append(i)
        # Improvements: 1.0 (first), 0.8, 0.6
        assert saved_iterations == [0, 1, 3]


# -------------------------------------------------------------------
# _is_improvement (direct tests)
# -------------------------------------------------------------------


class TestIsImprovement:
    """Direct tests of _is_improvement."""

    def test_first_value_always_improvement(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        assert mgr._is_improvement(999.0) is True

    def test_min_mode_lower_is_better(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        mgr._best_metric_value = 1.0
        assert mgr._is_improvement(0.5) is True
        assert mgr._is_improvement(1.5) is False
        assert mgr._is_improvement(1.0) is False  # equal is not improvement

    def test_max_mode_higher_is_better(
        self, max_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(max_config)
        mgr._best_metric_value = 0.5
        assert mgr._is_improvement(1.0) is True
        assert mgr._is_improvement(0.1) is False
        assert mgr._is_improvement(0.5) is False  # equal is not improvement


# -------------------------------------------------------------------
# Checkpoint rotation (keep_last_n)
# -------------------------------------------------------------------


class TestCheckpointRotation:
    """Tests for _cleanup_old_checkpoints."""

    def test_keeps_last_n(self, ckpt_dir: Path) -> None:
        config = CheckpointConfig(
            checkpoint_dir=str(ckpt_dir),
            keep_last_n=2,
            save_best=False,
        )
        mgr = CheckpointManager(config)
        for i in range(5):
            mgr.save(
                iteration=i,
                network_state={"w": i},
                optimizer_state={"lr": 0.01},
            )
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 2
        # Should keep the two most recent (iteration 3 and 4)
        assert remaining[0].name == "checkpoint_00000003.pt"
        assert remaining[1].name == "checkpoint_00000004.pt"

    def test_no_cleanup_when_under_limit(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        """keep_last_n=3, saving only 2 should not delete anything."""
        manager.save(1, sample_network_state, sample_optimizer_state)
        manager.save(2, sample_network_state, sample_optimizer_state)
        assert len(manager.list_checkpoints()) == 2

    def test_best_model_not_deleted_by_rotation(
        self, ckpt_dir: Path,
    ) -> None:
        """The best_model.pt should survive rotation."""
        config = CheckpointConfig(
            checkpoint_dir=str(ckpt_dir),
            keep_last_n=2,
            save_best=True,
            best_metric_mode="min",
        )
        mgr = CheckpointManager(config)
        mgr.save_best(metric_value=0.1, network_state={"w": 0}, iteration=0)
        for i in range(5):
            mgr.save(
                iteration=i,
                network_state={"w": i},
                optimizer_state={"lr": 0.01},
            )
        best_path = ckpt_dir / _BEST_FILENAME
        assert best_path.exists()

    def test_keep_last_n_equals_1(self, ckpt_dir: Path) -> None:
        config = CheckpointConfig(
            checkpoint_dir=str(ckpt_dir),
            keep_last_n=1,
            save_best=False,
        )
        mgr = CheckpointManager(config)
        for i in range(10):
            mgr.save(iteration=i, network_state={}, optimizer_state={})
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 1
        assert remaining[0].name == "checkpoint_00000009.pt"


# -------------------------------------------------------------------
# Latest checkpoint discovery
# -------------------------------------------------------------------


class TestLatestCheckpoint:
    """Tests for _latest_checkpoint."""

    def test_empty_directory(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        assert mgr._latest_checkpoint() is None

    def test_returns_highest_iteration(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        mgr.save(iteration=10, network_state={}, optimizer_state={})
        mgr.save(iteration=5, network_state={}, optimizer_state={})
        mgr.save(iteration=20, network_state={}, optimizer_state={})
        latest = mgr._latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint_00000020.pt"

    def test_ignores_non_checkpoint_files(
        self, ckpt_dir: Path, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        # Create non-matching files
        (ckpt_dir / "notes.txt").write_text("hello")
        (ckpt_dir / "best_model.pt").write_text("fake")
        (ckpt_dir / "checkpoint_abc.pt").write_text("bad")
        (ckpt_dir / "checkpoint.pt").write_text("bad")
        assert mgr._latest_checkpoint() is None

    def test_load_without_path_uses_latest(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        mgr.save(iteration=1, network_state={"v": 1}, optimizer_state={})
        mgr.save(iteration=2, network_state={"v": 2}, optimizer_state={})
        loaded = mgr.load()  # path=None
        assert loaded["iteration"] == 2


# -------------------------------------------------------------------
# list_checkpoints
# -------------------------------------------------------------------


class TestListCheckpoints:
    """Tests for list_checkpoints."""

    def test_empty_directory(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        assert mgr.list_checkpoints() == []

    def test_sorted_by_iteration(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        # Save in non-sequential order. keep_last_n=3 means all 3 are kept.
        mgr.save(iteration=30, network_state={}, optimizer_state={})
        mgr.save(iteration=10, network_state={}, optimizer_state={})
        mgr.save(iteration=20, network_state={}, optimizer_state={})
        result = mgr.list_checkpoints()
        names = [p.name for p in result]
        assert names == [
            "checkpoint_00000010.pt",
            "checkpoint_00000020.pt",
            "checkpoint_00000030.pt",
        ]

    def test_ignores_best_model(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        mgr.save(iteration=1, network_state={}, optimizer_state={})
        mgr.save_best(metric_value=0.1, network_state={}, iteration=1)
        checkpoints = mgr.list_checkpoints()
        names = [p.name for p in checkpoints]
        assert _BEST_FILENAME not in names

    def test_ignores_unrelated_files(
        self, ckpt_dir: Path, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        mgr.save(iteration=1, network_state={}, optimizer_state={})
        (ckpt_dir / "readme.txt").write_text("ignore me")
        (ckpt_dir / "model.onnx").write_text("ignore me")
        assert len(mgr.list_checkpoints()) == 1


# -------------------------------------------------------------------
# Load edge cases
# -------------------------------------------------------------------


class TestLoadEdgeCases:
    """Edge cases for the load method."""

    def test_load_empty_dir_raises(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            mgr.load()

    def test_load_explicit_path(
        self,
        manager: CheckpointManager,
        sample_network_state: dict[str, Any],
        sample_optimizer_state: dict[str, Any],
    ) -> None:
        path = manager.save(
            iteration=5,
            network_state=sample_network_state,
            optimizer_state=sample_optimizer_state,
        )
        loaded = manager.load(path)
        assert loaded["iteration"] == 5

    def test_load_best_model_by_path(
        self, min_config: CheckpointConfig,
    ) -> None:
        mgr = CheckpointManager(min_config)
        best_path = mgr.save_best(
            metric_value=0.01,
            network_state={"w": torch.randn(2)},
            iteration=99,
        )
        assert best_path is not None
        loaded = mgr.load(best_path)
        assert loaded["iteration"] == 99
        assert loaded["best_metric_value"] == 0.01


# -------------------------------------------------------------------
# Migration system
# -------------------------------------------------------------------


class TestMigrations:
    """Tests for _apply_migrations."""

    def test_current_version_no_migration(self) -> None:
        """A checkpoint at CURRENT_VERSION should pass through unchanged."""
        ckpt = {"version": CURRENT_VERSION, "data": 42}
        result = _apply_migrations(ckpt)
        assert result["data"] == 42
        assert result["version"] == CURRENT_VERSION

    def test_missing_migration_raises(self) -> None:
        """A checkpoint at an old version without migration should raise."""
        old_ckpt = {"version": 0, "data": "old"}
        # CURRENT_VERSION is 1, so version 0 needs a migration that
        # doesn't exist in MIGRATIONS (it's empty)
        with pytest.raises(ValueError, match="No migration from checkpoint version 0"):
            _apply_migrations(old_ckpt)

    def test_missing_version_key_defaults_to_zero(self) -> None:
        """Checkpoint without a version key should be treated as version 0."""
        ckpt: dict[str, Any] = {"data": "legacy"}
        with pytest.raises(ValueError, match="No migration from checkpoint version 0"):
            _apply_migrations(ckpt)

    def test_migration_chain_applied_sequentially(self) -> None:
        """When migrations exist, they are applied in order."""

        def migrate_v0_to_v1(ckpt: dict[str, Any]) -> dict[str, Any]:
            ckpt["migrated_v0_v1"] = True
            return ckpt

        def migrate_v1_to_v2(ckpt: dict[str, Any]) -> dict[str, Any]:
            ckpt["migrated_v1_v2"] = True
            return ckpt

        with patch.dict(MIGRATIONS, {0: migrate_v0_to_v1, 1: migrate_v1_to_v2}):
            with patch(
                "src.alphagalerkin.training.checkpointing.CURRENT_VERSION", 2,
            ):
                ckpt: dict[str, Any] = {"version": 0, "data": "old"}
                result = _apply_migrations(ckpt)
                assert result["migrated_v0_v1"] is True
                assert result["migrated_v1_v2"] is True
                assert result["version"] == 2

    def test_single_migration_step(self) -> None:
        """A single migration step should work."""

        def migrate_v0_to_v1(ckpt: dict[str, Any]) -> dict[str, Any]:
            ckpt["upgraded"] = True
            return ckpt

        with patch.dict(MIGRATIONS, {0: migrate_v0_to_v1}):
            ckpt: dict[str, Any] = {"version": 0}
            result = _apply_migrations(ckpt)
            assert result["upgraded"] is True
            assert result["version"] == CURRENT_VERSION


# -------------------------------------------------------------------
# Checkpoint filename pattern
# -------------------------------------------------------------------


class TestCheckpointPattern:
    """Tests for the _CHECKPOINT_PATTERN regex."""

    @pytest.mark.parametrize(
        "filename",
        [
            "checkpoint_00000001.pt",
            "checkpoint_00000000.pt",
            "checkpoint_99999999.pt",
            "checkpoint_12345678.pt",
        ],
    )
    def test_valid_filenames_match(self, filename: str) -> None:
        assert _CHECKPOINT_PATTERN.match(filename) is not None

    @pytest.mark.parametrize(
        "filename",
        [
            "checkpoint.pt",
            "checkpoint_abc.pt",
            "best_model.pt",
            "checkpoint_00000001.onnx",
            "checkpoint_1.pt",  # not zero-padded to 8, but still digits
            "not_a_checkpoint_00000001.pt",
            "checkpoint_00000001.pt.bak",
        ],
    )
    def test_invalid_filenames_do_not_match(self, filename: str) -> None:
        # Note: "checkpoint_1.pt" actually matches because \d+ is greedy.
        # We test the actual regex behavior. Let's check.
        match = _CHECKPOINT_PATTERN.match(filename)
        # The regex is ^checkpoint_(\d+)\.pt$ so "checkpoint_1.pt" DOES match.
        # Skip that one from "no match" list and handle separately.
        if filename == "checkpoint_1.pt":
            assert match is not None  # This actually matches the regex
        else:
            assert match is None

    def test_extracts_iteration_number(self) -> None:
        match = _CHECKPOINT_PATTERN.match("checkpoint_00042000.pt")
        assert match is not None
        assert int(match.group(1)) == 42000


# -------------------------------------------------------------------
# Integration: save, rotate, load latest
# -------------------------------------------------------------------


class TestEndToEnd:
    """Integration-style tests combining multiple operations."""

    def test_save_rotate_load_latest(self, ckpt_dir: Path) -> None:
        config = CheckpointConfig(
            checkpoint_dir=str(ckpt_dir),
            keep_last_n=2,
            save_best=True,
            best_metric_mode="min",
        )
        mgr = CheckpointManager(config)
        states: dict[str, Any] = {"w": torch.randn(3, 3)}
        opt: dict[str, Any] = {"lr": 0.001}

        # Save 5 regular checkpoints + best
        for i in range(5):
            mgr.save(iteration=i, network_state=states, optimizer_state=opt)
            mgr.save_best(
                metric_value=1.0 / (i + 1),
                network_state=states,
                iteration=i,
            )

        # Only 2 regular checkpoints should remain
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 2

        # Load latest should give iteration 4
        loaded = mgr.load()
        assert loaded["iteration"] == 4

        # Best model should still exist with best metric
        best_path = ckpt_dir / _BEST_FILENAME
        assert best_path.exists()
        best_loaded = torch.load(
            best_path, map_location="cpu", weights_only=False,
        )
        assert best_loaded["best_metric_value"] == pytest.approx(0.2)
        assert best_loaded["iteration"] == 4

    def test_multiple_managers_same_directory(
        self, ckpt_dir: Path,
    ) -> None:
        """Simulate resume: a second manager sees earlier checkpoints."""
        config = CheckpointConfig(
            checkpoint_dir=str(ckpt_dir),
            keep_last_n=5,
            save_best=False,
        )
        mgr1 = CheckpointManager(config)
        mgr1.save(iteration=1, network_state={"v": 1}, optimizer_state={})
        mgr1.save(iteration=2, network_state={"v": 2}, optimizer_state={})

        mgr2 = CheckpointManager(config)
        loaded = mgr2.load()
        assert loaded["iteration"] == 2
        assert len(mgr2.list_checkpoints()) == 2
