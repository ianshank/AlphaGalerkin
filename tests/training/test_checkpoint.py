"""Tests for checkpoint management."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from config.schemas import AlphaGalerkinConfig, OperatorConfig
from src.training.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointManager,
    CheckpointState,
    create_model_from_checkpoint,
    load_checkpoint_with_config,
    load_model_only,
    save_model_only,
)


@pytest.fixture
def small_model() -> nn.Module:
    """Create small model for testing."""
    from src.modeling.model import AlphaGalerkinModel

    config = OperatorConfig(
        d_model=32,
        d_key=16,
        d_value=16,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=16,
        use_fnet_mixing=False,
    )
    return AlphaGalerkinModel(config)


@pytest.fixture
def checkpoint_dir() -> Path:
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCheckpointState:
    """Tests for CheckpointState."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization roundtrip."""
        state = CheckpointState(
            step=100,
            model_state_dict={"layer.weight": torch.randn(10, 10)},
            optimizer_state_dict={"param_groups": []},
            scheduler_state_dict={"last_epoch": 99},
            config={"d_model": 256},
            metrics={"loss": 0.5},
            timestamp="2024-01-01T00:00:00",
        )

        state_dict = state.to_dict()
        restored = CheckpointState.from_dict(state_dict)

        assert restored.step == 100
        assert restored.metrics["loss"] == 0.5
        assert restored.version == CHECKPOINT_VERSION

    def test_from_dict_defaults(self) -> None:
        """Test from_dict with minimal data (missing optional fields)."""
        data = {
            "step": 50,
            "model_state_dict": {"w": torch.zeros(2)},
            "version": CHECKPOINT_VERSION,
        }
        state = CheckpointState.from_dict(data)
        assert state.step == 50
        assert state.optimizer_state_dict is None
        assert state.scheduler_state_dict is None
        assert state.config is None
        assert state.metrics == {}
        assert state.timestamp == ""


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_checkpoint(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test saving a checkpoint."""
        manager = CheckpointManager(checkpoint_dir)

        optimizer = torch.optim.Adam(small_model.parameters())

        path = manager.save(
            step=100,
            model=small_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )

        assert path.exists()
        assert "checkpoint_00000100" in path.name

    def test_load_checkpoint(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading a checkpoint."""
        manager = CheckpointManager(checkpoint_dir)
        optimizer = torch.optim.Adam(small_model.parameters())

        # Save
        manager.save(step=100, model=small_model, optimizer=optimizer)

        # Load
        state = manager.load()

        assert state.step == 100
        assert "policy_head.net.0.weight" in state.model_state_dict

    def test_restore_model(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test restoring model state."""
        manager = CheckpointManager(checkpoint_dir)
        optimizer = torch.optim.Adam(small_model.parameters())

        # Save original state
        original_weight = small_model.policy_head.net[0].weight.clone()
        manager.save(step=100, model=small_model, optimizer=optimizer)

        # Modify model
        with torch.no_grad():
            small_model.policy_head.net[0].weight.fill_(0.0)

        assert not torch.allclose(
            small_model.policy_head.net[0].weight,
            original_weight,
        )

        # Restore
        step = manager.restore(model=small_model, optimizer=optimizer)

        assert step == 100
        assert torch.allclose(
            small_model.policy_head.net[0].weight,
            original_weight,
        )

    def test_get_latest(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test getting latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir)

        # Save multiple checkpoints
        manager.save(step=100, model=small_model)
        manager.save(step=200, model=small_model)
        manager.save(step=150, model=small_model)

        latest = manager.get_latest()

        assert latest is not None
        assert "00000200" in latest.name

    def test_checkpoint_rotation(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test that old checkpoints are rotated."""
        manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)

        # Save more than max
        for step in range(0, 500, 100):
            manager.save(step=step, model=small_model)

        checkpoints = manager.get_all_checkpoints()

        # Should only keep max_checkpoints
        assert len(checkpoints) == 3

        # Should keep most recent
        steps = [int(p.stem.split("_")[1]) for p in checkpoints]
        assert 400 in steps
        assert 300 in steps
        assert 200 in steps

    def test_best_checkpoint_tracking(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test best checkpoint tracking."""
        manager = CheckpointManager(
            checkpoint_dir,
            keep_best=True,
            best_metric="loss",
            best_mode="min",
        )

        # Save with different losses
        manager.save(step=100, model=small_model, metrics={"loss": 0.5})
        manager.save(step=200, model=small_model, metrics={"loss": 0.3})  # Best
        manager.save(step=300, model=small_model, metrics={"loss": 0.4})

        best_path = checkpoint_dir / "best.pt"
        assert best_path.exists()

        # Load best and check it's step 200
        state = manager.load(load_best=True)
        assert state.step == 200

    def test_load_nonexistent_raises(self, checkpoint_dir: Path) -> None:
        """Test that loading nonexistent checkpoint raises error."""
        manager = CheckpointManager(checkpoint_dir)

        with pytest.raises(FileNotFoundError):
            manager.load()

    def test_version_compatibility(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test version compatibility checking."""
        manager = CheckpointManager(checkpoint_dir)
        manager.save(step=100, model=small_model)

        # Modify saved checkpoint to have incompatible version
        ckpt_path = manager.get_latest()
        data = torch.load(ckpt_path, weights_only=False)
        data["version"] = "999.0.0"
        torch.save(data, ckpt_path)

        with pytest.raises(ValueError, match="not compatible"):
            manager.load()

    # --- Tests for previously uncovered lines ---

    def test_load_specific_path_not_found(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading a specific path that does not exist raises FileNotFoundError (line 240)."""
        manager = CheckpointManager(checkpoint_dir)
        fake_path = checkpoint_dir / "nonexistent_checkpoint.pt"

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            manager.load(path=fake_path)

    def test_is_compatible_invalid_version_string(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test _is_compatible returns False for non-numeric version (lines 395-396)."""
        manager = CheckpointManager(checkpoint_dir)
        # Invalid version strings should return False
        assert manager._is_compatible("abc.def.ghi") is False
        assert manager._is_compatible("") is False
        assert manager._is_compatible("not_a_version") is False

    def test_save_metadata(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test saving metadata to checkpoint directory (lines 405-407)."""
        manager = CheckpointManager(checkpoint_dir)
        metadata = {"experiment": "test", "notes": "hello", "epoch": 5}
        manager.save_metadata(metadata)

        metadata_path = checkpoint_dir / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            loaded = json.load(f)
        assert loaded["experiment"] == "test"
        assert loaded["notes"] == "hello"
        assert loaded["epoch"] == 5

    def test_load_metadata_exists(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading existing metadata (lines 416-420)."""
        manager = CheckpointManager(checkpoint_dir)
        metadata = {"key": "value", "count": 42}
        manager.save_metadata(metadata)

        loaded = manager.load_metadata()
        assert loaded["key"] == "value"
        assert loaded["count"] == 42

    def test_load_metadata_not_exists(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading metadata when file does not exist returns empty dict (line 420)."""
        manager = CheckpointManager(checkpoint_dir)
        loaded = manager.load_metadata()
        assert loaded == {}

    def test_best_checkpoint_max_mode(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test best checkpoint tracking with max mode."""
        manager = CheckpointManager(
            checkpoint_dir,
            keep_best=True,
            best_metric="accuracy",
            best_mode="max",
        )

        manager.save(step=100, model=small_model, metrics={"accuracy": 0.8})
        manager.save(step=200, model=small_model, metrics={"accuracy": 0.95})  # Best
        manager.save(step=300, model=small_model, metrics={"accuracy": 0.9})

        state = manager.load(load_best=True)
        assert state.step == 200

    def test_save_without_metrics_no_best_update(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test that save without the best_metric key does not update best."""
        manager = CheckpointManager(
            checkpoint_dir,
            keep_best=True,
            best_metric="loss",
            best_mode="min",
        )
        manager.save(step=100, model=small_model, metrics={"other": 1.0})
        best_path = checkpoint_dir / "best.pt"
        assert not best_path.exists()

    def test_restore_with_scheduler(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test restoring with scheduler state."""
        manager = CheckpointManager(checkpoint_dir)
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Step scheduler a few times
        for _ in range(5):
            scheduler.step()

        original_last_epoch = scheduler.last_epoch
        manager.save(
            step=50,
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Reset scheduler
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        assert scheduler2.last_epoch != original_last_epoch

        step = manager.restore(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler2,
        )
        assert step == 50
        assert scheduler2.last_epoch == original_last_epoch

    def test_save_with_config(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test saving checkpoint with config."""
        manager = CheckpointManager(checkpoint_dir)
        config = AlphaGalerkinConfig()
        path = manager.save(step=10, model=small_model, config=config)

        state = manager.load(path=path)
        assert state.config is not None
        assert "operator" in state.config

    def test_get_latest_empty(self, checkpoint_dir: Path) -> None:
        """Test get_latest returns None when no checkpoints exist."""
        manager = CheckpointManager(checkpoint_dir)
        assert manager.get_latest() is None

    def test_get_all_checkpoints_empty(self, checkpoint_dir: Path) -> None:
        """Test get_all_checkpoints returns empty list when none exist."""
        manager = CheckpointManager(checkpoint_dir)
        assert manager.get_all_checkpoints() == []


class TestSaveLoadModelOnly:
    """Tests for save/load model only functions."""

    def test_save_load_roundtrip(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test save and load model only."""
        path = checkpoint_dir / "model.pt"

        # Save original state
        original_weight = small_model.policy_head.net[0].weight.clone()
        save_model_only(small_model, path)

        # Modify model
        with torch.no_grad():
            small_model.policy_head.net[0].weight.fill_(0.0)

        # Load
        load_model_only(small_model, path)

        assert torch.allclose(
            small_model.policy_head.net[0].weight,
            original_weight,
        )

    def test_save_model_only_with_config(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test save_model_only with a config object."""
        path = checkpoint_dir / "model_with_config.pt"
        config = AlphaGalerkinConfig()
        save_model_only(small_model, path, config=config)

        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["config"] is not None
        assert "operator" in data["config"]
        assert data["version"] == CHECKPOINT_VERSION
        assert "timestamp" in data

    def test_load_model_only_fallback_to_legacy(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test load_model_only falls back to weights_only=False on failure (lines 463-470).

        We mock torch.load so the first call (weights_only=True) raises,
        forcing the fallback path that uses weights_only=False.
        """
        path = checkpoint_dir / "legacy_model.pt"
        save_model_only(small_model, path)

        original_weight = small_model.policy_head.net[0].weight.clone()

        # Modify model
        with torch.no_grad():
            small_model.policy_head.net[0].weight.fill_(0.0)

        original_torch_load = torch.load

        call_count = 0

        def patched_load(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if kwargs.get("weights_only", False) is True:
                raise RuntimeError("Simulated weights_only failure")
            return original_torch_load(*args, **kwargs)

        with patch("src.training.checkpoint.torch.load", side_effect=patched_load):
            load_model_only(small_model, path)

        # Verify the fallback was actually used (at least 2 calls)
        assert call_count >= 2

        assert torch.allclose(
            small_model.policy_head.net[0].weight,
            original_weight,
        )

    def test_load_model_only_non_strict(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test load_model_only with strict=False."""
        path = checkpoint_dir / "model.pt"
        save_model_only(small_model, path)

        # Create a model that has a subset of keys (non-strict load)
        load_model_only(small_model, path, strict=False)


class TestLoadCheckpointWithConfig:
    """Tests for load_checkpoint_with_config (lines 506-526)."""

    def test_load_with_config(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading checkpoint that contains config."""
        path = checkpoint_dir / "full_ckpt.pt"
        config = AlphaGalerkinConfig()
        state = {
            "model_state_dict": small_model.state_dict(),
            "config": config.model_dump(),
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        checkpoint, config_dict = load_checkpoint_with_config(path)
        assert checkpoint is not None
        assert config_dict is not None
        assert "operator" in config_dict

    def test_load_without_config(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading checkpoint that has no config field."""
        path = checkpoint_dir / "no_config_ckpt.pt"
        state = {
            "model_state_dict": small_model.state_dict(),
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        checkpoint, config_dict = load_checkpoint_with_config(path)
        assert checkpoint is not None
        assert config_dict is None

    def test_load_nonexistent_path(self, checkpoint_dir: Path) -> None:
        """Test FileNotFoundError for nonexistent path."""
        fake_path = checkpoint_dir / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint_with_config(fake_path)

    def test_load_corrupted_file(self, checkpoint_dir: Path) -> None:
        """Test RuntimeError for corrupted checkpoint file."""
        path = checkpoint_dir / "corrupted.pt"
        path.write_bytes(b"this is not a valid pytorch checkpoint")

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_checkpoint_with_config(path)

    def test_load_with_device(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test loading checkpoint with explicit device argument."""
        path = checkpoint_dir / "ckpt_device.pt"
        state = {
            "model_state_dict": small_model.state_dict(),
            "config": {"operator": {"d_model": 32}},
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        checkpoint, config_dict = load_checkpoint_with_config(path, device="cpu")
        assert checkpoint is not None
        assert config_dict is not None


class TestCreateModelFromCheckpoint:
    """Tests for create_model_from_checkpoint (lines 569-617)."""

    def test_create_from_checkpoint_with_config(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating model from checkpoint that has operator config."""
        path = checkpoint_dir / "model_ckpt.pt"
        op_config = OperatorConfig(
            d_model=32,
            d_key=16,
            d_value=16,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
        )
        full_config = AlphaGalerkinConfig(operator=op_config)

        state = {
            "model_state_dict": small_model.state_dict(),
            "config": full_config.model_dump(),
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        model, config_dict = create_model_from_checkpoint(path, device="cpu")
        assert model is not None
        assert config_dict is not None
        assert "operator" in config_dict
        # Model should be in eval mode
        assert not model.training

    def test_create_from_checkpoint_default_config(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating model from checkpoint without config uses defaults (line 594-595)."""
        from src.modeling.model import AlphaGalerkinModel

        # Create a model with default config, save checkpoint without config
        default_config = OperatorConfig()
        model = AlphaGalerkinModel(default_config)

        path = checkpoint_dir / "no_config_ckpt.pt"
        state = {
            "model_state_dict": model.state_dict(),
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        loaded_model, config_dict = create_model_from_checkpoint(path, device="cpu")
        assert loaded_model is not None
        assert config_dict is None
        assert not loaded_model.training

    def test_create_from_checkpoint_legacy_format(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating model from legacy checkpoint (state dict directly, line 604)."""
        from src.modeling.model import AlphaGalerkinModel

        default_config = OperatorConfig()
        model = AlphaGalerkinModel(default_config)

        path = checkpoint_dir / "legacy_ckpt.pt"
        # Legacy format: checkpoint IS the state dict (no "model_state_dict" key)
        state_dict = model.state_dict()
        torch.save(state_dict, path)

        loaded_model, config_dict = create_model_from_checkpoint(path, device="cpu")
        assert loaded_model is not None
        assert config_dict is None
        assert not loaded_model.training

    def test_create_from_checkpoint_with_explicit_classes(
        self,
        small_model: nn.Module,
        checkpoint_dir: Path,
    ) -> None:
        """Test create_model_from_checkpoint with explicit model_class and config_class."""
        from src.modeling.model import AlphaGalerkinModel

        op_config = OperatorConfig(
            d_model=32,
            d_key=16,
            d_value=16,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
        )
        full_config = AlphaGalerkinConfig(operator=op_config)

        path = checkpoint_dir / "explicit_class_ckpt.pt"
        state = {
            "model_state_dict": small_model.state_dict(),
            "config": full_config.model_dump(),
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        model, config_dict = create_model_from_checkpoint(
            path,
            device="cpu",
            model_class=AlphaGalerkinModel,
            config_class=OperatorConfig,
        )
        assert model is not None
        assert config_dict is not None
        assert not model.training

    def test_create_from_checkpoint_bad_operator_config_fallback(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test fallback to default config when operator config parsing fails (lines 587-592)."""
        from src.modeling.model import AlphaGalerkinModel

        # Create model with default config
        default_config = OperatorConfig()
        model = AlphaGalerkinModel(default_config)

        path = checkpoint_dir / "bad_op_config_ckpt.pt"
        state = {
            "model_state_dict": model.state_dict(),
            "config": {
                "operator": {
                    "d_model": "not_a_number",  # Invalid value to trigger parse failure
                    "invalid_field_xyz": True,
                }
            },
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        # Should fall back to default config and still create a model
        loaded_model, config_dict = create_model_from_checkpoint(path, device="cpu")
        assert loaded_model is not None
        assert config_dict is not None
        assert not loaded_model.training

    def test_create_from_checkpoint_nonexistent(self, checkpoint_dir: Path) -> None:
        """Test FileNotFoundError for nonexistent checkpoint."""
        fake_path = checkpoint_dir / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            create_model_from_checkpoint(fake_path)

    def test_create_from_checkpoint_config_without_operator_key(
        self,
        checkpoint_dir: Path,
    ) -> None:
        """Test config dict present but missing 'operator' key uses defaults (line 593-595)."""
        from src.modeling.model import AlphaGalerkinModel

        default_config = OperatorConfig()
        model = AlphaGalerkinModel(default_config)

        path = checkpoint_dir / "no_operator_ckpt.pt"
        state = {
            "model_state_dict": model.state_dict(),
            "config": {"training": {"batch_size": 32}},  # has config but no "operator" key
            "version": CHECKPOINT_VERSION,
        }
        torch.save(state, path)

        loaded_model, config_dict = create_model_from_checkpoint(path, device="cpu")
        assert loaded_model is not None
        assert config_dict is not None
        assert "training" in config_dict
        assert not loaded_model.training
