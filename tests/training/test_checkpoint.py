"""Tests for checkpoint management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel
from src.training.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointManager,
    CheckpointState,
    load_model_only,
    save_model_only,
)


@pytest.fixture
def small_model() -> AlphaGalerkinModel:
    """Create small model for testing."""
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


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_checkpoint(
        self,
        small_model: AlphaGalerkinModel,
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
        small_model: AlphaGalerkinModel,
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
        small_model: AlphaGalerkinModel,
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
        small_model: AlphaGalerkinModel,
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
        small_model: AlphaGalerkinModel,
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
        small_model: AlphaGalerkinModel,
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
        small_model: AlphaGalerkinModel,
        checkpoint_dir: Path,
    ) -> None:
        """Test version compatibility checking."""
        manager = CheckpointManager(checkpoint_dir)
        manager.save(step=100, model=small_model)

        # Modify saved checkpoint to have incompatible version
        ckpt_path = manager.get_latest()
        data = torch.load(ckpt_path)
        data["version"] = "999.0.0"
        torch.save(data, ckpt_path)

        with pytest.raises(ValueError, match="not compatible"):
            manager.load()


class TestSaveLoadModelOnly:
    """Tests for save/load model only functions."""

    def test_save_load_roundtrip(
        self,
        small_model: AlphaGalerkinModel,
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
