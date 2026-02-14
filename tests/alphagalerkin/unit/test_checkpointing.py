"""Tests for checkpoint management.

Note: The AlphaGalerkin checkpoint infrastructure uses torch.save/load
with the CheckpointConfig. These tests verify the config validation
and basic save/load patterns via the existing utils.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.alphagalerkin.core.config import CheckpointConfig
from src.alphagalerkin.core.exceptions import CheckpointError


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
        config = CheckpointConfig(
            resume_from="/path/to/ckpt.pt",
        )
        assert config.resume_from == "/path/to/ckpt.pt"


class TestCheckpointSaveLoad:
    """Tests for manual checkpoint save/load via torch."""

    def test_save_and_load_state_dict(
        self, tmp_path: Path,
    ) -> None:
        """Verify basic torch save/load roundtrip."""
        path = tmp_path / "ckpt.pt"
        state = {
            "iteration": 42,
            "network_state_dict": {"weight": 1.0},
            "optimizer_state_dict": {"lr": 0.001},
            "metrics": {"loss": 0.5},
        }
        torch.save(state, path)
        loaded = torch.load(
            path, map_location="cpu", weights_only=False,
        )
        assert loaded["iteration"] == 42
        assert loaded["network_state_dict"]["weight"] == 1.0

    def test_load_nonexistent_raises(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            torch.load(str(path))

    def test_multiple_checkpoints(
        self, tmp_path: Path,
    ) -> None:
        """Save multiple checkpoints, load the latest."""
        for i in range(5):
            path = tmp_path / f"ckpt_{i:04d}.pt"
            torch.save({"iteration": i}, path)

        ckpts = sorted(tmp_path.glob("ckpt_*.pt"))
        latest = ckpts[-1]
        loaded = torch.load(
            latest, map_location="cpu", weights_only=False,
        )
        assert loaded["iteration"] == 4

    def test_checkpoint_keeps_n_latest(
        self, tmp_path: Path,
    ) -> None:
        """Simulate keep_last_n rotation."""
        keep_n = 2
        paths: list[Path] = []
        for i in range(5):
            path = tmp_path / f"ckpt_{i:04d}.pt"
            torch.save({"iteration": i}, path)
            paths.append(path)
            # Rotate: delete oldest if we exceed keep_n
            ckpts = sorted(tmp_path.glob("ckpt_*.pt"))
            while len(ckpts) > keep_n:
                ckpts[0].unlink()
                ckpts = ckpts[1:]

        remaining = list(tmp_path.glob("ckpt_*.pt"))
        assert len(remaining) == keep_n


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
