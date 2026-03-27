"""Coverage tests for video compression trainer.

Targets uncovered lines in src/video_compression/training/trainer.py:
    - VideoCompressionTrainer init
    - train_step / eval_step
    - save_checkpoint / load_checkpoint
    - TrainingState / TrainingMetrics
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.video_compression.training.trainer import (
    TrainingMetrics,
    TrainingState,
    VideoCompressionTrainer,
)


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class _FakeCodec(nn.Module):
    """Minimal codec mock that behaves like VideoCodec for training."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)  # Needs parameters

    def forward(self, x: torch.Tensor):
        # Return (reconstructed, rate, distortion)
        return x, torch.tensor(0.5), torch.tensor(0.01)


def _fake_training_config():
    """Create a minimal TrainingConfig-like object via mock."""
    config = MagicMock()
    config.device = "cpu"
    config.lambda_rd = 0.01
    config.lambda_values = [0.01, 0.02]
    config.distortion_metric = "mse"
    config.ms_ssim_weight = 0.84
    config.use_perceptual_loss = False
    config.perceptual_weight = 0.0
    config.learning_rate = 1e-4
    config.weight_decay = 0.0
    config.total_steps = 10
    config.warmup_steps = 2
    config.use_amp = False
    config.gradient_clip = 1.0
    config.eval_interval = 5
    config.checkpoint_interval = 5
    config.model_dump = MagicMock(return_value={"lr": 1e-4})
    return config


# ---------------------------------------------------------------------------
# Tests: dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_training_state_defaults(self) -> None:
        state = TrainingState()
        assert state.step == 0
        assert state.epoch == 0
        assert state.best_rd_loss == float("inf")
        assert state.lambda_idx == 0

    def test_training_metrics(self) -> None:
        m = TrainingMetrics(loss=0.5, rate=0.3, distortion=0.2, psnr=30.0)
        assert m.loss == 0.5
        assert m.ms_ssim is None
        assert m.lr is None


# ---------------------------------------------------------------------------
# Tests: VideoCompressionTrainer init
# ---------------------------------------------------------------------------


class TestTrainerInit:
    def test_init_cpu(self, tmp_path: Path) -> None:
        codec = _FakeCodec()
        config = _fake_training_config()

        trainer = VideoCompressionTrainer(
            codec=codec, config=config, output_dir=tmp_path
        )

        assert trainer.device == torch.device("cpu")
        assert trainer.state.step == 0
        assert trainer.output_dir == tmp_path


# ---------------------------------------------------------------------------
# Tests: train_step / eval_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def _make_trainer(self, tmp_path: Path) -> VideoCompressionTrainer:
        codec = _FakeCodec()
        config = _fake_training_config()
        return VideoCompressionTrainer(
            codec=codec, config=config, output_dir=tmp_path
        )

    def test_train_step_returns_metrics(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        batch = torch.randn(1, 3, 16, 16)
        metrics = trainer.train_step(batch)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss >= 0
        assert metrics.lr is not None
        assert trainer.state.step == 1

    def test_warmup_lr(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        batch = torch.randn(1, 3, 16, 16)
        # Step 0: warmup_factor = 0/2 = 0
        metrics = trainer.train_step(batch)
        # Should have set lr according to warmup factor (step was 0 when computed)
        # After step increments to 1
        assert trainer.state.step == 1

    def test_eval_step(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        batch = torch.randn(1, 3, 16, 16)
        metrics = trainer.eval_step(batch)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss >= 0

    def test_lambda_cycling(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        batch = torch.randn(1, 3, 16, 16)

        # Run 100 steps to trigger lambda cycling
        for _ in range(100):
            trainer.train_step(batch)

        assert trainer.state.lambda_idx == 1  # Cycled once


# ---------------------------------------------------------------------------
# Tests: save/load checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_save_checkpoint(self, tmp_path: Path) -> None:
        codec = _FakeCodec()
        config = _fake_training_config()
        trainer = VideoCompressionTrainer(
            codec=codec, config=config, output_dir=tmp_path
        )

        path = trainer.save_checkpoint("test.pt")
        assert path.exists()

    def test_load_checkpoint(self, tmp_path: Path) -> None:
        codec = _FakeCodec()
        config = _fake_training_config()
        trainer = VideoCompressionTrainer(
            codec=codec, config=config, output_dir=tmp_path
        )

        # Do a step, save, reset, load
        batch = torch.randn(1, 3, 16, 16)
        trainer.train_step(batch)
        save_path = trainer.save_checkpoint("ckpt.pt")

        # Create fresh trainer and load
        trainer2 = VideoCompressionTrainer(
            codec=_FakeCodec(), config=config, output_dir=tmp_path
        )
        trainer2.load_checkpoint(save_path)
        assert trainer2.state.step == 1
