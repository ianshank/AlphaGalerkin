"""Tests for the main Trainer class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.modeling.model import AlphaGalerkinModel
from src.training.trainer import Trainer, TrainingMetrics, create_trainer


def _make_fake_experiences(trainer: Trainer, n: int = 10) -> list:
    """Create fake experiences matching the model's expected input shape."""
    from src.training.replay_buffer import Experience

    board_size = 9
    input_channels = trainer.config.operator.input_channels
    action_space = board_size * board_size + 1

    return [
        Experience(
            board_state=torch.randn(input_channels, board_size, board_size),
            board_size=board_size,
            target_policy=torch.softmax(torch.randn(action_space), dim=0),
            target_value=float(torch.randn(1).tanh().item()),
        )
        for _ in range(n)
    ]


def _prefill_and_mock(trainer: Trainer, n: int = 100):
    """Pre-fill buffer and return a context manager that mocks self-play.

    Usage::

        trainer = Trainer(...)
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3)
    """
    from contextlib import contextmanager

    for exp in _make_fake_experiences(trainer, n):
        trainer.buffer.add(exp)

    @contextmanager
    def _ctx():
        fake = _make_fake_experiences(trainer, 5)
        with (
            patch.object(trainer, "_fill_buffer"),
            patch.object(
                trainer.self_play_worker,
                "generate_experiences",
                return_value=fake,
            ),
        ):
            yield

    return _ctx()


@pytest.fixture
def small_config() -> AlphaGalerkinConfig:
    """Create small config for fast testing."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=OperatorConfig(
            d_model=32,
            d_key=16,
            d_value=16,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
        ),
        mcts=MCTSConfig(
            n_simulations=5,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=4,
            gradient_clip=1.0,
            lr_scheduler="constant",
            warmup_steps=0,
            total_steps=5,
            n_self_play_games=2,
            replay_buffer_size=50,
            checkpoint_interval=3,
            use_amp=False,
        ),
        experiment_name="test",
        seed=42,
    )


@pytest.fixture
def small_model(small_config: AlphaGalerkinConfig) -> AlphaGalerkinModel:
    """Create small model."""
    return AlphaGalerkinModel(small_config.operator)


@pytest.fixture
def checkpoint_dir() -> Path:
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test trainer initialization."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        assert trainer.model is small_model
        assert trainer.global_step == 0
        assert trainer.device == torch.device("cpu")

    def test_training_step_increments(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that training steps increment correctly."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        initial_step = trainer.global_step
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

        assert trainer.global_step == initial_step + 3

    def test_metrics_logged(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that metrics are logged during training."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

        history = trainer.get_metrics_history()
        assert len(history) == 3

        # Check metrics structure
        first_metrics = history[0]
        assert "total_loss" in first_metrics
        assert "policy_loss" in first_metrics
        assert "value_loss" in first_metrics
        assert "learning_rate" in first_metrics

    def test_checkpoint_saved(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that checkpoints are saved at intervals."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=5, log_interval=1, checkpoint_interval=2)

        # Should have checkpoints
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 1

    def test_resume_from_checkpoint(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test resuming training from checkpoint."""
        # Initial training
        trainer1 = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        with _prefill_and_mock(trainer1):
            trainer1.train(n_steps=3, log_interval=1, checkpoint_interval=1)
        saved_step = trainer1.global_step

        # Create new trainer and resume
        new_model = AlphaGalerkinModel(small_config.operator)
        trainer2 = Trainer(
            model=new_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        trainer2.load_checkpoint()

        assert trainer2.global_step == saved_step

    def test_lr_schedule_applied(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that learning rate schedule is applied."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        initial_lr = trainer.get_current_lr()
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)
        final_lr = trainer.get_current_lr()

        # With constant scheduler, LR should be same
        # With cosine, it would decrease
        assert final_lr > 0


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = TrainingMetrics(
            step=100,
            total_loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
            lbb_loss=0.0,
            learning_rate=1e-4,
        )

        d = metrics.to_dict()

        assert d["step"] == 100
        assert d["total_loss"] == 0.5
        assert d["learning_rate"] == 1e-4


class TestCreateTrainer:
    """Tests for create_trainer factory function."""

    def test_create_trainer_basic(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating trainer with factory function."""
        trainer = create_trainer(
            model=small_model,
            config=small_config,
            checkpoint_dir=checkpoint_dir,
            device="cpu",
        )

        assert isinstance(trainer, Trainer)
        assert trainer.global_step == 0

    def test_create_trainer_with_resume(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating trainer with checkpoint resumption."""
        # First, create and save a checkpoint
        trainer1 = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        with _prefill_and_mock(trainer1):
            trainer1.train(n_steps=2, log_interval=1, checkpoint_interval=1)
        ckpt_path = trainer1.checkpoint_manager.get_latest()

        # Create new trainer resuming from checkpoint
        new_model = AlphaGalerkinModel(small_config.operator)
        trainer2 = create_trainer(
            model=new_model,
            config=small_config,
            checkpoint_dir=checkpoint_dir,
            resume_from=ckpt_path,
            device="cpu",
        )

        assert trainer2.global_step == trainer1.global_step
