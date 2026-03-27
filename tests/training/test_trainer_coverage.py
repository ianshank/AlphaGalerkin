"""Coverage tests for the Trainer class.

Tests cover additional paths not covered by the existing test_trainer.py:
- TrainingMetrics: to_dict with physics fields
- Trainer: _create_optimizer, _create_scheduler variants,
  _create_loss_balancer, _create_physics_loss
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
from src.training.trainer import Trainer, TrainingMetrics

SEED = 42


@pytest.fixture
def small_operator_config() -> OperatorConfig:
    """Create small operator config for fast testing."""
    return OperatorConfig(
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


def _make_config(
    operator: OperatorConfig,
    lr_scheduler: str = "constant",
    warmup_steps: int = 0,
    loss_balancing_strategy: str = "relobralo",
    physics_informed: bool = False,
) -> AlphaGalerkinConfig:
    """Create a full config with customizable training params."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=operator,
        mcts=MCTSConfig(
            n_simulations=2,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=4,
            gradient_clip=1.0,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            total_steps=10,
            n_self_play_games=1,
            replay_buffer_size=50,
            checkpoint_interval=100,
            use_amp=False,
            loss_balancing_strategy=loss_balancing_strategy,
            physics_informed=physics_informed,
        ),
        experiment_name="test_coverage",
        seed=SEED,
    )


@pytest.fixture
def checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTrainingMetricsExtended:
    """Extended tests for TrainingMetrics."""

    def test_to_dict_without_physics(self) -> None:
        metrics = TrainingMetrics(
            step=50,
            total_loss=1.0,
            policy_loss=0.5,
            value_loss=0.3,
            lbb_loss=0.02,
            lbb_constant=0.1,
            learning_rate=1e-4,
            gradient_norm=0.5,
            buffer_size=100,
            games_generated=10,
            step_time_ms=25.0,
            physics_weight=0.0,  # No physics
        )
        d = metrics.to_dict()
        assert "physics_loss" not in d
        assert "physics_residual_loss" not in d

    def test_to_dict_with_physics(self) -> None:
        metrics = TrainingMetrics(
            step=50,
            total_loss=1.0,
            policy_loss=0.5,
            value_loss=0.3,
            lbb_loss=0.02,
            learning_rate=1e-4,
            physics_loss=0.1,
            physics_residual_loss=0.05,
            physics_boundary_loss=0.05,
            physics_weight=0.1,
        )
        d = metrics.to_dict()
        assert "physics_loss" in d
        assert d["physics_loss"] == 0.1
        assert d["physics_residual_loss"] == 0.05
        assert d["physics_boundary_loss"] == 0.05


class TestTrainerSchedulers:
    """Tests for different scheduler configurations."""

    def test_cosine_scheduler(
        self, small_operator_config: OperatorConfig, checkpoint_dir: Path
    ) -> None:
        config = _make_config(small_operator_config, lr_scheduler="cosine", warmup_steps=0)
        model = AlphaGalerkinModel(config.operator)
        trainer = Trainer(model=model, config=config, device="cpu", checkpoint_dir=checkpoint_dir)
        initial_lr = trainer.get_current_lr()
        trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)
        # Cosine should decrease LR
        assert trainer.get_current_lr() <= initial_lr + 1e-8

    def test_linear_scheduler(
        self, small_operator_config: OperatorConfig, checkpoint_dir: Path
    ) -> None:
        config = _make_config(small_operator_config, lr_scheduler="linear", warmup_steps=0)
        model = AlphaGalerkinModel(config.operator)
        trainer = Trainer(model=model, config=config, device="cpu", checkpoint_dir=checkpoint_dir)
        assert trainer.scheduler is not None

    def test_warmup_scheduler(
        self, small_operator_config: OperatorConfig, checkpoint_dir: Path
    ) -> None:
        config = _make_config(small_operator_config, lr_scheduler="cosine", warmup_steps=3)
        model = AlphaGalerkinModel(config.operator)
        trainer = Trainer(model=model, config=config, device="cpu", checkpoint_dir=checkpoint_dir)
        # During warmup LR should be low
        initial_lr = trainer.get_current_lr()
        assert initial_lr < config.training.learning_rate


class TestTrainerLossBalancer:
    """Tests for different loss balancing strategies."""

    @pytest.mark.parametrize(
        "strategy",
        ["static", "relobralo", "gradnorm", "uncertainty", "softadapt"],
    )
    def test_loss_balancer_strategies(
        self,
        strategy: str,
        small_operator_config: OperatorConfig,
        checkpoint_dir: Path,
    ) -> None:
        config = _make_config(small_operator_config, loss_balancing_strategy=strategy)
        model = AlphaGalerkinModel(config.operator)
        trainer = Trainer(model=model, config=config, device="cpu", checkpoint_dir=checkpoint_dir)
        assert trainer.loss_balancer is not None


class TestTrainerOptimizer:
    """Tests for optimizer creation."""

    def test_creates_adamw(
        self, small_operator_config: OperatorConfig, checkpoint_dir: Path
    ) -> None:
        config = _make_config(small_operator_config)
        model = AlphaGalerkinModel(config.operator)
        trainer = Trainer(model=model, config=config, device="cpu", checkpoint_dir=checkpoint_dir)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        # Check LR matches config
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == config.training.learning_rate


class TestTrainerPhysicsLoss:
    """Tests for physics loss integration."""

    def test_physics_loss_creation(
        self, small_operator_config: OperatorConfig, checkpoint_dir: Path
    ) -> None:
        config = _make_config(small_operator_config, physics_informed=True)
        model = AlphaGalerkinModel(config.operator)
        trainer = Trainer(model=model, config=config, device="cpu", checkpoint_dir=checkpoint_dir)
        # Physics loss may or may not be created depending on imports
        # But use_physics_loss flag should be set
        assert trainer.use_physics_loss is True
