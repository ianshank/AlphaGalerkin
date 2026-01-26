"""Pytest fixtures for training tests."""

from __future__ import annotations

import pytest
import torch

from config.schemas import (
    AlphaGalerkinConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
    WandbConfig,
)
from src.modeling.model import AlphaGalerkinModel
from src.training.replay_buffer import Experience


@pytest.fixture
def small_config() -> OperatorConfig:
    """Create small config for fast testing."""
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


@pytest.fixture
def small_model(small_config: OperatorConfig) -> AlphaGalerkinModel:
    """Create small model for testing."""
    return AlphaGalerkinModel(small_config)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create training config for testing."""
    return TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=4,
        gradient_clip=1.0,
        lr_scheduler="constant",
        warmup_steps=0,
        total_steps=10,
        n_self_play_games=2,
        replay_buffer_size=100,
        checkpoint_interval=5,
        use_amp=False,
    )


@pytest.fixture
def mcts_config() -> MCTSConfig:
    """Create MCTS config for testing."""
    return MCTSConfig(
        n_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_size=2,
    )


@pytest.fixture
def wandb_config() -> WandbConfig:
    """Create disabled W&B config for testing."""
    return WandbConfig(
        enabled=False,
        project="test-project",
        mode="disabled",
        log_code=False,
        log_model=False,
        watch_model=False,
    )


@pytest.fixture
def full_config(
    small_config: OperatorConfig,
    training_config: TrainingConfig,
    mcts_config: MCTSConfig,
    wandb_config: WandbConfig,
) -> AlphaGalerkinConfig:
    """Create full config for testing."""
    return AlphaGalerkinConfig(
        operator=small_config,
        training=training_config,
        mcts=mcts_config,
        wandb=wandb_config,
        experiment_name="test",
        seed=42,
        device="cpu",
        board_sizes=[9],
    )


@pytest.fixture
def sample_experience_9x9() -> Experience:
    """Create sample experience for 9x9 board."""
    board_size = 9
    n_channels = 17
    n_actions = board_size ** 2 + 1

    return Experience(
        board_state=torch.randn(n_channels, board_size, board_size),
        board_size=board_size,
        target_policy=torch.softmax(torch.randn(n_actions), dim=0),
        target_value=0.5,
        metadata={"test": True},
    )


@pytest.fixture
def sample_experience_19x19() -> Experience:
    """Create sample experience for 19x19 board."""
    board_size = 19
    n_channels = 17
    n_actions = board_size ** 2 + 1

    return Experience(
        board_state=torch.randn(n_channels, board_size, board_size),
        board_size=board_size,
        target_policy=torch.softmax(torch.randn(n_actions), dim=0),
        target_value=-0.3,
        metadata={"test": True},
    )


@pytest.fixture
def sample_experiences(
    sample_experience_9x9: Experience,
    sample_experience_19x19: Experience,
) -> list[Experience]:
    """Create list of sample experiences with mixed board sizes."""
    experiences = []
    for _ in range(3):
        experiences.append(sample_experience_9x9)
        experiences.append(sample_experience_19x19)
    return experiences
