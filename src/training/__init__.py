"""AlphaGalerkin Training Module.

This module provides training infrastructure for the AlphaGalerkin
resolution-independent Go AI, including:

- Loss computation (policy + value + LBB regularization)
- Replay buffer with priority sampling
- Self-play game generation
- Training loop orchestration
- Checkpoint management
"""

from __future__ import annotations

from src.training.checkpoint import CheckpointManager, CheckpointState
from src.training.evaluation import EvaluationResult, Evaluator, quick_evaluate
from src.training.loss import AlphaGalerkinLoss, EntropyRegularizer, LossOutput
from src.training.replay_buffer import (
    Experience,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    UniformReplayBuffer,
    create_replay_buffer,
)
from src.training.self_play import GameRecord, SelfPlayWorker
from src.training.trainer import Trainer, TrainingMetrics, create_trainer

__all__ = [
    # Loss
    "AlphaGalerkinLoss",
    "EntropyRegularizer",
    "LossOutput",
    # Replay Buffer
    "Experience",
    "ReplayBuffer",
    "UniformReplayBuffer",
    "PrioritizedReplayBuffer",
    "create_replay_buffer",
    # Self-play
    "GameRecord",
    "SelfPlayWorker",
    # Trainer
    "Trainer",
    "TrainingMetrics",
    "create_trainer",
    # Checkpoint
    "CheckpointManager",
    "CheckpointState",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "quick_evaluate",
]
