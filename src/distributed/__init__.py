"""Distributed training infrastructure for AlphaGalerkin.

This module provides multi-node training capabilities using PyTorch's
distributed training utilities with NCCL backend for gradient synchronization.

Key Components:
    - DistributedConfig: Pydantic configuration for distributed settings
    - DistributedTrainer: Coordinator for distributed training
    - SelfPlayCoordinator: Distributed self-play game generation
    - GradientSynchronizer: NCCL-based gradient aggregation
    - ModelZoo: Checkpoint management for curriculum learning

Usage:
    from src.distributed import DistributedConfig, DistributedTrainer

    config = DistributedConfig(world_size=4, backend="nccl")
    trainer = DistributedTrainer(model, config)
    trainer.setup(rank=0, world_size=4)
    trainer.train()
"""

from src.distributed.config import (
    DistributedConfig,
    LauncherConfig,
    SelfPlayDistributedConfig,
)
from src.distributed.gradient_sync import GradientSynchronizer
from src.distributed.launcher import DistributedLauncher
from src.distributed.model_zoo import ModelZoo, ModelZooConfig
from src.distributed.trainer import DistributedTrainer
from src.distributed.worker import SelfPlayCoordinator

__all__ = [
    "DistributedConfig",
    "DistributedTrainer",
    "GradientSynchronizer",
    "LauncherConfig",
    "DistributedLauncher",
    "ModelZoo",
    "ModelZooConfig",
    "SelfPlayCoordinator",
    "SelfPlayDistributedConfig",
]
