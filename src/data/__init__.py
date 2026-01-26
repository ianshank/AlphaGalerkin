"""AlphaGalerkin Data Module.

This module provides data loading and preprocessing for training,
including:

- Dataset classes for replay buffer data
- Variable-size board collation
- Data augmentation utilities
"""

from __future__ import annotations

from src.data.collate import (
    SameSizeCollator,
    TrainingBatch,
    VariableSizeCollator,
    create_collator,
)
from src.data.dataset import (
    AugmentedExperience,
    BoardSizeBatchSampler,
    ExperienceListDataset,
    ReplayDataset,
    StreamingReplayDataset,
)

__all__ = [
    # Collation
    "TrainingBatch",
    "VariableSizeCollator",
    "SameSizeCollator",
    "create_collator",
    # Datasets
    "ReplayDataset",
    "StreamingReplayDataset",
    "ExperienceListDataset",
    "BoardSizeBatchSampler",
    "AugmentedExperience",
]
