"""Curriculum Learning module for AlphaGalerkin.

Provides progressive training with:
- Board size progression (9x9 → 13x13 → 19x19)
- Performance-based stage transitions
- Opponent selection strategies
- Training schedule management
"""

from __future__ import annotations

from src.curriculum.config import (
    CurriculumConfig,
    ProgressionCriterion,
    ProgressionOperator,
    StageConfig,
)
from src.curriculum.manager import CurriculumManager
from src.curriculum.scheduler import CurriculumScheduler
from src.curriculum.stage import CurriculumStage, StageStatus

__all__ = [
    # Configuration
    "CurriculumConfig",
    "StageConfig",
    "ProgressionCriterion",
    "ProgressionOperator",
    # Stage management
    "CurriculumStage",
    "StageStatus",
    # Scheduler
    "CurriculumScheduler",
    # Manager
    "CurriculumManager",
]
