"""Pytest fixtures for curriculum learning tests."""

from __future__ import annotations

import pytest

from src.curriculum.config import (
    CurriculumConfig,
    ProgressionCriterion,
    ProgressionOperator,
    StageConfig,
    create_default_curriculum,
)
from src.curriculum.manager import CurriculumManager
from src.curriculum.scheduler import CurriculumScheduler
from src.curriculum.stage import CurriculumStage


@pytest.fixture
def simple_stage_config() -> StageConfig:
    """Create a simple stage configuration."""
    return StageConfig(
        name="test_stage",
        board_size=9,
        min_games=10,
        min_steps=10,
        progression_criteria=[
            ProgressionCriterion(
                metric="win_rate",
                operator=ProgressionOperator.GREATER_EQUAL,
                threshold=0.6,
                min_samples=5,
            )
        ],
    )


@pytest.fixture
def two_stage_config() -> CurriculumConfig:
    """Create a two-stage curriculum configuration."""
    return CurriculumConfig(
        name="test_curriculum",
        stages=[
            StageConfig(
                name="stage_9x9",
                board_size=9,
                min_games=10,
                min_steps=5,
                progression_criteria=[
                    ProgressionCriterion(
                        metric="win_rate",
                        operator=ProgressionOperator.GREATER_EQUAL,
                        threshold=0.6,
                        min_samples=5,
                    )
                ],
            ),
            StageConfig(
                name="stage_13x13",
                board_size=13,
                min_games=10,
                min_steps=5,
                # No criteria - terminal stage
            ),
        ],
        warmup_games_per_stage=2,
        evaluation_interval=5,
    )


@pytest.fixture
def three_stage_config() -> CurriculumConfig:
    """Create a three-stage curriculum configuration."""
    return CurriculumConfig(
        name="full_curriculum",
        stages=[
            StageConfig(
                name="beginner",
                board_size=9,
                min_games=10,
                min_steps=5,
                progression_criteria=[
                    ProgressionCriterion(
                        metric="win_rate",
                        operator=ProgressionOperator.GREATER_EQUAL,
                        threshold=0.55,
                        min_samples=5,
                    )
                ],
            ),
            StageConfig(
                name="intermediate",
                board_size=13,
                min_games=10,
                min_steps=5,
                progression_criteria=[
                    ProgressionCriterion(
                        metric="win_rate",
                        operator=ProgressionOperator.GREATER_EQUAL,
                        threshold=0.55,
                        min_samples=5,
                    )
                ],
            ),
            StageConfig(
                name="advanced",
                board_size=19,
                min_games=10,
                min_steps=5,
            ),
        ],
        warmup_games_per_stage=0,
        evaluation_interval=3,
    )


@pytest.fixture
def curriculum_stage(simple_stage_config: StageConfig) -> CurriculumStage:
    """Create a curriculum stage instance."""
    return CurriculumStage(config=simple_stage_config)


@pytest.fixture
def curriculum_scheduler(two_stage_config: CurriculumConfig) -> CurriculumScheduler:
    """Create a curriculum scheduler instance."""
    return CurriculumScheduler(two_stage_config)


@pytest.fixture
def curriculum_manager(two_stage_config: CurriculumConfig) -> CurriculumManager:
    """Create a curriculum manager instance."""
    return CurriculumManager(config=two_stage_config)


@pytest.fixture
def default_curriculum_config() -> CurriculumConfig:
    """Create a default curriculum configuration."""
    return create_default_curriculum()
