"""Configuration schemas for Curriculum Learning.

Provides Pydantic-validated configuration with:
- No hardcoded values
- Validation constraints
- Hash computation for reproducibility
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProgressionOperator(str, Enum):
    """Operators for progression criterion evaluation."""

    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    EQUAL = "eq"

    def evaluate(self, value: float, threshold: float) -> bool:
        """Evaluate the operator against value and threshold.

        Args:
            value: Current metric value.
            threshold: Threshold to compare against.

        Returns:
            True if condition is satisfied.
        """
        if self == ProgressionOperator.GREATER_THAN:
            return value > threshold
        elif self == ProgressionOperator.GREATER_EQUAL:
            return value >= threshold
        elif self == ProgressionOperator.LESS_THAN:
            return value < threshold
        elif self == ProgressionOperator.LESS_EQUAL:
            return value <= threshold
        elif self == ProgressionOperator.EQUAL:
            return abs(value - threshold) < 1e-6
        return False


class ProgressionCriterion(BaseModel):
    """Single criterion for stage progression.

    Defines when a training stage should advance to the next.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    metric: str = Field(
        ...,
        min_length=1,
        description="Metric name to evaluate (e.g., 'win_rate', 'loss')",
    )
    operator: ProgressionOperator = Field(
        ...,
        description="Comparison operator",
    )
    threshold: float = Field(
        ...,
        description="Threshold value for progression",
    )
    min_samples: int = Field(
        default=10,
        ge=1,
        description="Minimum samples before criterion is evaluated",
    )
    window_size: int = Field(
        default=100,
        ge=1,
        description="Rolling window size for metric computation",
    )

    def is_satisfied(self, value: float, n_samples: int) -> bool:
        """Check if progression criterion is satisfied.

        Args:
            value: Current metric value.
            n_samples: Number of samples collected.

        Returns:
            True if criterion is satisfied.
        """
        if n_samples < self.min_samples:
            return False
        return self.operator.evaluate(value, self.threshold)


class StageConfig(BaseModel):
    """Configuration for a single curriculum stage.

    Each stage represents a training phase with specific
    board size and progression requirements.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        description="Stage name (e.g., 'beginner_9x9')",
    )
    board_size: int = Field(
        ...,
        ge=5,
        le=25,
        description="Board size for this stage",
    )
    min_games: int = Field(
        default=1000,
        ge=1,
        description="Minimum games before stage can complete",
    )
    max_games: int | None = Field(
        default=None,
        ge=1,
        description="Maximum games before forced progression (None for unlimited)",
    )
    min_steps: int = Field(
        default=1000,
        ge=1,
        description="Minimum training steps before stage can complete",
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        description="Maximum training steps before forced progression",
    )
    progression_criteria: list[ProgressionCriterion] = Field(
        default_factory=list,
        description="Criteria for advancing to next stage",
    )
    criteria_mode: str = Field(
        default="all",
        pattern="^(all|any)$",
        description="Mode for evaluating criteria: 'all' or 'any'",
    )
    learning_rate_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Learning rate multiplier for this stage",
    )
    batch_size_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Batch size multiplier for this stage",
    )
    mcts_simulations: int | None = Field(
        default=None,
        ge=1,
        description="MCTS simulations override (None for default)",
    )

    @model_validator(mode="after")
    def validate_max_greater_than_min(self) -> "StageConfig":
        """Validate that max values are greater than min values."""
        if self.max_games is not None and self.max_games < self.min_games:
            raise ValueError("max_games must be >= min_games")
        if self.max_steps is not None and self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")
        return self


class CurriculumConfig(BaseModel):
    """Complete configuration for curriculum learning.

    Defines the entire curriculum with multiple stages
    and global settings.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        description="Curriculum name for identification",
    )
    stages: list[StageConfig] = Field(
        ...,
        min_length=1,
        description="Ordered list of curriculum stages",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    allow_regression: bool = Field(
        default=False,
        description="Allow returning to earlier stages on performance drop",
    )
    regression_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Win rate drop threshold to trigger regression",
    )
    warmup_games_per_stage: int = Field(
        default=100,
        ge=0,
        description="Games to play without evaluation after stage transition",
    )
    evaluation_interval: int = Field(
        default=100,
        ge=1,
        description="Games between progression evaluations",
    )
    checkpoint_on_stage_complete: bool = Field(
        default=True,
        description="Save checkpoint when completing a stage",
    )
    log_detailed_metrics: bool = Field(
        default=True,
        description="Log detailed per-stage metrics",
    )

    @field_validator("stages")
    @classmethod
    def validate_stages_ordered(cls, v: list[StageConfig]) -> list[StageConfig]:
        """Validate stages are ordered by board size."""
        board_sizes = [s.board_size for s in v]
        if board_sizes != sorted(board_sizes):
            raise ValueError("Stages must be ordered by increasing board size")
        return v

    @field_validator("stages")
    @classmethod
    def validate_unique_stage_names(cls, v: list[StageConfig]) -> list[StageConfig]:
        """Validate stage names are unique."""
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")
        return v

    def compute_hash(self) -> str:
        """Compute unique hash of configuration.

        Returns:
            Hexadecimal hash string.
        """
        data = self.model_dump(mode="json")
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def get_stage(self, name: str) -> StageConfig | None:
        """Get stage by name.

        Args:
            name: Stage name to find.

        Returns:
            StageConfig or None if not found.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def get_stage_index(self, name: str) -> int:
        """Get stage index by name.

        Args:
            name: Stage name to find.

        Returns:
            Stage index or -1 if not found.
        """
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                return i
        return -1


def create_default_curriculum(
    name: str = "default",
    board_sizes: list[int] | None = None,
    win_rate_threshold: float = 0.55,
    min_games_per_stage: int = 1000,
) -> CurriculumConfig:
    """Create a default curriculum configuration.

    Args:
        name: Curriculum name.
        board_sizes: List of board sizes (default: [9, 13, 19]).
        win_rate_threshold: Win rate for progression.
        min_games_per_stage: Minimum games per stage.

    Returns:
        CurriculumConfig with default stages.
    """
    if board_sizes is None:
        board_sizes = [9, 13, 19]

    stages = []
    for i, size in enumerate(board_sizes):
        stage_name = f"stage_{size}x{size}"

        # Last stage has no progression criteria (terminal)
        criteria = []
        if i < len(board_sizes) - 1:
            criteria = [
                ProgressionCriterion(
                    metric="win_rate",
                    operator=ProgressionOperator.GREATER_EQUAL,
                    threshold=win_rate_threshold,
                    min_samples=50,
                    window_size=100,
                )
            ]

        stages.append(
            StageConfig(
                name=stage_name,
                board_size=size,
                min_games=min_games_per_stage,
                progression_criteria=criteria,
            )
        )

    return CurriculumConfig(
        name=name,
        stages=stages,
    )
