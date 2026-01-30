"""Curriculum learning for board size progression.

Implements progressive introduction of board sizes during training,
starting with smaller boards and gradually adding larger ones.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CurriculumStage:
    """A stage in the curriculum.

    Attributes:
        start_step: Training step when this stage activates.
        board_sizes: Board sizes available in this stage.
        size_weights: Optional sampling weights for each size.

    """

    start_step: int
    board_sizes: list[int]
    size_weights: list[float] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize weights."""
        if self.size_weights is not None:
            if len(self.size_weights) != len(self.board_sizes):
                raise ValueError(
                    f"size_weights length ({len(self.size_weights)}) must match "
                    f"board_sizes length ({len(self.board_sizes)})"
                )
            # Normalize weights
            total = sum(self.size_weights)
            self.size_weights = [w / total for w in self.size_weights]


class BoardSizeCurriculum:
    """Curriculum scheduler for board sizes.

    Progressively introduces larger board sizes as training progresses.
    This helps the model learn fundamental patterns on smaller boards
    before tackling the complexity of larger ones.

    Example:
        curriculum = BoardSizeCurriculum.from_schedule({
            0: [9],           # Start with 9x9 only
            10000: [9, 13],   # Add 13x13 at step 10k
            50000: [9, 13, 19]  # Add 19x19 at step 50k
        })

        for step in range(100000):
            board_size = curriculum.sample_board_size(step)
            # Use board_size for self-play

    """

    def __init__(self, stages: list[CurriculumStage]) -> None:
        """Initialize curriculum.

        Args:
            stages: List of curriculum stages, sorted by start_step.

        """
        self.stages = sorted(stages, key=lambda s: s.start_step)
        if not self.stages:
            raise ValueError("At least one curriculum stage is required")

        logger.info(
            "curriculum_initialized",
            n_stages=len(self.stages),
            stages=[
                {"step": s.start_step, "sizes": s.board_sizes} for s in self.stages
            ],
        )

    @classmethod
    def from_schedule(
        cls,
        schedule: dict[int, list[int]],
    ) -> BoardSizeCurriculum:
        """Create curriculum from schedule dictionary.

        Args:
            schedule: Mapping from start_step to board_sizes.
                Example: {0: [9], 10000: [9, 13], 50000: [9, 13, 19]}

        Returns:
            Configured curriculum.

        """
        stages = [
            CurriculumStage(start_step=step, board_sizes=sizes)
            for step, sizes in schedule.items()
        ]
        return cls(stages)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> BoardSizeCurriculum:
        """Create curriculum from config dictionary.

        Handles both integer and string keys (YAML may use strings).

        Args:
            config: Schedule config, may have string keys.

        Returns:
            Configured curriculum.

        """
        schedule = {int(k): v for k, v in config.items()}
        return cls.from_schedule(schedule)

    def get_current_stage(self, step: int) -> CurriculumStage:
        """Get the active curriculum stage for a step.

        Args:
            step: Current training step.

        Returns:
            Active curriculum stage.

        """
        active_stage = self.stages[0]
        for stage in self.stages:
            if step >= stage.start_step:
                active_stage = stage
            else:
                break
        return active_stage

    def get_board_sizes(self, step: int) -> list[int]:
        """Get available board sizes for current step.

        Args:
            step: Current training step.

        Returns:
            List of available board sizes.

        """
        return self.get_current_stage(step).board_sizes

    def sample_board_size(self, step: int) -> int:
        """Sample a board size for current step.

        Args:
            step: Current training step.

        Returns:
            Sampled board size.

        """
        stage = self.get_current_stage(step)
        if stage.size_weights is not None:
            return random.choices(stage.board_sizes, weights=stage.size_weights, k=1)[0]
        return random.choice(stage.board_sizes)

    def is_transition_step(self, step: int) -> bool:
        """Check if this step is a curriculum transition.

        Args:
            step: Current training step.

        Returns:
            True if this step starts a new curriculum stage.

        """
        return any(stage.start_step == step for stage in self.stages)

    def get_schedule_info(self) -> list[dict[str, Any]]:
        """Get schedule information for logging.

        Returns:
            List of stage info dictionaries.

        """
        return [
            {
                "start_step": stage.start_step,
                "board_sizes": stage.board_sizes,
                "weights": stage.size_weights,
            }
            for stage in self.stages
        ]


def create_default_curriculum() -> BoardSizeCurriculum:
    """Create default curriculum for AlphaGalerkin.

    Default schedule:
    - Steps 0-9999: 9x9 only
    - Steps 10000-49999: 9x9 and 13x13
    - Steps 50000+: 9x9, 13x13, and 19x19

    Returns:
        Default curriculum.

    """
    return BoardSizeCurriculum.from_schedule(
        {
            0: [9],
            10000: [9, 13],
            50000: [9, 13, 19],
        }
    )
