"""Tests for curriculum schedule configuration and trainer integration.

Validates the ``curriculum_schedule`` field on ``TrainingConfig`` and
curriculum transition logging in the trainer.
"""

from __future__ import annotations

from config.schemas import TrainingConfig
from src.training.curriculum import BoardSizeCurriculum


class TestTrainingConfigCurriculum:
    """Tests for curriculum fields on TrainingConfig."""

    def test_curriculum_disabled_by_default(self) -> None:
        """curriculum_enabled defaults to False."""
        config = TrainingConfig()
        assert config.curriculum_enabled is False

    def test_curriculum_schedule_none_by_default(self) -> None:
        """curriculum_schedule defaults to None."""
        config = TrainingConfig()
        assert config.curriculum_schedule is None

    def test_curriculum_schedule_round_trip(self) -> None:
        """Schedule can be set and retrieved."""
        schedule = {0: [9], 10000: [9, 13], 50000: [9, 13, 19]}
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule=schedule,
        )
        assert config.curriculum_enabled is True
        assert config.curriculum_schedule == schedule

    def test_curriculum_schedule_creates_valid_curriculum(self) -> None:
        """Schedule from config creates a working BoardSizeCurriculum."""
        schedule = {0: [9], 5000: [9, 13]}
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule=schedule,
        )
        curriculum = BoardSizeCurriculum.from_config(config.curriculum_schedule)

        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(5000) == [9, 13]
        assert curriculum.get_board_sizes(10000) == [9, 13]


class TestCurriculumTransitionDetection:
    """Test transition detection used for logging."""

    def test_is_transition_step_at_start(self) -> None:
        """Step 0 is always a transition step."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9], 100: [9, 13]})
        assert curriculum.is_transition_step(0)

    def test_is_transition_step_at_boundary(self) -> None:
        """Exactly at stage boundary is a transition."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9], 100: [9, 13], 500: [9, 13, 19]})
        assert curriculum.is_transition_step(100)
        assert curriculum.is_transition_step(500)

    def test_is_not_transition_step_between(self) -> None:
        """Steps between boundaries are not transitions."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9], 100: [9, 13]})
        assert not curriculum.is_transition_step(50)
        assert not curriculum.is_transition_step(99)
        assert not curriculum.is_transition_step(101)

    def test_get_schedule_info(self) -> None:
        """get_schedule_info returns list of stage dicts."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9], 100: [9, 13]})
        info = curriculum.get_schedule_info()
        assert len(info) == 2
        assert info[0]["start_step"] == 0
        assert info[0]["board_sizes"] == [9]
        assert info[1]["start_step"] == 100
        assert info[1]["board_sizes"] == [9, 13]
