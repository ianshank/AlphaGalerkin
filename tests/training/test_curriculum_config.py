"""Tests for curriculum schedule configuration and trainer integration.

Validates the ``curriculum_schedule`` field on ``TrainingConfig`` and
curriculum transition logging in the trainer.
"""

from __future__ import annotations

import pytest

from config.schemas import TrainingConfig
from src.training.curriculum import (
    BoardSizeCurriculum,
    CurriculumStage,
    create_default_curriculum,
)


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


class TestCurriculumStage:
    """Tests for CurriculumStage dataclass."""

    def test_stage_without_weights(self) -> None:
        """Stage without weights stores None."""
        stage = CurriculumStage(start_step=0, board_sizes=[9, 13])
        assert stage.size_weights is None

    def test_stage_with_valid_weights(self) -> None:
        """Stage normalizes valid weights."""
        stage = CurriculumStage(start_step=0, board_sizes=[9, 13], size_weights=[1.0, 3.0])
        # Should be normalized to sum to 1
        assert stage.size_weights is not None
        assert abs(sum(stage.size_weights) - 1.0) < 1e-6
        assert abs(stage.size_weights[0] - 0.25) < 1e-6
        assert abs(stage.size_weights[1] - 0.75) < 1e-6

    def test_stage_weight_length_mismatch_raises(self) -> None:
        """Mismatched weights length raises ValueError."""
        with pytest.raises(ValueError, match="size_weights length"):
            CurriculumStage(start_step=0, board_sizes=[9, 13], size_weights=[1.0])

    def test_stage_zero_total_weight_raises(self) -> None:
        """Zero total weight raises ValueError."""
        with pytest.raises(ValueError, match="must sum to a positive value"):
            CurriculumStage(start_step=0, board_sizes=[9, 13], size_weights=[0.0, 0.0])

    def test_stage_negative_total_weight_raises(self) -> None:
        """Negative total weight raises ValueError."""
        with pytest.raises(ValueError, match="must sum to a positive value"):
            CurriculumStage(start_step=0, board_sizes=[9, 13], size_weights=[-1.0, 0.5])


class TestBoardSizeCurriculumSampling:
    """Tests for weighted sampling in BoardSizeCurriculum."""

    def test_sample_without_weights_uniform(self) -> None:
        """Sampling without weights is uniform."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9, 13]})
        samples = [curriculum.sample_board_size(0) for _ in range(100)]
        # Both sizes should appear
        assert 9 in samples
        assert 13 in samples

    def test_sample_with_weights(self) -> None:
        """Sampling respects weights."""
        stages = [CurriculumStage(start_step=0, board_sizes=[9, 13], size_weights=[0.99, 0.01])]
        curriculum = BoardSizeCurriculum(stages)
        samples = [curriculum.sample_board_size(0) for _ in range(100)]
        # 9 should dominate
        count_9 = samples.count(9)
        assert count_9 > 80  # With 99% weight, should be heavily biased

    def test_get_current_stage_at_exact_boundary(self) -> None:
        """get_current_stage at exact boundary returns new stage."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9], 100: [9, 13]})
        stage = curriculum.get_current_stage(100)
        assert stage.board_sizes == [9, 13]

    def test_get_current_stage_before_boundary(self) -> None:
        """get_current_stage before boundary returns previous stage."""
        curriculum = BoardSizeCurriculum.from_schedule({0: [9], 100: [9, 13]})
        stage = curriculum.get_current_stage(99)
        assert stage.board_sizes == [9]

    def test_empty_stages_raises(self) -> None:
        """Empty stages list raises ValueError."""
        with pytest.raises(ValueError, match="At least one curriculum stage"):
            BoardSizeCurriculum([])


class TestBoardSizeCurriculumFromConfig:
    """Tests for from_config with string keys."""

    def test_from_config_string_keys(self) -> None:
        """from_config handles string keys from YAML."""
        config = {"0": [9], "100": [9, 13], "500": [9, 13, 19]}
        curriculum = BoardSizeCurriculum.from_config(config)
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(100) == [9, 13]
        assert curriculum.get_board_sizes(500) == [9, 13, 19]

    def test_from_config_int_keys(self) -> None:
        """from_config also handles int keys."""
        config = {0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_config(config)
        assert curriculum.get_board_sizes(0) == [9]


class TestCreateDefaultCurriculum:
    """Tests for create_default_curriculum factory."""

    def test_default_curriculum_stages(self) -> None:
        """Default curriculum has expected stages."""
        curriculum = create_default_curriculum()
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(10000) == [9, 13]
        assert curriculum.get_board_sizes(50000) == [9, 13, 19]

    def test_default_curriculum_transitions(self) -> None:
        """Default curriculum has expected transitions."""
        curriculum = create_default_curriculum()
        assert curriculum.is_transition_step(0)
        assert curriculum.is_transition_step(10000)
        assert curriculum.is_transition_step(50000)
        assert not curriculum.is_transition_step(1)
        assert not curriculum.is_transition_step(25000)

    def test_default_curriculum_far_future(self) -> None:
        """Default curriculum uses last stage for far future."""
        curriculum = create_default_curriculum()
        assert curriculum.get_board_sizes(1000000) == [9, 13, 19]
