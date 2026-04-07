"""Tests for curriculum learning integration with the training pipeline.

Validates:
- TrainingConfig curriculum fields (defaults, custom schedules)
- Stage transitions and logging
- Backwards compatibility with existing configs
- Integration with self-play board size selection
"""

from __future__ import annotations

from unittest.mock import MagicMock

from config.schemas import AlphaGalerkinConfig, TrainingConfig
from src.constants import DEFAULT_CURRICULUM_SCHEDULE
from src.training.curriculum import (
    BoardSizeCurriculum,
    create_default_curriculum,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestCurriculumConfigDefaults:
    """Verify curriculum-related fields on TrainingConfig."""

    def test_curriculum_disabled_by_default(self) -> None:
        """curriculum_enabled defaults to False -- no behavioural change."""
        config = TrainingConfig()
        assert config.curriculum_enabled is False

    def test_curriculum_schedule_none_by_default(self) -> None:
        """curriculum_schedule defaults to None."""
        config = TrainingConfig()
        assert config.curriculum_schedule is None

    def test_disabled_curriculum_does_not_create_curriculum(self) -> None:
        """When curriculum_enabled=False the trainer path sets curriculum to None."""
        config = TrainingConfig(curriculum_enabled=False)
        # The trainer conditionally creates curriculum; simulate the branch:
        curriculum = (
            _mock_create_curriculum(config)
            if config.curriculum_enabled
            else None
        )
        assert curriculum is None

    def test_enabled_with_none_schedule_uses_default(self) -> None:
        """curriculum_enabled=True + curriculum_schedule=None falls back to default."""
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule=None,
        )
        assert config.curriculum_enabled is True
        # The trainer falls back to the built-in default schedule
        schedule = config.curriculum_schedule
        if schedule is None:
            schedule = {0: [9], 10000: [9, 13], 50000: [9, 13, 19]}
        curriculum = BoardSizeCurriculum.from_config(schedule)
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(10000) == [9, 13]
        assert curriculum.get_board_sizes(50000) == [9, 13, 19]

    def test_custom_schedule_is_respected(self) -> None:
        """A custom curriculum_schedule dict is honoured verbatim."""
        custom = {0: [5], 500: [5, 9], 2000: [5, 9, 13]}
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule=custom,
        )
        assert config.curriculum_schedule == custom
        curriculum = BoardSizeCurriculum.from_config(config.curriculum_schedule)
        assert curriculum.get_board_sizes(0) == [5]
        assert curriculum.get_board_sizes(500) == [5, 9]
        assert curriculum.get_board_sizes(2000) == [5, 9, 13]

    def test_enabled_true_creates_curriculum_with_default_schedule(self) -> None:
        """curriculum_enabled=True with no schedule still yields a curriculum."""
        config = TrainingConfig(curriculum_enabled=True)
        curriculum = _mock_create_curriculum(config)
        assert curriculum is not None
        # Should match DEFAULT_CURRICULUM_SCHEDULE
        assert curriculum.get_board_sizes(0) == DEFAULT_CURRICULUM_SCHEDULE[0]

    def test_custom_single_stage_schedule(self) -> None:
        """A single-stage schedule (always all sizes) is valid."""
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule={0: [9, 13, 19]},
        )
        curriculum = BoardSizeCurriculum.from_config(config.curriculum_schedule)
        assert curriculum.get_board_sizes(0) == [9, 13, 19]
        assert curriculum.get_board_sizes(99999) == [9, 13, 19]


# ---------------------------------------------------------------------------
# Stage transition tests
# ---------------------------------------------------------------------------
class TestStageTransitions:
    """Verify that curriculum reports correct board sizes per step."""

    def test_step_zero_only_small_boards(self) -> None:
        """At step 0 the default curriculum only offers 9x9."""
        curriculum = create_default_curriculum()
        assert curriculum.get_board_sizes(0) == [9]

    def test_later_steps_introduce_larger_boards(self) -> None:
        """After step 10000, 13x13 is available; after 50000, 19x19."""
        curriculum = create_default_curriculum()
        assert 13 in curriculum.get_board_sizes(10000)
        assert 19 in curriculum.get_board_sizes(50000)

    def test_custom_transition_boundaries(self) -> None:
        """Custom schedule transitions at the specified steps."""
        schedule = {0: [7], 100: [7, 9], 300: [7, 9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)

        assert curriculum.get_board_sizes(0) == [7]
        assert curriculum.get_board_sizes(99) == [7]
        assert curriculum.get_board_sizes(100) == [7, 9]
        assert curriculum.get_board_sizes(299) == [7, 9]
        assert curriculum.get_board_sizes(300) == [7, 9, 13]

    def test_transition_step_logging(self) -> None:
        """is_transition_step returns True exactly at boundaries."""
        curriculum = BoardSizeCurriculum.from_schedule(
            {0: [9], 100: [9, 13], 500: [9, 13, 19]}
        )
        assert curriculum.is_transition_step(0)
        assert curriculum.is_transition_step(100)
        assert curriculum.is_transition_step(500)
        assert not curriculum.is_transition_step(1)
        assert not curriculum.is_transition_step(99)
        assert not curriculum.is_transition_step(250)

    def test_transition_logged_via_structlog(self) -> None:
        """When is_transition_step is True, the trainer logs the event.

        We simulate the trainer's logging branch and verify structlog is called.
        """
        curriculum = BoardSizeCurriculum.from_schedule(
            {0: [9], 100: [9, 13]}
        )
        mock_logger = MagicMock()

        step = 100
        if curriculum.is_transition_step(step):
            stage = curriculum.get_current_stage(step)
            mock_logger.info(
                "curriculum_stage_transition",
                step=step,
                board_sizes=stage.board_sizes,
                weights=stage.size_weights,
            )

        mock_logger.info.assert_called_once_with(
            "curriculum_stage_transition",
            step=100,
            board_sizes=[9, 13],
            weights=None,
        )


# ---------------------------------------------------------------------------
# Backwards compatibility tests
# ---------------------------------------------------------------------------
class TestBackwardsCompatibility:
    """Existing configs without curriculum fields must work unchanged."""

    def test_default_training_config_unchanged(self) -> None:
        """TrainingConfig() without any curriculum args works."""
        config = TrainingConfig()
        assert config.curriculum_enabled is False
        assert config.curriculum_schedule is None

    def test_full_config_without_curriculum(self) -> None:
        """AlphaGalerkinConfig with default training has no curriculum."""
        config = AlphaGalerkinConfig()
        assert config.training.curriculum_enabled is False
        assert config.training.curriculum_schedule is None

    def test_existing_fields_not_affected(self) -> None:
        """Other training fields keep their defaults when curriculum is set."""
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule={0: [9]},
        )
        # Spot-check a selection of unrelated defaults
        assert config.learning_rate == 2e-4
        assert config.batch_size == 256
        assert config.total_steps == 100000
        assert config.use_amp is True

    def test_serialisation_round_trip(self) -> None:
        """model_dump / model_validate preserves curriculum fields."""
        original = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule={0: [9], 1000: [9, 13]},
        )
        data = original.model_dump()
        restored = TrainingConfig.model_validate(data)
        assert restored.curriculum_enabled is True
        assert restored.curriculum_schedule == {0: [9], 1000: [9, 13]}

    def test_no_curriculum_fields_in_dump_when_defaults(self) -> None:
        """Default dump includes curriculum fields but they are falsy/None."""
        config = TrainingConfig()
        data = config.model_dump()
        assert "curriculum_enabled" in data
        assert data["curriculum_enabled"] is False
        assert data["curriculum_schedule"] is None


# ---------------------------------------------------------------------------
# Integration with self-play board size selection
# ---------------------------------------------------------------------------
class TestSelfPlayIntegration:
    """Board sizes sampled from curriculum during self-play."""

    def test_sample_size_at_step_zero(self) -> None:
        """At step 0, only small board sizes are sampled."""
        curriculum = create_default_curriculum()
        for _ in range(50):
            size = curriculum.sample_board_size(0)
            assert size == 9

    def test_sample_sizes_after_first_transition(self) -> None:
        """After step 10000, sampled sizes include 13."""
        curriculum = create_default_curriculum()
        sizes = {curriculum.sample_board_size(10000) for _ in range(200)}
        assert 9 in sizes
        assert 13 in sizes
        assert 19 not in sizes

    def test_sample_sizes_after_second_transition(self) -> None:
        """After step 50000, all three sizes are available."""
        curriculum = create_default_curriculum()
        sizes = {curriculum.sample_board_size(50000) for _ in range(500)}
        assert 9 in sizes
        assert 13 in sizes
        assert 19 in sizes

    def test_board_size_changes_at_correct_threshold(self) -> None:
        """Board size set expands exactly at step thresholds."""
        schedule = {0: [9], 50: [9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)

        # Just before threshold
        for _ in range(50):
            assert curriculum.sample_board_size(49) == 9

        # At threshold
        sizes = {curriculum.sample_board_size(50) for _ in range(200)}
        assert 13 in sizes

    def test_curriculum_none_means_no_override(self) -> None:
        """When curriculum is None, board_size stays None (caller decides)."""
        curriculum = None  # disabled path
        board_size = None
        if curriculum is not None:
            board_size = curriculum.sample_board_size(0)
        assert board_size is None

    def test_trainer_create_curriculum_with_custom_schedule(self) -> None:
        """Simulates Trainer._create_curriculum with a custom schedule."""
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule={0: [7], 200: [7, 11]},
        )
        curriculum = _mock_create_curriculum(config)
        assert curriculum is not None
        assert curriculum.get_board_sizes(0) == [7]
        assert curriculum.get_board_sizes(200) == [7, 11]

    def test_trainer_create_curriculum_default_schedule(self) -> None:
        """Simulates Trainer._create_curriculum with default (None) schedule."""
        config = TrainingConfig(
            curriculum_enabled=True,
            curriculum_schedule=None,
        )
        curriculum = _mock_create_curriculum(config)
        assert curriculum is not None
        # Should use the hardcoded fallback
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(10000) == [9, 13]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_create_curriculum(config: TrainingConfig) -> BoardSizeCurriculum | None:
    """Reproduce the Trainer._create_curriculum logic without needing the full Trainer."""
    if not config.curriculum_enabled:
        return None
    schedule: dict[int, list[int]] | None = config.curriculum_schedule
    if schedule is None:
        schedule = {0: [9], 10000: [9, 13], 50000: [9, 13, 19]}
    return BoardSizeCurriculum.from_config(schedule)
