"""Integration tests for curriculum learning.

Tests:
- BoardSizeCurriculum.from_config() parses schedule correctly
- sample_board_size(step) returns correct size for each phase
- Edge cases: step exactly on boundary, step before first boundary
- CurriculumManager end-to-end stage transitions with structlog verification
- CurriculumScheduler integration with stage progression and callbacks
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.curriculum.config import (
    CurriculumConfig,
    ProgressionCriterion,
    ProgressionOperator,
    StageConfig,
)
from src.curriculum.manager import CurriculumManager
from src.curriculum.scheduler import CurriculumScheduler
from src.training.curriculum import (
    BoardSizeCurriculum,
)
from src.training.curriculum import (
    create_default_curriculum as create_training_default,
)


# ---------------------------------------------------------------------------
# BoardSizeCurriculum.from_config() parsing tests
# ---------------------------------------------------------------------------
class TestBoardSizeCurriculumFromConfig:
    """Verify that from_config correctly parses schedule dictionaries."""

    def test_from_config_integer_keys(self) -> None:
        """Integer keys are parsed directly."""
        config: dict[str, Any] = {0: [9], 10000: [9, 13], 50000: [9, 13, 19]}
        curriculum = BoardSizeCurriculum.from_config(config)
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(10000) == [9, 13]
        assert curriculum.get_board_sizes(50000) == [9, 13, 19]

    def test_from_config_string_keys(self) -> None:
        """String keys (from YAML parsing) are converted to int."""
        config: dict[str, Any] = {"0": [9], "5000": [9, 13]}
        curriculum = BoardSizeCurriculum.from_config(config)
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(5000) == [9, 13]

    def test_from_config_single_stage(self) -> None:
        """Single-stage schedule is valid."""
        config: dict[str, Any] = {0: [9, 13, 19]}
        curriculum = BoardSizeCurriculum.from_config(config)
        assert curriculum.get_board_sizes(0) == [9, 13, 19]
        assert curriculum.get_board_sizes(999999) == [9, 13, 19]

    def test_from_config_custom_sizes(self) -> None:
        """Non-standard board sizes are accepted."""
        config: dict[str, Any] = {0: [5], 100: [5, 7], 500: [5, 7, 11]}
        curriculum = BoardSizeCurriculum.from_config(config)
        assert curriculum.get_board_sizes(0) == [5]
        assert curriculum.get_board_sizes(100) == [5, 7]
        assert curriculum.get_board_sizes(500) == [5, 7, 11]

    def test_from_config_preserves_stage_order(self) -> None:
        """Stages are sorted by start_step regardless of dict insertion order."""
        config: dict[str, Any] = {500: [9, 13], 0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_config(config)
        # stage order: 0, 100, 500
        assert curriculum.stages[0].start_step == 0
        assert curriculum.stages[1].start_step == 100
        assert curriculum.stages[2].start_step == 500


# ---------------------------------------------------------------------------
# sample_board_size() per-phase tests
# ---------------------------------------------------------------------------
class TestSampleBoardSizePerPhase:
    """Verify sample_board_size returns correct sizes for each phase."""

    def test_phase_one_only_small_boards(self) -> None:
        """During phase 1 (step 0-9999), only 9x9 is sampled."""
        curriculum = create_training_default()
        for _ in range(100):
            assert curriculum.sample_board_size(0) == 9
            assert curriculum.sample_board_size(5000) == 9
            assert curriculum.sample_board_size(9999) == 9

    def test_phase_two_includes_medium_boards(self) -> None:
        """During phase 2 (step 10000-49999), 9x9 and 13x13 are available."""
        curriculum = create_training_default()
        sizes = {curriculum.sample_board_size(25000) for _ in range(300)}
        assert sizes == {9, 13}

    def test_phase_three_all_boards(self) -> None:
        """During phase 3 (step 50000+), all three sizes are available."""
        curriculum = create_training_default()
        sizes = {curriculum.sample_board_size(75000) for _ in range(500)}
        assert sizes == {9, 13, 19}

    def test_custom_schedule_phases(self) -> None:
        """Custom schedule correctly constrains sampling per phase."""
        schedule = {0: [7], 50: [7, 9], 200: [7, 9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)

        # Phase 1: only 7
        for _ in range(50):
            assert curriculum.sample_board_size(25) == 7

        # Phase 2: 7 or 9
        sizes = {curriculum.sample_board_size(100) for _ in range(200)}
        assert sizes == {7, 9}

        # Phase 3: 7, 9, or 13
        sizes = {curriculum.sample_board_size(300) for _ in range(500)}
        assert sizes == {7, 9, 13}


# ---------------------------------------------------------------------------
# Edge cases: boundary steps
# ---------------------------------------------------------------------------
class TestBoundaryEdgeCases:
    """Edge cases at step boundaries."""

    def test_step_exactly_on_boundary(self) -> None:
        """Step exactly on a boundary uses the new stage."""
        schedule = {0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)
        # At step 100, 13 should be available
        sizes_at_100 = curriculum.get_board_sizes(100)
        assert 13 in sizes_at_100

    def test_step_one_before_boundary(self) -> None:
        """Step just before a boundary uses the previous stage."""
        schedule = {0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)
        sizes_at_99 = curriculum.get_board_sizes(99)
        assert sizes_at_99 == [9]

    def test_step_one_after_boundary(self) -> None:
        """Step just after a boundary uses the new stage."""
        schedule = {0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)
        sizes_at_101 = curriculum.get_board_sizes(101)
        assert sizes_at_101 == [9, 13]

    def test_step_before_first_boundary(self) -> None:
        """Step 0 (before any later boundaries) uses the first stage."""
        schedule = {0: [9], 100: [9, 13], 500: [9, 13, 19]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)
        assert curriculum.get_board_sizes(0) == [9]

    def test_step_way_past_last_boundary(self) -> None:
        """Steps far beyond the last boundary keep using the last stage."""
        schedule = {0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)
        assert curriculum.get_board_sizes(1_000_000) == [9, 13]

    def test_negative_step_uses_first_stage(self) -> None:
        """A negative step still returns the first stage (no crash)."""
        schedule = {0: [9], 100: [9, 13]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)
        # Step -1 is before stage 0, should still get first stage
        sizes = curriculum.get_board_sizes(-1)
        assert sizes == [9]

    def test_transition_detected_exactly_on_boundary(self) -> None:
        """is_transition_step returns True only at exact boundaries."""
        schedule = {0: [9], 100: [9, 13], 500: [9, 13, 19]}
        curriculum = BoardSizeCurriculum.from_schedule(schedule)

        assert curriculum.is_transition_step(0)
        assert curriculum.is_transition_step(100)
        assert curriculum.is_transition_step(500)
        assert not curriculum.is_transition_step(1)
        assert not curriculum.is_transition_step(99)
        assert not curriculum.is_transition_step(101)
        assert not curriculum.is_transition_step(499)
        assert not curriculum.is_transition_step(501)

    def test_empty_stages_rejected(self) -> None:
        """An empty schedule raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            BoardSizeCurriculum(stages=[])


# ---------------------------------------------------------------------------
# CurriculumManager end-to-end integration with structlog
# ---------------------------------------------------------------------------
class TestCurriculumManagerIntegration:
    """End-to-end tests for CurriculumManager with stage transitions."""

    @pytest.fixture
    def fast_curriculum_config(self) -> CurriculumConfig:
        """Create a fast curriculum for testing (low thresholds)."""
        return CurriculumConfig(
            name="integration_test",
            stages=[
                StageConfig(
                    name="stage_9x9",
                    board_size=9,
                    min_games=5,
                    min_steps=3,
                    progression_criteria=[
                        ProgressionCriterion(
                            metric="win_rate",
                            operator=ProgressionOperator.GREATER_EQUAL,
                            threshold=0.6,
                            min_samples=3,
                        )
                    ],
                ),
                StageConfig(
                    name="stage_13x13",
                    board_size=13,
                    min_games=5,
                    min_steps=3,
                ),
            ],
            warmup_games_per_stage=0,
            evaluation_interval=3,
        )

    def test_full_lifecycle(self, fast_curriculum_config: CurriculumConfig) -> None:
        """Manager completes full lifecycle: start -> play -> transition."""
        manager = CurriculumManager(config=fast_curriculum_config)
        manager.start()

        assert manager.is_started
        assert not manager.is_completed
        assert manager.current_board_size == 9

        # Play enough winning games to trigger progression
        # Need min_games=5, min_steps=3, evaluated every 3 games
        for _ in range(3):
            manager.step(training_metrics={"total_loss": 0.5})

        for i in range(6):
            transitioned = manager.step(game_result={"won": True, "drawn": False})
            if transitioned:
                break

        # Should have advanced to stage_13x13
        assert manager.current_board_size == 13
        assert manager.current_stage_name == "stage_13x13"

    def test_board_size_changes_on_transition(
        self, fast_curriculum_config: CurriculumConfig
    ) -> None:
        """Board size changes when stage transitions."""
        manager = CurriculumManager(config=fast_curriculum_config)
        manager.start()

        initial_size = manager.current_board_size
        assert initial_size == 9

        # Meet progression criteria
        for _ in range(3):
            manager.step(training_metrics={"total_loss": 0.5})

        for _ in range(6):
            manager.step(game_result={"won": True, "drawn": False})

        assert manager.current_board_size == 13

    def test_stage_transition_logged(self, fast_curriculum_config: CurriculumConfig) -> None:
        """Stage transitions produce structlog output."""
        mock_logger = MagicMock()
        manager = CurriculumManager(
            config=fast_curriculum_config,
            logger=mock_logger,
        )
        manager.start()

        # Verify start was logged
        mock_logger.info.assert_any_call(
            "curriculum_manager_started",
            stages=2,
            first_board_size=9,
        )

        # Meet progression criteria
        for _ in range(3):
            manager.step(training_metrics={"total_loss": 0.5})
        for _ in range(6):
            manager.step(game_result={"won": True, "drawn": False})

        # Verify transition was logged via the scheduler's logger
        # The manager passes its logger to the scheduler
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "new_stage_started" in info_calls

    def test_metrics_reflect_current_stage(self, fast_curriculum_config: CurriculumConfig) -> None:
        """get_metrics returns correct stage info."""
        manager = CurriculumManager(config=fast_curriculum_config)
        manager.start()

        metrics = manager.get_metrics()
        assert metrics.current_board_size == 9
        assert metrics.current_stage == "stage_9x9"
        assert metrics.total_stages == 2

    def test_curriculum_not_started_raises(self, fast_curriculum_config: CurriculumConfig) -> None:
        """Calling step before start raises RuntimeError."""
        manager = CurriculumManager(config=fast_curriculum_config)
        with pytest.raises(RuntimeError, match="not started"):
            manager.step(game_result={"won": True})


# ---------------------------------------------------------------------------
# CurriculumScheduler integration with callbacks
# ---------------------------------------------------------------------------
class TestSchedulerCallbackIntegration:
    """Test scheduler fires callbacks at correct times."""

    @pytest.fixture
    def scheduler_config(self) -> CurriculumConfig:
        """Create config with fast progression."""
        return CurriculumConfig(
            name="callback_test",
            stages=[
                StageConfig(
                    name="small",
                    board_size=9,
                    min_games=3,
                    min_steps=1,
                    progression_criteria=[
                        ProgressionCriterion(
                            metric="win_rate",
                            operator=ProgressionOperator.GREATER_EQUAL,
                            threshold=0.5,
                            min_samples=2,
                        )
                    ],
                ),
                StageConfig(
                    name="medium",
                    board_size=13,
                    min_games=3,
                    min_steps=1,
                    progression_criteria=[
                        ProgressionCriterion(
                            metric="win_rate",
                            operator=ProgressionOperator.GREATER_EQUAL,
                            threshold=0.5,
                            min_samples=2,
                        )
                    ],
                ),
                StageConfig(
                    name="large",
                    board_size=19,
                    min_games=3,
                    min_steps=1,
                ),
            ],
            warmup_games_per_stage=0,
            evaluation_interval=3,
        )

    def test_stage_start_callback_fires(self, scheduler_config: CurriculumConfig) -> None:
        """on_stage_start callback fires when a new stage begins."""
        scheduler = CurriculumScheduler(scheduler_config)
        start_events: list[str] = []
        scheduler.on_stage_start(lambda stage: start_events.append(stage.config.name))

        scheduler.start()
        assert "small" in start_events

    def test_stage_complete_callback_fires(self, scheduler_config: CurriculumConfig) -> None:
        """on_stage_complete callback fires when stage completes."""
        scheduler = CurriculumScheduler(scheduler_config)
        complete_events: list[str] = []
        scheduler.on_stage_complete(lambda stage: complete_events.append(stage.config.name))

        scheduler.start()
        # Record training step to satisfy min_steps
        scheduler.record_training_step(0.5)

        # Win enough games to progress
        for _ in range(6):
            scheduler.record_game(won=True)

        assert "small" in complete_events

    def test_three_stage_full_progression(self, scheduler_config: CurriculumConfig) -> None:
        """Full three-stage progression fires callbacks in order."""
        scheduler = CurriculumScheduler(scheduler_config)
        transitions: list[tuple[str, str]] = []

        scheduler.on_stage_complete(
            lambda stage: transitions.append(("complete", stage.config.name))
        )
        scheduler.on_stage_start(lambda stage: transitions.append(("start", stage.config.name)))

        scheduler.start()
        # Starts with "small"
        assert ("start", "small") in transitions

        # Progress through small
        scheduler.record_training_step(0.5)
        for _ in range(6):
            scheduler.record_game(won=True)

        assert ("complete", "small") in transitions
        assert ("start", "medium") in transitions

        # Progress through medium
        scheduler.record_training_step(0.5)
        for _ in range(6):
            scheduler.record_game(won=True)

        assert ("complete", "medium") in transitions
        assert ("start", "large") in transitions

    def test_board_size_progression(self, scheduler_config: CurriculumConfig) -> None:
        """Board size increases through stages."""
        scheduler = CurriculumScheduler(scheduler_config)
        scheduler.start()

        assert scheduler.current_board_size == 9

        scheduler.record_training_step(0.5)
        for _ in range(6):
            scheduler.record_game(won=True)

        assert scheduler.current_board_size == 13

        scheduler.record_training_step(0.5)
        for _ in range(6):
            scheduler.record_game(won=True)

        assert scheduler.current_board_size == 19
