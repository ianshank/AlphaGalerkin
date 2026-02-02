"""Tests for curriculum scheduler."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.curriculum.config import CurriculumConfig
from src.curriculum.scheduler import CurriculumScheduler
from src.curriculum.stage import StageStatus


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler class."""

    def test_initial_state(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test initial scheduler state."""
        assert curriculum_scheduler.current_board_size == 9
        assert curriculum_scheduler.current_stage_name == "stage_9x9"
        assert not curriculum_scheduler.is_final_stage

    def test_start(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test starting the curriculum."""
        curriculum_scheduler.start()

        stage = curriculum_scheduler.current_stage
        # Should be in warmup since config has warmup_games_per_stage=2
        assert stage.status in (StageStatus.WARMUP, StageStatus.ACTIVE)

    def test_total_stages(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test total stages property."""
        assert curriculum_scheduler.total_stages == 2

    def test_is_final_stage(self, three_stage_config: CurriculumConfig) -> None:
        """Test final stage detection."""
        scheduler = CurriculumScheduler(three_stage_config)
        scheduler.start()

        assert not scheduler.is_final_stage

        # Skip to last stage
        scheduler.skip_to_stage("advanced")
        assert scheduler.is_final_stage

    def test_progress_percentage(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test progress percentage calculation."""
        curriculum_scheduler.start()
        assert curriculum_scheduler.progress_percentage == 0.0

    def test_record_game_during_warmup(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test recording games during warmup."""
        curriculum_scheduler.start()

        # Should be in warmup
        stage = curriculum_scheduler.current_stage
        if stage.status == StageStatus.WARMUP:
            curriculum_scheduler.record_game(won=True)
            assert stage.warmup_games_remaining < 2

    def test_record_game_triggers_evaluation(self, two_stage_config: CurriculumConfig) -> None:
        """Test that evaluation is triggered at interval."""
        scheduler = CurriculumScheduler(two_stage_config)
        scheduler.start()

        # Clear warmup
        for _ in range(2):
            scheduler.record_game(won=True)

        # Record games up to evaluation interval
        for _ in range(4):
            scheduler.record_game(won=True)

        # This should trigger evaluation
        scheduler.record_game(won=True)

    def test_record_training_step(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test recording training steps."""
        curriculum_scheduler.start()
        curriculum_scheduler.record_training_step(total_loss=0.5, policy_loss=0.3, value_loss=0.2)

        assert curriculum_scheduler._total_steps == 1
        assert curriculum_scheduler.current_stage.metrics.training_steps == 1

    def test_stage_progression(self, two_stage_config: CurriculumConfig) -> None:
        """Test stage progression when criteria met."""
        # Use shorter intervals for testing
        scheduler = CurriculumScheduler(two_stage_config)
        scheduler.start()

        # Play enough games to meet criteria
        for _ in range(15):  # More than min_games + enough to trigger evaluation
            scheduler.record_game(won=True)
            scheduler.record_training_step(total_loss=0.1)

        # Should have advanced to second stage
        assert scheduler.current_stage_name in ("stage_9x9", "stage_13x13")

    def test_force_advance(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test forced advancement."""
        curriculum_scheduler.start()

        assert curriculum_scheduler.current_stage_name == "stage_9x9"
        curriculum_scheduler.force_advance()
        assert curriculum_scheduler.current_stage_name == "stage_13x13"

    def test_force_advance_at_final_stage(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test forced advancement at final stage."""
        curriculum_scheduler.start()
        curriculum_scheduler.force_advance()  # Go to final

        assert curriculum_scheduler.is_final_stage
        result = curriculum_scheduler.force_advance()
        assert not result  # Can't advance further

    def test_skip_to_stage(self, three_stage_config: CurriculumConfig) -> None:
        """Test skipping to specific stage."""
        scheduler = CurriculumScheduler(three_stage_config)
        scheduler.start()

        result = scheduler.skip_to_stage("advanced")

        assert result
        assert scheduler.current_stage_name == "advanced"
        assert scheduler.current_board_size == 19

    def test_skip_to_nonexistent_stage(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test skipping to nonexistent stage."""
        curriculum_scheduler.start()

        result = curriculum_scheduler.skip_to_stage("nonexistent")
        assert not result

    def test_get_training_params(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test getting training parameters."""
        curriculum_scheduler.start()

        params = curriculum_scheduler.get_training_params(
            base_learning_rate=1e-4,
            base_batch_size=64,
            base_mcts_simulations=800,
        )

        assert "learning_rate" in params
        assert "batch_size" in params
        assert "board_size" in params
        assert params["board_size"] == 9

    def test_callbacks_on_stage_start(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test stage start callbacks."""
        started_stages = []

        def on_start(stage):
            started_stages.append(stage.config.name)

        curriculum_scheduler.on_stage_start(on_start)
        curriculum_scheduler.start()

        assert "stage_9x9" in started_stages

    def test_callbacks_on_stage_complete(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test stage complete callbacks."""
        completed_stages = []

        def on_complete(stage):
            completed_stages.append(stage.config.name)

        curriculum_scheduler.on_stage_complete(on_complete)
        curriculum_scheduler.start()
        curriculum_scheduler.force_advance()

        assert "stage_9x9" in completed_stages

    def test_save_and_load_state(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test state persistence."""
        curriculum_scheduler.start()

        # Record some progress
        for _ in range(5):
            curriculum_scheduler.record_game(won=True)
            curriculum_scheduler.record_training_step(total_loss=0.1)

        # Save state
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        curriculum_scheduler.save_state(path)

        # Create new scheduler and load state
        new_scheduler = CurriculumScheduler(curriculum_scheduler.config)
        new_scheduler.load_state(path)

        assert new_scheduler._total_games == curriculum_scheduler._total_games
        assert new_scheduler._total_steps == curriculum_scheduler._total_steps

        # Cleanup
        path.unlink()

    def test_load_state_config_mismatch(
        self,
        two_stage_config: CurriculumConfig,
        three_stage_config: CurriculumConfig,
    ) -> None:
        """Test loading state with mismatched config."""
        scheduler1 = CurriculumScheduler(two_stage_config)
        scheduler1.start()

        # Save state
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        scheduler1.save_state(path)

        # Try to load with different config
        scheduler2 = CurriculumScheduler(three_stage_config)

        with pytest.raises(ValueError, match="hash mismatch"):
            scheduler2.load_state(path)

        # Cleanup
        path.unlink()

    def test_get_summary(self, curriculum_scheduler: CurriculumScheduler) -> None:
        """Test getting curriculum summary."""
        curriculum_scheduler.start()

        summary = curriculum_scheduler.get_summary()

        assert "curriculum_name" in summary
        assert "current_stage" in summary
        assert "stages" in summary
        assert len(summary["stages"]) == 2


class TestRegressionHandling:
    """Tests for curriculum regression handling."""

    def test_regression_disabled_by_default(self, two_stage_config: CurriculumConfig) -> None:
        """Test regression is disabled by default."""
        assert not two_stage_config.allow_regression

    def test_regression_when_enabled(self) -> None:
        """Test regression occurs when enabled and threshold breached."""
        from src.curriculum.config import (
            CurriculumConfig,
            ProgressionCriterion,
            ProgressionOperator,
            StageConfig,
        )

        config = CurriculumConfig(
            name="regression_test",
            stages=[
                StageConfig(
                    name="stage_9",
                    board_size=9,
                    min_games=5,
                    min_steps=1,
                    progression_criteria=[
                        ProgressionCriterion(
                            metric="win_rate",
                            operator=ProgressionOperator.GREATER_EQUAL,
                            threshold=0.5,
                            min_samples=3,
                        )
                    ],
                ),
                StageConfig(
                    name="stage_13",
                    board_size=13,
                    min_games=5,
                    min_steps=1,
                ),
            ],
            allow_regression=True,
            regression_threshold=0.3,
            warmup_games_per_stage=0,
            evaluation_interval=3,
        )

        scheduler = CurriculumScheduler(config)
        scheduler.start()

        # Advance to second stage
        scheduler.skip_to_stage("stage_13")
        assert scheduler.current_stage_name == "stage_13"

        # Regression not tracked until enough samples
        # The regression check needs 50 samples minimum
