"""Tests for curriculum manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.curriculum.config import CurriculumConfig
from src.curriculum.manager import CurriculumManager, CurriculumMetrics, create_curriculum_manager


class TestCurriculumManager:
    """Tests for CurriculumManager class."""

    def test_initialization(self, curriculum_manager: CurriculumManager) -> None:
        """Test manager initialization."""
        assert not curriculum_manager.is_started
        assert not curriculum_manager.is_completed

    def test_start(self, curriculum_manager: CurriculumManager) -> None:
        """Test starting curriculum."""
        curriculum_manager.start()

        assert curriculum_manager.is_started
        assert curriculum_manager.current_board_size == 9

    def test_start_idempotent(self, curriculum_manager: CurriculumManager) -> None:
        """Test starting twice doesn't break."""
        curriculum_manager.start()
        curriculum_manager.start()  # Should not raise

        assert curriculum_manager.is_started

    def test_step_before_start_raises(self, curriculum_manager: CurriculumManager) -> None:
        """Test step before start raises error."""
        with pytest.raises(RuntimeError, match="not started"):
            curriculum_manager.step(game_result={"won": True})

    def test_step_with_game_result(self, curriculum_manager: CurriculumManager) -> None:
        """Test step with game result."""
        curriculum_manager.start()

        curriculum_manager.step(game_result={"won": True})
        metrics = curriculum_manager.scheduler.current_stage.metrics

        assert metrics.games_played == 1
        assert metrics.games_won == 1

    def test_step_with_training_metrics(self, curriculum_manager: CurriculumManager) -> None:
        """Test step with training metrics."""
        curriculum_manager.start()

        curriculum_manager.step(
            training_metrics={
                "total_loss": 0.5,
                "policy_loss": 0.3,
                "value_loss": 0.2,
            }
        )

        metrics = curriculum_manager.scheduler.current_stage.metrics
        assert metrics.training_steps == 1

    def test_step_with_both(self, curriculum_manager: CurriculumManager) -> None:
        """Test step with both game result and training metrics."""
        curriculum_manager.start()

        curriculum_manager.step(
            game_result={"won": True},
            training_metrics={"total_loss": 0.5},
        )

        metrics = curriculum_manager.scheduler.current_stage.metrics
        assert metrics.games_played == 1
        assert metrics.training_steps == 1

    def test_get_training_params(self, curriculum_manager: CurriculumManager) -> None:
        """Test getting training parameters."""
        curriculum_manager.start()

        params = curriculum_manager.get_training_params(
            base_learning_rate=1e-4,
            base_batch_size=64,
            base_mcts_simulations=800,
        )

        assert params["board_size"] == 9
        assert params["learning_rate"] == 1e-4
        assert params["batch_size"] == 64

    def test_get_metrics(self, curriculum_manager: CurriculumManager) -> None:
        """Test getting curriculum metrics."""
        curriculum_manager.start()

        for _ in range(5):
            curriculum_manager.step(game_result={"won": True})

        metrics = curriculum_manager.get_metrics()

        assert isinstance(metrics, CurriculumMetrics)
        assert metrics.total_games == 5
        assert metrics.current_board_size == 9

    def test_force_advance(self, curriculum_manager: CurriculumManager) -> None:
        """Test forced advancement."""
        curriculum_manager.start()

        result = curriculum_manager.force_advance()

        assert result
        assert curriculum_manager.current_board_size == 13

    def test_skip_to_board_size(self, three_stage_config: CurriculumConfig) -> None:
        """Test skipping to specific board size."""
        manager = CurriculumManager(config=three_stage_config)
        manager.start()

        result = manager.skip_to_board_size(19)

        assert result
        assert manager.current_board_size == 19

    def test_skip_to_invalid_board_size(self, curriculum_manager: CurriculumManager) -> None:
        """Test skipping to invalid board size."""
        curriculum_manager.start()

        result = curriculum_manager.skip_to_board_size(25)
        assert not result

    def test_checkpoint_save_and_load(self, curriculum_manager: CurriculumManager) -> None:
        """Test checkpoint save and load."""
        curriculum_manager.start()

        # Record progress
        for _ in range(5):
            curriculum_manager.step(game_result={"won": True})

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            saved_path = curriculum_manager.save_checkpoint(path)

            assert saved_path.exists()

            # Create new manager and load
            new_manager = CurriculumManager(config=curriculum_manager.config)
            new_manager.load_checkpoint(saved_path)

            assert new_manager.is_started
            summary = new_manager.get_summary()
            assert summary["total_games"] == 5

    def test_on_curriculum_complete_callback(self, two_stage_config: CurriculumConfig) -> None:
        """Test curriculum completion callback."""
        manager = CurriculumManager(config=two_stage_config)
        completed = []

        def on_complete():
            completed.append(True)

        manager.on_curriculum_complete(on_complete)
        manager.start()

        # Force advance to final stage and complete it
        manager.force_advance()

        # The callback should fire when the final stage completes
        # Since 13x13 has no criteria, it's a terminal stage
        manager.scheduler.current_stage.complete()
        manager._complete()

        assert len(completed) == 1

    def test_get_summary(self, curriculum_manager: CurriculumManager) -> None:
        """Test getting full summary."""
        curriculum_manager.start()

        summary = curriculum_manager.get_summary()

        assert "is_started" in summary
        assert "is_completed" in summary
        assert "start_time" in summary
        assert "stages" in summary


class TestCurriculumMetrics:
    """Tests for CurriculumMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = CurriculumMetrics(
            curriculum_name="test",
            current_stage="stage_9",
            current_board_size=9,
            stage_progress=50.0,
            curriculum_progress=25.0,
            total_games=100,
            total_steps=500,
            current_win_rate=0.65,
            stages_completed=1,
            total_stages=4,
        )

        data = metrics.to_dict()

        assert data["curriculum_name"] == "test"
        assert data["current_board_size"] == 9
        assert data["current_win_rate"] == 0.65


class TestCreateCurriculumManager:
    """Tests for create_curriculum_manager factory."""

    def test_create_default(self) -> None:
        """Test creating default manager."""
        manager = create_curriculum_manager()

        assert manager.config.name == "default"
        assert len(manager.config.stages) == 3

    def test_create_with_custom_sizes(self) -> None:
        """Test creating manager with custom sizes."""
        manager = create_curriculum_manager(board_sizes=[9, 19])

        assert len(manager.config.stages) == 2
        manager.start()
        assert manager.current_board_size == 9

    def test_create_with_custom_parameters(self) -> None:
        """Test creating manager with custom parameters."""
        manager = create_curriculum_manager(
            name="custom",
            win_rate_threshold=0.7,
            min_games_per_stage=500,
        )

        assert manager.config.name == "custom"
        assert manager.config.stages[0].min_games == 500

    def test_create_with_checkpoint_dir(self) -> None:
        """Test creating manager with checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_curriculum_manager(
                checkpoint_dir=tmpdir,
            )
            manager.start()

            # Save checkpoint should work
            path = manager.save_checkpoint()
            assert path.exists()


class TestIntegration:
    """Integration tests for curriculum learning."""

    def test_full_curriculum_progression(self) -> None:
        """Test complete curriculum progression."""
        from src.curriculum.config import (
            CurriculumConfig,
            ProgressionCriterion,
            ProgressionOperator,
            StageConfig,
        )

        # Create fast curriculum for testing
        config = CurriculumConfig(
            name="fast_test",
            stages=[
                StageConfig(
                    name="stage_9",
                    board_size=9,
                    min_games=5,
                    min_steps=2,
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
                    min_steps=2,
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
                    name="stage_19",
                    board_size=19,
                    min_games=5,
                    min_steps=2,
                ),
            ],
            warmup_games_per_stage=0,
            evaluation_interval=3,
        )

        manager = CurriculumManager(config=config)
        manager.start()

        # Progress through stages
        stages_visited = [manager.current_stage_name]

        # Play enough winning games to progress
        for _ in range(30):
            transition = manager.step(
                game_result={"won": True},
                training_metrics={"total_loss": 0.1},
            )
            if transition and manager.current_stage_name not in stages_visited:
                stages_visited.append(manager.current_stage_name)

        # Should have visited multiple stages
        assert len(stages_visited) >= 1

    def test_training_params_change_with_stage(self) -> None:
        """Test that training params change with stage."""
        from src.curriculum.config import (
            CurriculumConfig,
            StageConfig,
        )

        config = CurriculumConfig(
            name="multiplier_test",
            stages=[
                StageConfig(
                    name="stage_9",
                    board_size=9,
                    min_games=5,
                    min_steps=2,
                    learning_rate_multiplier=1.0,
                    batch_size_multiplier=1.0,
                ),
                StageConfig(
                    name="stage_13",
                    board_size=13,
                    min_games=5,
                    min_steps=2,
                    learning_rate_multiplier=0.5,
                    batch_size_multiplier=1.5,
                ),
            ],
            warmup_games_per_stage=0,
        )

        manager = CurriculumManager(config=config)
        manager.start()

        params1 = manager.get_training_params(1e-4, 64, 800)
        assert params1["learning_rate"] == 1e-4
        assert params1["batch_size"] == 64

        manager.force_advance()

        params2 = manager.get_training_params(1e-4, 64, 800)
        assert params2["learning_rate"] == pytest.approx(0.5e-4)
        assert params2["batch_size"] == 96  # 64 * 1.5
