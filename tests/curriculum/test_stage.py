"""Tests for curriculum stage management."""

from __future__ import annotations

import pytest

from src.curriculum.config import (
    ProgressionCriterion,
    ProgressionOperator,
    StageConfig,
)
from src.curriculum.stage import CurriculumStage, StageMetrics, StageStatus


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_terminal_states(self) -> None:
        """Test terminal state detection."""
        terminal = [StageStatus.COMPLETED, StageStatus.SKIPPED]
        non_terminal = [
            StageStatus.PENDING,
            StageStatus.ACTIVE,
            StageStatus.WARMUP,
            StageStatus.EVALUATING,
        ]

        for status in terminal:
            assert status.is_terminal()

        for status in non_terminal:
            assert not status.is_terminal()

    def test_active_states(self) -> None:
        """Test active state detection."""
        active = [StageStatus.ACTIVE, StageStatus.WARMUP, StageStatus.EVALUATING]
        inactive = [StageStatus.PENDING, StageStatus.COMPLETED, StageStatus.SKIPPED]

        for status in active:
            assert status.is_active()

        for status in inactive:
            assert not status.is_active()


class TestStageMetrics:
    """Tests for StageMetrics dataclass."""

    def test_initial_values(self) -> None:
        """Test initial metric values."""
        metrics = StageMetrics()
        assert metrics.games_played == 0
        assert metrics.win_rate == 0.0
        assert metrics.recent_win_rate == 0.0

    def test_record_win(self) -> None:
        """Test recording a win."""
        metrics = StageMetrics()
        metrics.record_game(won=True)

        assert metrics.games_played == 1
        assert metrics.games_won == 1
        assert metrics.win_rate == 1.0

    def test_record_loss(self) -> None:
        """Test recording a loss."""
        metrics = StageMetrics()
        metrics.record_game(won=False)

        assert metrics.games_played == 1
        assert metrics.games_lost == 1
        assert metrics.win_rate == 0.0

    def test_record_draw(self) -> None:
        """Test recording a draw."""
        metrics = StageMetrics()
        metrics.record_game(won=False, drawn=True)

        assert metrics.games_played == 1
        assert metrics.games_drawn == 1
        assert metrics.win_rate == 0.0  # Draws don't count as wins

    def test_win_rate_calculation(self) -> None:
        """Test win rate calculation."""
        metrics = StageMetrics()

        # 3 wins, 2 losses = 60% win rate
        for _ in range(3):
            metrics.record_game(won=True)
        for _ in range(2):
            metrics.record_game(won=False)

        assert metrics.win_rate == pytest.approx(0.6)

    def test_recent_win_rate(self) -> None:
        """Test recent win rate calculation."""
        metrics = StageMetrics()

        # Record some games
        for _ in range(5):
            metrics.record_game(won=True)
        for _ in range(5):
            metrics.record_game(won=False)

        assert metrics.recent_win_rate == pytest.approx(0.5)

    def test_record_training_step(self) -> None:
        """Test recording training step."""
        metrics = StageMetrics()
        metrics.record_training_step(total_loss=0.5, policy_loss=0.3, value_loss=0.2)

        assert metrics.training_steps == 1
        assert metrics.total_loss == 0.5
        assert metrics.policy_loss == 0.3
        assert metrics.value_loss == 0.2

    def test_average_loss(self) -> None:
        """Test average loss calculation."""
        metrics = StageMetrics()

        for loss in [0.5, 0.4, 0.3]:
            metrics.record_training_step(total_loss=loss)

        assert metrics.average_loss == pytest.approx(0.4)

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        metrics = StageMetrics()
        metrics.record_game(won=True)
        metrics.record_training_step(total_loss=0.5)

        data = metrics.to_dict()

        assert data["games_played"] == 1
        assert data["games_won"] == 1
        assert data["training_steps"] == 1
        assert "win_rate" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "games_played": 10,
            "games_won": 6,
            "games_lost": 4,
            "games_drawn": 0,
            "training_steps": 50,
        }

        metrics = StageMetrics.from_dict(data)

        assert metrics.games_played == 10
        assert metrics.games_won == 6
        assert metrics.win_rate == 0.6


class TestCurriculumStage:
    """Tests for CurriculumStage class."""

    def test_initial_state(self, curriculum_stage: CurriculumStage) -> None:
        """Test initial stage state."""
        assert curriculum_stage.status == StageStatus.PENDING
        assert curriculum_stage.metrics.games_played == 0
        assert curriculum_stage.start_time is None

    def test_activate(self, curriculum_stage: CurriculumStage) -> None:
        """Test stage activation."""
        curriculum_stage.activate()

        assert curriculum_stage.status == StageStatus.ACTIVE
        assert curriculum_stage.start_time is not None

    def test_activate_with_warmup(self, curriculum_stage: CurriculumStage) -> None:
        """Test stage activation with warmup."""
        curriculum_stage.activate(warmup_games=5)

        assert curriculum_stage.status == StageStatus.WARMUP
        assert curriculum_stage.warmup_games_remaining == 5

    def test_warmup_completion(self, curriculum_stage: CurriculumStage) -> None:
        """Test warmup period completion."""
        curriculum_stage.activate(warmup_games=3)

        for i in range(3):
            curriculum_stage.record_game(won=True)

        assert curriculum_stage.status == StageStatus.ACTIVE
        assert curriculum_stage.warmup_games_remaining <= 0

    def test_complete(self, curriculum_stage: CurriculumStage) -> None:
        """Test stage completion."""
        curriculum_stage.activate()
        curriculum_stage.complete()

        assert curriculum_stage.status == StageStatus.COMPLETED
        assert curriculum_stage.end_time is not None

    def test_skip(self, curriculum_stage: CurriculumStage) -> None:
        """Test stage skip."""
        curriculum_stage.skip()

        assert curriculum_stage.status == StageStatus.SKIPPED
        assert curriculum_stage.end_time is not None

    def test_board_size_property(self, curriculum_stage: CurriculumStage) -> None:
        """Test board size property."""
        assert curriculum_stage.board_size == 9

    def test_duration(self, curriculum_stage: CurriculumStage) -> None:
        """Test duration calculation."""
        assert curriculum_stage.duration_seconds is None

        curriculum_stage.activate()
        assert curriculum_stage.duration_seconds is not None
        assert curriculum_stage.duration_seconds >= 0

    def test_record_game(self, curriculum_stage: CurriculumStage) -> None:
        """Test recording games."""
        curriculum_stage.activate()
        curriculum_stage.record_game(won=True)

        assert curriculum_stage.metrics.games_played == 1
        assert curriculum_stage.metrics.games_won == 1

    def test_record_training_step(self, curriculum_stage: CurriculumStage) -> None:
        """Test recording training steps."""
        curriculum_stage.activate()
        curriculum_stage.record_training_step(total_loss=0.5)

        assert curriculum_stage.metrics.training_steps == 1

    def test_check_min_requirements_not_met(
        self, curriculum_stage: CurriculumStage
    ) -> None:
        """Test minimum requirements not met."""
        curriculum_stage.activate()

        # Play less than required
        for _ in range(5):
            curriculum_stage.record_game(won=True)
            curriculum_stage.record_training_step(total_loss=0.1)

        assert not curriculum_stage.check_min_requirements()

    def test_check_min_requirements_met(
        self, curriculum_stage: CurriculumStage
    ) -> None:
        """Test minimum requirements met."""
        curriculum_stage.activate()

        # Meet minimum games and steps
        for _ in range(12):
            curriculum_stage.record_game(won=True)
            curriculum_stage.record_training_step(total_loss=0.1)

        assert curriculum_stage.check_min_requirements()

    def test_check_max_limits_not_exceeded(
        self, simple_stage_config: StageConfig
    ) -> None:
        """Test max limits not exceeded."""
        config = StageConfig(
            name="limited",
            board_size=9,
            min_games=5,
            max_games=10,
            min_steps=5,
        )
        stage = CurriculumStage(config=config)
        stage.activate()

        for _ in range(5):
            stage.record_game(won=True)

        assert not stage.check_max_limits()

    def test_check_max_limits_exceeded(self) -> None:
        """Test max limits exceeded."""
        config = StageConfig(
            name="limited",
            board_size=9,
            min_games=5,
            max_games=10,
            min_steps=5,
        )
        stage = CurriculumStage(config=config)
        stage.activate()

        for _ in range(11):
            stage.record_game(won=True)

        assert stage.check_max_limits()

    def test_check_progression_criteria_during_warmup(
        self, curriculum_stage: CurriculumStage
    ) -> None:
        """Test progression check during warmup."""
        curriculum_stage.activate(warmup_games=10)

        # Even with good metrics, can't progress during warmup
        for _ in range(5):
            curriculum_stage.record_game(won=True)

        assert not curriculum_stage.check_progression_criteria()

    def test_check_progression_criteria_satisfied(
        self, curriculum_stage: CurriculumStage
    ) -> None:
        """Test progression criteria satisfied."""
        curriculum_stage.activate()

        # Meet minimum requirements with good win rate
        for _ in range(12):
            curriculum_stage.record_game(won=True)
            curriculum_stage.record_training_step(total_loss=0.1)

        assert curriculum_stage.check_progression_criteria()

    def test_check_progression_criteria_not_satisfied(
        self, curriculum_stage: CurriculumStage
    ) -> None:
        """Test progression criteria not satisfied."""
        curriculum_stage.activate()

        # Meet minimum requirements but poor win rate
        for _ in range(12):
            curriculum_stage.record_game(won=False)
            curriculum_stage.record_training_step(total_loss=0.1)

        assert not curriculum_stage.check_progression_criteria()

    def test_effective_learning_rate(
        self, curriculum_stage: CurriculumStage
    ) -> None:
        """Test effective learning rate calculation."""
        base_lr = 1e-4
        effective = curriculum_stage.get_effective_learning_rate(base_lr)
        assert effective == base_lr  # Default multiplier is 1.0

    def test_effective_batch_size(self, curriculum_stage: CurriculumStage) -> None:
        """Test effective batch size calculation."""
        base_batch = 64
        effective = curriculum_stage.get_effective_batch_size(base_batch)
        assert effective == base_batch  # Default multiplier is 1.0

    def test_to_dict(self, curriculum_stage: CurriculumStage) -> None:
        """Test serialization to dictionary."""
        curriculum_stage.activate()
        curriculum_stage.record_game(won=True)

        data = curriculum_stage.to_dict()

        assert data["config_name"] == "test_stage"
        assert data["board_size"] == 9
        assert data["status"] == "active"
        assert data["start_time"] is not None
