"""Tests for curriculum learning configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.curriculum.config import (
    CurriculumConfig,
    ProgressionCriterion,
    ProgressionOperator,
    StageConfig,
    create_default_curriculum,
)


class TestProgressionOperator:
    """Tests for ProgressionOperator enum."""

    def test_greater_than(self) -> None:
        """Test greater than operator."""
        op = ProgressionOperator.GREATER_THAN
        assert op.evaluate(0.6, 0.5)
        assert not op.evaluate(0.5, 0.5)
        assert not op.evaluate(0.4, 0.5)

    def test_greater_equal(self) -> None:
        """Test greater or equal operator."""
        op = ProgressionOperator.GREATER_EQUAL
        assert op.evaluate(0.6, 0.5)
        assert op.evaluate(0.5, 0.5)
        assert not op.evaluate(0.4, 0.5)

    def test_less_than(self) -> None:
        """Test less than operator."""
        op = ProgressionOperator.LESS_THAN
        assert op.evaluate(0.4, 0.5)
        assert not op.evaluate(0.5, 0.5)
        assert not op.evaluate(0.6, 0.5)

    def test_less_equal(self) -> None:
        """Test less or equal operator."""
        op = ProgressionOperator.LESS_EQUAL
        assert op.evaluate(0.4, 0.5)
        assert op.evaluate(0.5, 0.5)
        assert not op.evaluate(0.6, 0.5)

    def test_equal(self) -> None:
        """Test equal operator."""
        op = ProgressionOperator.EQUAL
        assert op.evaluate(0.5, 0.5)
        assert op.evaluate(0.50000001, 0.5)  # Within epsilon
        assert not op.evaluate(0.6, 0.5)


class TestProgressionCriterion:
    """Tests for ProgressionCriterion model."""

    def test_valid_criterion(self) -> None:
        """Test creating valid criterion."""
        criterion = ProgressionCriterion(
            metric="win_rate",
            operator=ProgressionOperator.GREATER_EQUAL,
            threshold=0.55,
        )
        assert criterion.metric == "win_rate"
        assert criterion.threshold == 0.55

    def test_is_satisfied_with_enough_samples(self) -> None:
        """Test criterion satisfaction check."""
        criterion = ProgressionCriterion(
            metric="win_rate",
            operator=ProgressionOperator.GREATER_EQUAL,
            threshold=0.55,
            min_samples=10,
        )
        assert criterion.is_satisfied(0.6, n_samples=15)
        assert not criterion.is_satisfied(0.5, n_samples=15)

    def test_is_satisfied_not_enough_samples(self) -> None:
        """Test criterion requires minimum samples."""
        criterion = ProgressionCriterion(
            metric="win_rate",
            operator=ProgressionOperator.GREATER_EQUAL,
            threshold=0.55,
            min_samples=10,
        )
        # Even if value passes, not enough samples
        assert not criterion.is_satisfied(0.9, n_samples=5)

    def test_default_values(self) -> None:
        """Test default criterion values."""
        criterion = ProgressionCriterion(
            metric="loss",
            operator=ProgressionOperator.LESS_THAN,
            threshold=0.1,
        )
        assert criterion.min_samples == 10
        assert criterion.window_size == 100

    def test_invalid_min_samples(self) -> None:
        """Test validation rejects invalid min_samples."""
        with pytest.raises(ValidationError):
            ProgressionCriterion(
                metric="win_rate",
                operator=ProgressionOperator.GREATER_EQUAL,
                threshold=0.5,
                min_samples=0,
            )


class TestStageConfig:
    """Tests for StageConfig model."""

    def test_valid_stage(self) -> None:
        """Test creating valid stage config."""
        stage = StageConfig(
            name="stage_9x9",
            board_size=9,
            min_games=100,
            min_steps=50,
        )
        assert stage.name == "stage_9x9"
        assert stage.board_size == 9

    def test_board_size_constraints(self) -> None:
        """Test board size validation."""
        with pytest.raises(ValidationError):
            StageConfig(name="invalid", board_size=4)  # Too small

        with pytest.raises(ValidationError):
            StageConfig(name="invalid", board_size=30)  # Too large

    def test_max_greater_than_min_games(self) -> None:
        """Test max_games must be >= min_games."""
        with pytest.raises(ValidationError):
            StageConfig(
                name="invalid",
                board_size=9,
                min_games=100,
                max_games=50,
            )

    def test_max_greater_than_min_steps(self) -> None:
        """Test max_steps must be >= min_steps."""
        with pytest.raises(ValidationError):
            StageConfig(
                name="invalid",
                board_size=9,
                min_steps=100,
                max_steps=50,
            )

    def test_criteria_mode_validation(self) -> None:
        """Test criteria_mode validation."""
        # Valid modes
        StageConfig(name="test", board_size=9, criteria_mode="all")
        StageConfig(name="test", board_size=9, criteria_mode="any")

        # Invalid mode
        with pytest.raises(ValidationError):
            StageConfig(name="test", board_size=9, criteria_mode="invalid")

    def test_multiplier_constraints(self) -> None:
        """Test learning rate and batch size multiplier constraints."""
        with pytest.raises(ValidationError):
            StageConfig(
                name="test",
                board_size=9,
                learning_rate_multiplier=0,
            )

        with pytest.raises(ValidationError):
            StageConfig(
                name="test",
                board_size=9,
                batch_size_multiplier=20,
            )


class TestCurriculumConfig:
    """Tests for CurriculumConfig model."""

    def test_valid_curriculum(self, two_stage_config: CurriculumConfig) -> None:
        """Test creating valid curriculum config."""
        assert two_stage_config.name == "test_curriculum"
        assert len(two_stage_config.stages) == 2

    def test_stages_must_be_ordered(self) -> None:
        """Test stages must be ordered by board size."""
        with pytest.raises(ValidationError, match="ordered"):
            CurriculumConfig(
                name="invalid",
                stages=[
                    StageConfig(name="large", board_size=19),
                    StageConfig(name="small", board_size=9),
                ],
            )

    def test_unique_stage_names(self) -> None:
        """Test stage names must be unique."""
        with pytest.raises(ValidationError, match="unique"):
            CurriculumConfig(
                name="invalid",
                stages=[
                    StageConfig(name="same", board_size=9),
                    StageConfig(name="same", board_size=13),
                ],
            )

    def test_minimum_one_stage(self) -> None:
        """Test curriculum requires at least one stage."""
        with pytest.raises(ValidationError):
            CurriculumConfig(name="empty", stages=[])

    def test_compute_hash(self, two_stage_config: CurriculumConfig) -> None:
        """Test hash computation."""
        hash1 = two_stage_config.compute_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Same config should produce same hash
        hash2 = two_stage_config.compute_hash()
        assert hash1 == hash2

    def test_get_stage(self, two_stage_config: CurriculumConfig) -> None:
        """Test getting stage by name."""
        stage = two_stage_config.get_stage("stage_9x9")
        assert stage is not None
        assert stage.board_size == 9

        assert two_stage_config.get_stage("nonexistent") is None

    def test_get_stage_index(self, two_stage_config: CurriculumConfig) -> None:
        """Test getting stage index."""
        assert two_stage_config.get_stage_index("stage_9x9") == 0
        assert two_stage_config.get_stage_index("stage_13x13") == 1
        assert two_stage_config.get_stage_index("nonexistent") == -1


class TestCreateDefaultCurriculum:
    """Tests for create_default_curriculum factory."""

    def test_default_curriculum(self) -> None:
        """Test creating default curriculum."""
        config = create_default_curriculum()
        assert config.name == "default"
        assert len(config.stages) == 3

        sizes = [s.board_size for s in config.stages]
        assert sizes == [9, 13, 19]

    def test_custom_board_sizes(self) -> None:
        """Test custom board sizes."""
        config = create_default_curriculum(board_sizes=[9, 19])
        assert len(config.stages) == 2
        assert config.stages[0].board_size == 9
        assert config.stages[1].board_size == 19

    def test_custom_parameters(self) -> None:
        """Test custom curriculum parameters."""
        config = create_default_curriculum(
            name="custom",
            win_rate_threshold=0.7,
            min_games_per_stage=500,
        )
        assert config.name == "custom"
        assert config.stages[0].min_games == 500

        # Check criterion threshold
        criterion = config.stages[0].progression_criteria[0]
        assert criterion.threshold == 0.7

    def test_terminal_stage_no_criteria(self) -> None:
        """Test terminal stage has no progression criteria."""
        config = create_default_curriculum()
        last_stage = config.stages[-1]
        assert len(last_stage.progression_criteria) == 0
