"""Tests for curriculum learning."""

from __future__ import annotations

import pytest

from src.training.curriculum import (
    BoardSizeCurriculum,
    CurriculumStage,
    create_default_curriculum,
)


class TestCurriculumStage:
    """Tests for CurriculumStage."""

    def test_basic_creation(self) -> None:
        """Test basic stage creation."""
        stage = CurriculumStage(start_step=0, board_sizes=[9])
        assert stage.start_step == 0
        assert stage.board_sizes == [9]
        assert stage.size_weights is None

    def test_with_weights(self) -> None:
        """Test stage with weights."""
        stage = CurriculumStage(
            start_step=100,
            board_sizes=[9, 13],
            size_weights=[0.7, 0.3],
        )
        assert stage.size_weights is not None
        # Weights should be normalized
        assert abs(sum(stage.size_weights) - 1.0) < 0.001

    def test_weight_length_mismatch(self) -> None:
        """Test error on weight length mismatch."""
        with pytest.raises(ValueError, match="size_weights length"):
            CurriculumStage(
                start_step=0,
                board_sizes=[9, 13],
                size_weights=[0.5],  # Wrong length
            )


class TestBoardSizeCurriculum:
    """Tests for BoardSizeCurriculum."""

    def test_from_schedule(self) -> None:
        """Test creation from schedule."""
        curriculum = BoardSizeCurriculum.from_schedule({
            0: [9],
            100: [9, 13],
        })
        assert len(curriculum.stages) == 2

    def test_from_config(self) -> None:
        """Test creation from config with string keys."""
        curriculum = BoardSizeCurriculum.from_config({
            "0": [9],
            "100": [9, 13],
        })
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(100) == [9, 13]

    def test_get_board_sizes(self) -> None:
        """Test getting board sizes at different steps."""
        curriculum = BoardSizeCurriculum.from_schedule({
            0: [9],
            100: [9, 13],
            200: [9, 13, 19],
        })
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(50) == [9]
        assert curriculum.get_board_sizes(100) == [9, 13]
        assert curriculum.get_board_sizes(150) == [9, 13]
        assert curriculum.get_board_sizes(200) == [9, 13, 19]
        assert curriculum.get_board_sizes(1000) == [9, 13, 19]

    def test_sample_board_size(self) -> None:
        """Test sampling board sizes."""
        curriculum = BoardSizeCurriculum.from_schedule({
            0: [9],
            100: [9, 13],
        })
        # At step 0, only 9 available
        for _ in range(10):
            assert curriculum.sample_board_size(0) == 9

        # At step 100, both available
        samples = [curriculum.sample_board_size(100) for _ in range(100)]
        assert 9 in samples
        assert 13 in samples

    def test_is_transition_step(self) -> None:
        """Test transition step detection."""
        curriculum = BoardSizeCurriculum.from_schedule({
            0: [9],
            100: [9, 13],
        })
        assert curriculum.is_transition_step(0)
        assert curriculum.is_transition_step(100)
        assert not curriculum.is_transition_step(50)

    def test_empty_curriculum_error(self) -> None:
        """Test error on empty curriculum."""
        with pytest.raises(ValueError, match="At least one"):
            BoardSizeCurriculum([])


class TestDefaultCurriculum:
    """Tests for default curriculum."""

    def test_create_default(self) -> None:
        """Test default curriculum creation."""
        curriculum = create_default_curriculum()
        # Should have 3 stages: 0, 10000, 50000
        assert len(curriculum.stages) == 3

    def test_default_progression(self) -> None:
        """Test default curriculum progression."""
        curriculum = create_default_curriculum()
        assert curriculum.get_board_sizes(0) == [9]
        assert curriculum.get_board_sizes(10000) == [9, 13]
        assert curriculum.get_board_sizes(50000) == [9, 13, 19]
