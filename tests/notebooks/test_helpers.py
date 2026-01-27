"""Tests for notebook helper utilities."""

from __future__ import annotations

import pytest
import torch

from notebooks.utils.helpers import (
    create_sample_board,
    validate_board_sizes,
    format_model_summary,
    ModelForwardResult,
)


class TestCreateSampleBoard:
    """Tests for create_sample_board function."""

    def test_default_positions(self) -> None:
        """Test board creation with default positions."""
        board = create_sample_board(size=19)
        assert board.shape == (1, 17, 19, 19)
        # Check black stones exist
        assert board[0, 0, 3, 3] == 1
        # Check white stones exist
        assert board[0, 1, 3, 4] == 1

    def test_custom_size(self) -> None:
        """Test board creation with custom size."""
        board = create_sample_board(size=9)
        assert board.shape == (1, 17, 9, 9)

    def test_custom_positions(self) -> None:
        """Test board creation with custom positions."""
        board = create_sample_board(
            size=9,
            black_positions=[(0, 0), (1, 1)],
            white_positions=[(2, 2)],
        )
        assert board[0, 0, 0, 0] == 1
        assert board[0, 0, 1, 1] == 1
        assert board[0, 1, 2, 2] == 1

    def test_positions_out_of_bounds_ignored(self) -> None:
        """Test that out-of-bounds positions are ignored."""
        board = create_sample_board(
            size=5,
            black_positions=[(10, 10)],  # Out of bounds
            white_positions=[],
        )
        # Should not raise error, just skip the position
        assert board.sum() == 0  # No stones placed

    def test_custom_channels(self) -> None:
        """Test board creation with custom channels."""
        board = create_sample_board(size=9, n_channels=3)
        assert board.shape == (1, 3, 9, 9)

    def test_device_placement(self) -> None:
        """Test board is placed on correct device."""
        board = create_sample_board(size=9, device="cpu")
        assert board.device.type == "cpu"


class TestValidateBoardSizes:
    """Tests for validate_board_sizes function."""

    def test_valid_sizes(self) -> None:
        """Test validation of valid sizes."""
        assert validate_board_sizes([5, 9, 13, 19]) is True

    def test_invalid_size_too_small(self) -> None:
        """Test validation fails for too-small sizes."""
        with pytest.raises(ValueError, match="out of range"):
            validate_board_sizes([2])

    def test_invalid_size_too_large(self) -> None:
        """Test validation fails for too-large sizes."""
        with pytest.raises(ValueError, match="out of range"):
            validate_board_sizes([30])

    def test_invalid_type(self) -> None:
        """Test validation fails for non-integer types."""
        with pytest.raises(ValueError, match="must be int"):
            validate_board_sizes([9.5])  # type: ignore

    def test_custom_range(self) -> None:
        """Test validation with custom range."""
        assert validate_board_sizes([3], min_size=3, max_size=5) is True
        with pytest.raises(ValueError):
            validate_board_sizes([6], min_size=3, max_size=5)


class TestFormatModelSummary:
    """Tests for format_model_summary function."""

    def test_format_simple_model(self) -> None:
        """Test formatting a simple model."""
        model = torch.nn.Linear(10, 5)
        summary = format_model_summary(model)
        assert "Linear" in summary
        assert "Total parameters" in summary
        assert "Trainable parameters" in summary

    def test_format_contains_param_count(self) -> None:
        """Test that summary contains parameter count."""
        model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        summary = format_model_summary(model)
        assert "55" in summary


class TestModelForwardResult:
    """Tests for ModelForwardResult dataclass."""

    def test_successful_result(self) -> None:
        """Test successful forward result."""
        result = ModelForwardResult(
            success=True,
            policy_logits=torch.randn(1, 82),
            value=torch.tensor([[0.5]]),
            lbb_constant=None,
            error=None,
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed forward result."""
        result = ModelForwardResult(
            success=False,
            policy_logits=None,
            value=None,
            lbb_constant=None,
            error="Shape mismatch",
        )
        assert result.success is False
        assert result.error == "Shape mismatch"
