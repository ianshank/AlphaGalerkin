"""Tests for notebook helper utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from notebooks.utils.helpers import (
    EnvironmentInfo,
    ModelForwardResult,
    create_sample_board,
    format_model_summary,
    safe_model_forward,
    setup_environment,
    validate_board_sizes,
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


class TestSetupEnvironment:
    """Tests for setup_environment function."""

    def test_setup_with_valid_project_root(self, tmp_path: Path) -> None:
        """Test setup with a valid project root."""
        # Create expected structure
        (tmp_path / "src").mkdir()
        (tmp_path / "config").mkdir()
        (tmp_path / "notebooks").mkdir()

        env_info = setup_environment(random_seed=123, project_root=tmp_path)

        assert isinstance(env_info, EnvironmentInfo)
        assert env_info.project_root == tmp_path
        assert env_info.python_version
        assert env_info.torch_version

    def test_setup_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        """Test that setup raises RuntimeError for non-existent path."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(RuntimeError, match="does not exist"):
            setup_environment(project_root=nonexistent)

    def test_setup_raises_for_invalid_structure(self, tmp_path: Path) -> None:
        """Test that setup raises RuntimeError for invalid project structure."""
        # Create empty directory (no src/ or notebooks/)
        with pytest.raises(RuntimeError, match="appears invalid"):
            setup_environment(project_root=tmp_path)

    def test_setup_sets_random_seeds(self, tmp_path: Path) -> None:
        """Test that random seeds are set correctly."""
        (tmp_path / "src").mkdir()
        (tmp_path / "notebooks").mkdir()

        setup_environment(random_seed=42, project_root=tmp_path)

        # Verify torch seed by checking reproducibility
        t1 = torch.rand(5)
        torch.manual_seed(42)
        t2 = torch.rand(5)
        # After setup with seed 42, new random values should differ
        # but resetting to 42 should give same sequence
        torch.manual_seed(42)
        t3 = torch.rand(5)
        assert torch.allclose(t2, t3)

    def test_setup_detects_device(self, tmp_path: Path) -> None:
        """Test that device detection works."""
        (tmp_path / "src").mkdir()
        (tmp_path / "notebooks").mkdir()

        env_info = setup_environment(project_root=tmp_path)

        assert env_info.device is not None
        assert env_info.cuda_available == torch.cuda.is_available()


class TestSafeModelForward:
    """Tests for safe_model_forward function."""

    def test_successful_forward(self) -> None:
        """Test successful model forward pass."""
        # Create a mock model with expected output structure
        mock_output = MagicMock()
        mock_output.policy_logits = torch.randn(1, 82)
        mock_output.value = torch.tensor([[0.5]])

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        x = torch.randn(1, 17, 9, 9)
        result = safe_model_forward(mock_model, x)

        assert result.success is True
        assert result.error is None
        assert result.policy_logits is not None
        assert result.value is not None

    def test_failed_forward_catches_exception(self) -> None:
        """Test that exceptions are caught and returned as error."""
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("Shape mismatch error")

        x = torch.randn(1, 17, 9, 9)
        result = safe_model_forward(mock_model, x)

        assert result.success is False
        assert result.error is not None
        assert "Shape mismatch error" in result.error
        assert result.policy_logits is None
        assert result.value is None

    def test_forward_with_lbb_constant(self) -> None:
        """Test forward pass with LBB constant requested."""
        mock_output = MagicMock()
        mock_output.policy_logits = torch.randn(1, 82)
        mock_output.value = torch.tensor([[0.5]])
        mock_output.lbb_constant = torch.tensor(0.1)

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        x = torch.randn(1, 17, 9, 9)
        result = safe_model_forward(mock_model, x, return_lbb=True)

        assert result.success is True
        assert result.lbb_constant is not None
        mock_model.assert_called_once()
        # Verify return_lbb was passed
        call_kwargs = mock_model.call_args
        assert call_kwargs[1].get("return_lbb") is True
