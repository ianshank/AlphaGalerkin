"""Tests for notebook visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestPlotAttentionComparison:
    """Tests for plot_attention_comparison function."""

    def test_valid_inputs(self) -> None:
        """Test plotting with valid inputs."""
        from notebooks.utils.visualization import plot_attention_comparison

        galerkin_times = [1.0, 2.0, 3.0]
        softmax_times = [2.0, 4.0, 8.0]
        board_labels = ["5×5", "9×9", "13×13"]

        fig = plot_attention_comparison(galerkin_times, softmax_times, board_labels)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_sequences_raise_error(self) -> None:
        """Test that empty sequences raise ValueError."""
        from notebooks.utils.visualization import plot_attention_comparison

        with pytest.raises(ValueError, match="cannot be empty"):
            plot_attention_comparison([], [], [])

    def test_mismatched_lengths_raise_error(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        from notebooks.utils.visualization import plot_attention_comparison

        with pytest.raises(ValueError, match="mismatch"):
            plot_attention_comparison([1.0, 2.0], [1.0], ["a", "b"])

    def test_label_count_mismatch_raises_error(self) -> None:
        """Test that label count mismatch raises ValueError."""
        from notebooks.utils.visualization import plot_attention_comparison

        with pytest.raises(ValueError, match="Label count mismatch"):
            plot_attention_comparison([1.0, 2.0], [2.0, 4.0], ["a"])


class TestPlotPoissonSamples:
    """Tests for plot_poisson_samples function."""

    def test_valid_samples(self) -> None:
        """Test plotting with valid samples."""
        from notebooks.utils.visualization import plot_poisson_samples

        @dataclass
        class MockSample:
            grid_size: int
            charges: np.ndarray
            potential: np.ndarray

        samples = [
            MockSample(grid_size=9, charges=np.random.randn(81), potential=np.random.randn(81)),
            MockSample(
                grid_size=13,
                charges=np.random.randn(169),
                potential=np.random.randn(169),
            ),
        ]

        fig = plot_poisson_samples(samples)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_samples_raise_error(self) -> None:
        """Test that empty samples raise ValueError."""
        from notebooks.utils.visualization import plot_poisson_samples

        with pytest.raises(ValueError, match="cannot be empty"):
            plot_poisson_samples([])

    def test_missing_attributes_raise_error(self) -> None:
        """Test that samples missing attributes raise AttributeError."""
        from notebooks.utils.visualization import plot_poisson_samples

        class IncompletesSample:
            grid_size: int = 9
            # Missing charges and potential

        with pytest.raises(AttributeError, match="missing required attribute"):
            plot_poisson_samples([IncompletesSample()])


class TestPlotGoBoard:
    """Tests for plot_go_board function."""

    def test_valid_board(self) -> None:
        """Test plotting a valid board."""
        import matplotlib.pyplot as plt

        from notebooks.utils.visualization import plot_go_board

        fig, ax = plt.subplots()
        board = torch.zeros(1, 17, 9, 9)
        board[0, 0, 3, 3] = 1  # Black stone
        board[0, 1, 4, 4] = 1  # White stone

        # Should not raise
        plot_go_board(board, ax)

        plt.close(fig)

    def test_empty_board(self) -> None:
        """Test plotting an empty board."""
        import matplotlib.pyplot as plt

        from notebooks.utils.visualization import plot_go_board

        fig, ax = plt.subplots()
        board = torch.zeros(1, 17, 9, 9)

        # Should not raise
        plot_go_board(board, ax)

        plt.close(fig)


class TestPlotPolicyHeatmap:
    """Tests for plot_policy_heatmap function."""

    def test_valid_policy(self) -> None:
        """Test plotting with valid policy logits."""
        import matplotlib.pyplot as plt

        from notebooks.utils.visualization import plot_policy_heatmap

        fig, ax = plt.subplots()
        policy_logits = torch.randn(1, 82)  # 9x9 board + pass

        im = plot_policy_heatmap(policy_logits, board_size=9, ax=ax)

        assert im is not None
        plt.close(fig)

    def test_returns_axes_image(self) -> None:
        """Test that function returns AxesImage for colorbar attachment."""
        import matplotlib.pyplot as plt
        from matplotlib.image import AxesImage

        from notebooks.utils.visualization import plot_policy_heatmap

        fig, ax = plt.subplots()
        policy_logits = torch.randn(1, 82)

        im = plot_policy_heatmap(policy_logits, board_size=9, ax=ax)

        assert isinstance(im, AxesImage)
        plt.close(fig)

    def test_shape_mismatch_logs_warning(self) -> None:
        """Test that shape mismatch logs warning but continues."""
        import matplotlib.pyplot as plt

        from notebooks.utils.visualization import plot_policy_heatmap

        fig, ax = plt.subplots()
        # Wrong size: 25 + 1 = 26, but passing board_size=9 expects 82
        policy_logits = torch.randn(1, 26)

        # Should not raise, just warn
        im = plot_policy_heatmap(policy_logits, board_size=5, ax=ax)

        assert im is not None
        plt.close(fig)


class TestPlotFourierFeatures:
    """Tests for plot_fourier_features function."""

    def test_empty_board_sizes_raise_error(self) -> None:
        """Test that empty board sizes raise ValueError."""
        from notebooks.utils.visualization import plot_fourier_features

        mock_encoder = MagicMock()

        with pytest.raises(ValueError, match="cannot be empty"):
            plot_fourier_features(mock_encoder, board_sizes=[])


class TestPlotMultiBoardVisualization:
    """Tests for plot_multi_board_visualization function."""

    def test_valid_boards(self) -> None:
        """Test plotting multiple boards."""
        from notebooks.utils.visualization import plot_multi_board_visualization

        boards = [
            torch.zeros(1, 17, 9, 9),
            torch.zeros(1, 17, 13, 13),
        ]
        board_sizes = [9, 13]

        fig = plot_multi_board_visualization(boards, board_sizes)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_board(self) -> None:
        """Test plotting a single board."""
        from notebooks.utils.visualization import plot_multi_board_visualization

        boards = [torch.zeros(1, 17, 9, 9)]
        board_sizes = [9]

        fig = plot_multi_board_visualization(boards, board_sizes)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)
