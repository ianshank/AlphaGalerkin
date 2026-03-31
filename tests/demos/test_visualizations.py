"""Tests for demo visualization components.

Tests cover:
- Board rendering
- Field heatmaps
- Chart generation
- Attention visualization
- Memory cleanup
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from src.demos.config import ColorScheme, VisualizationConfig
from src.demos.visualizations import (
    AttentionVisualizer,
    BoardVisualizer,
    ChartVisualizer,
    FieldVisualizer,
    PlotResult,
    figure_to_image,
    get_colormap,
)


class TestPlotResult:
    """Tests for PlotResult dataclass."""

    def test_creation(self) -> None:
        """Test PlotResult creation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = PlotResult(image=image)
        assert result.image.shape == (100, 100, 3)
        assert result.figure is None
        assert result.metadata == {}

    def test_with_metadata(self) -> None:
        """Test PlotResult with metadata."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = PlotResult(
            image=image,
            metadata={"board_size": 9, "has_policy": True},
        )
        assert result.metadata["board_size"] == 9
        assert result.metadata["has_policy"] is True

    def test_close_without_figure(self) -> None:
        """Test close() with no figure."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = PlotResult(image=image)
        # Should not raise
        result.close()


class TestGetColormap:
    """Tests for get_colormap utility."""

    def test_all_color_schemes(self) -> None:
        """Test all color schemes return valid colormaps."""
        for scheme in ColorScheme:
            cmap = get_colormap(scheme)
            assert cmap is not None
            # Test colormap can be called
            color = cmap(0.5)
            assert len(color) == 4  # RGBA

    def test_viridis(self) -> None:
        """Test viridis colormap specifically."""
        cmap = get_colormap(ColorScheme.VIRIDIS)
        assert cmap.name == "viridis"


class TestBoardVisualizer:
    """Tests for BoardVisualizer."""

    @pytest.fixture
    def visualizer(self) -> BoardVisualizer:
        """Create a BoardVisualizer instance."""
        return BoardVisualizer()

    def test_initialization(self, visualizer: BoardVisualizer) -> None:
        """Test visualizer initialization."""
        assert visualizer.config is not None
        assert isinstance(visualizer.config, VisualizationConfig)

    def test_render_empty_board(self, visualizer: BoardVisualizer) -> None:
        """Test rendering an empty board."""
        board = np.full((9, 9), -1, dtype=np.int8)  # -1 = empty
        result = visualizer.render_board(board)

        assert isinstance(result, PlotResult)
        assert result.image.shape[2] == 3  # RGB
        assert result.metadata["board_size"] == 9
        assert result.metadata["has_policy"] is False

        result.close()

    def test_render_board_with_stones(self, visualizer: BoardVisualizer) -> None:
        """Test rendering a board with stones."""
        board = np.full((9, 9), -1, dtype=np.int8)
        board[4, 4] = 0  # Black stone
        board[4, 5] = 1  # White stone

        result = visualizer.render_board(board)
        assert result.image.shape[2] == 3
        result.close()

    def test_render_board_with_last_move(self, visualizer: BoardVisualizer) -> None:
        """Test rendering with last move marker."""
        board = np.full((9, 9), -1, dtype=np.int8)
        board[4, 4] = 0

        result = visualizer.render_board(board, last_move=(4, 4))
        assert result.image.shape[2] == 3
        result.close()

    def test_render_board_with_policy(self, visualizer: BoardVisualizer) -> None:
        """Test rendering with policy heatmap."""
        board = np.full((9, 9), -1, dtype=np.int8)
        policy = np.random.rand(81).astype(np.float32)
        policy = policy / policy.sum()

        result = visualizer.render_board(board, policy=policy)
        assert result.metadata["has_policy"] is True
        result.close()

    def test_render_board_with_suggestions(self, visualizer: BoardVisualizer) -> None:
        """Test rendering with move suggestions."""
        board = np.full((9, 9), -1, dtype=np.int8)
        suggestions = [(4, 4, 0.5), (3, 3, 0.3), (5, 5, 0.2)]

        result = visualizer.render_board(board, suggested_moves=suggestions)
        assert result.metadata["has_suggestions"] is True
        result.close()

    def test_star_points_9x9(self, visualizer: BoardVisualizer) -> None:
        """Test star points for 9x9 board."""
        stars = visualizer._get_star_points(9)
        assert len(stars) == 5
        assert (4, 4) in stars  # Center

    def test_star_points_19x19(self, visualizer: BoardVisualizer) -> None:
        """Test star points for 19x19 board."""
        stars = visualizer._get_star_points(19)
        assert len(stars) == 9
        assert (9, 9) in stars  # Tengen


class TestFieldVisualizer:
    """Tests for FieldVisualizer."""

    @pytest.fixture
    def visualizer(self) -> FieldVisualizer:
        """Create a FieldVisualizer instance."""
        return FieldVisualizer()

    def test_initialization(self, visualizer: FieldVisualizer) -> None:
        """Test visualizer initialization."""
        assert visualizer.config is not None

    def test_render_field(self, visualizer: FieldVisualizer) -> None:
        """Test rendering a scalar field."""
        field = np.random.randn(16, 16).astype(np.float32)

        result = visualizer.render_field(field, title="Test Field")
        assert result.image.shape[2] == 3
        assert result.metadata["field_shape"] == (16, 16)
        assert "field_min" in result.metadata
        assert "field_max" in result.metadata
        assert "field_mean" in result.metadata
        result.close()

    def test_render_field_with_limits(self, visualizer: FieldVisualizer) -> None:
        """Test rendering with explicit value limits."""
        field = np.random.randn(16, 16).astype(np.float32)

        result = visualizer.render_field(field, vmin=-1.0, vmax=1.0)
        assert result.image.shape[2] == 3
        result.close()

    def test_render_comparison(self, visualizer: FieldVisualizer) -> None:
        """Test rendering ground truth vs prediction comparison."""
        gt = np.random.randn(16, 16).astype(np.float32)
        pred = gt + 0.1 * np.random.randn(16, 16).astype(np.float32)

        result = visualizer.render_comparison(
            ground_truth=gt,
            prediction=pred,
            title="Test Comparison",
        )

        assert result.image.shape[2] == 3
        assert "mse" in result.metadata
        assert "mae" in result.metadata
        assert result.metadata["mse"] > 0
        result.close()

    def test_render_comparison_without_diff(self, visualizer: FieldVisualizer) -> None:
        """Test comparison without difference map."""
        gt = np.random.randn(8, 8).astype(np.float32)
        pred = gt.copy()

        result = visualizer.render_comparison(
            ground_truth=gt,
            prediction=pred,
            show_difference=False,
        )
        assert result.image.shape[2] == 3
        result.close()

    def test_render_transfer_comparison(self, visualizer: FieldVisualizer) -> None:
        """Test multi-resolution transfer comparison."""
        results = {
            9: {
                "ground_truth": np.random.randn(9, 9).astype(np.float32),
                "prediction": np.random.randn(9, 9).astype(np.float32),
                "mse": 0.001,
            },
            13: {
                "ground_truth": np.random.randn(13, 13).astype(np.float32),
                "prediction": np.random.randn(13, 13).astype(np.float32),
                "mse": 0.002,
            },
        }

        result = visualizer.render_transfer_comparison(results)
        assert result.image.shape[2] == 3
        assert result.metadata["grid_sizes"] == [9, 13]
        result.close()


class TestChartVisualizer:
    """Tests for ChartVisualizer."""

    @pytest.fixture
    def visualizer(self) -> ChartVisualizer:
        """Create a ChartVisualizer instance."""
        return ChartVisualizer()

    def test_initialization(self, visualizer: ChartVisualizer) -> None:
        """Test visualizer initialization."""
        assert visualizer.config is not None

    def test_render_scaling_comparison(self, visualizer: ChartVisualizer) -> None:
        """Test scaling comparison chart."""
        sizes = [81, 169, 361]
        fnet_times = [1.0, 2.0, 3.5]
        softmax_times = [2.0, 8.0, 25.0]

        result = visualizer.render_scaling_comparison(
            sizes=sizes,
            fnet_times=fnet_times,
            softmax_times=softmax_times,
        )

        assert result.image.shape[2] == 3
        assert result.metadata["sizes"] == sizes
        assert result.metadata["speedups"] == [2.0, 4.0, pytest.approx(7.14, rel=0.1)]
        result.close()

    def test_render_mse_bar_chart(self, visualizer: ChartVisualizer) -> None:
        """Test MSE bar chart."""
        labels = ["9x9", "13x13", "19x19"]
        mse_values = [0.001, 0.002, 0.003]

        result = visualizer.render_mse_bar_chart(
            labels=labels,
            mse_values=mse_values,
            threshold=0.05,
        )

        assert result.image.shape[2] == 3
        assert result.metadata["all_below_threshold"] is True
        result.close()

    def test_render_mse_bar_chart_failure(self, visualizer: ChartVisualizer) -> None:
        """Test MSE bar chart with failures."""
        labels = ["9x9", "19x19"]
        mse_values = [0.01, 0.1]  # Second fails

        result = visualizer.render_mse_bar_chart(
            labels=labels,
            mse_values=mse_values,
            threshold=0.05,
        )

        assert result.metadata["all_below_threshold"] is False
        result.close()

    def test_render_attention_heatmap(self, visualizer: ChartVisualizer) -> None:
        """Test attention weight heatmap."""
        attention = np.random.rand(4, 64, 64).astype(np.float32)
        attention = attention / attention.sum(axis=-1, keepdims=True)

        result = visualizer.render_attention_heatmap(
            attention_weights=attention,
            head_idx=0,
        )

        assert result.image.shape[2] == 3
        assert result.metadata["head_idx"] == 0
        assert result.metadata["seq_length"] == 64
        result.close()

    def test_render_fourier_spectrum(self, visualizer: ChartVisualizer) -> None:
        """Test Fourier spectrum visualization."""
        frequencies = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        amplitudes = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float32)

        result = visualizer.render_fourier_spectrum(
            frequencies=frequencies,
            amplitudes=amplitudes,
        )

        assert result.image.shape[2] == 3
        assert result.metadata["n_frequencies"] == 5
        assert result.metadata["max_frequency"] == 8.0
        assert result.metadata["dominant_frequency"] == 0.5
        result.close()


class TestAttentionVisualizer:
    """Tests for AttentionVisualizer."""

    @pytest.fixture
    def visualizer(self) -> AttentionVisualizer:
        """Create an AttentionVisualizer instance."""
        return AttentionVisualizer()

    def test_initialization(self, visualizer: AttentionVisualizer) -> None:
        """Test visualizer initialization."""
        assert visualizer.config is not None
        assert visualizer.chart_viz is not None

    def test_render_galerkin_vs_softmax(self, visualizer: AttentionVisualizer) -> None:
        """Test Galerkin vs Softmax comparison."""
        seq_len = 64
        n_heads = 4

        galerkin_attn = np.random.rand(n_heads, seq_len, seq_len).astype(np.float32)
        galerkin_attn = galerkin_attn / galerkin_attn.sum(axis=-1, keepdims=True)

        softmax_attn = np.random.rand(n_heads, seq_len, seq_len).astype(np.float32)
        softmax_attn = np.exp(softmax_attn) / np.exp(softmax_attn).sum(axis=-1, keepdims=True)

        result = visualizer.render_galerkin_vs_softmax(
            galerkin_attn=galerkin_attn,
            softmax_attn=softmax_attn,
        )

        assert result.image.shape[2] == 3
        assert "galerkin_sparsity" in result.metadata
        assert "softmax_sparsity" in result.metadata
        assert "galerkin_entropy" in result.metadata
        assert "softmax_entropy" in result.metadata
        result.close()


class TestFigureToImage:
    """Tests for figure_to_image utility."""

    def test_basic_conversion(self) -> None:
        """Test basic figure to image conversion."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1])

        image = figure_to_image(fig)
        plt.close(fig)

        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_image_dimensions(self) -> None:
        """Test image dimensions match figure size."""
        import matplotlib.pyplot as plt

        dpi = 100
        width, height = 6, 4
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.plot([0, 1], [0, 1])

        image = figure_to_image(fig)
        plt.close(fig)

        # Image dimensions should be approximately width*dpi x height*dpi
        assert abs(image.shape[1] - width * dpi) < 10
        assert abs(image.shape[0] - height * dpi) < 10


class TestCustomConfig:
    """Tests for visualizers with custom configuration."""

    def test_board_visualizer_custom_config(self) -> None:
        """Test BoardVisualizer with custom config."""
        config = VisualizationConfig(
            figure_width=10.0,
            figure_height=10.0,
            dpi=150,
            board_wood_color="#d4a76a",
        )
        visualizer = BoardVisualizer(config)

        board = np.full((9, 9), -1, dtype=np.int8)
        result = visualizer.render_board(board)

        # Image should be larger due to higher resolution
        assert result.image.shape[0] > 500
        assert result.image.shape[1] > 500
        result.close()

    def test_field_visualizer_custom_colormap(self) -> None:
        """Test FieldVisualizer with custom color scheme."""
        config = VisualizationConfig(color_scheme=ColorScheme.PLASMA)
        visualizer = FieldVisualizer(config)

        field = np.random.randn(16, 16).astype(np.float32)
        result = visualizer.render_field(field)

        assert result.image.shape[2] == 3
        result.close()
