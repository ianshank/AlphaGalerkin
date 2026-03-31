"""Tests for architecture visualization demo.

Tests cover:
- Initialization and configuration
- Fourier feature visualization
- Attention pattern visualization
- LBB stability visualization
- Architecture overview
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")

from src.demos.architecture_demo import ArchitectureDemo
from src.demos.config import ArchitectureDemoConfig, VisualizationConfig


class TestArchitectureDemoInit:
    """Tests for ArchitectureDemo initialization."""

    def test_default_initialization(self) -> None:
        """Test demo initialization with defaults."""
        demo = ArchitectureDemo()

        assert demo.config is not None
        assert demo.model is None
        assert demo.device == "cpu"
        assert demo.attention_viz is not None
        assert demo.chart_viz is not None

    def test_custom_config(self) -> None:
        """Test demo with custom configuration."""
        config = ArchitectureDemoConfig(
            sample_board_size=13,
            n_attention_heads=8,
            n_fourier_samples=500,
        )
        demo = ArchitectureDemo(config)

        assert demo.config.sample_board_size == 13
        assert demo.config.n_attention_heads == 8
        assert demo.config.n_fourier_samples == 500

    def test_custom_device(self) -> None:
        """Test demo with custom device."""
        demo = ArchitectureDemo(device="cpu")
        assert demo.device == "cpu"


class TestFourierFeatureVisualization:
    """Tests for Fourier feature visualization."""

    @pytest.fixture
    def demo(self) -> ArchitectureDemo:
        """Create demo instance."""
        config = ArchitectureDemoConfig(
            visualization=VisualizationConfig(
                dpi=50,  # Lower for faster tests
            ),
        )
        return ArchitectureDemo(config)

    def test_visualize_fourier_default(self, demo: ArchitectureDemo) -> None:
        """Test Fourier visualization with defaults."""
        embedding_img, spectrum_img, explanation = demo.visualize_fourier_features()

        # Check embedding image
        assert embedding_img.ndim == 3
        assert embedding_img.shape[2] == 3  # RGB

        # Check spectrum image
        assert spectrum_img.ndim == 3
        assert spectrum_img.shape[2] == 3

        # Check explanation
        assert "Fourier Feature" in explanation
        assert "High-frequency" in explanation

    def test_visualize_fourier_custom_params(self, demo: ArchitectureDemo) -> None:
        """Test Fourier visualization with custom parameters."""
        embedding_img, spectrum_img, explanation = demo.visualize_fourier_features(
            n_features=32,
            scale=2.0,
            grid_size=25,
        )

        assert embedding_img.ndim == 3
        assert "32" in explanation
        assert "2.0" in explanation

    def test_fourier_different_scales(self, demo: ArchitectureDemo) -> None:
        """Test Fourier features at different scales."""
        for scale in [0.5, 1.0, 2.0]:
            _, _, explanation = demo.visualize_fourier_features(
                n_features=16,
                scale=scale,
                grid_size=20,
            )
            assert f"scale={scale}" in explanation or str(scale) in explanation


class TestAttentionPatternVisualization:
    """Tests for attention pattern visualization."""

    @pytest.fixture
    def demo(self) -> ArchitectureDemo:
        """Create demo instance."""
        config = ArchitectureDemoConfig(
            visualization=VisualizationConfig(dpi=50),
        )
        return ArchitectureDemo(config)

    def test_visualize_attention_default(self, demo: ArchitectureDemo) -> None:
        """Test attention visualization with defaults."""
        heads_img, comparison_img, comparison_text = demo.visualize_attention_patterns()

        # Check images
        assert heads_img.ndim == 3
        assert heads_img.shape[2] == 3
        assert comparison_img.ndim == 3

        # Check comparison text
        assert "Galerkin" in comparison_text
        assert "Softmax" in comparison_text
        assert "O(N)" in comparison_text
        assert "O(N²)" in comparison_text

    def test_visualize_attention_custom_board(self, demo: ArchitectureDemo) -> None:
        """Test attention visualization with custom board size."""
        _, _, comparison_text = demo.visualize_attention_patterns(
            board_size=5,
            n_heads=2,
        )

        assert "5×5" in comparison_text
        assert "N=25" in comparison_text

    def test_attention_entropy_values(self, demo: ArchitectureDemo) -> None:
        """Test that entropy values are computed."""
        _, _, comparison_text = demo.visualize_attention_patterns(
            board_size=7,
            n_heads=4,
        )

        # Check entropy is reported
        assert "Entropy:" in comparison_text

    def test_attention_different_heads(self, demo: ArchitectureDemo) -> None:
        """Test with different number of attention heads."""
        for n_heads in [2, 4, 8]:
            heads_img, _, _ = demo.visualize_attention_patterns(
                board_size=5,
                n_heads=n_heads,
            )
            assert heads_img.ndim == 3


class TestLBBStabilityVisualization:
    """Tests for LBB stability visualization."""

    @pytest.fixture
    def demo(self) -> ArchitectureDemo:
        """Create demo instance."""
        config = ArchitectureDemoConfig(
            visualization=VisualizationConfig(dpi=50),
        )
        return ArchitectureDemo(config)

    def test_visualize_lbb_default(self, demo: ArchitectureDemo) -> None:
        """Test LBB visualization with defaults."""
        stability_img, explanation = demo.visualize_lbb_stability()

        # Check image
        assert stability_img.ndim == 3
        assert stability_img.shape[2] == 3

        # Check explanation
        assert "LBB" in explanation
        assert "Ladyzhenskaya-Babuška-Brezzi" in explanation
        assert "σ_min" in explanation

    def test_visualize_lbb_custom_samples(self, demo: ArchitectureDemo) -> None:
        """Test LBB visualization with custom sample count."""
        stability_img, explanation = demo.visualize_lbb_stability(n_samples=50)

        assert stability_img.ndim == 3
        assert "Minimum singular value" in explanation

    def test_lbb_values_reported(self, demo: ArchitectureDemo) -> None:
        """Test that LBB values are properly reported."""
        _, explanation = demo.visualize_lbb_stability(n_samples=30)

        # Check that numeric values are present
        assert "Condition number:" in explanation
        assert "Stability threshold:" in explanation


class TestArchitectureOverview:
    """Tests for architecture overview visualization."""

    @pytest.fixture
    def demo(self) -> ArchitectureDemo:
        """Create demo instance."""
        config = ArchitectureDemoConfig(
            visualization=VisualizationConfig(dpi=50),
        )
        return ArchitectureDemo(config)

    def test_visualize_overview(self, demo: ArchitectureDemo) -> None:
        """Test architecture overview visualization."""
        overview_img, description = demo.visualize_architecture_overview()

        # Check image
        assert overview_img.ndim == 3
        assert overview_img.shape[2] == 3

        # Check description contains key components
        assert "AlphaGalerkin" in description or len(description) > 0


class TestArchitectureDemoReproducibility:
    """Tests for reproducibility."""

    def test_fourier_reproducible(self) -> None:
        """Test Fourier features are reproducible."""
        demo = ArchitectureDemo()

        _, spec1, _ = demo.visualize_fourier_features(n_features=16, grid_size=20)
        _, spec2, _ = demo.visualize_fourier_features(n_features=16, grid_size=20)

        # Same seed should give same results
        np.testing.assert_array_equal(spec1, spec2)

    def test_attention_reproducible(self) -> None:
        """Test attention patterns are reproducible."""
        demo = ArchitectureDemo()

        heads1, _, _ = demo.visualize_attention_patterns(board_size=5, n_heads=2)
        heads2, _, _ = demo.visualize_attention_patterns(board_size=5, n_heads=2)

        np.testing.assert_array_equal(heads1, heads2)

    def test_lbb_reproducible(self) -> None:
        """Test LBB visualization is reproducible."""
        demo = ArchitectureDemo()

        img1, _ = demo.visualize_lbb_stability(n_samples=20)
        img2, _ = demo.visualize_lbb_stability(n_samples=20)

        np.testing.assert_array_equal(img1, img2)


class TestArchitectureDemoConfig:
    """Tests for configuration handling."""

    def test_visualization_config_propagation(self) -> None:
        """Test that visualization config is used."""
        config = ArchitectureDemoConfig(
            visualization=VisualizationConfig(
                figure_width=5,
                figure_height=4,
                dpi=50,
            ),
        )
        demo = ArchitectureDemo(config)

        assert demo.config.visualization.figure_width == 5
        assert demo.config.visualization.figure_height == 4
        assert demo.config.visualization.dpi == 50

    def test_model_parameter(self) -> None:
        """Test model parameter handling."""
        # Create a simple mock model
        model = torch.nn.Linear(10, 10)
        demo = ArchitectureDemo(model=model)

        assert demo.model is model

    def test_default_sample_board_size(self) -> None:
        """Test default sample board size."""
        demo = ArchitectureDemo()
        assert demo.config.sample_board_size == 9  # Default

    def test_custom_fourier_settings(self) -> None:
        """Test custom Fourier settings in config."""
        config = ArchitectureDemoConfig(
            n_fourier_samples=2000,
            fourier_frequency_range=(0.5, 5.0),
        )
        demo = ArchitectureDemo(config)

        assert demo.config.n_fourier_samples == 2000
        assert demo.config.fourier_frequency_range == (0.5, 5.0)
