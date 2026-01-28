"""Tests for multi-scale Fourier features."""

import pytest
import torch

from src.modeling.multiscale_fourier import (
    AdaptiveFourierFeatures,
    FourierFeaturesConfig,
    MultiScaleFourierFeatures,
    PositionalEncoding,
    ProgressiveFourierFeatures,
    SpatialPositionalEncoding,
)


class TestFourierFeaturesConfig:
    """Tests for FourierFeaturesConfig."""

    def test_create_default_config(self) -> None:
        """Test creating default config."""
        config = FourierFeaturesConfig(name="test")
        assert config.n_features == 128
        assert len(config.scales) == 5
        assert config.learnable is True

    def test_custom_scales(self) -> None:
        """Test custom frequency scales."""
        config = FourierFeaturesConfig(
            name="test",
            scales=[1.0, 10.0, 100.0],
        )
        assert config.scales == [1.0, 10.0, 100.0]


class TestMultiScaleFourierFeatures:
    """Tests for MultiScaleFourierFeatures."""

    def test_output_dimension(self) -> None:
        """Test output dimension calculation."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=64,
            scales=[1.0, 2.0, 4.0],
            include_input=True,
        )
        # 2 * 64 * 3 (sin + cos at each scale) + 2 (input)
        expected = 2 * 64 * 3 + 2
        assert features.output_dim == expected

    def test_output_dimension_no_input(self) -> None:
        """Test output dimension without raw input."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=64,
            scales=[1.0, 2.0],
            include_input=False,
        )
        # 2 * 64 * 2 (sin + cos at each scale)
        expected = 2 * 64 * 2
        assert features.output_dim == expected

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0, 2.0, 4.0],
        )
        x = torch.randn(8, 100, 2)  # batch, n_points, input_dim
        output = features(x)

        assert output.shape[0] == 8
        assert output.shape[1] == 100
        assert output.shape[2] == features.output_dim

    def test_forward_2d_input(self) -> None:
        """Test forward pass with 2D input."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0],
        )
        x = torch.randn(100, 2)  # n_points, input_dim
        output = features(x)

        assert output.shape[0] == 100
        assert output.shape[1] == features.output_dim

    def test_learnable_frequencies(self) -> None:
        """Test that frequencies are learnable when configured."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=16,
            scales=[1.0, 2.0],
            learnable=True,
        )
        assert features.frequency_matrices is not None
        assert all(p.requires_grad for p in features.frequency_matrices)

    def test_fixed_frequencies(self) -> None:
        """Test fixed (non-learnable) frequencies."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=16,
            scales=[1.0, 2.0],
            learnable=False,
        )
        assert features.frequency_matrices is None
        assert hasattr(features, "B_0")
        assert hasattr(features, "B_1")

    def test_get_scale_features(self) -> None:
        """Test getting features from specific scale."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0, 2.0, 4.0],
        )
        x = torch.randn(8, 100, 2)
        scale_feat = features.get_scale_features(x, scale_idx=1)

        assert scale_feat.shape[0] == 8
        assert scale_feat.shape[1] == 100
        assert scale_feat.shape[2] == 2 * 32  # sin + cos

    def test_invalid_scale_index_raises(self) -> None:
        """Test invalid scale index raises error."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0, 2.0],
        )
        x = torch.randn(8, 100, 2)

        with pytest.raises(ValueError):
            features.get_scale_features(x, scale_idx=5)

    def test_different_scales_produce_different_features(self) -> None:
        """Test that different scales produce different features."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0, 10.0],
            learnable=False,
        )
        x = torch.randn(4, 10, 2)

        feat_scale0 = features.get_scale_features(x, scale_idx=0)
        feat_scale1 = features.get_scale_features(x, scale_idx=1)

        # Features should be different
        assert not torch.allclose(feat_scale0, feat_scale1)


class TestAdaptiveFourierFeatures:
    """Tests for AdaptiveFourierFeatures."""

    def test_output_dimension(self) -> None:
        """Test output dimension."""
        features = AdaptiveFourierFeatures(
            input_dim=2,
            n_features=64,
            n_frequency_banks=4,
        )
        # 2 * n_features + input_dim
        expected = 2 * 64 + 2
        assert features.output_dim == expected

    def test_forward_shape(self) -> None:
        """Test forward pass shape."""
        features = AdaptiveFourierFeatures(
            input_dim=2,
            n_features=32,
            n_frequency_banks=4,
        )
        x = torch.randn(4, 50, 2)
        output = features(x)

        assert output.shape == (4, 50, features.output_dim)

    def test_attention_weights(self) -> None:
        """Test attention-based weighting."""
        features = AdaptiveFourierFeatures(
            input_dim=2,
            n_features=32,
            n_frequency_banks=4,
            use_attention=True,
        )
        x = torch.randn(4, 50, 2)
        output = features(x)

        # Output should be valid
        assert not torch.isnan(output).any()

    def test_no_attention_mode(self) -> None:
        """Test without attention (simple average)."""
        features = AdaptiveFourierFeatures(
            input_dim=2,
            n_features=32,
            n_frequency_banks=4,
            use_attention=False,
        )
        x = torch.randn(4, 50, 2)
        output = features(x)

        assert output.shape == (4, 50, features.output_dim)


class TestProgressiveFourierFeatures:
    """Tests for ProgressiveFourierFeatures."""

    def test_initial_progress(self) -> None:
        """Test initial progress is 1.0 (all scales active)."""
        features = ProgressiveFourierFeatures(input_dim=2, n_features=32)
        assert features.progress == 1.0

    def test_set_progress(self) -> None:
        """Test setting progress."""
        features = ProgressiveFourierFeatures(input_dim=2, n_features=32)
        features.set_progress(0.5)
        assert features.progress == 0.5

    def test_progress_clamping(self) -> None:
        """Test progress is clamped to [0, 1]."""
        features = ProgressiveFourierFeatures(input_dim=2, n_features=32)

        features.set_progress(-0.5)
        assert features.progress == 0.0

        features.set_progress(1.5)
        assert features.progress == 1.0

    def test_low_progress_reduces_high_frequencies(self) -> None:
        """Test that low progress reduces high-frequency components."""
        features = ProgressiveFourierFeatures(
            input_dim=2,
            n_features=16,
            scales=[1.0, 10.0, 100.0],
        )
        x = torch.randn(4, 20, 2)

        # Full progress
        features.set_progress(1.0)
        output_full = features(x)

        # Low progress
        features.set_progress(0.1)
        output_low = features(x)

        # Low progress should have smaller high-frequency contributions
        # (gated by progress)
        assert not torch.allclose(output_full, output_low)

    def test_output_shape(self) -> None:
        """Test output shape."""
        features = ProgressiveFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0, 2.0, 4.0],
        )
        x = torch.randn(4, 50, 2)
        output = features(x)

        assert output.shape == (4, 50, features.output_dim)


class TestPositionalEncoding:
    """Tests for standard positional encoding."""

    def test_output_shape(self) -> None:
        """Test output shape matches input."""
        pe = PositionalEncoding(d_model=64, max_len=1000)
        x = torch.randn(4, 100, 64)
        output = pe(x)

        assert output.shape == x.shape

    def test_encoding_added(self) -> None:
        """Test that encoding is added (not just returned)."""
        pe = PositionalEncoding(d_model=64, max_len=1000)
        x = torch.zeros(4, 50, 64)
        output = pe(x)

        # Output should have non-zero values from positional encoding
        assert not torch.allclose(output, x)

    def test_positional_pattern(self) -> None:
        """Test that different positions have different encodings."""
        pe = PositionalEncoding(d_model=64, max_len=1000)
        x = torch.zeros(1, 10, 64)
        output = pe(x)

        # Different positions should have different encodings
        assert not torch.allclose(output[0, 0], output[0, 5])


class TestSpatialPositionalEncoding:
    """Tests for 2D spatial positional encoding."""

    def test_dimension_requirement(self) -> None:
        """Test that d_model must be divisible by 4."""
        with pytest.raises(ValueError):
            SpatialPositionalEncoding(d_model=65)

        # Should not raise
        SpatialPositionalEncoding(d_model=64)

    def test_output_shape(self) -> None:
        """Test output shape."""
        spe = SpatialPositionalEncoding(d_model=64, max_size=100)
        x = torch.randn(4, 64, 32, 32)
        output = spe(x)

        assert output.shape == x.shape

    def test_encoding_varies_spatially(self) -> None:
        """Test that encoding varies across spatial dimensions."""
        spe = SpatialPositionalEncoding(d_model=64, max_size=100)
        x = torch.zeros(1, 64, 10, 10)
        output = spe(x)

        # Different spatial positions should have different encodings
        assert not torch.allclose(output[0, :, 0, 0], output[0, :, 5, 5])


class TestIntegration:
    """Integration tests for Fourier features."""

    def test_in_neural_network(self) -> None:
        """Test using Fourier features in a neural network."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0, 5.0],
        )

        # Simple MLP after Fourier features
        model = torch.nn.Sequential(
            features,
            torch.nn.Linear(features.output_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        x = torch.randn(16, 100, 2)
        output = model(x)

        assert output.shape == (16, 100, 1)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through Fourier features."""
        features = MultiScaleFourierFeatures(
            input_dim=2,
            n_features=32,
            scales=[1.0],
            learnable=True,
        )

        x = torch.randn(4, 10, 2, requires_grad=True)
        output = features(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to input and parameters
        assert x.grad is not None
        for param in features.parameters():
            assert param.grad is not None

    def test_progressive_curriculum(self) -> None:
        """Test progressive features for curriculum learning."""
        features = ProgressiveFourierFeatures(
            input_dim=2,
            n_features=16,
            scales=[1.0, 5.0, 25.0],
        )

        x = torch.randn(4, 20, 2)

        # Simulate curriculum
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            features.set_progress(progress)
            output = features(x)
            assert not torch.isnan(output).any()
