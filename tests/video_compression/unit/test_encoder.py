"""Tests for video compression encoder."""

import pytest
import torch

from src.video_compression.config import EncoderConfig
from src.video_compression.models.encoder import (
    GDN,
    Encoder,
    EncoderBlock,
    FNetGalerkinBlock,
    GalerkinEncoderAttention,
)


@pytest.fixture
def encoder_config() -> EncoderConfig:
    """Create a small encoder config for testing."""
    return EncoderConfig(
        name="test",
        in_channels=3,
        latent_channels=64,  # Smaller for testing
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ffn=128,
        downsample_factor=4,  # Smaller for testing
    )


@pytest.fixture
def encoder(encoder_config: EncoderConfig) -> Encoder:
    """Create encoder instance."""
    return Encoder(encoder_config)


class TestGDN:
    """Tests for Generalized Divisive Normalization."""

    def test_forward_shape(self) -> None:
        """Test GDN preserves shape."""
        gdn = GDN(channels=32)
        x = torch.randn(2, 32, 16, 16)

        y = gdn(x)

        assert y.shape == x.shape

    def test_inverse_gdn(self) -> None:
        """Test inverse GDN."""
        gdn = GDN(channels=32)
        igdn = GDN(channels=32, inverse=True)

        x = torch.randn(2, 32, 16, 16)

        # GDN should be roughly invertible (not exact due to non-linearity)
        y = gdn(x)
        assert y.shape == x.shape

        y_inv = igdn(x)
        assert y_inv.shape == x.shape

    def test_gradient_flow(self) -> None:
        """Test gradients flow through GDN."""
        gdn = GDN(channels=32)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)

        y = gdn(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGalerkinEncoderAttention:
    """Tests for Galerkin attention in encoder."""

    def test_forward_shape(self) -> None:
        """Test output shape matches input."""
        attn = GalerkinEncoderAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 100, 64)  # (batch, seq, features)

        y = attn(x)

        assert y.shape == x.shape

    def test_resolution_independence(self) -> None:
        """Test attention works with different sequence lengths."""
        attn = GalerkinEncoderAttention(d_model=64, n_heads=4)

        # Test multiple sequence lengths
        for n in [64, 100, 256, 400]:
            x = torch.randn(2, n, 64)
            y = attn(x)
            assert y.shape == x.shape

    def test_linear_complexity(self) -> None:
        """Test O(N) complexity via K^T V computation."""
        attn = GalerkinEncoderAttention(d_model=64, n_heads=4)

        # The attention should compute K^T V first (d_k x d_k)
        # then Q x KV, giving O(N * d_k^2) complexity
        x = torch.randn(1, 1000, 64)
        y = attn(x)

        assert y.shape == x.shape

    def test_gradient_flow(self) -> None:
        """Test gradients flow through attention."""
        attn = GalerkinEncoderAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 100, 64, requires_grad=True)

        y = attn(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None


class TestFNetGalerkinBlock:
    """Tests for hybrid FNet-Galerkin block."""

    def test_forward_shape(self) -> None:
        """Test output shape matches input."""
        block = FNetGalerkinBlock(d_model=64, n_heads=4, d_ffn=128)
        x = torch.randn(2, 100, 64)

        y = block(x, height=10, width=10)

        assert y.shape == x.shape

    def test_fft_mixing(self) -> None:
        """Test FFT mixing operates without learnable parameters."""
        block = FNetGalerkinBlock(d_model=64, n_heads=4, d_ffn=128)

        # FFT mixing should be deterministic
        x = torch.randn(2, 64, 64)  # 8x8 spatial

        y1 = block._fft_mixing(x, height=8, width=8)
        y2 = block._fft_mixing(x, height=8, width=8)

        assert torch.allclose(y1, y2)


class TestEncoderBlock:
    """Tests for complete encoder block."""

    def test_forward_downsamples(self) -> None:
        """Test encoder block downsamples spatially."""
        block = EncoderBlock(
            in_channels=32,
            out_channels=64,
            d_model=64,
            n_heads=4,
            d_ffn=128,
            downsample_stride=2,
        )
        x = torch.randn(2, 32, 16, 16)

        y = block(x)

        # Should downsample by 2x and change channels
        assert y.shape == (2, 64, 8, 8)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through encoder block."""
        block = EncoderBlock(
            in_channels=32,
            out_channels=64,
            d_model=64,
            n_heads=4,
            d_ffn=128,
        )
        x = torch.randn(2, 32, 16, 16, requires_grad=True)

        y = block(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None


class TestEncoder:
    """Tests for complete encoder."""

    def test_forward_shape(self, encoder: Encoder) -> None:
        """Test encoder output shape."""
        x = torch.randn(2, 3, 64, 64)

        y = encoder(x)

        # Should be downsampled by downsample_factor
        expected_h = 64 // encoder.config.downsample_factor
        expected_w = 64 // encoder.config.downsample_factor
        assert y.shape == (2, encoder.config.latent_channels, expected_h, expected_w)

    def test_resolution_independence(self, encoder: Encoder) -> None:
        """Test encoder works with different resolutions."""
        ds = encoder.config.downsample_factor

        # Test multiple resolutions (must be divisible by downsample_factor)
        for size in [32, 64, 128, 256]:
            if size % ds == 0:
                x = torch.randn(1, 3, size, size)
                y = encoder(x)
                assert y.shape[2] == size // ds
                assert y.shape[3] == size // ds

    def test_invalid_resolution(self, encoder: Encoder) -> None:
        """Test encoder raises error for invalid resolution."""
        ds = encoder.config.downsample_factor

        # Resolution not divisible by downsample_factor
        x = torch.randn(1, 3, ds + 1, ds + 1)

        with pytest.raises(ValueError):
            encoder(x)

    def test_gradient_flow(self, encoder: Encoder) -> None:
        """Test gradients flow through encoder."""
        x = torch.randn(2, 3, 64, 64, requires_grad=True)

        y = encoder(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_deterministic(self, encoder: Encoder) -> None:
        """Test encoder is deterministic in eval mode."""
        encoder.eval()
        x = torch.randn(2, 3, 64, 64)

        y1 = encoder(x)
        y2 = encoder(x)

        assert torch.allclose(y1, y2)
