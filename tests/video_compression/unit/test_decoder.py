"""Tests for video compression decoder."""

import pytest
import torch

from src.video_compression.config import DecoderConfig
from src.video_compression.models.decoder import (
    Decoder,
    DecoderBlock,
    TemporalDecoder,
    UpsampleBlock,
)


@pytest.fixture
def decoder_config() -> DecoderConfig:
    """Create a small decoder config for testing."""
    return DecoderConfig(
        name="test",
        latent_channels=64,
        out_channels=3,
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ffn=128,
        upsample_factor=4,
    )


@pytest.fixture
def decoder(decoder_config: DecoderConfig) -> Decoder:
    """Create decoder instance."""
    return Decoder(decoder_config)


class TestUpsampleBlock:
    """Tests for upsampling block."""

    def test_forward_upsamples(self) -> None:
        """Test upsample block increases spatial size."""
        block = UpsampleBlock(in_channels=64, out_channels=32, stride=2)
        x = torch.randn(2, 64, 8, 8)

        y = block(x)

        assert y.shape == (2, 32, 16, 16)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through upsample block."""
        block = UpsampleBlock(in_channels=64, out_channels=32, stride=2)
        x = torch.randn(2, 64, 8, 8, requires_grad=True)

        y = block(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None


class TestDecoderBlock:
    """Tests for decoder block."""

    def test_forward_upsamples(self) -> None:
        """Test decoder block upsamples spatially."""
        block = DecoderBlock(
            in_channels=64,
            out_channels=32,
            d_model=64,
            n_heads=4,
            d_ffn=128,
            upsample_stride=2,
        )
        x = torch.randn(2, 64, 8, 8)

        y = block(x)

        assert y.shape == (2, 32, 16, 16)


class TestDecoder:
    """Tests for complete decoder."""

    def test_forward_shape(self, decoder: Decoder) -> None:
        """Test decoder output shape."""
        us = decoder.config.upsample_factor
        x = torch.randn(2, decoder.config.latent_channels, 16, 16)

        y = decoder(x)

        assert y.shape == (2, 3, 16 * us, 16 * us)

    def test_output_range(self, decoder: Decoder) -> None:
        """Test decoder output is in [0, 1] range."""
        x = torch.randn(2, decoder.config.latent_channels, 16, 16)

        y = decoder(x)

        # Should be in [0, 1] due to sigmoid
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_resolution_independence(self, decoder: Decoder) -> None:
        """Test decoder works with different latent sizes."""
        for h, w in [(8, 8), (16, 16), (32, 32)]:
            x = torch.randn(1, decoder.config.latent_channels, h, w)
            y = decoder(x)

            us = decoder.config.upsample_factor
            assert y.shape == (1, 3, h * us, w * us)

    def test_gradient_flow(self, decoder: Decoder) -> None:
        """Test gradients flow through decoder."""
        x = torch.randn(2, decoder.config.latent_channels, 16, 16, requires_grad=True)

        y = decoder(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None


class TestTemporalDecoder:
    """Tests for temporal decoder."""

    def test_forward_without_reference(self, decoder_config: DecoderConfig) -> None:
        """Test temporal decoder without reference frame."""
        temporal = TemporalDecoder(decoder_config)
        x = torch.randn(2, decoder_config.latent_channels, 16, 16)

        y = temporal(x, reference=None)

        us = decoder_config.upsample_factor
        assert y.shape == (2, 3, 16 * us, 16 * us)

    def test_forward_with_reference(self, decoder_config: DecoderConfig) -> None:
        """Test temporal decoder with reference frame."""
        temporal = TemporalDecoder(decoder_config)
        x = torch.randn(2, decoder_config.latent_channels, 16, 16)
        ref = torch.randn(2, decoder_config.latent_channels, 16, 16)

        y = temporal(x, reference=ref)

        us = decoder_config.upsample_factor
        assert y.shape == (2, 3, 16 * us, 16 * us)

    def test_reference_affects_output(self, decoder_config: DecoderConfig) -> None:
        """Test that reference frame affects output."""
        temporal = TemporalDecoder(decoder_config)
        temporal.eval()

        x = torch.randn(2, decoder_config.latent_channels, 16, 16)
        ref1 = torch.randn(2, decoder_config.latent_channels, 16, 16)
        ref2 = torch.randn(2, decoder_config.latent_channels, 16, 16)

        y_no_ref = temporal(x, reference=None)
        y_ref1 = temporal(x, reference=ref1)
        y_ref2 = temporal(x, reference=ref2)

        # Outputs should differ
        assert not torch.allclose(y_no_ref, y_ref1)
        assert not torch.allclose(y_ref1, y_ref2)
