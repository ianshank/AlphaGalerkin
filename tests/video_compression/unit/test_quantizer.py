"""Tests for video compression quantizers."""

import pytest
import torch

from src.video_compression.config import QuantizerConfig, QuantizationMode
from src.video_compression.models.quantizer import (
    NoiseQuantizer,
    STEQuantizer,
    SoftQuantizer,
    create_quantizer,
)


class TestNoiseQuantizer:
    """Tests for noise-based quantization."""

    def test_training_adds_noise(self) -> None:
        """Test that training mode adds noise."""
        quantizer = NoiseQuantizer(noise_scale=0.5)
        quantizer.train()

        x = torch.zeros(100, 100)
        y = quantizer(x)

        # Should add noise, so not all zeros
        assert not torch.allclose(y, x)

    def test_inference_rounds(self) -> None:
        """Test that inference mode rounds."""
        quantizer = NoiseQuantizer()
        quantizer.eval()

        x = torch.tensor([0.3, 0.7, 1.4, -0.2])
        y = quantizer(x)

        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        assert torch.allclose(y, expected)

    def test_noise_scale(self) -> None:
        """Test noise scale affects variance."""
        q_small = NoiseQuantizer(noise_scale=0.1)
        q_large = NoiseQuantizer(noise_scale=0.9)
        q_small.train()
        q_large.train()

        x = torch.zeros(10000)

        y_small = q_small(x)
        y_large = q_large(x)

        # Larger noise scale should have larger variance
        assert y_large.var() > y_small.var()


class TestSTEQuantizer:
    """Tests for straight-through estimator quantization."""

    def test_forward_rounds(self) -> None:
        """Test forward pass rounds values."""
        quantizer = STEQuantizer()

        x = torch.tensor([0.3, 0.7, 1.4, -0.2])
        y = quantizer(x)

        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        assert torch.allclose(y, expected)

    def test_gradient_passes_through(self) -> None:
        """Test gradient passes through unchanged."""
        quantizer = STEQuantizer()

        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = quantizer(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be 1.0 for each element (identity)
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)


class TestSoftQuantizer:
    """Tests for soft quantization."""

    def test_training_is_soft(self) -> None:
        """Test training uses soft quantization."""
        quantizer = SoftQuantizer(temperature=1.0)
        quantizer.train()

        x = torch.tensor([0.5])
        y = quantizer(x)

        # Soft quantization should not be exactly 0 or 1
        assert 0.0 < y.item() < 1.0

    def test_inference_is_hard(self) -> None:
        """Test inference uses hard quantization."""
        quantizer = SoftQuantizer()
        quantizer.eval()

        x = torch.tensor([0.3, 0.7])
        y = quantizer(x)

        expected = torch.tensor([0.0, 1.0])
        assert torch.allclose(y, expected)

    def test_temperature_annealing(self) -> None:
        """Test temperature annealing."""
        quantizer = SoftQuantizer(temperature=1.0, min_temperature=0.5)

        initial_temp = quantizer.temperature.item()
        quantizer.anneal_temperature(factor=0.5)
        final_temp = quantizer.temperature.item()

        assert final_temp < initial_temp
        assert final_temp >= 0.5  # Min temperature


class TestCreateQuantizer:
    """Tests for quantizer factory function."""

    def test_creates_noise_quantizer(self) -> None:
        """Test creating noise quantizer from config."""
        config = QuantizerConfig(name="test", mode=QuantizationMode.NOISE)
        quantizer = create_quantizer(config)

        assert isinstance(quantizer, NoiseQuantizer)

    def test_creates_ste_quantizer(self) -> None:
        """Test creating STE quantizer from config."""
        config = QuantizerConfig(name="test", mode=QuantizationMode.STE)
        quantizer = create_quantizer(config)

        assert isinstance(quantizer, STEQuantizer)

    def test_creates_soft_quantizer(self) -> None:
        """Test creating soft quantizer from config."""
        config = QuantizerConfig(name="test", mode=QuantizationMode.SOFT)
        quantizer = create_quantizer(config)

        assert isinstance(quantizer, SoftQuantizer)


class TestQuantizationLossless:
    """Tests for lossless encode/decode."""

    def test_encode_decode_lossless(self) -> None:
        """Test encode/decode is lossless."""
        quantizer = STEQuantizer()

        # Random integers
        original = torch.randint(-10, 10, (100,)).float()

        encoded = quantizer.encode(original)
        decoded = quantizer.decode(encoded)

        assert torch.equal(original, decoded)

    def test_encode_produces_integers(self) -> None:
        """Test encode produces integer tensor."""
        quantizer = STEQuantizer()

        x = torch.randn(100)
        encoded = quantizer.encode(x)

        assert encoded.dtype == torch.int32
