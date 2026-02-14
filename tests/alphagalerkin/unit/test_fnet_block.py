"""Tests for FNet block."""
from __future__ import annotations

import pytest
import torch

from src.alphagalerkin.nn.fnet_block import FNetBlock, FNetMixing


class TestFNetMixing:
    """Unit tests for FFT mixing layer."""

    def test_output_shape(self) -> None:
        """FFT mixing preserves shape."""
        mixing = FNetMixing()
        x = torch.randn(2, 10, 64)
        out = mixing(x)
        assert out.shape == (2, 10, 64)

    def test_output_is_real(self) -> None:
        """Output should be real-valued (not complex)."""
        mixing = FNetMixing()
        x = torch.randn(2, 10, 64)
        out = mixing(x)
        assert not out.is_complex()

    def test_differs_from_input(self) -> None:
        """FFT mixing should transform the input."""
        mixing = FNetMixing()
        x = torch.randn(2, 10, 64)
        out = mixing(x)
        assert not torch.allclose(x, out, atol=1e-3)

    def test_no_learnable_parameters(self) -> None:
        """FNetMixing has zero learnable parameters."""
        mixing = FNetMixing()
        params = list(mixing.parameters())
        assert len(params) == 0

    def test_output_finite(self) -> None:
        """Output contains no NaN or Inf."""
        mixing = FNetMixing()
        x = torch.randn(3, 16, 128)
        out = mixing(x)
        assert torch.isfinite(out).all()


class TestFNetBlock:
    """Unit tests for complete FNet block."""

    def test_output_shape_batched(self) -> None:
        """Batched output shape matches input."""
        block = FNetBlock(hidden_dim=64)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_output_shape_unbatched(self) -> None:
        """Unbatched input returns same shape."""
        block = FNetBlock(hidden_dim=64)
        x = torch.randn(10, 64)
        out = block(x)
        assert out.shape == (10, 64)

    def test_residual_connection(self) -> None:
        """Output differs from input but is in same space."""
        block = FNetBlock(hidden_dim=32)
        x = torch.randn(1, 5, 32)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.allclose(x, out, atol=1e-3)

    def test_gradient_flows(self) -> None:
        """Gradients flow through the FNet block."""
        block = FNetBlock(hidden_dim=32)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    @pytest.mark.parametrize("seq_len", [1, 5, 20, 100])
    def test_different_sequence_lengths(self, seq_len: int) -> None:
        """Works with various sequence lengths."""
        block = FNetBlock(hidden_dim=32, dropout=0.0)
        x = torch.randn(2, seq_len, 32)
        out = block(x)
        assert out.shape == (2, seq_len, 32)

    def test_custom_expansion_factor(self) -> None:
        """Custom FFN expansion factor works correctly."""
        block = FNetBlock(hidden_dim=32, expansion_factor=2, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)
