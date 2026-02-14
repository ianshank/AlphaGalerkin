"""Tests for Fourier positional encoding."""

from __future__ import annotations

import torch

from src.alphagalerkin.nn.fourier_features import (
    FourierPositionalEncoding,
)


class TestFourierPositionalEncoding:
    """Unit tests for Fourier features."""

    def test_output_dim_correct(self) -> None:
        """Output dimension matches formula: d + 2*nf*d."""
        enc = FourierPositionalEncoding(coord_dim=2, num_frequencies=16)
        expected = 2 + 2 * 16 * 2  # 66
        assert enc.output_dim == expected

    def test_output_shape(self) -> None:
        """Output shape is (..., output_dim)."""
        enc = FourierPositionalEncoding(coord_dim=2, num_frequencies=8)
        coords = torch.randn(10, 2)
        out = enc(coords)
        assert out.shape == (10, enc.output_dim)

    def test_batched_input(self) -> None:
        """Works with batched coordinates."""
        enc = FourierPositionalEncoding(coord_dim=2, num_frequencies=8)
        coords = torch.randn(4, 10, 2)
        out = enc(coords)
        assert out.shape == (4, 10, enc.output_dim)

    def test_different_coords_give_different_features(self) -> None:
        """Different input coordinates produce different features."""
        enc = FourierPositionalEncoding(coord_dim=2, num_frequencies=16)
        c1 = torch.tensor([[0.0, 0.0]])
        c2 = torch.tensor([[1.0, 1.0]])
        f1 = enc(c1)
        f2 = enc(c2)
        assert not torch.allclose(f1, f2)

    def test_learnable_frequencies(self) -> None:
        """Learnable mode has B as a parameter."""
        enc = FourierPositionalEncoding(
            coord_dim=2,
            num_frequencies=8,
            learnable=True,
        )
        param_names = [n for n, _ in enc.named_parameters()]
        assert "B" in param_names

    def test_fixed_frequencies_no_gradient(self) -> None:
        """Fixed mode has B as buffer (no gradient)."""
        enc = FourierPositionalEncoding(
            coord_dim=2,
            num_frequencies=8,
            learnable=False,
        )
        param_names = [n for n, _ in enc.named_parameters()]
        assert "B" not in param_names
        buffer_names = [n for n, _ in enc.named_buffers()]
        assert "B" in buffer_names

    def test_1d_coordinates(self) -> None:
        """Works with 1D coordinates."""
        enc = FourierPositionalEncoding(coord_dim=1, num_frequencies=4)
        coords = torch.randn(5, 1)
        out = enc(coords)
        assert out.shape == (5, 1 + 2 * 4 * 1)

    def test_output_finite(self) -> None:
        """Output contains no NaN or Inf."""
        enc = FourierPositionalEncoding(coord_dim=2, num_frequencies=32)
        coords = torch.randn(8, 20, 2)
        out = enc(coords)
        assert torch.isfinite(out).all()

    def test_original_coords_preserved(self) -> None:
        """First coord_dim values in output are the original coordinates."""
        enc = FourierPositionalEncoding(coord_dim=2, num_frequencies=8)
        coords = torch.randn(5, 2)
        out = enc(coords)
        assert torch.allclose(out[:, :2], coords)

    def test_learnable_gradients_flow(self) -> None:
        """Gradients flow to learnable frequency matrix B."""
        enc = FourierPositionalEncoding(
            coord_dim=2,
            num_frequencies=16,
            learnable=True,
        )
        coords = torch.randn(4, 10, 2)
        out = enc(coords)
        out.sum().backward()
        assert enc.B.grad is not None
