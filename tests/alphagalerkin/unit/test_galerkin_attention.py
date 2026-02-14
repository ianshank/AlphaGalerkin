"""Tests for Galerkin linear attention."""

from __future__ import annotations

import pytest
import torch

from src.alphagalerkin.nn.galerkin_attention import (
    GalerkinLinearAttention,
)


class TestGalerkinLinearAttention:
    """Unit tests for GalerkinLinearAttention."""

    def test_output_shape_batched(self) -> None:
        """Output shape matches input shape (batch, seq, hidden)."""
        attn = GalerkinLinearAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_output_shape_unbatched(self) -> None:
        """Unbatched input (seq, hidden) returns same shape."""
        attn = GalerkinLinearAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        out = attn(x)
        assert out.shape == (10, 64)

    def test_linear_complexity_output_differs(self) -> None:
        """Output should differ from input (non-identity transform)."""
        attn = GalerkinLinearAttention(hidden_dim=32, num_heads=2)
        x = torch.randn(1, 5, 32)
        out = attn(x)
        assert not torch.allclose(x, out, atol=1e-3)

    def test_lbb_diagnostic_keys(self) -> None:
        """Diagnostic dict contains expected keys."""
        attn = GalerkinLinearAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        diag = attn.compute_lbb_diagnostic(x)
        assert "sigma_min" in diag
        assert "sigma_max" in diag
        assert "condition_number" in diag

    def test_lbb_diagnostic_positive_values(self) -> None:
        """Singular values should be positive."""
        attn = GalerkinLinearAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        diag = attn.compute_lbb_diagnostic(x)
        assert diag["sigma_min"] >= 0.0
        assert diag["sigma_max"] >= 0.0

    def test_dropout_zero_is_deterministic(self) -> None:
        """With dropout=0, same input gives same output."""
        attn = GalerkinLinearAttention(hidden_dim=32, num_heads=2, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 5, 32)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self) -> None:
        """Gradients flow through the attention layer."""
        attn = GalerkinLinearAttention(hidden_dim=32, num_heads=2)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_output_finite(self) -> None:
        """Output contains no NaN or Inf values."""
        attn = GalerkinLinearAttention(hidden_dim=128, num_heads=8)
        x = torch.randn(3, 16, 128)
        out = attn(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("seq_len", [1, 5, 20, 100])
    def test_different_sequence_lengths(self, seq_len: int) -> None:
        """Works with various sequence lengths."""
        attn = GalerkinLinearAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, seq_len, 64)
        out = attn(x)
        assert out.shape == (2, seq_len, 64)

    def test_lbb_diagnostic_condition_number(self) -> None:
        """Condition number >= 1.0 (sigma_max >= sigma_min)."""
        attn = GalerkinLinearAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(4, 12, 64)
        diag = attn.compute_lbb_diagnostic(x)
        assert diag["condition_number"] >= 1.0

    def test_long_sequence_works(self) -> None:
        """O(N) complexity allows very long sequences."""
        attn = GalerkinLinearAttention(hidden_dim=32, num_heads=2)
        x = torch.randn(1, 500, 32)
        out = attn(x)
        assert out.shape == (1, 500, 32)
        assert torch.isfinite(out).all()
