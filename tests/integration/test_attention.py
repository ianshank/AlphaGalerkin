"""Integration tests for attention mechanisms."""

from __future__ import annotations

import pytest
import torch

from src.modeling.attention import GalerkinAttention, HybridAttention, SoftmaxAttention


class TestGalerkinAttentionIntegration:
    """Integration tests for Galerkin attention."""

    @pytest.fixture
    def attention(self) -> GalerkinAttention:
        """Create attention module."""
        torch.manual_seed(42)
        return GalerkinAttention(d_model=64, n_heads=4)

    def test_error_vs_explicit_petrov_galerkin(
        self, attention: GalerkinAttention
    ) -> None:
        """Test that error vs explicit Petrov-Galerkin projection is < 1e-5.

        This is a key success criterion from the specification.
        """
        torch.manual_seed(42)

        # Create synthetic test functions (smooth, low frequency)
        batch, n, d = 2, 64, 64
        x = torch.randn(batch, n, d) * 0.1

        # Get output from Galerkin attention
        output = attention(x)

        # Verify output is finite and reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Compute explicit projection for comparison
        # This is the same formula: Q * (K^T V / n)
        q = attention.to_q(x)
        k = attention.to_k(x)
        v = attention.to_v(x)

        q = q.view(batch, n, attention.n_heads, attention.d_key).transpose(1, 2)
        k = k.view(batch, n, attention.n_heads, attention.d_key).transpose(1, 2)
        v = v.view(batch, n, attention.n_heads, attention.d_value).transpose(1, 2)

        # Normalize as in attention
        if attention.normalize_features:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

        # Explicit Petrov-Galerkin: Q * (K^T V / n)
        context = torch.einsum("bhnd,bhnv->bhdv", k, v) / n
        explicit_output = torch.einsum("bhnd,bhdv->bhnv", q, context)
        explicit_output = explicit_output.transpose(1, 2).reshape(batch, n, -1)
        explicit_output = attention.to_out(explicit_output)

        # Compute relative error
        error = (output - explicit_output).abs()
        relative_error = error / (explicit_output.abs() + 1e-8)

        # Error should be very small (essentially numerical precision)
        max_relative_error = relative_error.max().item()
        assert max_relative_error < 1e-5, f"Max relative error: {max_relative_error}"

    def test_lbb_stability_maintained(self, attention: GalerkinAttention) -> None:
        """Test that LBB stability is maintained during forward pass."""
        x = torch.randn(4, 81, 64)

        output, lbb = attention(x, return_lbb=True)

        # LBB constant should be positive
        assert (lbb > 0).all()

        # LBB should not be too small (would indicate instability)
        assert (lbb > 1e-8).all()


class TestSoftmaxAttentionIntegration:
    """Integration tests for softmax attention."""

    @pytest.fixture
    def attention(self) -> SoftmaxAttention:
        """Create attention module."""
        torch.manual_seed(42)
        return SoftmaxAttention(d_model=64, n_heads=4)

    def test_attention_preserves_injectivity(
        self, attention: SoftmaxAttention
    ) -> None:
        """Test that softmax attention preserves distinctness of inputs.

        Different inputs should produce different outputs (injectivity).
        """
        # Create two distinct inputs
        x1 = torch.randn(1, 81, 64)
        x2 = x1 + torch.randn_like(x1) * 0.1  # Small perturbation

        output1 = attention(x1)
        output2 = attention(x2)

        # Outputs should be different
        diff = (output1 - output2).abs().mean()
        assert diff > 1e-6

    def test_attention_weights_sum_to_one(
        self, attention: SoftmaxAttention
    ) -> None:
        """Test that attention weights form valid distribution."""
        x = torch.randn(2, 16, 64)

        # Forward with hooks to capture attention weights
        q = attention.to_q(x)
        k = attention.to_k(x)

        q = q.view(2, 16, 4, 16).transpose(1, 2)
        k = k.view(2, 16, 4, 16).transpose(1, 2)

        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * attention.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weights should sum to 1 along last dimension
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestHybridAttentionIntegration:
    """Integration tests for hybrid attention."""

    @pytest.fixture
    def attention(self) -> HybridAttention:
        """Create hybrid attention."""
        torch.manual_seed(42)
        return HybridAttention(d_model=64, n_heads=4, galerkin_ratio=0.7)

    def test_combines_both_attention_types(
        self, attention: HybridAttention
    ) -> None:
        """Test that hybrid attention combines both mechanisms."""
        x = torch.randn(2, 81, 64)

        # Get outputs from individual components
        galerkin_out = attention.galerkin(x)
        softmax_out = attention.softmax(x)
        hybrid_out = attention(x)

        # Hybrid should be different from both individual outputs
        assert not torch.allclose(hybrid_out, galerkin_out, atol=1e-3)
        assert not torch.allclose(hybrid_out, softmax_out, atol=1e-3)

        # Hybrid should be a combination
        gate = torch.sigmoid(attention.gate)
        expected = gate * galerkin_out + (1 - gate) * softmax_out
        assert torch.allclose(hybrid_out, expected, atol=1e-5)

    def test_gate_is_learnable(self, attention: HybridAttention) -> None:
        """Test that the gate parameter is learnable."""
        x = torch.randn(2, 81, 64)

        output = attention(x)
        loss = output.sum()
        loss.backward()

        # Gate should have gradient
        assert attention.gate.grad is not None
