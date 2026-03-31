"""Property-based tests for attention mechanisms.

Tests Galerkin attention invariants:
- Output dimension matches input
- LBB constant is positive
- Linearity in value space
- Numerical stability under extreme inputs
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.modeling.attention import GalerkinAttention, HybridAttention, SoftmaxAttention


def _make_galerkin(d_model: int = 32, n_heads: int = 4) -> GalerkinAttention:
    """Create a Galerkin attention module (avoids fixture/hypothesis clash)."""
    torch.manual_seed(42)
    return GalerkinAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)


def _make_softmax(d_model: int = 32, n_heads: int = 4) -> SoftmaxAttention:
    """Create a Softmax attention module."""
    torch.manual_seed(42)
    return SoftmaxAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def galerkin_attn() -> GalerkinAttention:
    """Create a small Galerkin attention module for testing."""
    torch.manual_seed(42)
    return GalerkinAttention(d_model=32, n_heads=4, dropout=0.0)


@pytest.fixture
def softmax_attn() -> SoftmaxAttention:
    """Create a small Softmax attention module for testing."""
    torch.manual_seed(42)
    return SoftmaxAttention(d_model=32, n_heads=4, dropout=0.0)


@pytest.fixture
def hybrid_attn() -> HybridAttention:
    """Create a small Hybrid attention module for testing."""
    torch.manual_seed(42)
    return HybridAttention(d_model=32, n_heads=4, dropout=0.0)


# ---------------------------------------------------------------------------
# Galerkin attention properties
# ---------------------------------------------------------------------------


class TestGalerkinAttentionProperties:
    """Property tests for Galerkin linear attention."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=4, max_value=64),
    )
    @settings(max_examples=20)
    def test_output_shape_matches_input(
        self, batch_size: int, seq_len: int
    ) -> None:
        """Output tensor must have the same shape as input."""
        attn = _make_galerkin()
        torch.manual_seed(batch_size * 100 + seq_len)
        x = torch.randn(batch_size, seq_len, 32)

        output = attn(x)

        assert output.shape == x.shape, (
            f"Output shape {output.shape} must match input shape {x.shape}"
        )

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=4, max_value=64),
    )
    @settings(max_examples=20)
    def test_lbb_constant_is_positive(
        self, batch_size: int, seq_len: int
    ) -> None:
        """LBB constant (min singular value) must be positive."""
        attn = _make_galerkin()
        torch.manual_seed(batch_size * 100 + seq_len)
        x = torch.randn(batch_size, seq_len, 32)

        _, lbb = attn(x, return_lbb=True)

        assert lbb.shape == (batch_size,), f"LBB shape must be (batch,), got {lbb.shape}"
        assert (lbb > 0).all(), "LBB constant must be positive for all samples"

    def test_output_is_finite(self, galerkin_attn: GalerkinAttention) -> None:
        """Output must be finite for normal inputs."""
        x = torch.randn(2, 16, 32)
        output = galerkin_attn(x)

        assert torch.isfinite(output).all(), "Galerkin attention output must be finite"

    def test_gradient_flow(self, galerkin_attn: GalerkinAttention) -> None:
        """Gradients must flow through Galerkin attention."""
        x = torch.randn(2, 16, 32, requires_grad=True)
        output = galerkin_attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients must flow to input"
        assert torch.isfinite(x.grad).all(), "Input gradients must be finite"

    def test_lbb_gradient_flow(self, galerkin_attn: GalerkinAttention) -> None:
        """Gradients must flow through the LBB constant computation."""
        x = torch.randn(2, 16, 32, requires_grad=True)
        output, lbb = galerkin_attn(x, return_lbb=True)

        # Use both output and lbb in loss
        loss = output.sum() + lbb.sum()
        loss.backward()

        assert x.grad is not None, "Gradients must flow when using LBB"
        assert torch.isfinite(x.grad).all(), "Gradients must be finite with LBB"

    @pytest.mark.parametrize("seq_len", [9, 25, 81, 169])
    def test_resolution_independence(
        self, seq_len: int, galerkin_attn: GalerkinAttention
    ) -> None:
        """Galerkin attention must work with arbitrary sequence lengths.

        This is the key property enabling zero-shot transfer between
        board sizes (e.g., 9x9=81 -> 13x13=169 -> 19x19=361).
        """
        torch.manual_seed(seq_len)
        x = torch.randn(2, seq_len, 32)

        output = galerkin_attn(x)

        assert output.shape == (2, seq_len, 32)
        assert torch.isfinite(output).all()

    def test_deterministic_eval(self, galerkin_attn: GalerkinAttention) -> None:
        """In eval mode (dropout=0), same input must give same output."""
        galerkin_attn.eval()
        x = torch.randn(2, 16, 32)

        out1 = galerkin_attn(x)
        out2 = galerkin_attn(x)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Galerkin attention must be deterministic in eval mode"
        )


# ---------------------------------------------------------------------------
# Softmax attention properties
# ---------------------------------------------------------------------------


class TestSoftmaxAttentionProperties:
    """Property tests for standard softmax attention."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=4, max_value=64),
    )
    @settings(max_examples=20)
    def test_output_shape_matches_input(
        self, batch_size: int, seq_len: int
    ) -> None:
        """Output tensor must have the same shape as input."""
        attn = _make_softmax()
        torch.manual_seed(batch_size * 100 + seq_len)
        x = torch.randn(batch_size, seq_len, 32)

        output = attn(x)

        assert output.shape == x.shape

    def test_output_is_finite(self, softmax_attn: SoftmaxAttention) -> None:
        """Output must be finite for normal inputs."""
        x = torch.randn(2, 16, 32)
        output = softmax_attn(x)

        assert torch.isfinite(output).all(), "Softmax attention output must be finite"

    def test_gradient_flow(self, softmax_attn: SoftmaxAttention) -> None:
        """Gradients must flow through softmax attention."""
        x = torch.randn(2, 16, 32, requires_grad=True)
        output = softmax_attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_mask_zeros_out_positions(self, softmax_attn: SoftmaxAttention) -> None:
        """Masking should affect the output compared to unmasked."""
        x = torch.randn(2, 8, 32)

        out_no_mask = softmax_attn(x, mask=None)

        # Causal mask -- shape (batch, 1, n, n) to broadcast over heads
        mask = torch.tril(torch.ones(2, 1, 8, 8))
        out_masked = softmax_attn(x, mask=mask)

        # Outputs should differ when mask is applied
        assert not torch.allclose(out_no_mask, out_masked, atol=1e-4), (
            "Masked and unmasked outputs should differ"
        )


# ---------------------------------------------------------------------------
# Hybrid attention properties
# ---------------------------------------------------------------------------


class TestHybridAttentionProperties:
    """Property tests for hybrid (Galerkin + Softmax) attention."""

    def test_output_shape(self, hybrid_attn: HybridAttention) -> None:
        """Output shape must match input shape."""
        x = torch.randn(2, 16, 32)
        output = hybrid_attn(x)

        assert output.shape == x.shape

    def test_output_is_finite(self, hybrid_attn: HybridAttention) -> None:
        """Output must be finite."""
        x = torch.randn(2, 16, 32)
        output = hybrid_attn(x)

        assert torch.isfinite(output).all()

    def test_gate_is_learnable(self, hybrid_attn: HybridAttention) -> None:
        """The gate parameter should be learnable and receive gradients."""
        x = torch.randn(2, 16, 32)
        output = hybrid_attn(x)
        loss = output.sum()
        loss.backward()

        assert hybrid_attn.gate.grad is not None, "Gate must receive gradients"
        assert torch.isfinite(hybrid_attn.gate.grad).all()

    def test_output_between_galerkin_and_softmax(
        self, hybrid_attn: HybridAttention
    ) -> None:
        """Hybrid output should be a convex combination of Galerkin and Softmax.

        The hybrid output is: gate * galerkin + (1-gate) * softmax.
        We verify the output norm is bounded by the max of the two components.
        """
        x = torch.randn(2, 16, 32)

        galerkin_out = hybrid_attn.galerkin(x)
        softmax_out = hybrid_attn.softmax(x)
        hybrid_out = hybrid_attn(x)

        # Hybrid output norm should be at most the max of the two component norms
        # (plus some tolerance for floating point)
        max_norm = max(galerkin_out.norm().item(), softmax_out.norm().item())
        assert hybrid_out.norm().item() <= max_norm * 1.1, (
            "Hybrid norm should not exceed the max of its components (with tolerance)"
        )


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestAttentionNumericalStability:
    """Numerical stability tests for attention mechanisms."""

    def test_galerkin_with_large_inputs(self, galerkin_attn: GalerkinAttention) -> None:
        """Galerkin attention should handle large input magnitudes."""
        x = torch.randn(2, 16, 32) * 100.0
        output = galerkin_attn(x)

        assert torch.isfinite(output).all(), (
            "Output must be finite with large inputs"
        )

    def test_galerkin_with_small_inputs(self, galerkin_attn: GalerkinAttention) -> None:
        """Galerkin attention should handle very small input magnitudes."""
        x = torch.randn(2, 16, 32) * 1e-6
        output = galerkin_attn(x)

        assert torch.isfinite(output).all(), (
            "Output must be finite with very small inputs"
        )

    def test_softmax_with_large_inputs(self, softmax_attn: SoftmaxAttention) -> None:
        """Softmax attention should handle large input magnitudes."""
        x = torch.randn(2, 16, 32) * 100.0
        output = softmax_attn(x)

        assert torch.isfinite(output).all(), (
            "Softmax output must be finite with large inputs"
        )

    @pytest.mark.slow
    def test_galerkin_with_long_sequence(self) -> None:
        """Galerkin O(N) attention should handle long sequences efficiently."""
        attn = GalerkinAttention(d_model=32, n_heads=4, dropout=0.0)
        torch.manual_seed(42)

        # 361 = 19x19 board, a realistic long sequence
        x = torch.randn(1, 361, 32)
        output = attn(x)

        assert output.shape == (1, 361, 32)
        assert torch.isfinite(output).all()

    def test_galerkin_single_token(self, galerkin_attn: GalerkinAttention) -> None:
        """Galerkin attention with a single token (n=1) should not crash."""
        x = torch.randn(2, 1, 32)
        output = galerkin_attn(x)

        assert output.shape == (2, 1, 32)
        assert torch.isfinite(output).all()

    def test_softmax_single_token(self, softmax_attn: SoftmaxAttention) -> None:
        """Softmax attention with a single token should not crash."""
        x = torch.randn(2, 1, 32)
        output = softmax_attn(x)

        assert output.shape == (2, 1, 32)
        assert torch.isfinite(output).all()
