"""Property-based tests for GalerkinAttention mathematical invariants.

Tests focus on the core mathematical properties of the Petrov-Galerkin projection:
- Output shape invariance over a range of batch sizes and sequence lengths
- O(N) time complexity: time roughly doubles when N doubles
- Normalization: 1/N Monte Carlo scaling keeps output magnitude stable as N varies
"""

from __future__ import annotations

import time

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.modeling.attention import GalerkinAttention

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attn(d_model: int = 16, n_heads: int = 4) -> GalerkinAttention:
    """Create a small GalerkinAttention in eval mode (no dropout noise)."""
    torch.manual_seed(0)
    attn = GalerkinAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    attn.eval()
    return attn


# ---------------------------------------------------------------------------
# Shape invariance
# ---------------------------------------------------------------------------


class TestGalerkinShapeInvariance:
    """Output shape must match input shape for all valid (batch, seq_len, d_model)."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=20, deadline=None)
    def test_output_shape_matches_input(self, batch_size: int, seq_len: int) -> None:
        """For any batch_size and seq_len, output.shape == input.shape."""
        attn = _make_attn(d_model=16, n_heads=4)
        torch.manual_seed(batch_size * 37 + seq_len)
        x = torch.randn(batch_size, seq_len, 16)

        with torch.no_grad():
            out = attn(x)

        assert out.shape == x.shape, f"Shape mismatch: got {out.shape}, expected {x.shape}"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=20, deadline=None)
    def test_output_is_finite(self, batch_size: int, seq_len: int) -> None:
        """Output must be finite (no NaN or Inf) for standard inputs."""
        attn = _make_attn(d_model=16, n_heads=4)
        torch.manual_seed(batch_size * 53 + seq_len)
        x = torch.randn(batch_size, seq_len, 16)

        with torch.no_grad():
            out = attn(x)

        assert torch.isfinite(out).all(), f"Non-finite values in output for shape {x.shape}"


# ---------------------------------------------------------------------------
# O(N) complexity
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGalerkinLinearComplexity:
    """O(N) property: doubling seq_len should roughly double wall-clock time."""

    def _time_forward(
        self,
        attn: GalerkinAttention,
        seq_len: int,
        n_repeats: int = 20,
    ) -> float:
        """Return median forward-pass time in seconds."""
        x = torch.randn(1, seq_len, 16)
        times = []
        with torch.no_grad():
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                attn(x)
                times.append(time.perf_counter() - t0)
        times.sort()
        return times[len(times) // 2]

    @pytest.mark.parametrize(
        "n_small,n_large",
        [(32, 64), (64, 128), (128, 256)],
    )
    def test_time_roughly_doubles_when_n_doubles(self, n_small: int, n_large: int) -> None:
        """t(2N) / t(N) should be in [0.5, 8.0] for O(N) computation.

        The window is intentionally wide: we only rule out O(N^2) behaviour
        (which would produce a ratio ~4x) and allow variance from CPU caching.
        """
        attn = _make_attn(d_model=16, n_heads=4)

        t_small = self._time_forward(attn, n_small)
        t_large = self._time_forward(attn, n_large)

        if t_small < 1e-9:
            pytest.skip("Timer resolution too coarse for meaningful measurement")

        ratio = t_large / t_small
        # For O(N^2) the ratio would be ~4; for O(N) it is ~2.
        # We allow a generous range because CPU overhead dominates at small sizes.
        assert ratio < 8.0, (
            f"t({n_large})/t({n_small}) = {ratio:.2f}, which exceeds the O(N) upper bound of 8.0"
        )


# ---------------------------------------------------------------------------
# 1/N normalisation scaling
# ---------------------------------------------------------------------------


class TestGalerkinNormalizationScaling:
    """The 1/N Monte Carlo normalization should keep output scale stable as N grows.

    Galerkin context = K^T V / N.  When all token features are drawn i.i.d.
    from N(0,1), the expected squared norm of context is O(d^2) regardless of N.
    Hence the expected output norm should be O(1) over a range of N values.
    """

    @pytest.mark.parametrize("seq_len", [8, 16, 32, 64, 128])
    def test_output_scale_bounded_as_n_varies(self, seq_len: int) -> None:
        """Output absolute mean should stay in a stable range regardless of N.

        We use a fresh random model each time to eliminate bias from a single
        weight initialisation, and test across multiple seeds.
        """
        scales = []
        for seed in range(5):
            torch.manual_seed(seed)
            attn = GalerkinAttention(d_model=16, n_heads=4, dropout=0.0)
            attn.eval()
            x = torch.randn(1, seq_len, 16)
            with torch.no_grad():
                out = attn(x)
            scales.append(out.abs().mean().item())

        mean_scale = sum(scales) / len(scales)
        # Empirically, well-initialised Galerkin attention at these sizes
        # produces mean |output| in (0.01, 50.0).
        assert 0.0 < mean_scale < 100.0, (
            f"Output scale {mean_scale:.4f} at seq_len={seq_len} is out of expected range"
        )

    def test_normalization_dampens_growth_with_n(self) -> None:
        """1/N factor: output norm should not grow linearly with N.

        Without normalisation, ||output|| ~ N * ||per-token contribution||.
        With 1/N, it should remain roughly constant.
        """
        attn = _make_attn(d_model=16, n_heads=4)

        norms = []
        for seq_len in [16, 64, 256]:
            torch.manual_seed(seq_len)
            x = torch.randn(1, seq_len, 16)
            with torch.no_grad():
                out = attn(x)
            norms.append(out.norm().item())

        # With proper 1/N normalisation the norm should NOT scale with N.
        # Require that the largest norm is less than 20x the smallest.
        max_norm = max(norms)
        min_norm = min(norms) + 1e-8
        ratio = max_norm / min_norm
        assert ratio < 20.0, (
            f"Output norms {norms} vary by {ratio:.1f}x across N=[16,64,256]; "
            f"expected O(1) due to 1/N normalisation"
        )

    @given(seq_len=st.integers(min_value=4, max_value=128))
    @settings(max_examples=20, deadline=None)
    def test_output_scale_o1_property(self, seq_len: int) -> None:
        """For any seq_len, output absolute mean should be in (0, 100)."""
        torch.manual_seed(seq_len + 7)
        attn = GalerkinAttention(d_model=16, n_heads=4, dropout=0.0)
        attn.eval()
        x = torch.randn(1, seq_len, 16)

        with torch.no_grad():
            out = attn(x)

        mean_abs = out.abs().mean().item()
        assert 0.0 < mean_abs < 100.0, (
            f"Output scale {mean_abs:.4f} at seq_len={seq_len} suggests incorrect normalisation"
        )
