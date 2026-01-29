"""Property-based tests for Galerkin Neural Operator using Hypothesis.

Tests mathematical invariants and properties that should hold for any valid input.
"""

from __future__ import annotations

import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch")

from hypothesis import given, settings, strategies as st

from src.modeling.galerkin_operator import Galerkin2d, GalerkinOperatorBlock


class TestGalerkinResolutionIndependence:
    """Property tests for resolution independence."""

    @pytest.fixture(autouse=True)
    def set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(42)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        resolution=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=20, deadline=None)
    def test_resolution_independence(self, batch: int, resolution: int) -> None:
        """Model should handle any resolution without errors."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )
        model.eval()

        x = torch.randn(batch, 1, resolution, resolution)
        y = model(x)

        assert y.shape == (batch, 1, resolution, resolution)
        assert torch.isfinite(y).all(), "Output should not contain NaN or Inf"

    @given(
        h=st.integers(min_value=8, max_value=24),
        w=st.integers(min_value=8, max_value=24),
    )
    @settings(max_examples=15, deadline=None)
    def test_non_square_resolution(self, h: int, w: int) -> None:
        """Model should handle non-square resolutions."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )
        model.eval()

        x = torch.randn(1, 1, h, w)
        y = model(x)

        assert y.shape == (1, 1, h, w)
        assert torch.isfinite(y).all()


class TestGalerkinLBBStability:
    """Property tests for LBB stability."""

    @pytest.fixture(autouse=True)
    def set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(42)

    @given(resolution=st.integers(min_value=8, max_value=24))
    @settings(max_examples=15, deadline=None)
    def test_lbb_positivity(self, resolution: int) -> None:
        """LBB constant should be positive (stability condition)."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=2,
            n_heads=2,
            fourier_features=8,
        )
        model.eval()

        x = torch.randn(2, 1, resolution, resolution)
        _, lbb_list = model(x, return_lbb=True)

        for i, lbb in enumerate(lbb_list):
            assert (lbb > 0).all(), f"LBB constant at layer {i} should be positive"

    @given(
        n_heads=st.integers(min_value=1, max_value=8),
        d_model=st.sampled_from([16, 32, 64]),
    )
    @settings(max_examples=10, deadline=None)
    def test_block_lbb_positivity(self, n_heads: int, d_model: int) -> None:
        """Block LBB should be positive for various configurations."""
        # Ensure d_model is divisible by n_heads
        d_model = (d_model // n_heads) * n_heads
        if d_model == 0:
            d_model = n_heads

        block = GalerkinOperatorBlock(
            d_model=d_model,
            n_heads=n_heads,
        )
        block.eval()

        x = torch.randn(2, 16, d_model)
        _, lbb = block(x, return_lbb=True)

        assert (lbb > 0).all(), "Block LBB should be positive"


class TestGalerkinGradientFlow:
    """Property tests for gradient flow."""

    @pytest.fixture(autouse=True)
    def set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(42)

    @given(batch=st.integers(min_value=1, max_value=4))
    @settings(max_examples=10, deadline=None)
    def test_gradient_exists(self, batch: int) -> None:
        """Gradients should flow through all parameters."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )
        model.train()

        x = torch.randn(batch, 1, 12, 12, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check that input has gradient
        assert x.grad is not None, "Input should have gradient"
        assert (x.grad != 0).any(), "Input gradient should be non-zero"

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradient"

    @given(
        n_layers=st.integers(min_value=1, max_value=4),
        width=st.sampled_from([16, 32]),
    )
    @settings(max_examples=10, deadline=None)
    def test_gradient_magnitude_reasonable(self, n_layers: int, width: int) -> None:
        """Gradient magnitudes should be reasonable (not exploding/vanishing)."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=width,
            n_layers=n_layers,
            n_heads=2,
            fourier_features=8,
        )
        model.train()

        x = torch.randn(2, 1, 12, 12, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()

        # Check gradient magnitudes
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                # Gradients should not explode
                assert grad_norm < 1e6, f"Gradient for {name} is too large: {grad_norm}"

        # At least some gradients should be non-trivial
        assert max(grad_norms) > 1e-10, "Gradients appear to be vanishing"


class TestGalerkinOutputBounds:
    """Property tests for output characteristics."""

    @pytest.fixture(autouse=True)
    def set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(42)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        resolution=st.integers(min_value=8, max_value=20),
    )
    @settings(max_examples=15, deadline=None)
    def test_output_finite(self, batch: int, resolution: int) -> None:
        """Output should always be finite."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )
        model.eval()

        x = torch.randn(batch, 1, resolution, resolution)
        y = model(x)

        assert torch.isfinite(y).all(), "Output should not contain NaN or Inf"

    @given(scale=st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_output_scale_reasonable(self, scale: float) -> None:
        """Output scale should be reasonable for scaled inputs."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )
        model.eval()

        x = torch.randn(2, 1, 12, 12) * scale
        y = model(x)

        # Output should not explode for reasonable input scales
        output_scale = y.abs().max().item()
        assert output_scale < 1e6, f"Output scale {output_scale} is unreasonably large"


class TestGalerkinZeroShotTransfer:
    """Property tests for zero-shot resolution transfer."""

    @pytest.fixture(autouse=True)
    def set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(42)

    @given(
        train_res=st.integers(min_value=8, max_value=16),
        test_res=st.integers(min_value=20, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_transfer_output_valid(self, train_res: int, test_res: int) -> None:
        """Model trained at one resolution should produce valid output at another."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )

        # Simulate training (single step)
        model.train()
        x_train = torch.randn(2, 1, train_res, train_res)
        y_train = model(x_train)
        loss = y_train.mean()
        loss.backward()

        # Test at different resolution (zero-shot transfer)
        model.eval()
        x_test = torch.randn(2, 1, test_res, test_res)
        y_test = model(x_test)

        # Output should be valid
        assert y_test.shape == (2, 1, test_res, test_res)
        assert torch.isfinite(y_test).all()

    @given(
        res_multiplier=st.floats(min_value=1.5, max_value=3.0),
    )
    @settings(max_examples=10, deadline=None)
    def test_transfer_consistency(self, res_multiplier: float) -> None:
        """Output statistics should be somewhat consistent across resolutions."""
        model = Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=16,
            n_layers=1,
            n_heads=2,
            fourier_features=8,
        )
        model.eval()

        base_res = 10
        high_res = int(base_res * res_multiplier)

        # Same random seed for similar input patterns
        torch.manual_seed(123)
        x_base = torch.randn(2, 1, base_res, base_res)
        y_base = model(x_base)

        torch.manual_seed(123)
        x_high = torch.randn(2, 1, high_res, high_res)
        y_high = model(x_high)

        # Both outputs should be finite
        assert torch.isfinite(y_base).all()
        assert torch.isfinite(y_high).all()

        # Output statistics should be in similar ranges (rough consistency)
        base_std = y_base.std().item()
        high_std = y_high.std().item()

        # Allow for some variation but not orders of magnitude difference
        ratio = max(base_std, high_std) / (min(base_std, high_std) + 1e-8)
        assert ratio < 100, f"Output scale differs too much: {base_std} vs {high_std}"
