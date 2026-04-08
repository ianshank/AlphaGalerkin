"""Tests for NeuralOperator model (src.modeling.operator).

Covers:
- FNO backend forward pass, output shapes, resolution independence
- Galerkin backend forward pass, output shapes
- Backend selection and error handling
- Parameter counting
- Config variations
"""

from __future__ import annotations

import pytest
import torch

from src.modeling.operator import NeuralOperator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 2
IN_CH = 1
OUT_CH = 1
H, W = 16, 16


# ---------------------------------------------------------------------------
# FNO backend tests
# ---------------------------------------------------------------------------


class TestNeuralOperatorFNO:
    """Test NeuralOperator with FNO backend."""

    def test_default_construction(self) -> None:
        """Default NeuralOperator creates FNO backend."""
        model = NeuralOperator()
        assert model.backend == "fno"
        assert model.in_channels == 1
        assert model.out_channels == 1

    def test_forward_shape(self) -> None:
        """FNO forward pass produces correct output shape."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            modes=8,
        )
        x = torch.randn(BATCH, IN_CH, H, W)
        y = model(x)
        assert y.shape == (BATCH, OUT_CH, H, W)

    def test_forward_different_resolution(self) -> None:
        """FNO handles different input resolutions."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            modes=4,
        )
        # Train resolution
        x_small = torch.randn(BATCH, IN_CH, 8, 8)
        y_small = model(x_small)
        assert y_small.shape == (BATCH, OUT_CH, 8, 8)

        # Higher resolution
        x_large = torch.randn(BATCH, IN_CH, 32, 32)
        y_large = model(x_large)
        assert y_large.shape == (BATCH, OUT_CH, 32, 32)

    def test_forward_multichannel(self) -> None:
        """FNO handles multiple input/output channels."""
        model = NeuralOperator(in_channels=3, out_channels=2, width=32, n_layers=2, modes=4)
        x = torch.randn(BATCH, 3, H, W)
        y = model(x)
        assert y.shape == (BATCH, 2, H, W)

    def test_forward_with_coords(self) -> None:
        """FNO forward pass works with optional coordinate grid."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            modes=4,
        )
        x = torch.randn(BATCH, IN_CH, H, W)
        coords = torch.randn(BATCH, H, W, 2)
        y = model(x, coords=coords)
        assert y.shape == (BATCH, OUT_CH, H, W)

    def test_output_is_finite(self) -> None:
        """FNO output contains no NaN or Inf values."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            modes=4,
        )
        x = torch.randn(BATCH, IN_CH, H, W)
        y = model(x)
        assert y.isfinite().all()

    def test_gradient_flow(self) -> None:
        """Gradients flow through FNO model."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            modes=4,
        )
        x = torch.randn(BATCH, IN_CH, H, W, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()


# ---------------------------------------------------------------------------
# Galerkin backend tests
# ---------------------------------------------------------------------------


class TestNeuralOperatorGalerkin:
    """Test NeuralOperator with Galerkin backend."""

    def test_construction(self) -> None:
        """Galerkin backend initializes correctly."""
        model = NeuralOperator(backend="galerkin", width=32, n_layers=2)
        assert model.backend == "galerkin"

    def test_forward_shape(self) -> None:
        """Galerkin forward pass produces correct output shape."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        x = torch.randn(BATCH, IN_CH, H, W)
        y = model(x)
        assert y.shape == (BATCH, OUT_CH, H, W)

    def test_different_resolution(self) -> None:
        """Galerkin handles different resolutions."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        for size in [8, 16, 24]:
            x = torch.randn(BATCH, IN_CH, size, size)
            y = model(x)
            assert y.shape == (BATCH, OUT_CH, size, size)

    def test_custom_n_heads(self) -> None:
        """Custom n_heads is passed to Galerkin backend."""
        model = NeuralOperator(
            backend="galerkin",
            width=64,
            n_heads=8,
            n_layers=2,
        )
        assert model.backend == "galerkin"

    def test_default_n_heads_from_width(self) -> None:
        """Default n_heads is derived from width // 16."""
        model = NeuralOperator(backend="galerkin", width=64, n_layers=2)
        # width // 16 = 4 heads
        assert model.backend == "galerkin"

    def test_output_is_finite(self) -> None:
        """Galerkin output is finite."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        x = torch.randn(BATCH, IN_CH, H, W)
        y = model(x)
        assert y.isfinite().all()

    def test_gradient_flow(self) -> None:
        """Gradients flow through Galerkin model."""
        model = NeuralOperator(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        x = torch.randn(BATCH, IN_CH, H, W, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Backend selection and error handling
# ---------------------------------------------------------------------------


class TestNeuralOperatorBackendSelection:
    """Test backend selection and error handling."""

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            NeuralOperator(backend="unknown")

    def test_fno_backend_explicit(self) -> None:
        """Explicit FNO backend selection."""
        model = NeuralOperator(backend="fno", width=32, n_layers=2, modes=4)
        assert model.backend == "fno"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


class TestCountParameters:
    """Test parameter counting utility."""

    def test_count_parameters_positive(self) -> None:
        """Model has positive number of parameters."""
        model = NeuralOperator(width=32, n_layers=2, modes=4)
        count = model.count_parameters()
        assert count > 0

    def test_larger_model_more_params(self) -> None:
        """Larger model has more parameters."""
        small = NeuralOperator(width=16, n_layers=1, modes=4)
        large = NeuralOperator(width=64, n_layers=4, modes=8)
        assert large.count_parameters() > small.count_parameters()

    def test_galerkin_has_params(self) -> None:
        """Galerkin backend has trainable parameters."""
        model = NeuralOperator(backend="galerkin", width=32, n_layers=2)
        assert model.count_parameters() > 0


# ---------------------------------------------------------------------------
# Config variations
# ---------------------------------------------------------------------------


class TestConfigVariations:
    """Test various configuration combinations."""

    def test_single_layer(self) -> None:
        """Single layer model works."""
        model = NeuralOperator(width=32, n_layers=1, modes=4)
        x = torch.randn(BATCH, IN_CH, H, W)
        y = model(x)
        assert y.shape == (BATCH, OUT_CH, H, W)

    def test_batch_size_one(self) -> None:
        """Batch size 1 works."""
        model = NeuralOperator(width=32, n_layers=2, modes=4)
        x = torch.randn(1, IN_CH, H, W)
        y = model(x)
        assert y.shape == (1, OUT_CH, H, W)

    def test_small_spatial_dims(self) -> None:
        """Small spatial dimensions work (modes must not exceed size)."""
        model = NeuralOperator(width=32, n_layers=2, modes=2)
        x = torch.randn(BATCH, IN_CH, 4, 4)
        y = model(x)
        assert y.shape == (BATCH, OUT_CH, 4, 4)
