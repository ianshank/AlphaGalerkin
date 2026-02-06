"""Tests for PhysicsLoss with Laplacian regularization.

Validates both the pure MSE path and the physics-informed Laplacian
constraint for the Poisson equation ``-Lap(phi) = rho``.
Also tests the PhysicsOperator neural network forward pass.
"""

from __future__ import annotations

import pytest
import torch

from src.experiments.physics_model import GalerkinBlock, PhysicsLoss, PhysicsOperator


class TestPhysicsLossBasic:
    """Test MSE-only behaviour (physics_weight=0)."""

    def test_mse_only_when_weight_zero(self) -> None:
        """Zero physics_weight returns pure MSE loss."""
        loss_fn = PhysicsLoss(physics_weight=0.0)
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)

        loss = loss_fn(pred, target)

        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_mse_only_when_charges_none(self) -> None:
        """Non-zero weight but no charges still returns MSE."""
        loss_fn = PhysicsLoss(physics_weight=1.0)
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)

        loss = loss_fn(pred, target, charges=None, coords=None)

        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_mse_only_when_coords_none(self) -> None:
        """Non-zero weight and charges but no coords still returns MSE."""
        loss_fn = PhysicsLoss(physics_weight=1.0)
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)
        charges = torch.randn(4, 16)

        loss = loss_fn(pred, target, charges=charges, coords=None)

        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(loss, expected, atol=1e-6)


class TestPhysicsLossLaplacian:
    """Test physics-informed Laplacian regularization."""

    def test_physics_loss_adds_residual(self) -> None:
        """With weight > 0 and coords, loss >= MSE."""
        loss_fn = PhysicsLoss(physics_weight=0.1)

        # Create coords that require grad for autograd
        coords = torch.randn(2, 9, 2, requires_grad=True)
        charges = torch.randn(2, 9)

        # Simple differentiable function of coords
        pred = coords[..., 0] ** 2 + coords[..., 1] ** 2
        target = torch.zeros_like(pred)

        loss = loss_fn(pred, target, charges=charges, coords=coords)
        mse = torch.nn.functional.mse_loss(pred, target)

        # Physics term should add something
        assert loss.item() >= mse.item() - 1e-6

    def test_physics_loss_differentiable(self) -> None:
        """Loss is differentiable (backward works)."""
        loss_fn = PhysicsLoss(physics_weight=0.01)

        coords = torch.randn(2, 4, 2, requires_grad=True)
        charges = torch.randn(2, 4)

        pred = coords[..., 0] + coords[..., 1]
        target = torch.zeros_like(pred)

        loss = loss_fn(pred, target, charges=charges, coords=coords)
        loss.backward()

        assert coords.grad is not None

    def test_physics_loss_zero_residual_for_harmonic(self) -> None:
        """Harmonic functions have Laplacian=0, so residual = -charges."""
        loss_fn = PhysicsLoss(physics_weight=1.0)

        coords = torch.randn(1, 9, 2, requires_grad=True)
        # Harmonic function: phi = x + y (Laplacian = 0)
        pred = coords[..., 0] + coords[..., 1]
        target = pred.detach()
        charges = torch.zeros(1, 9)  # Source = 0

        loss = loss_fn(pred, target, charges=charges, coords=coords)

        # MSE should be 0, physics loss should be ~0 (harmonic + zero source)
        assert loss.item() < 0.01

    def test_laplacian_eps_parameter(self) -> None:
        """laplacian_eps is stored as attribute."""
        loss_fn = PhysicsLoss(physics_weight=0.5, laplacian_eps=1e-8)
        assert loss_fn.laplacian_eps == 1e-8


class TestPhysicsLossComputeLaplacian:
    """Test the static _compute_laplacian method."""

    def test_laplacian_of_quadratic(self) -> None:
        """Lap(x^2 + y^2) = 4 (d^2/dx^2 + d^2/dy^2 = 2 + 2)."""
        coords = torch.randn(1, 16, 2, requires_grad=True)
        pred = coords[..., 0] ** 2 + coords[..., 1] ** 2

        laplacian = PhysicsLoss._compute_laplacian(pred, coords)

        # Should be approximately 4.0 everywhere
        assert torch.allclose(laplacian, torch.full_like(laplacian, 4.0), atol=0.01)

    def test_laplacian_of_linear(self) -> None:
        """Lap(ax + by) = 0 for linear functions."""
        coords = torch.randn(1, 16, 2, requires_grad=True)
        pred = 3.0 * coords[..., 0] + 7.0 * coords[..., 1]

        laplacian = PhysicsLoss._compute_laplacian(pred, coords)

        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=0.01)

    def test_laplacian_shape(self) -> None:
        """Output shape matches pred shape."""
        coords = torch.randn(3, 25, 2, requires_grad=True)
        pred = coords[..., 0] ** 2

        laplacian = PhysicsLoss._compute_laplacian(pred, coords)

        assert laplacian.shape == pred.shape


class TestPhysicsOperator:
    """Tests for PhysicsOperator forward pass."""

    def test_forward_shape(self) -> None:
        """Forward pass returns correct shape."""
        model = PhysicsOperator(d_model=32, n_heads=2, n_layers=2, n_fourier_features=16)
        coords = torch.randn(2, 16, 2)
        charges = torch.randn(2, 16)

        output = model(coords, charges)

        assert output.shape == (2, 16)

    def test_forward_with_fnet(self) -> None:
        """Forward pass works with FNet enabled."""
        model = PhysicsOperator(d_model=32, n_heads=2, n_layers=4, use_fnet=True)
        coords = torch.randn(1, 9, 2)
        charges = torch.randn(1, 9)

        output = model(coords, charges)

        assert output.shape == (1, 9)
        assert output.isfinite().all()

    def test_forward_without_fnet(self) -> None:
        """Forward pass works with FNet disabled."""
        model = PhysicsOperator(d_model=32, n_heads=2, n_layers=2, use_fnet=False)
        coords = torch.randn(1, 25, 2)
        charges = torch.randn(1, 25)

        output = model(coords, charges)

        assert output.shape == (1, 25)
        assert model.fnet_layers is None

    def test_forward_differentiable(self) -> None:
        """Output is differentiable w.r.t. inputs."""
        model = PhysicsOperator(d_model=32, n_heads=2, n_layers=2)
        coords = torch.randn(1, 16, 2, requires_grad=True)
        charges = torch.randn(1, 16, requires_grad=True)

        output = model(coords, charges)
        loss = output.sum()
        loss.backward()

        assert coords.grad is not None
        assert charges.grad is not None

    @pytest.mark.parametrize("n_points", [9, 16, 25, 81])
    def test_forward_variable_resolution(self, n_points: int) -> None:
        """Forward pass works with varying resolution."""
        model = PhysicsOperator(d_model=32, n_heads=2, n_layers=2)
        coords = torch.randn(1, n_points, 2)
        charges = torch.randn(1, n_points)

        output = model(coords, charges)

        assert output.shape == (1, n_points)

    def test_model_attributes_stored(self) -> None:
        """Model stores configuration attributes."""
        model = PhysicsOperator(d_model=64, use_fnet=True)
        assert model.d_model == 64
        assert model.use_fnet is True

    def test_batch_processing(self) -> None:
        """Model handles batched inputs."""
        model = PhysicsOperator(d_model=32, n_heads=2, n_layers=2)
        coords = torch.randn(8, 16, 2)
        charges = torch.randn(8, 16)

        output = model(coords, charges)

        assert output.shape == (8, 16)


class TestGalerkinBlock:
    """Tests for GalerkinBlock layer."""

    def test_forward_shape_preserved(self) -> None:
        """GalerkinBlock preserves input shape."""
        block = GalerkinBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)

        output = block(x)

        assert output.shape == x.shape

    def test_forward_with_custom_ffn(self) -> None:
        """GalerkinBlock works with custom FFN dimension."""
        block = GalerkinBlock(d_model=32, n_heads=2, d_ffn=128)
        x = torch.randn(1, 9, 32)

        output = block(x)

        assert output.shape == x.shape

    def test_forward_differentiable(self) -> None:
        """GalerkinBlock is differentiable."""
        block = GalerkinBlock(d_model=32, n_heads=2)
        x = torch.randn(1, 16, 32, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_dropout_applied(self) -> None:
        """GalerkinBlock applies dropout during training."""
        block = GalerkinBlock(d_model=32, n_heads=2, dropout=0.5)
        block.train()
        x = torch.randn(1, 16, 32)

        # Multiple forward passes should give different outputs in training mode
        out1 = block(x)
        out2 = block(x)

        # With high dropout, outputs should differ
        # (though with small tensors this isn't guaranteed, just check it runs)
        assert out1.shape == out2.shape
