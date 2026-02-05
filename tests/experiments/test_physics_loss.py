"""Tests for PhysicsLoss with Laplacian regularization.

Validates both the pure MSE path and the physics-informed Laplacian
constraint for the Poisson equation ``-Lap(phi) = rho``.
"""

from __future__ import annotations

import torch

from src.experiments.physics_model import PhysicsLoss


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
