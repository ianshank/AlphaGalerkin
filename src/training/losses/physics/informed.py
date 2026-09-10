"""Combined physics-informed loss with adaptive balancing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
import torch
from torch import Tensor, nn

from src.training.loss_balancing import LossBalancingConfig, create_loss_balancer
from src.training.losses.base import register_loss
from src.training.losses.physics.boundary import BoundaryLoss
from src.training.losses.physics.config import PhysicsLossConfig, PhysicsLossOutput
from src.training.losses.physics.conservation import ConservationLoss
from src.training.losses.physics.device import _get_device_from_model
from src.training.losses.physics.initial_condition import InitialConditionLoss
from src.training.losses.physics.residual import ResidualLoss

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


@register_loss("physics_informed")
class PhysicsInformedLoss(nn.Module):
    """Combined physics-informed loss with adaptive balancing.

    Combines multiple physics loss terms:
        L = w_r * L_residual + w_b * L_boundary + w_ic * L_initial + w_c * L_conservation

    Supports:
    - Fixed weights
    - Adaptive balancing (ReLoBRaLo, GradNorm, etc.)
    - Curriculum learning (gradual constraint introduction)
    """

    def __init__(
        self,
        pde_operator: PDEOperator,
        config: PhysicsLossConfig,
    ) -> None:
        """Initialize physics-informed loss.

        Args:
            pde_operator: PDE operator for residual/BC computation.
            config: Loss configuration.

        """
        super().__init__()
        self.pde_operator = pde_operator
        self.config = config

        # Individual loss components
        self.residual_loss = ResidualLoss(pde_operator)
        self.boundary_loss = BoundaryLoss(pde_operator)

        if pde_operator.is_time_dependent:
            self.initial_loss = InitialConditionLoss(pde_operator)
        else:
            self.initial_loss = None

        self.conservation_loss = ConservationLoss()

        # Loss balancer
        loss_names = ["residual", "boundary"]
        if pde_operator.is_time_dependent:
            loss_names.append("initial")
        if config.conservation_weight > 0:
            loss_names.append("conservation")

        if config.use_adaptive_weights:
            balancing_config = config.balancing_config or LossBalancingConfig(
                name="physics_balancing"
            )
            self.balancer = create_loss_balancer(balancing_config, loss_names)
        else:
            # Static weights
            self._static_weights = {
                "residual": config.residual_weight,
                "boundary": config.boundary_weight,
                "initial": config.initial_weight,
                "conservation": config.conservation_weight,
            }
            self.balancer = None

        # Collocation point cache
        self._collocation_points: Tensor | None = None
        self._boundary_points: Tensor | None = None

    def _generate_collocation_points(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Generate collocation and boundary points.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            Tuple of (interior_points, boundary_points).

        """
        # Interior points
        interior_np = self.pde_operator.generate_collocation_points(
            n_points=self.config.n_collocation_points,
            method=self.config.sampling_method,
        )
        interior = torch.from_numpy(interior_np).to(device)
        interior = interior.unsqueeze(0).expand(batch_size, -1, -1)

        # Boundary points
        boundary_np = self.pde_operator.generate_boundary_points(
            n_points_per_face=self.config.n_boundary_points,
        )
        boundary = torch.from_numpy(boundary_np).to(device)
        boundary = boundary.unsqueeze(0).expand(batch_size, -1, -1)

        return interior, boundary

    def forward(
        self,
        model: nn.Module,
        coords_interior: Tensor | None = None,
        coords_boundary: Tensor | None = None,
        coords_initial: Tensor | None = None,
        time: float | None = None,
    ) -> PhysicsLossOutput:
        """Compute combined physics-informed loss.

        Args:
            model: Neural network model (coords -> solution).
            coords_interior: Interior collocation points (auto-generated if None).
            coords_boundary: Boundary points (auto-generated if None).
            coords_initial: Initial time points (for time-dependent).
            time: Current time value.

        Returns:
            PhysicsLossOutput with all loss components.

        """
        # Get device from model (safely)
        device = _get_device_from_model(model)

        # Determine batch size from provided coordinates or default to 1
        if coords_interior is not None:
            batch_size = coords_interior.shape[0]
        elif coords_boundary is not None:
            batch_size = coords_boundary.shape[0]
        else:
            batch_size = 1

        # Generate or use provided collocation points
        if coords_interior is None:
            coords_interior, auto_boundary = self._generate_collocation_points(batch_size, device)
            if coords_boundary is None:
                coords_boundary = auto_boundary
        elif coords_boundary is None:
            _, coords_boundary = self._generate_collocation_points(batch_size, device)

        # Clone coords to avoid modifying input tensor in-place
        # Enable gradients for coords (needed for residual computation)
        coords_interior_grad = coords_interior.clone().requires_grad_(True)

        logger.debug(
            "computing_physics_loss",
            batch_size=batch_size,
            n_interior=coords_interior.shape[1],
            n_boundary=coords_boundary.shape[1],
            device=str(device),
        )

        # Forward pass through model
        u_interior = model(coords_interior_grad)
        u_boundary = model(coords_boundary)

        # Compute individual losses
        loss_residual = self.residual_loss(u_interior, coords_interior_grad, time)
        loss_boundary = self.boundary_loss(u_boundary, coords_boundary, time)

        losses = {
            "residual": loss_residual,
            "boundary": loss_boundary,
        }

        # Initial condition loss (time-dependent only)
        loss_initial = None
        if self.initial_loss is not None and coords_initial is not None:
            u_initial = model(coords_initial)
            loss_initial = self.initial_loss(u_initial, coords_initial)
            losses["initial"] = loss_initial

        # Conservation loss
        loss_conservation = None
        if self.config.conservation_weight > 0:
            loss_conservation = self.conservation_loss(u_interior, coords_interior_grad)
            losses["conservation"] = loss_conservation

        # Compute weighted total
        if self.balancer is not None:
            result = self.balancer.compute_weighted_loss(losses)
            total_loss = result.weighted_sum
            weights = result.weights
        else:
            weights = self._static_weights
            total_loss = sum(weights.get(name, 0.0) * loss for name, loss in losses.items())

        return PhysicsLossOutput(
            total=total_loss,
            residual=loss_residual,
            boundary=loss_boundary,
            initial=loss_initial,
            conservation=loss_conservation,
            weights=weights,
        )
