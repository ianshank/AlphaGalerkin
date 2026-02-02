"""Physics-informed loss components for AlphaGalerkin.

This module provides physics-informed loss terms that can be
combined with the standard policy/value losses. These losses
encode physical constraints and PDE residuals to provide
denser training signal.

Components:
- ResidualLoss: PDE residual minimization
- BoundaryLoss: Boundary condition enforcement
- InitialConditionLoss: Initial condition enforcement
- ConservationLoss: Conservation law satisfaction
- PhysicsInformedLoss: Combined physics loss with balancing

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks: A deep learning framework
    for solving forward and inverse problems.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor, nn

from src.templates.config import BaseModuleConfig
from src.training.loss_balancing import (
    LossBalancingConfig,
    create_loss_balancer,
)

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


def _get_device_from_model(model: nn.Module) -> torch.device:
    """Safely get device from model parameters.

    Args:
        model: The neural network model.

    Returns:
        Device the model is on, or CPU if model has no parameters.

    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters (e.g., DataParallel wrapper or frozen model)
        logger.warning("model_has_no_parameters", defaulting_to="cpu")
        return torch.device("cpu")


class PhysicsLossConfig(BaseModuleConfig):
    """Configuration for physics-informed loss.

    Attributes:
        residual_weight: Weight for PDE residual loss.
        boundary_weight: Weight for boundary condition loss.
        initial_weight: Weight for initial condition loss.
        conservation_weight: Weight for conservation loss.
        use_adaptive_weights: Whether to use adaptive balancing.
        balancing_config: Configuration for loss balancing.
        n_collocation_points: Number of interior collocation points.
        n_boundary_points: Number of boundary points.
        sampling_method: How to sample collocation points.

    """

    residual_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for PDE residual loss",
    )
    boundary_weight: float = Field(
        default=10.0,
        ge=0.0,
        description="Weight for boundary condition loss",
    )
    initial_weight: float = Field(
        default=10.0,
        ge=0.0,
        description="Weight for initial condition loss",
    )
    conservation_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for conservation loss",
    )

    use_adaptive_weights: bool = Field(
        default=True,
        description="Use adaptive loss balancing",
    )
    balancing_config: LossBalancingConfig | None = Field(
        default=None,
        description="Configuration for loss balancing",
    )

    n_collocation_points: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Number of interior collocation points",
    )
    n_boundary_points: int = Field(
        default=200,
        ge=10,
        le=10000,
        description="Number of boundary points per face",
    )
    sampling_method: str = Field(
        default="lhs",
        description="Collocation point sampling method",
    )


@dataclass
class PhysicsLossOutput:
    """Output from physics loss computation.

    Attributes:
        total: Total weighted loss.
        residual: PDE residual loss.
        boundary: Boundary condition loss.
        initial: Initial condition loss.
        conservation: Conservation loss.
        weights: Current loss weights.

    """

    total: Float[Tensor, ""]
    residual: Float[Tensor, ""]
    boundary: Float[Tensor, ""]
    initial: Float[Tensor, ""] | None
    conservation: Float[Tensor, ""] | None
    weights: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of scalar values."""
        result = {
            "total": self.total.item(),
            "residual": self.residual.item(),
            "boundary": self.boundary.item(),
        }
        if self.initial is not None:
            result["initial"] = self.initial.item()
        if self.conservation is not None:
            result["conservation"] = self.conservation.item()
        result.update({f"weight_{k}": v for k, v in self.weights.items()})
        return result


class ResidualLoss(nn.Module):
    """PDE residual loss.

    Minimizes the L2 norm of the PDE residual:
        L_r = 1/N Σ |L(u)(x_i) - f(x_i)|²

    where L is the differential operator and f is the source term.
    """

    VALID_REDUCTIONS: set[str] = {"mean", "sum", "none"}

    def __init__(
        self,
        pde_operator: PDEOperator,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize residual loss.

        Args:
            pde_operator: PDE operator for residual computation.
            reduction: Reduction method ('mean', 'sum', 'none').

        Raises:
            ValueError: If reduction is not valid.

        """
        super().__init__()
        if reduction not in self.VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Must be one of: {self.VALID_REDUCTIONS}"
            )
        self.pde_operator = pde_operator
        self.reduction = reduction

    def forward(
        self,
        u: Float[Tensor, "batch n"],
        coords: Float[Tensor, "batch n d"],
        time: float | None = None,
    ) -> Float[Tensor, ""]:
        """Compute residual loss.

        Args:
            u: Solution values at collocation points.
            coords: Collocation point coordinates.
            time: Time value for time-dependent PDEs.

        Returns:
            Residual loss.

        Raises:
            RuntimeError: If PDE operator residual computation fails.

        """
        batch_size = u.shape[0]
        total_loss = torch.tensor(0.0, device=u.device)

        for b in range(batch_size):
            # Compute residual for this batch element
            try:
                residual = self.pde_operator.residual(
                    u[b],
                    coords[b],
                    compute_derivatives=False,
                )
            except Exception as e:
                logger.error(
                    "residual_computation_failed",
                    batch_index=b,
                    error=str(e),
                )
                raise RuntimeError(f"PDE operator residual computation failed: {e}") from e

            # Convert to tensor if needed
            if isinstance(residual.values, Tensor):
                res_sq = residual.values**2
            elif isinstance(residual.values, np.ndarray):
                res_sq = torch.from_numpy(residual.values).to(u.device) ** 2
            else:
                raise TypeError(f"Unexpected residual type: {type(residual.values)}")

            if self.reduction == "mean":
                total_loss = total_loss + res_sq.mean()
            elif self.reduction == "sum":
                total_loss = total_loss + res_sq.sum()
            else:
                total_loss = total_loss + res_sq

        return total_loss / batch_size


class BoundaryLoss(nn.Module):
    """Boundary condition loss.

    Enforces boundary conditions:
        L_b = 1/N_b Σ |u(x_b) - g(x_b)|²

    where g is the boundary condition function.

    Supports:
    - Dirichlet: u = g on ∂Ω
    - Neumann: ∂u/∂n = g on ∂Ω (requires gradient computation)
    - Robin: αu + β∂u/∂n = g on ∂Ω
    """

    VALID_REDUCTIONS: set[str] = {"mean", "sum", "none"}
    VALID_BC_TYPES: set[str] = {"dirichlet", "neumann", "robin"}

    def __init__(
        self,
        pde_operator: PDEOperator,
        bc_type: Literal["dirichlet", "neumann", "robin"] = "dirichlet",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize boundary loss.

        Args:
            pde_operator: PDE operator for boundary values.
            bc_type: Boundary condition type.
            reduction: Reduction method.

        Raises:
            ValueError: If bc_type or reduction is invalid.

        """
        super().__init__()
        if bc_type not in self.VALID_BC_TYPES:
            raise ValueError(f"Invalid bc_type '{bc_type}'. Must be one of: {self.VALID_BC_TYPES}")
        if reduction not in self.VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Must be one of: {self.VALID_REDUCTIONS}"
            )
        self.pde_operator = pde_operator
        self.bc_type = bc_type
        self.reduction = reduction

    def forward(
        self,
        u_boundary: Float[Tensor, "batch n_b"],
        coords_boundary: Float[Tensor, "batch n_b d"],
        time: float | None = None,
    ) -> Float[Tensor, ""]:
        """Compute boundary loss.

        Args:
            u_boundary: Solution values at boundary points.
            coords_boundary: Boundary point coordinates.
            time: Time value for time-dependent PDEs.

        Returns:
            Boundary loss.

        Raises:
            RuntimeError: If boundary value computation fails.

        """
        batch_size = u_boundary.shape[0]
        total_loss = torch.tensor(0.0, device=u_boundary.device)

        for b in range(batch_size):
            # Get target boundary values
            try:
                target = self.pde_operator.boundary_value(
                    coords_boundary[b],
                    time=time,
                )
            except Exception as e:
                logger.error(
                    "boundary_value_computation_failed",
                    batch_index=b,
                    error=str(e),
                )
                raise RuntimeError(f"PDE operator boundary_value failed: {e}") from e

            # Convert to tensor if needed
            if not isinstance(target, Tensor):
                if isinstance(target, np.ndarray):
                    target = torch.from_numpy(target).to(u_boundary.device)
                else:
                    raise TypeError(f"Unexpected target type: {type(target)}")

            # Compute MSE
            bc_error = (u_boundary[b] - target) ** 2

            if self.reduction == "mean":
                total_loss = total_loss + bc_error.mean()
            elif self.reduction == "sum":
                total_loss = total_loss + bc_error.sum()
            else:
                total_loss = total_loss + bc_error

        return total_loss / batch_size


class InitialConditionLoss(nn.Module):
    """Initial condition loss for time-dependent PDEs.

    Enforces initial conditions:
        L_ic = 1/N_0 Σ |u(x, t=0) - u_0(x)|²
    """

    def __init__(
        self,
        pde_operator: PDEOperator,
        reduction: str = "mean",
    ) -> None:
        """Initialize initial condition loss.

        Args:
            pde_operator: PDE operator for initial values.
            reduction: Reduction method.

        """
        super().__init__()
        self.pde_operator = pde_operator
        self.reduction = reduction

    def forward(
        self,
        u_initial: Float[Tensor, "batch n"],
        coords_initial: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, ""]:
        """Compute initial condition loss.

        Args:
            u_initial: Solution values at t=0.
            coords_initial: Spatial coordinates at t=0.

        Returns:
            Initial condition loss.

        """
        batch_size = u_initial.shape[0]
        total_loss = torch.tensor(0.0, device=u_initial.device)

        for b in range(batch_size):
            # Get target initial values
            target = self.pde_operator.initial_condition(coords_initial[b])

            if not isinstance(target, Tensor):
                target = torch.from_numpy(target).to(u_initial.device)

            # Compute MSE
            ic_error = (u_initial[b] - target) ** 2

            if self.reduction == "mean":
                total_loss = total_loss + ic_error.mean()
            elif self.reduction == "sum":
                total_loss = total_loss + ic_error.sum()
            else:
                total_loss = total_loss + ic_error

        return total_loss / batch_size


class ConservationLoss(nn.Module):
    """Conservation law loss.

    Enforces integral conservation properties:
        L_c = |∫_Ω u(x,t) dx - ∫_Ω u_0(x) dx|²

    for conserved quantities (mass, energy, etc.).
    """

    def __init__(
        self,
        conserved_quantity: Callable[[Tensor, Tensor], Tensor] | None = None,
        initial_integral: float | None = None,
    ) -> None:
        """Initialize conservation loss.

        Args:
            conserved_quantity: Function computing conserved quantity.
            initial_integral: Expected integral value (default: preserve from t=0).

        """
        super().__init__()
        self.conserved_quantity = conserved_quantity or self._default_mass
        self.initial_integral = initial_integral
        self._stored_initial: Tensor | None = None

    def _default_mass(
        self,
        u: Tensor,
        coords: Tensor,
    ) -> Tensor:
        """Default: compute total mass/integral."""
        # Simple quadrature approximation
        n_points = u.shape[-1]
        return u.sum(dim=-1) / n_points

    def forward(
        self,
        u: Float[Tensor, "batch n"],
        coords: Float[Tensor, "batch n d"],
        u_initial: Float[Tensor, "batch n"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute conservation loss.

        Args:
            u: Current solution values.
            coords: Coordinates.
            u_initial: Initial solution (for comparison).

        Returns:
            Conservation loss.

        """
        # Compute current integral
        current_integral = self.conserved_quantity(u, coords)

        # Determine target
        if self.initial_integral is not None:
            target = self.initial_integral
        elif u_initial is not None:
            target = self.conserved_quantity(u_initial, coords)
        elif self._stored_initial is not None:
            target = self._stored_initial
        else:
            # No target available - store current as initial
            self._stored_initial = current_integral.detach()
            return torch.tensor(0.0, device=u.device)

        if isinstance(target, Tensor):
            return ((current_integral - target) ** 2).mean()
        else:
            return ((current_integral - target) ** 2).mean()


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


class CombinedAlphaGalerkinPhysicsLoss(nn.Module):
    """Combined AlphaGalerkin + Physics loss.

    Integrates standard AlphaGalerkin losses (policy, value, LBB)
    with physics-informed losses (residual, boundary, IC).

    This enables:
    - Denser training signal from physics
    - Regularization through physical constraints
    - Better generalization through physical inductive bias
    """

    def __init__(
        self,
        pde_operator: PDEOperator | None = None,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        lbb_weight: float = 0.01,
        physics_weight: float = 0.1,
        physics_config: PhysicsLossConfig | None = None,
    ) -> None:
        """Initialize combined loss.

        Args:
            pde_operator: PDE operator (optional - physics losses disabled if None).
            policy_weight: Weight for policy loss.
            value_weight: Weight for value loss.
            lbb_weight: Weight for LBB regularization.
            physics_weight: Overall weight for physics losses.
            physics_config: Configuration for physics losses.

        """
        super().__init__()

        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.lbb_weight = lbb_weight
        self.physics_weight = physics_weight

        # Import here to avoid circular dependency
        from src.training.loss import AlphaGalerkinLoss

        # Standard AlphaGalerkin loss
        self.alphagalerkin_loss = AlphaGalerkinLoss(
            policy_weight=1.0,  # Weights applied externally
            value_weight=1.0,
            lbb_weight=1.0,
        )

        # Physics loss (optional)
        if pde_operator is not None:
            physics_config = physics_config or PhysicsLossConfig(name="physics")
            self.physics_loss = PhysicsInformedLoss(pde_operator, physics_config)
        else:
            self.physics_loss = None

        # Combined loss balancing
        loss_names = ["policy", "value", "lbb"]
        if self.physics_loss is not None:
            loss_names.append("physics")

        # Use adaptive balancing for all terms
        balancing_config = LossBalancingConfig(name="combined_balancing")
        self.balancer = create_loss_balancer(balancing_config, loss_names)

    def forward(
        self,
        policy_logits: Tensor,
        value: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
        lbb_constant: Tensor | None = None,
        action_mask: Tensor | None = None,
        model: nn.Module | None = None,
        coords: Tensor | None = None,
    ) -> dict[str, Tensor | float]:
        """Compute combined loss.

        Args:
            policy_logits: Predicted policy logits.
            value: Predicted value.
            target_policy: Target policy from MCTS.
            target_value: Target value (game outcome).
            lbb_constant: LBB stability constant.
            action_mask: Valid action mask.
            model: Model for physics loss (if using).
            coords: Coordinates for physics loss.

        Returns:
            Dictionary with all loss components and total.

        """
        # Compute AlphaGalerkin losses
        ag_loss = self.alphagalerkin_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            lbb_constant=lbb_constant,
            action_mask=action_mask,
        )

        losses = {
            "policy": ag_loss.policy * self.policy_weight,
            "value": ag_loss.value * self.value_weight,
            "lbb": ag_loss.lbb * self.lbb_weight,
        }

        # Physics loss (if available)
        if self.physics_loss is not None and model is not None:
            physics_output = self.physics_loss(model, coords_interior=coords)
            losses["physics"] = physics_output.total * self.physics_weight

        # Apply balancing
        result = self.balancer.compute_weighted_loss(losses)

        # Get device from first loss for default physics tensor
        device = next(iter(losses.values())).device

        return {
            "total": result.weighted_sum,
            "policy": losses["policy"],
            "value": losses["value"],
            "lbb": losses["lbb"],
            "physics": losses.get("physics", torch.tensor(0.0, device=device)),
            "weights": result.weights,
        }
