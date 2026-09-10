"""Pydantic config and output types for physics-informed losses."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from src.templates.config import BaseModuleConfig
from src.training.loss_balancing import LossBalancingConfig


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
