"""Boundary condition loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.training.losses.base import register_loss

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


@register_loss("boundary")
class BoundaryLoss(nn.Module):
    """Boundary condition loss.

    Enforces boundary conditions:
        L_b = 1/N_b Sigma |u(x_b) - g(x_b)|^2

    where g is the boundary condition function.

    Supports:
    - Dirichlet: u = g on boundary
    - Neumann: du/dn = g on boundary (requires gradient computation)
    - Robin: alpha*u + beta*du/dn = g on boundary
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
