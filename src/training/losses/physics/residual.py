"""PDE residual loss."""

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


@register_loss("residual")
class ResidualLoss(nn.Module):
    """PDE residual loss.

    Minimizes the L2 norm of the PDE residual:
        L_r = 1/N Sigma |L(u)(x_i) - f(x_i)|^2

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
