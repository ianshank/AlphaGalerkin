"""Initial-condition loss for time-dependent PDEs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.training.losses.base import register_loss

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator


@register_loss("initial_condition")
class InitialConditionLoss(nn.Module):
    """Initial condition loss for time-dependent PDEs.

    Enforces initial conditions:
        L_ic = 1/N_0 Sigma |u(x, t=0) - u_0(x)|^2
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
