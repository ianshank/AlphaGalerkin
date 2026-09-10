"""Conservation-law loss."""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.training.losses.base import register_loss


@register_loss("conservation")
class ConservationLoss(nn.Module):
    """Conservation law loss.

    Enforces integral conservation properties:
        L_c = |integral u(x,t) dx - integral u_0(x) dx|^2

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
