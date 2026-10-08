r"""One reentrant corner's singular harmonic ``c * r**lam * sin(lam * phi)``.

The building block :class:`~src.pde.operators.multi_corner_poisson.MultiCornerPoissonOperator`
sums over a polyomino's corners. That module's docstring derives the formula,
the angle convention (``psi = (theta - b) mod 2*pi``, wrapping on the exterior
bisector rather than on an edge) and why the operator verifies every corner
against the geometry. Nothing here knows about polyominoes: a term is a
position, a bisector, an interior angle and a strength, and its two evaluators
(numpy float64, torch in the input's dtype) are pure functions of the points.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from src.pde.geometry_polyomino import POLYOMINO_REENTRANT_INTERIOR_ANGLE

#: One full turn, the modulus of the angle wrap.
FULL_TURN: Final[float] = 2.0 * math.pi

#: Default interior angle of a singular corner: the 270-degree reentrant corner
#: every polyomino has (exponent ``pi / w = 2/3``).
DEFAULT_CORNER_INTERIOR_ANGLE: Final[float] = POLYOMINO_REENTRANT_INTERIOR_ANGLE

#: Default singular strength ``c``: the canonical benchmark's unit coefficient.
DEFAULT_SINGULAR_COEFFICIENT: Final[float] = 1.0


class SingularCornerTerm(BaseModel):
    """One corner's singular harmonic ``c * r**lam * sin(lam * phi)``."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    x: float = Field(description="Corner x coordinate.")
    y: float = Field(description="Corner y coordinate.")
    exterior_bisector: float = Field(
        description=(
            "Angle (radians) bisecting the exterior wedge at the corner. The branch "
            "cut runs along it, so it must point out of the domain."
        ),
    )
    interior_angle: float = Field(
        default=DEFAULT_CORNER_INTERIOR_ANGLE,
        gt=math.pi,
        lt=FULL_TURN,
        description="Interior angle w of the reentrant corner; the exponent is pi / w.",
    )
    coefficient: float = Field(
        default=DEFAULT_SINGULAR_COEFFICIENT,
        description="Singular strength c of this corner's term.",
    )

    @property
    def position(self) -> tuple[float, float]:
        """The corner as an ``(x, y)`` pair."""
        return (self.x, self.y)

    @property
    def exponent(self) -> float:
        """Singular exponent ``lam = pi / w`` (2/3 for a 270-degree corner)."""
        return math.pi / self.interior_angle

    @property
    def wedge_offset(self) -> float:
        """``(2*pi - w) / 2``: the angle from the cut to either edge of the corner."""
        return (FULL_TURN - self.interior_angle) / 2.0


def singular_term_values(
    points: NDArray[np.float64], term: SingularCornerTerm
) -> NDArray[np.float64]:
    """``s_i`` (no coefficient) at ``(n, 2)`` points, in float64; exactly 0 at the corner."""
    dx = points[:, 0] - term.x
    dy = points[:, 1] - term.y
    r = np.hypot(dx, dy)
    phi = np.mod(np.arctan2(dy, dx) - term.exterior_bisector, FULL_TURN) - term.wedge_offset
    lam = term.exponent
    values = np.power(r, lam) * np.sin(lam * phi)
    return np.asarray(np.where(r > 0.0, values, 0.0), dtype=np.float64)


def singular_term_tensor(points: Tensor, term: SingularCornerTerm) -> Tensor:
    """Torch twin of :func:`singular_term_values`, in the input's dtype.

    Double-``where``: the corner itself is replaced by a harmless stand-in
    before ``atan2``/``hypot``/``pow`` see it, so autograd returns a finite
    (zero) gradient there instead of a NaN that would poison a whole batch.
    """
    dx = points[:, 0] - term.x
    dy = points[:, 1] - term.y
    at_corner = (dx == 0) & (dy == 0)
    safe_dx = torch.where(at_corner, torch.ones_like(dx), dx)
    safe_dy = torch.where(at_corner, torch.zeros_like(dy), dy)
    r = torch.hypot(safe_dx, safe_dy)
    theta = torch.atan2(safe_dy, safe_dx)
    phi = torch.remainder(theta - term.exterior_bisector, FULL_TURN) - term.wedge_offset
    lam = term.exponent
    values = r.pow(lam) * torch.sin(lam * phi)
    return torch.where(at_corner, torch.zeros_like(values), values)


__all__ = [
    "DEFAULT_CORNER_INTERIOR_ANGLE",
    "DEFAULT_SINGULAR_COEFFICIENT",
    "FULL_TURN",
    "SingularCornerTerm",
    "singular_term_tensor",
    "singular_term_values",
]
