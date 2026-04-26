"""Slope limiters for MUSCL reconstruction.

All limiters satisfy the TVD (Total Variation Diminishing) property,
ensuring monotonicity preservation near shocks and contact discontinuities.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.solver import LimiterType


@runtime_checkable
class Limiter(Protocol):
    """Slope limiter interface."""

    def __call__(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute limiter value phi(r) for slope ratio r.

        Args:
            r: Ratio of consecutive gradients.

        Returns:
            Limiter value phi(r) in [0, 2].

        """
        ...


def minmod(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minmod limiter: phi(r) = max(0, min(1, r)).

    Most diffusive TVD limiter. Clips to first-order near extrema.
    """
    return np.maximum(0.0, np.minimum(1.0, r))


def van_leer(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Van Leer limiter: phi(r) = (r + |r|) / (1 + |r|).

    Smooth, symmetric limiter. Good balance of accuracy and stability.
    Preferred default for hypersonic flows.
    """
    abs_r = np.abs(r)
    return (r + abs_r) / (1.0 + abs_r)


def van_albada(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Van Albada limiter: phi(r) = (r^2 + r) / (r^2 + 1).

    Differentiable everywhere, which can help convergence.
    """
    r2 = r * r
    return (r2 + r) / (r2 + 1.0)


def superbee(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Superbee limiter: phi(r) = max(0, min(2r, 1), min(r, 2)).

    Least diffusive TVD limiter. Can cause overshoots in some cases.
    """
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


def no_limiter(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """No limiting (second-order central). NOT TVD — use for testing only."""
    return np.ones_like(r)


_LIMITERS: dict[LimiterType, Limiter] = {
    LimiterType.MINMOD: minmod,
    LimiterType.VAN_LEER: van_leer,
    LimiterType.VAN_ALBADA: van_albada,
    LimiterType.SUPERBEE: superbee,
    LimiterType.NONE: no_limiter,
}


def get_limiter(limiter_type: LimiterType) -> Limiter:
    """Get limiter function by type.

    Args:
        limiter_type: Which limiter to use.

    Returns:
        Callable limiter function.

    """
    return _LIMITERS[limiter_type]
