"""MUSCL reconstruction for second-order spatial accuracy.

Monotone Upstream-centered Schemes for Conservation Laws.
Reconstructs left/right interface states from cell-averaged
values using slope-limited linear interpolation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.solver import LimiterType
from src.reentry.flux.limiter import Limiter, get_limiter


class MUSCLReconstruction:
    """MUSCL reconstruction with configurable slope limiter.

    Given cell-averaged values U_i, reconstructs left and right
    interface states at i+1/2:
        U_L = U_i + 0.5 * phi(r) * (U_i - U_{i-1})
        U_R = U_{i+1} - 0.5 * phi(1/r) * (U_{i+1} - U_i)

    where phi is a TVD slope limiter and r is the slope ratio.
    """

    def __init__(self, limiter_type: LimiterType = LimiterType.VAN_LEER) -> None:
        self.limiter: Limiter = get_limiter(limiter_type)

    def reconstruct(
        self, u: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reconstruct left and right states at cell interfaces.

        For 1D data with n cells, produces n-1 interface pairs.

        Args:
            u: Cell-averaged values, shape (n,) or (n, n_vars).

        Returns:
            Tuple of (u_left, u_right), each shape (n-1,) or (n-1, n_vars).

        """
        if u.ndim == 1:
            return self._reconstruct_1d(u)
        return self._reconstruct_multivar(u)

    def _reconstruct_1d(
        self, u: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reconstruct 1D scalar field."""
        n = len(u)
        if n < 3:
            # Fall back to first order
            return u[:-1].copy(), u[1:].copy()

        # Forward and backward differences
        du_fwd = u[1:] - u[:-1]  # (n-1,)

        # Slope ratios for left reconstruction (at i+1/2, using i)
        # r_L = (U_i - U_{i-1}) / (U_{i+1} - U_i)
        eps = 1e-30  # Prevent division by zero
        r_left = np.zeros(n - 1, dtype=np.float64)
        r_left[1:] = du_fwd[:-1] / (du_fwd[1:] + eps * np.sign(du_fwd[1:] + eps))

        # Slope ratios for right reconstruction (at i+1/2, using i+1)
        r_right = np.zeros(n - 1, dtype=np.float64)
        r_right[:-1] = du_fwd[1:] / (du_fwd[:-1] + eps * np.sign(du_fwd[:-1] + eps))

        phi_left = self.limiter(r_left)
        phi_right = self.limiter(r_right)

        # Reconstructed states
        u_left = u[:-1] + 0.5 * phi_left * du_fwd
        u_right = u[1:] - 0.5 * phi_right * du_fwd

        return u_left, u_right

    def _reconstruct_multivar(
        self, u: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reconstruct multi-variable field (n, n_vars)."""
        n, n_vars = u.shape
        u_left = np.zeros((n - 1, n_vars), dtype=np.float64)
        u_right = np.zeros((n - 1, n_vars), dtype=np.float64)
        for v in range(n_vars):
            u_left[:, v], u_right[:, v] = self._reconstruct_1d(u[:, v])
        return u_left, u_right
