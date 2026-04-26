"""Level-set fire perimeter tracking.

Tracks the fire boundary as the zero level set of a signed
distance function, enabling smooth perimeter evolution and
accurate area/perimeter computations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LevelSetPerimeter:
    """Level-set method for fire perimeter tracking.

    The fire perimeter is the zero contour of phi:
    - phi < 0: inside fire (burned/burning)
    - phi > 0: outside fire (unburned)
    - phi = 0: fire perimeter

    Evolution equation:
        phi_t + F * |grad(phi)| = 0

    where F is the local spread rate (function of wind,
    slope, fuel, and fire physics).
    """

    def __init__(self, dx: float, dy: float) -> None:
        self.dx = dx
        self.dy = dy
        self.phi: NDArray[np.float64] | None = None

    def initialize_from_mask(self, burning_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Initialize level set from a boolean burning mask.

        Computes an approximate signed distance function.

        Args:
            burning_mask: Boolean mask where True = burning.

        Returns:
            Level set function phi (ny, nx).

        """
        # Simple initialization: +1 for unburned, -1 for burned
        self.phi = np.where(burning_mask, -1.0, 1.0)
        # Reinitialize to approximate signed distance
        self._reinitialize()
        return self.phi

    def advance(
        self,
        spread_rate: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance the level set by one timestep.

        Uses first-order upwind scheme for the Hamilton-Jacobi equation:
            phi_t + F * |grad(phi)| = 0

        Args:
            spread_rate: Local fire spread rate (ny, nx) in m/s.
            dt: Timestep in seconds.

        Returns:
            Updated level set function.

        """
        if self.phi is None:
            msg = "Level set not initialized. Call initialize_from_mask first."
            raise RuntimeError(msg)

        # Compute upwind gradient magnitude
        grad_mag = self._upwind_gradient_magnitude(self.phi, spread_rate)

        # Hamilton-Jacobi update
        self.phi = self.phi - dt * spread_rate * grad_mag
        return self.phi

    def get_perimeter_mask(self, band_width: float = 1.5) -> NDArray[np.bool_]:
        """Get cells near the fire perimeter (narrow band).

        Args:
            band_width: Half-width of narrow band in cell units.

        Returns:
            Boolean mask of cells near perimeter.

        """
        if self.phi is None:
            msg = "Level set not initialized."
            raise RuntimeError(msg)
        threshold = band_width * max(self.dx, self.dy)
        return np.abs(self.phi) < threshold

    def get_burned_mask(self) -> NDArray[np.bool_]:
        """Get mask of burned/burning cells (phi <= 0)."""
        if self.phi is None:
            msg = "Level set not initialized."
            raise RuntimeError(msg)
        return self.phi <= 0.0

    def get_burned_area_m2(self) -> float:
        """Compute total burned area in m^2."""
        mask = self.get_burned_mask()
        return float(np.sum(mask) * self.dx * self.dy)

    def _upwind_gradient_magnitude(
        self,
        phi: NDArray[np.float64],
        speed: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute |grad(phi)| using Godunov's upwind scheme."""
        ny, nx = phi.shape

        # Forward/backward differences in x
        dx_fwd = np.zeros_like(phi)
        dx_bwd = np.zeros_like(phi)
        if nx > 1:
            dx_fwd[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / self.dx
            dx_bwd[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / self.dx

        # Forward/backward differences in y
        dy_fwd = np.zeros_like(phi)
        dy_bwd = np.zeros_like(phi)
        if ny > 1:
            dy_fwd[:-1, :] = (phi[1:, :] - phi[:-1, :]) / self.dy
            dy_bwd[1:, :] = (phi[1:, :] - phi[:-1, :]) / self.dy

        # Godunov upwind: pick appropriate one-sided difference
        # For positive speed (fire expanding outward)
        pos_mask = speed >= 0
        grad_x = np.where(
            pos_mask,
            np.sqrt(np.maximum(dx_bwd, 0.0) ** 2 + np.minimum(dx_fwd, 0.0) ** 2),
            np.sqrt(np.minimum(dx_bwd, 0.0) ** 2 + np.maximum(dx_fwd, 0.0) ** 2),
        )
        grad_y = np.where(
            pos_mask,
            np.sqrt(np.maximum(dy_bwd, 0.0) ** 2 + np.minimum(dy_fwd, 0.0) ** 2),
            np.sqrt(np.minimum(dy_bwd, 0.0) ** 2 + np.maximum(dy_fwd, 0.0) ** 2),
        )

        return np.sqrt(grad_x**2 + grad_y**2)

    def _reinitialize(self, n_iter: int = 20) -> None:
        """Reinitialize phi to a signed distance function.

        Uses the iterative PDE-based reinitialization:
            phi_t + sign(phi_0) * (|grad(phi)| - 1) = 0
        """
        if self.phi is None:
            return

        sign_phi = np.sign(self.phi)
        dt_reinit = 0.3 * min(self.dx, self.dy)

        for _ in range(n_iter):
            grad_mag = self._upwind_gradient_magnitude(self.phi, sign_phi)
            self.phi -= dt_reinit * sign_phi * (grad_mag - 1.0)
