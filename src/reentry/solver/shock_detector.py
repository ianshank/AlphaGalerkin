"""Shock detection for adaptive limiting and AMR flagging.

Identifies shock waves using pressure and density gradients
with optional Ducros sensor for shock/vortex discrimination.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ShockDetector:
    """Detects shock waves in 2D compressible flow fields.

    Provides a shock indicator field sigma in [0, 1] where:
    - sigma ~ 1: strong shock present
    - sigma ~ 0: smooth flow

    Uses normalized pressure gradient as primary indicator with
    optional Ducros sensor to distinguish shocks from vortices.
    """

    def __init__(
        self,
        pressure_threshold: float = 0.3,
        enable_ducros: bool = True,
    ) -> None:
        self.pressure_threshold = pressure_threshold
        self.enable_ducros = enable_ducros

    def detect(
        self,
        pressure: NDArray[np.float64],
        velocity_x: NDArray[np.float64] | None = None,
        velocity_y: NDArray[np.float64] | None = None,
        dx: NDArray[np.float64] | None = None,
        dy: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute shock indicator field.

        Args:
            pressure: Pressure field (ny, nx).
            velocity_x: x-velocity (ny, nx) — needed for Ducros sensor.
            velocity_y: y-velocity (ny, nx) — needed for Ducros sensor.
            dx: Cell widths (ny, nx) or scalar.
            dy: Cell heights (ny, nx) or scalar.

        Returns:
            Shock indicator sigma (ny, nx) in [0, 1].

        """
        sigma = self._pressure_gradient_indicator(pressure)

        if self.enable_ducros and velocity_x is not None and velocity_y is not None:
            ducros = self._ducros_sensor(velocity_x, velocity_y)
            sigma *= ducros

        return sigma

    def _pressure_gradient_indicator(self, pressure: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalized pressure gradient: |dp| / (p + eps).

        Large values indicate shocks; small values indicate smooth flow.
        """
        ny, nx = pressure.shape
        dp = np.zeros_like(pressure)

        if nx > 2:
            dp[:, 1:-1] += np.abs(pressure[:, 2:] - 2 * pressure[:, 1:-1] + pressure[:, :-2])
        if ny > 2:
            dp[1:-1, :] += np.abs(pressure[2:, :] - 2 * pressure[1:-1, :] + pressure[:-2, :])

        # Normalize by local pressure
        p_norm = np.maximum(np.abs(pressure), 1e-30)
        indicator = dp / p_norm

        # Threshold to [0, 1]
        return np.clip(indicator / self.pressure_threshold, 0.0, 1.0)

    @staticmethod
    def _ducros_sensor(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ducros sensor: distinguishes shocks (compressive) from vortices.

        phi_ducros = (div(u))^2 / ((div(u))^2 + |curl(u)|^2 + eps)

        Near 1 for shocks (compressive), near 0 for vortices (rotational).
        """
        ny, nx = u.shape
        div = np.zeros_like(u)
        curl_sq = np.zeros_like(u)

        if nx > 2:
            du_dx = np.zeros_like(u)
            du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * 0.5
            dv_dx = np.zeros_like(v)
            dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        else:
            du_dx = np.zeros_like(u)
            dv_dx = np.zeros_like(v)

        if ny > 2:
            du_dy = np.zeros_like(u)
            du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
            dv_dy = np.zeros_like(v)
            dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) * 0.5
        else:
            du_dy = np.zeros_like(u)
            dv_dy = np.zeros_like(v)

        div = du_dx + dv_dy
        curl = dv_dx - du_dy

        div_sq = div**2
        curl_sq = curl**2
        eps = 1e-30

        return div_sq / (div_sq + curl_sq + eps)
