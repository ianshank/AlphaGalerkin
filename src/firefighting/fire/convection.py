"""Convective heat transfer for wildfire spread.

Implements wind-driven convective transport of heat
from burning to unburned regions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.fire import FireConfig


class ConvectiveHeatTransfer:
    """Wind-driven convective heat transport.

    Models the advection of heat by wind using upwind differencing.
    The convective flux at each cell face is proportional to wind
    velocity and temperature gradient.
    """

    def __init__(self, config: FireConfig) -> None:
        self.config = config

    def compute(
        self,
        temperature: NDArray[np.float64],
        wind_u: NDArray[np.float64],
        wind_v: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> NDArray[np.float64]:
        """Compute convective heat flux divergence.

        Uses first-order upwind differencing for stability.

        Args:
            temperature: Temperature field (ny, nx) in K.
            wind_u: x-component of wind velocity (ny, nx) in m/s.
            wind_v: y-component of wind velocity (ny, nx) in m/s.
            dx: Cell width in meters.
            dy: Cell height in meters.

        Returns:
            Convective flux divergence (ny, nx) in K/s (to be multiplied by dt).

        """
        ny, nx = temperature.shape

        # Upwind scheme for x-direction
        flux_x = np.zeros_like(temperature)
        # Positive wind (left to right)
        pos_u = np.maximum(wind_u, 0.0)
        neg_u = np.minimum(wind_u, 0.0)

        if nx > 2:
            # Forward difference for negative velocity
            flux_x[:, 1:-1] = (
                pos_u[:, 1:-1] * (temperature[:, 1:-1] - temperature[:, :-2]) / dx
                + neg_u[:, 1:-1] * (temperature[:, 2:] - temperature[:, 1:-1]) / dx
            )

        # Upwind scheme for y-direction
        flux_y = np.zeros_like(temperature)
        pos_v = np.maximum(wind_v, 0.0)
        neg_v = np.minimum(wind_v, 0.0)

        if ny > 2:
            flux_y[1:-1, :] = (
                pos_v[1:-1, :] * (temperature[1:-1, :] - temperature[:-2, :]) / dy
                + neg_v[1:-1, :] * (temperature[2:, :] - temperature[1:-1, :]) / dy
            )

        return -(flux_x + flux_y)
