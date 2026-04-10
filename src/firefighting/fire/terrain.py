"""Terrain slope effects on fire spread rate.

Uphill spread is accelerated, downhill is decelerated,
following the Rothermel slope correction factor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.terrain import TerrainConfig


class TerrainEffects:
    """Terrain-induced modification of fire spread rate.

    Implements the Rothermel slope factor:
        phi_s = slope_factor * (tan(slope))^2

    Fire spreads faster uphill due to preheating of fuel
    by radiative and convective heat from flames below.
    """

    def __init__(self, config: TerrainConfig) -> None:
        self.config = config

    def compute_slope_factor(
        self,
        elevation: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> NDArray[np.float64]:
        """Compute slope-based spread rate multiplier.

        Args:
            elevation: Elevation field (ny, nx) in meters.
            dx: Cell width in meters.
            dy: Cell height in meters.

        Returns:
            Spread rate multiplier (ny, nx), >= 1.0 for uphill.

        """
        if self.config.flat_terrain or not self.config.enable_slope_effects:
            return np.ones_like(elevation)

        # Compute slope magnitude via central differences
        ny, nx = elevation.shape
        dz_dx = np.zeros_like(elevation)
        dz_dy = np.zeros_like(elevation)

        if nx > 2:
            dz_dx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / (2.0 * dx)
        if ny > 2:
            dz_dy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / (2.0 * dy)

        slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)

        # Convert to angle and clamp
        slope_angle_rad = np.arctan(slope_magnitude)
        max_rad = np.radians(self.config.max_slope_deg)
        slope_angle_rad = np.clip(slope_angle_rad, 0.0, max_rad)

        # Rothermel slope factor
        tan_slope = np.tan(slope_angle_rad)
        phi_s = self.config.slope_factor * tan_slope**2

        return 1.0 + phi_s

    def compute_slope_direction(
        self,
        elevation: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute uphill direction unit vectors.

        Args:
            elevation: Elevation field (ny, nx) in meters.
            dx: Cell width in meters.
            dy: Cell height in meters.

        Returns:
            Tuple of (uphill_x, uphill_y) unit vector components.

        """
        ny, nx = elevation.shape
        dz_dx = np.zeros_like(elevation)
        dz_dy = np.zeros_like(elevation)

        if nx > 2:
            dz_dx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / (2.0 * dx)
        if ny > 2:
            dz_dy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / (2.0 * dy)

        mag = np.sqrt(dz_dx**2 + dz_dy**2)
        mag = np.maximum(mag, 1e-10)

        return dz_dx / mag, dz_dy / mag
