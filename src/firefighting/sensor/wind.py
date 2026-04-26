"""Wind sensor ingestion and spatial interpolation.

Combines point wind measurements from drone-mounted anemometers
and other sources into a spatially continuous wind field for
the fire spread solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class WindObservation:
    """A single wind measurement.

    Attributes:
        x: x-coordinate of measurement point (meters).
        y: y-coordinate of measurement point (meters).
        speed_m_s: Wind speed in m/s.
        direction_deg: Wind direction in degrees from north (clockwise).
        timestamp: Measurement time (Unix epoch).

    """

    x: float
    y: float
    speed_m_s: float
    direction_deg: float
    timestamp: float

    @property
    def u(self) -> float:
        """x-component of wind velocity (east-positive)."""
        return self.speed_m_s * np.sin(np.radians(self.direction_deg))

    @property
    def v(self) -> float:
        """y-component of wind velocity (north-positive)."""
        return self.speed_m_s * np.cos(np.radians(self.direction_deg))


class WindFieldInterpolator:
    """Interpolates point wind observations to a 2D grid.

    Uses inverse-distance-weighted (IDW) interpolation to create
    a spatially continuous wind field from sparse observations.
    Falls back to uniform field if no observations are available.
    """

    def __init__(
        self,
        default_speed_m_s: float = 5.0,
        default_direction_deg: float = 270.0,
        power: float = 2.0,
    ) -> None:
        self.default_speed = default_speed_m_s
        self.default_direction = default_direction_deg
        self.power = power

    def interpolate(
        self,
        observations: list[WindObservation],
        grid_x: NDArray[np.float64],
        grid_y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate wind observations onto simulation grid.

        Args:
            observations: List of point wind measurements.
            grid_x: x-coordinates of grid cell centers (ny, nx).
            grid_y: y-coordinates of grid cell centers (ny, nx).

        Returns:
            Tuple of (wind_u, wind_v) on the grid, in m/s.

        """
        ny, nx = grid_x.shape

        if not observations:
            # Uniform default wind
            dir_rad = np.radians(self.default_direction)
            wind_u = np.full((ny, nx), self.default_speed * np.sin(dir_rad))
            wind_v = np.full((ny, nx), self.default_speed * np.cos(dir_rad))
            return wind_u, wind_v

        # IDW interpolation
        obs_x = np.array([o.x for o in observations])
        obs_y = np.array([o.y for o in observations])
        obs_u = np.array([o.u for o in observations])
        obs_v = np.array([o.v for o in observations])

        wind_u = np.zeros((ny, nx), dtype=np.float64)
        wind_v = np.zeros((ny, nx), dtype=np.float64)

        for j in range(ny):
            for i in range(nx):
                dx = grid_x[j, i] - obs_x
                dy = grid_y[j, i] - obs_y
                dist = np.sqrt(dx**2 + dy**2)
                dist = np.maximum(dist, 1e-6)  # Avoid division by zero

                weights = 1.0 / dist**self.power
                w_sum = weights.sum()

                wind_u[j, i] = np.sum(weights * obs_u) / w_sum
                wind_v[j, i] = np.sum(weights * obs_v) / w_sum

        return wind_u, wind_v
