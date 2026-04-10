"""Wind field model for fire-atmosphere coupling.

Manages wind velocity fields with optional fire-induced
wind modification for buoyancy-driven flows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.wind import WindConfig


class WindField:
    """Wind field manager with fire-induced modification.

    Provides steady-state and time-varying wind fields with
    optional coupling to fire heat release for buoyancy effects.
    """

    def __init__(self, config: WindConfig) -> None:
        self.config = config

    def uniform_field(
        self, shape: tuple[int, int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create a uniform wind field from config defaults.

        Args:
            shape: Grid shape (ny, nx).

        Returns:
            Tuple of (wind_u, wind_v) in m/s.

        """
        direction_rad = np.radians(self.config.default_direction_deg)
        speed = self.config.default_speed_m_s * self.config.mid_flame_adjustment

        wind_u = np.full(shape, speed * np.sin(direction_rad), dtype=np.float64)
        wind_v = np.full(shape, speed * np.cos(direction_rad), dtype=np.float64)

        return wind_u, wind_v

    def apply_fire_modification(
        self,
        wind_u: NDArray[np.float64],
        wind_v: NDArray[np.float64],
        temperature: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Modify wind field due to fire-induced buoyancy.

        Hot air rises over the fire, creating an inflow at ground level
        toward the fire perimeter. This is modeled as a gradient of
        temperature driving horizontal convergence.

        Args:
            wind_u: Base x-wind (ny, nx) in m/s.
            wind_v: Base y-wind (ny, nx) in m/s.
            temperature: Temperature field (ny, nx) in K.
            dx: Cell width in meters.
            dy: Cell height in meters.

        Returns:
            Modified (wind_u, wind_v).

        """
        if not self.config.enable_fire_induced_wind:
            return wind_u, wind_v

        alpha = self.config.fire_wind_coupling_strength
        ny, nx = temperature.shape

        # Temperature gradient drives convergence toward hot spots
        dt_dx = np.zeros_like(temperature)
        dt_dy = np.zeros_like(temperature)

        if nx > 2:
            dt_dx[:, 1:-1] = (temperature[:, 2:] - temperature[:, :-2]) / (2.0 * dx)
        if ny > 2:
            dt_dy[1:-1, :] = (temperature[2:, :] - temperature[:-2, :]) / (2.0 * dy)

        # Scale by coupling strength and normalize
        fire_u = alpha * dt_dx / np.maximum(np.abs(dt_dx).max(), 1e-10)
        fire_v = alpha * dt_dy / np.maximum(np.abs(dt_dy).max(), 1e-10)

        mod_u = np.clip(
            wind_u + fire_u, -self.config.max_wind_speed_m_s, self.config.max_wind_speed_m_s
        )
        mod_v = np.clip(
            wind_v + fire_v, -self.config.max_wind_speed_m_s, self.config.max_wind_speed_m_s
        )

        return mod_u, mod_v
