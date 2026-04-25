"""Sensor data to PDE boundary condition encoder.

Converts raw sensor readings (thermal images, wind measurements)
into boundary conditions for the fire spread solver, with
confidence-weighted blending when data is partially stale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.sensor import SensorConfig
from src.firefighting.sensor.staleness import StalenessTracker


@dataclass
class BoundaryConditions:
    """PDE boundary conditions derived from sensor data.

    Attributes:
        temperature_field: Temperature observations mapped to grid (ny, nx) in K.
        wind_u: x-component of wind velocity (ny, nx) in m/s.
        wind_v: y-component of wind velocity (ny, nx) in m/s.
        confidence: Overall confidence in these BCs [0, 1].
        thermal_confidence: Confidence in thermal data specifically.
        wind_confidence: Confidence in wind data specifically.

    """

    temperature_field: NDArray[np.float64]
    wind_u: NDArray[np.float64]
    wind_v: NDArray[np.float64]
    confidence: float
    thermal_confidence: float
    wind_confidence: float


class BoundaryEncoder:
    """Encodes sensor data into PDE boundary conditions.

    Maintains staleness trackers for thermal and wind sensors.
    When data goes stale, blends toward last-known-good values
    with decreasing confidence.
    """

    def __init__(self, config: SensorConfig, grid_shape: tuple[int, int]) -> None:
        self.config = config
        self.grid_shape = grid_shape

        self.thermal_tracker = StalenessTracker(config)
        self.wind_tracker = StalenessTracker(config)

        # Last known good values
        self._last_thermal: NDArray[np.float64] | None = None
        self._last_wind_u: NDArray[np.float64] | None = None
        self._last_wind_v: NDArray[np.float64] | None = None

    def update_thermal(
        self,
        thermal_image: NDArray[np.float64],
        timestamp: float | None = None,
    ) -> None:
        """Update thermal sensor reading.

        Args:
            thermal_image: Temperature map (ny, nx) in K.
            timestamp: Reading timestamp.

        """
        self._last_thermal = thermal_image.copy()
        self.thermal_tracker.update(timestamp)

    def update_wind(
        self,
        wind_u: NDArray[np.float64],
        wind_v: NDArray[np.float64],
        timestamp: float | None = None,
    ) -> None:
        """Update wind sensor reading.

        Args:
            wind_u: x-component of wind (ny, nx) in m/s.
            wind_v: y-component of wind (ny, nx) in m/s.
            timestamp: Reading timestamp.

        """
        self._last_wind_u = wind_u.copy()
        self._last_wind_v = wind_v.copy()
        self.wind_tracker.update(timestamp)

    def get_boundary_conditions(
        self,
        current_time: float | None = None,
        default_temperature: float = 300.0,
        default_wind_u: float = 0.0,
        default_wind_v: float = 0.0,
    ) -> BoundaryConditions:
        """Get current boundary conditions with confidence weighting.

        Uses last-known-good values when sensors are stale, with
        confidence decaying toward defaults.

        Args:
            current_time: Current time for staleness computation.
            default_temperature: Fallback ambient temperature (K).
            default_wind_u: Fallback x-wind (m/s).
            default_wind_v: Fallback y-wind (m/s).

        Returns:
            BoundaryConditions with confidence scores.

        """
        ny, nx = self.grid_shape

        # Thermal
        t_conf = self.thermal_tracker.confidence(current_time)
        if self._last_thermal is not None:
            temp = t_conf * self._last_thermal + (1 - t_conf) * default_temperature
        else:
            temp = np.full((ny, nx), default_temperature, dtype=np.float64)
            t_conf = 0.0

        # Wind
        w_conf = self.wind_tracker.confidence(current_time)
        if self._last_wind_u is not None and self._last_wind_v is not None:
            wu = w_conf * self._last_wind_u + (1 - w_conf) * default_wind_u
            wv = w_conf * self._last_wind_v + (1 - w_conf) * default_wind_v
        else:
            wu = np.full((ny, nx), default_wind_u, dtype=np.float64)
            wv = np.full((ny, nx), default_wind_v, dtype=np.float64)
            w_conf = 0.0

        overall = min(t_conf, w_conf)

        return BoundaryConditions(
            temperature_field=temp,
            wind_u=wu,
            wind_v=wv,
            confidence=overall,
            thermal_confidence=t_conf,
            wind_confidence=w_conf,
        )
