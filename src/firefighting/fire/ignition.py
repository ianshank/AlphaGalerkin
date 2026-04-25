"""Ignition and pyrolysis model for wildfire spread.

Determines when fuel reaches ignition temperature and
begins pyrolysis based on thermal exposure history.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.fire import FireConfig


class IgnitionModel:
    """Temperature-threshold ignition with thermal inertia.

    Tracks cumulative heat exposure to model delayed ignition
    for thick fuel beds where surface heating must penetrate
    to the fuel interior.
    """

    def __init__(self, config: FireConfig) -> None:
        self.config = config
        self._heat_accumulator: NDArray[np.float64] | None = None

    def initialize(self, shape: tuple[int, int]) -> None:
        """Initialize heat accumulator for the grid.

        Args:
            shape: Grid shape (ny, nx).

        """
        self._heat_accumulator = np.zeros(shape, dtype=np.float64)

    def update(
        self,
        temperature: NDArray[np.float64],
        fuel_available: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.bool_]:
        """Update ignition state and return newly ignited cells.

        Args:
            temperature: Current temperature field (K).
            fuel_available: Available fuel fraction [0, 1].
            dt: Timestep in seconds.

        Returns:
            Boolean mask of newly ignited cells this step.

        """
        if self._heat_accumulator is None:
            self.initialize(temperature.shape)
            assert self._heat_accumulator is not None

        t_ign = self.config.ignition_temperature_K
        t_amb = self.config.ambient_temperature_K

        # Accumulate heat above ambient
        excess = np.maximum(temperature - t_amb, 0.0)
        self._heat_accumulator += excess * dt

        # Ignition threshold: temperature above ignition AND fuel available
        threshold = (t_ign - t_amb) * 10.0  # ~10s of exposure at ignition temp
        newly_ignited = (
            (self._heat_accumulator >= threshold)
            & (temperature >= t_ign * 0.9)  # Within 90% of ignition temp
            & (fuel_available > 1e-6)
        )
        return newly_ignited

    def reset(self) -> None:
        """Reset the heat accumulator."""
        self._heat_accumulator = None
