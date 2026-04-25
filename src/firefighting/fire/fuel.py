"""Fuel models for wildfire simulation.

Provides fuel loading, moisture, and consumption rate computations
based on LANDFIRE fuel categories or custom parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.fire import FireConfig


@runtime_checkable
class FirePhysics(Protocol):
    """Pluggable fire physics model interface."""

    def heat_source(
        self,
        temperature: NDArray[np.float64],
        fuel: NDArray[np.float64],
        wind: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute heat release rate per unit area (W/m^2)."""
        ...

    def fuel_consumption_rate(
        self,
        temperature: NDArray[np.float64],
        fuel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute fuel consumption rate (kg/m^2/s)."""
        ...

    def ignition_mask(
        self,
        temperature: NDArray[np.float64],
        fuel: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return boolean mask of cells at or above ignition temperature."""
        ...


@dataclass
class FuelState:
    """Current fuel field state on the computational grid.

    Attributes:
        loading: Fuel loading at each cell (kg/m^2), shape (ny, nx).
        moisture: Fuel moisture fraction at each cell, shape (ny, nx).
        consumed: Fraction of fuel consumed [0, 1], shape (ny, nx).

    """

    loading: NDArray[np.float64]
    moisture: NDArray[np.float64]
    consumed: NDArray[np.float64]

    @property
    def available(self) -> NDArray[np.float64]:
        """Remaining fuel fraction."""
        return np.clip(1.0 - self.consumed, 0.0, 1.0)

    @property
    def effective_loading(self) -> NDArray[np.float64]:
        """Remaining fuel loading (kg/m^2)."""
        return self.loading * self.available


class FuelModel:
    """Fuel consumption and heat release model.

    Implements the FirePhysics protocol for a given fuel configuration.
    Uses a simple Arrhenius-style consumption rate combined with
    heat of combustion for heat release.
    """

    def __init__(self, config: FireConfig) -> None:
        self.config = config

    def heat_source(
        self,
        temperature: NDArray[np.float64],
        fuel: NDArray[np.float64],
        wind: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Heat release = consumption_rate * heat_of_combustion.

        Args:
            temperature: Temperature field (ny, nx) in K.
            fuel: Available fuel loading (ny, nx) in kg/m^2.
            wind: Wind speed field (ny, nx) in m/s.

        Returns:
            Heat source (ny, nx) in W/m^2.

        """
        consumption = self.fuel_consumption_rate(temperature, fuel)
        # Wind enhancement: spread rate increases with wind
        wind_factor = 1.0 + 0.5 * np.clip(wind, 0.0, self.config.surface_area_to_volume_1_m)
        return consumption * self.config.heat_of_combustion_J_kg * wind_factor

    def fuel_consumption_rate(
        self,
        temperature: NDArray[np.float64],
        fuel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fuel consumption rate using threshold + Arrhenius model.

        Below ignition temperature: zero consumption.
        Above ignition: rate proportional to exp(-Ea/RT) * fuel_available.

        Args:
            temperature: Temperature field in K.
            fuel: Available fuel loading in kg/m^2.

        Returns:
            Consumption rate in kg/(m^2·s).

        """
        t_ign = self.config.ignition_temperature_K
        # Simple threshold model with smooth transition
        excess = np.maximum(temperature - t_ign, 0.0)
        # Rate coefficient: characteristic time ~10s for full consumption
        rate_coeff = 0.1 * excess / np.maximum(temperature, 1.0)
        # Moisture suppression
        moisture_factor = np.maximum(1.0 - self.config.fuel_moisture_fraction * 2.5, 0.0)
        return rate_coeff * fuel * moisture_factor

    def ignition_mask(
        self,
        temperature: NDArray[np.float64],
        fuel: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Cells above ignition temperature with available fuel."""
        return (temperature >= self.config.ignition_temperature_K) & (fuel > 1e-6)
