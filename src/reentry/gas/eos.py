"""Equation of state implementations for gas mixtures.

Provides three levels of thermodynamic modeling:
1. CaloricallyPerfectEOS: constant gamma, single temperature
2. ThermallyPerfectEOS: temperature-dependent cp/cv, single temperature
3. TwoTemperatureEOS: separate T_tr and T_ve (Phase 3)

All implementations satisfy the EquationOfState Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.gas import GasConfig
from src.reentry.gas.species import get_species_data


@runtime_checkable
class EquationOfState(Protocol):
    """Thermodynamic equation of state interface."""

    def pressure(
        self,
        density: NDArray[np.float64],
        internal_energy: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute pressure from density and internal energy."""
        ...

    def temperature(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute temperature from density and pressure."""
        ...

    def sound_speed(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute sound speed."""
        ...

    def internal_energy(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute specific internal energy from density and pressure."""
        ...


class CaloricallyPerfectEOS:
    """Calorically perfect gas EOS with constant gamma.

    p = rho * R_mix * T
    e = p / (rho * (gamma - 1))
    a = sqrt(gamma * p / rho)

    Suitable for low-temperature flows (T < 2000 K) where
    vibrational modes are not excited.
    """

    def __init__(self, config: GasConfig) -> None:
        self.config = config
        self.gamma = config.gamma
        self._species_data = get_species_data(config)

    def _mixture_gas_constant(self, mass_fractions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute mixture-averaged gas constant R_mix = sum(Yi * Ri)."""
        r_mix = np.zeros(mass_fractions.shape[0], dtype=np.float64)
        for i, sp_name in enumerate(self.config.species):
            r_mix += mass_fractions[:, i] * self._species_data[sp_name].gas_constant
        return r_mix

    def pressure(
        self,
        density: NDArray[np.float64],
        internal_energy: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """P = rho * e * (gamma - 1)."""
        return density * internal_energy * (self.gamma - 1.0)

    def temperature(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """T = p / (rho * R_mix)."""
        r_mix = self._mixture_gas_constant(mass_fractions)
        return pressure / (density * r_mix)

    def sound_speed(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """A = sqrt(gamma * p / rho)."""
        return np.sqrt(self.gamma * pressure / density)

    def internal_energy(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """E = p / (rho * (gamma - 1))."""
        return pressure / (density * (self.gamma - 1.0))


class ThermallyPerfectEOS:
    """Thermally perfect gas EOS with temperature-dependent specific heats.

    Uses the harmonic oscillator model for vibrational contributions.
    Suitable for moderate temperatures (2000 K < T < 8000 K) before
    significant dissociation occurs.
    """

    def __init__(self, config: GasConfig) -> None:
        self.config = config
        self._species_data = get_species_data(config)

    def _mixture_gas_constant(self, mass_fractions: NDArray[np.float64]) -> NDArray[np.float64]:
        r_mix = np.zeros(mass_fractions.shape[0], dtype=np.float64)
        for i, sp_name in enumerate(self.config.species):
            r_mix += mass_fractions[:, i] * self._species_data[sp_name].gas_constant
        return r_mix

    def _mixture_cv(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mixture cv = sum(Yi * cv_s(T))."""
        cv = np.zeros_like(temperature)
        for i, sp_name in enumerate(self.config.species):
            sp = self._species_data[sp_name]
            cv_s = sp.cv_trans_rot + sp.cv_vib(temperature)
            cv += mass_fractions[:, i] * cv_s
        return cv

    def _mixture_gamma(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Effective gamma = cp/cv = (cv + R_mix) / cv."""
        cv = self._mixture_cv(temperature, mass_fractions)
        r_mix = self._mixture_gas_constant(mass_fractions)
        return (cv + r_mix) / cv

    def pressure(
        self,
        density: NDArray[np.float64],
        internal_energy: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute pressure iteratively: p = rho * R_mix * T."""
        r_mix = self._mixture_gas_constant(mass_fractions)
        # First estimate with constant gamma
        t_guess = internal_energy / self._mixture_cv(np.full_like(density, 1000.0), mass_fractions)
        t_guess = np.clip(t_guess, 100.0, 50000.0)
        # Newton iteration to find T from e = integral(cv dT)
        for _ in range(20):
            cv = self._mixture_cv(t_guess, mass_fractions)
            e_guess = cv * t_guess  # Approximate: e ≈ cv * T
            de_dt = cv
            dt = (internal_energy - e_guess) / np.maximum(de_dt, 1e-30)
            t_guess = np.clip(t_guess + dt, 100.0, 50000.0)
            if np.max(np.abs(dt)) < 0.1:
                break
        return density * r_mix * t_guess

    def temperature(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """T = p / (rho * R_mix)."""
        r_mix = self._mixture_gas_constant(mass_fractions)
        return pressure / (density * r_mix)

    def sound_speed(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """A = sqrt(gamma(T) * p / rho)."""
        temp = self.temperature(density, pressure, mass_fractions)
        gamma = self._mixture_gamma(temp, mass_fractions)
        return np.sqrt(gamma * pressure / density)

    def internal_energy(
        self,
        density: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """E = cv(T) * T (approximate for thermally perfect gas)."""
        temp = self.temperature(density, pressure, mass_fractions)
        cv = self._mixture_cv(temp, mass_fractions)
        return cv * temp
