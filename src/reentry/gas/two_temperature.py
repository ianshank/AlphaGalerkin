"""Two-temperature thermodynamic model for thermochemical nonequilibrium.

Separates internal energy into:
- Translational-rotational (T_tr): fully equilibrated at all temperatures
- Vibrational-electronic (T_ve): frozen or relaxing via Landau-Teller

The two temperatures equilibrate through energy exchange:
    dE_ve/dt = Q_TV = sum_s(rho_s * (e_v_eq(T_tr) - e_v(T_ve)) / tau_s)

where tau_s is the species-dependent vibrational relaxation time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.gas import UNIVERSAL_GAS_CONSTANT, GasConfig
from src.reentry.gas.species import SpeciesData, get_species_data


@dataclass
class TwoTemperatureState:
    """State for two-temperature model.

    Attributes:
        t_tr: Translational-rotational temperature (N,) in K.
        t_ve: Vibrational-electronic temperature (N,) in K.
        e_tr: Translational-rotational energy (N,) in J/kg.
        e_ve: Vibrational-electronic energy (N,) in J/kg.

    """

    t_tr: NDArray[np.float64]
    t_ve: NDArray[np.float64]
    e_tr: NDArray[np.float64]
    e_ve: NDArray[np.float64]


class TwoTemperatureModel:
    """Two-temperature model for vibrational nonequilibrium.

    In hypersonic flows, the vibrational temperature T_ve can
    differ significantly from T_tr behind shocks, where
    translational modes equilibrate quickly but vibrational
    modes require many collisions to relax.
    """

    def __init__(self, config: GasConfig) -> None:
        self.config = config
        self._species = get_species_data(config)

    def compute_temperatures(
        self,
        density: NDArray[np.float64],
        internal_energy: NDArray[np.float64],
        vibrational_energy: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> TwoTemperatureState:
        """Compute T_tr and T_ve from energy decomposition.

        Args:
            density: Mixture density (N,).
            internal_energy: Total specific internal energy (N,).
            vibrational_energy: Specific vibrational energy (N,).
            mass_fractions: Species mass fractions (N, n_species).

        Returns:
            TwoTemperatureState with both temperatures.

        """
        # Translational-rotational energy
        e_tr = internal_energy - vibrational_energy

        # T_tr from e_tr = cv_tr * T_tr
        cv_tr = self._mixture_cv_tr(mass_fractions)
        t_tr = e_tr / np.maximum(cv_tr, 1e-30)
        t_tr = np.clip(t_tr, 100.0, 60000.0)

        # T_ve from e_ve iteratively (invert e_vib(T_ve))
        t_ve = self._invert_vibrational_energy(vibrational_energy, mass_fractions)

        return TwoTemperatureState(t_tr=t_tr, t_ve=t_ve, e_tr=e_tr, e_ve=vibrational_energy)

    def energy_exchange_rate(
        self,
        density: NDArray[np.float64],
        t_tr: NDArray[np.float64],
        t_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute vibrational-translational energy exchange rate Q_TV.

        Q_TV = sum_s rho_s * (e_v_eq(T_tr) - e_v(T_ve)) / tau_s

        Positive Q_TV means energy flows from translational to vibrational.

        Args:
            density: Mixture density (N,).
            t_tr: Translational temperature (N,).
            t_ve: Vibrational temperature (N,).
            mass_fractions: Species mass fractions (N, n_species).

        Returns:
            Energy exchange rate (N,) in W/m^3.

        """
        n_pts = density.shape[0]
        q_tv = np.zeros(n_pts, dtype=np.float64)

        for i, sp_name in enumerate(self.config.species):
            sp = self._species[sp_name]
            if sp.theta_v <= 0 or sp.n_atoms < 2:
                continue

            rho_s = density * mass_fractions[:, i]
            e_v_eq = sp.e_vib(t_tr)
            e_v = sp.e_vib(t_ve)

            tau = self._relaxation_time(sp, t_tr, density)
            q_tv += rho_s * (e_v_eq - e_v) / np.maximum(tau, 1e-12)

        return q_tv

    def mixture_vibrational_energy(
        self,
        t_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute mixture vibrational energy at temperature T_ve."""
        e_ve = np.zeros_like(t_ve)
        for i, sp_name in enumerate(self.config.species):
            sp = self._species[sp_name]
            e_ve += mass_fractions[:, i] * sp.e_vib(t_ve)
        return e_ve

    def _mixture_cv_tr(self, mass_fractions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mixture translational-rotational cv."""
        cv = np.zeros(mass_fractions.shape[0], dtype=np.float64)
        for i, sp_name in enumerate(self.config.species):
            sp = self._species[sp_name]
            cv += mass_fractions[:, i] * sp.cv_trans_rot
        return cv

    def _invert_vibrational_energy(
        self,
        e_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Invert e_ve(T_ve) to find T_ve using Newton iteration."""
        t_ve = np.full_like(e_ve, 1000.0)  # Initial guess

        for _ in range(30):
            e_calc = self.mixture_vibrational_energy(t_ve, mass_fractions)
            # Derivative: de_ve/dT_ve
            cv_ve = np.zeros_like(t_ve)
            for i, sp_name in enumerate(self.config.species):
                sp = self._species[sp_name]
                cv_ve += mass_fractions[:, i] * sp.cv_vib(t_ve)

            cv_ve = np.maximum(cv_ve, 1e-30)
            dt = (e_ve - e_calc) / cv_ve
            t_ve = np.clip(t_ve + dt, 100.0, 60000.0)

            if np.max(np.abs(dt)) < 0.1:
                break

        return t_ve

    @staticmethod
    def _relaxation_time(
        sp: SpeciesData,
        temperature: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Millikan-White relaxation time with Park high-T correction.

        tau_MW = (1/p) * exp(A*(T^{-1/3} - B) - 18.42)
        tau_Park = 1 / (n * sigma * sqrt(8kT/(pi*m)))
        tau = tau_MW + tau_Park
        """
        mw_kg = sp.molecular_weight * 1000  # kg/mol -> g/mol for M-W formula
        a_mw = 1.16e-3 * np.sqrt(mw_kg) * sp.theta_v ** (4.0 / 3.0)
        b_mw = 0.015 * mw_kg**0.25

        # Approximate pressure
        # Approximate mixture R (use species-average molecular weight)
        r_mix = UNIVERSAL_GAS_CONSTANT / sp.molecular_weight
        p = density * r_mix * temperature
        p_atm = np.maximum(p / 101325.0, 1e-10)

        tau_mw = np.exp(a_mw * (temperature ** (-1.0 / 3.0) - b_mw) - 18.42) / p_atm

        # Park correction for high temperatures
        n_density = density / sp.molecular_weight * 6.022e23  # Number density
        sigma_park = 1e-21  # Cross-section in m^2 (order of magnitude)
        v_mean = np.sqrt(8.0 * 1.38e-23 * temperature / (np.pi * sp.molecular_weight / 6.022e23))
        tau_park = 1.0 / np.maximum(n_density * sigma_park * v_mean, 1e-30)

        return tau_mw + tau_park
