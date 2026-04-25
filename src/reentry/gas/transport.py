"""Transport property models for high-temperature gas mixtures.

Viscosity: Blottner curve fits (Blottner et al. 1971)
  ln(mu_s) = (A_s * ln(T) + B_s) * ln(T) + C_s

Conductivity: Eucken relation
  k_s = mu_s * (cv_tr + (9/4) * R_s) for monatomic
  k_s = mu_s * (cv_tr + (5/4) * R_s + cv_vib) for diatomic

Mixture rules: Wilke (1950) semi-empirical mixing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.gas import GasConfig
from src.reentry.gas.species import get_species_data


@dataclass(frozen=True)
class BlottnerCoefficients:
    """Blottner viscosity curve-fit coefficients.

    mu_s = 0.1 * exp((A * ln(T) + B) * ln(T) + C)  [Pa·s]
    """

    A: float  # noqa: N815
    B: float  # noqa: N815
    C: float  # noqa: N815


# Default Blottner coefficients (Blottner et al. 1971)
DEFAULT_BLOTTNER: dict[str, BlottnerCoefficients] = {
    "N2": BlottnerCoefficients(0.0268142, 0.3177838, -11.3155513),
    "O2": BlottnerCoefficients(0.0449290, -0.0826158, -9.2019475),
    "NO": BlottnerCoefficients(0.0436378, -0.0335511, -9.5767430),
    "N": BlottnerCoefficients(0.0115572, 0.6031679, -12.4327495),
    "O": BlottnerCoefficients(0.0203144, 0.4294404, -11.6031403),
}


class BlottnerTransport:
    """Transport properties using Blottner viscosity + Eucken conductivity.

    Mixture-averaged properties use Wilke's mixing rule.
    """

    def __init__(
        self,
        config: GasConfig,
        blottner_coeffs: dict[str, BlottnerCoefficients] | None = None,
    ) -> None:
        self.config = config
        self._species_data = get_species_data(config)
        self._blottner = blottner_coeffs or DEFAULT_BLOTTNER

    def species_viscosity(
        self,
        species: str,
        temperature: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute species viscosity using Blottner curve fits.

        Args:
            species: Species name.
            temperature: Temperature array in K.

        Returns:
            Dynamic viscosity in Pa·s.

        """
        coeffs = self._blottner[species]
        ln_t = np.log(np.maximum(temperature, 1.0))
        return 0.1 * np.exp((coeffs.A * ln_t + coeffs.B) * ln_t + coeffs.C)

    def mixture_viscosity(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mixture viscosity using Wilke's mixing rule.

        mu_mix = sum_s (X_s * mu_s / sum_r (X_r * phi_sr))

        Args:
            temperature: Temperature array (N,).
            mass_fractions: Species mass fractions (N, n_species).

        Returns:
            Mixture viscosity in Pa·s (N,).

        """
        species = self.config.species
        n_sp = len(species)
        n_pts = temperature.shape[0]

        # Compute mole fractions from mass fractions
        mole_fractions = self._mass_to_mole_fractions(mass_fractions)

        # Compute species viscosities
        mu_s = np.zeros((n_pts, n_sp), dtype=np.float64)
        for i, sp in enumerate(species):
            mu_s[:, i] = self.species_viscosity(sp, temperature)

        # Wilke mixing rule
        mu_mix = np.zeros(n_pts, dtype=np.float64)
        for i in range(n_sp):
            phi_sum = np.zeros(n_pts, dtype=np.float64)
            mi = self._species_data[species[i]].molecular_weight
            for j in range(n_sp):
                mj = self._species_data[species[j]].molecular_weight
                mu_ratio = mu_s[:, i] / np.maximum(mu_s[:, j], 1e-30)
                m_ratio = mj / mi
                phi_ij = (1.0 + np.sqrt(mu_ratio * np.sqrt(m_ratio))) ** 2 / np.sqrt(
                    8.0 * (1.0 + mi / mj)
                )
                phi_sum += mole_fractions[:, j] * phi_ij
            mu_mix += mole_fractions[:, i] * mu_s[:, i] / np.maximum(phi_sum, 1e-30)

        return mu_mix

    def species_conductivity(
        self,
        species: str,
        temperature: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Eucken relation for thermal conductivity.

        Args:
            species: Species name.
            temperature: Temperature array in K.

        Returns:
            Thermal conductivity in W/(m·K).

        """
        sp = self._species_data[species]
        mu = self.species_viscosity(species, temperature)
        cv_tr = sp.cv_trans_rot
        r_s = sp.gas_constant

        if sp.n_atoms == 1:
            # Monatomic: k = mu * (5/2) * cv_tr
            return mu * 2.5 * cv_tr
        # Diatomic: Eucken with vibrational contribution
        cv_v = sp.cv_vib(temperature)
        return mu * (cv_tr + 1.25 * r_s + cv_v)

    def mixture_conductivity(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mixture thermal conductivity (mass-fraction weighted average).

        Args:
            temperature: Temperature array (N,).
            mass_fractions: Species mass fractions (N, n_species).

        Returns:
            Mixture conductivity in W/(m·K) (N,).

        """
        k_mix = np.zeros(temperature.shape[0], dtype=np.float64)
        for i, sp in enumerate(self.config.species):
            k_s = self.species_conductivity(sp, temperature)
            k_mix += mass_fractions[:, i] * k_s
        return k_mix

    def _mass_to_mole_fractions(
        self,
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert mass fractions to mole fractions.

        X_s = (Y_s / M_s) / sum(Y_r / M_r)
        """
        species = self.config.species
        inv_m = np.zeros_like(mass_fractions)
        for i, sp in enumerate(species):
            inv_m[:, i] = mass_fractions[:, i] / self._species_data[sp].molecular_weight

        total = np.sum(inv_m, axis=1, keepdims=True)
        total = np.maximum(total, 1e-30)
        return inv_m / total
