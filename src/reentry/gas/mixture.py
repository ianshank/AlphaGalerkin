"""Gas mixture property computations.

Provides mixture-level thermodynamic and transport properties
combining individual species contributions using established
mixing rules (Wilke, Armaly-Sutton).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.gas import GasConfig
from src.reentry.gas.species import get_species_data


class GasMixture:
    """Thermodynamic mixture properties for multi-species gas.

    Combines species-level data to compute mixture-averaged
    properties needed by the flow solver.
    """

    def __init__(self, config: GasConfig) -> None:
        self.config = config
        self._species_data = get_species_data(config)

    @property
    def species_names(self) -> list[str]:
        return self.config.species

    @property
    def n_species(self) -> int:
        return len(self.config.species)

    def mixture_gas_constant(self, mass_fractions: NDArray[np.float64]) -> NDArray[np.float64]:
        """R_mix = sum(Yi * Ri)."""
        r_mix = np.zeros(mass_fractions.shape[0], dtype=np.float64)
        for i, sp_name in enumerate(self.config.species):
            r_mix += mass_fractions[:, i] * self._species_data[sp_name].gas_constant
        return r_mix

    def mixture_cv(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mixture cv = sum(Yi * cv_s(T)) including vibrational modes."""
        cv = np.zeros_like(temperature)
        for i, sp_name in enumerate(self.config.species):
            sp = self._species_data[sp_name]
            cv_s = sp.cv_trans_rot + sp.cv_vib(temperature)
            cv += mass_fractions[:, i] * cv_s
        return cv

    def mixture_cp(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mixture cp = cv + R_mix."""
        cv = self.mixture_cv(temperature, mass_fractions)
        r_mix = self.mixture_gas_constant(mass_fractions)
        return cv + r_mix

    def mixture_gamma(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Effective ratio of specific heats gamma = cp / cv."""
        cv = self.mixture_cv(temperature, mass_fractions)
        r_mix = self.mixture_gas_constant(mass_fractions)
        return (cv + r_mix) / np.maximum(cv, 1e-30)

    def mixture_enthalpy(
        self,
        temperature: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mixture specific enthalpy including formation enthalpies.

        h = sum(Yi * (cv_tr_s * T + e_vib_s(T) + h_f_s + R_s * T))
        """
        h = np.zeros_like(temperature)
        for i, sp_name in enumerate(self.config.species):
            sp = self._species_data[sp_name]
            h_s = (
                sp.cv_trans_rot * temperature
                + sp.e_vib(temperature)
                + sp.formation_enthalpy
                + sp.gas_constant * temperature
            )
            h += mass_fractions[:, i] * h_s
        return h

    def mass_to_mole_fractions(self, mass_fractions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert mass fractions Yi to mole fractions Xi."""
        inv_m = np.zeros_like(mass_fractions)
        for i, sp in enumerate(self.config.species):
            inv_m[:, i] = mass_fractions[:, i] / self._species_data[sp].molecular_weight
        total = np.sum(inv_m, axis=1, keepdims=True)
        return inv_m / np.maximum(total, 1e-30)

    def validate_mass_fractions(
        self, mass_fractions: NDArray[np.float64], rtol: float = 1e-6
    ) -> bool:
        """Check that mass fractions are valid (non-negative, sum to 1)."""
        if np.any(mass_fractions < -rtol):
            return False
        sums = np.sum(mass_fractions, axis=1)
        return bool(np.all(np.abs(sums - 1.0) < rtol))
