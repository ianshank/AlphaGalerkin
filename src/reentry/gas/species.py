"""Species thermodynamic data for high-temperature air.

Provides molecular weights, formation enthalpies, characteristic
temperatures, and NASA curve-fit coefficients for computing
temperature-dependent specific heats and enthalpies.

All data sourced from Park (1993) and Gupta (1990).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.gas import (
    DEFAULT_FORMATION_ENTHALPIES,
    DEFAULT_MOLECULAR_WEIGHTS,
    DEFAULT_THETA_V,
    UNIVERSAL_GAS_CONSTANT,
    GasConfig,
)


@dataclass(frozen=True)
class SpeciesData:
    """Thermodynamic data for a single chemical species."""

    name: str
    molecular_weight: float  # kg/mol
    formation_enthalpy: float  # J/kg
    theta_v: float  # Characteristic vibrational temperature (K)
    n_atoms: int  # Number of atoms (1=monatomic, 2=diatomic)

    @property
    def gas_constant(self) -> float:
        """Specific gas constant R_s = R_u / M_s in J/(kg·K)."""
        return UNIVERSAL_GAS_CONSTANT / self.molecular_weight

    @property
    def cv_trans(self) -> float:
        """Translational specific heat cv_tr = (3/2) R_s."""
        return 1.5 * self.gas_constant

    @property
    def cv_rot(self) -> float:
        """Rotational specific heat (0 for monatomic, R_s for diatomic)."""
        if self.n_atoms >= 2:
            return self.gas_constant
        return 0.0

    @property
    def cv_trans_rot(self) -> float:
        """Combined translational-rotational cv."""
        return self.cv_trans + self.cv_rot

    def cv_vib(self, temperature: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Vibrational specific heat from harmonic oscillator model.

        cv_vib = R_s * (theta_v/T)^2 * exp(theta_v/T) / (exp(theta_v/T) - 1)^2

        Args:
            temperature: Temperature in K.

        Returns:
            Vibrational specific heat in J/(kg·K).

        """
        if self.theta_v <= 0.0 or self.n_atoms < 2:
            if isinstance(temperature, np.ndarray):
                return np.zeros_like(temperature)
            return 0.0

        t = np.asarray(temperature, dtype=np.float64)
        # Clip to avoid overflow
        ratio = np.clip(self.theta_v / np.maximum(t, 1.0), 0.0, 100.0)
        exp_ratio = np.exp(ratio)
        denom = (exp_ratio - 1.0) ** 2
        # Avoid division by zero
        denom = np.maximum(denom, 1e-30)
        result = self.gas_constant * ratio**2 * exp_ratio / denom
        if isinstance(temperature, (int, float)):
            return float(result)
        return result

    def e_vib(self, temperature: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Vibrational energy per unit mass.

        e_vib = R_s * theta_v / (exp(theta_v/T) - 1)

        Args:
            temperature: Vibrational temperature in K.

        Returns:
            Vibrational energy in J/kg.

        """
        if self.theta_v <= 0.0 or self.n_atoms < 2:
            if isinstance(temperature, np.ndarray):
                return np.zeros_like(temperature)
            return 0.0

        t = np.asarray(temperature, dtype=np.float64)
        ratio = np.clip(self.theta_v / np.maximum(t, 1.0), 0.0, 100.0)
        result = self.gas_constant * self.theta_v / (np.exp(ratio) - 1.0)
        if isinstance(temperature, (int, float)):
            return float(result)
        return result


# Default species database
_DEFAULT_SPECIES: dict[str, SpeciesData] = {
    "N2": SpeciesData(
        "N2",
        DEFAULT_MOLECULAR_WEIGHTS["N2"],
        DEFAULT_FORMATION_ENTHALPIES["N2"],
        DEFAULT_THETA_V["N2"],
        2,
    ),
    "O2": SpeciesData(
        "O2",
        DEFAULT_MOLECULAR_WEIGHTS["O2"],
        DEFAULT_FORMATION_ENTHALPIES["O2"],
        DEFAULT_THETA_V["O2"],
        2,
    ),
    "NO": SpeciesData(
        "NO",
        DEFAULT_MOLECULAR_WEIGHTS["NO"],
        DEFAULT_FORMATION_ENTHALPIES["NO"],
        DEFAULT_THETA_V["NO"],
        2,
    ),
    "N": SpeciesData(
        "N",
        DEFAULT_MOLECULAR_WEIGHTS["N"],
        DEFAULT_FORMATION_ENTHALPIES["N"],
        DEFAULT_THETA_V["N"],
        1,
    ),
    "O": SpeciesData(
        "O",
        DEFAULT_MOLECULAR_WEIGHTS["O"],
        DEFAULT_FORMATION_ENTHALPIES["O"],
        DEFAULT_THETA_V["O"],
        1,
    ),
}


def get_species_data(config: GasConfig) -> dict[str, SpeciesData]:
    """Build species data dictionary from gas configuration.

    Args:
        config: Gas configuration with species list and properties.

    Returns:
        Dictionary mapping species names to SpeciesData.

    """
    species_data: dict[str, SpeciesData] = {}
    for name in config.species:
        n_atoms = 2 if len(name) > 1 and name[-1] != "+" and name != "e-" else 1
        # Override n_atoms for known species
        if name in ("N", "O", "N+", "O+", "e-"):
            n_atoms = 1
        species_data[name] = SpeciesData(
            name=name,
            molecular_weight=config.molecular_weights[name],
            formation_enthalpy=config.formation_enthalpies[name],
            theta_v=config.theta_v.get(name, 0.0),
            n_atoms=n_atoms,
        )
    return species_data
