"""Gas mixture configuration for reentry aerodynamics.

Defines species properties for high-temperature air:
- Molecular weights, formation enthalpies
- 5-species neutral (N2, O2, NO, N, O)
- 7-species with ions (+ N+, O+)
- 11-species with electrons (+ N+, O+, NO+, N2+, e-)
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig

# Universal constants (SI)
UNIVERSAL_GAS_CONSTANT = 8.314462  # J/(mol·K)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
AVOGADRO_NUMBER = 6.022140e23  # 1/mol


class SpeciesName(str, Enum):
    """Supported chemical species."""

    N2 = "N2"
    O2 = "O2"
    NO = "NO"
    N = "N"
    O = "O"
    N_PLUS = "N+"
    O_PLUS = "O+"
    NO_PLUS = "NO+"
    N2_PLUS = "N2+"
    ELECTRON = "e-"


# Default molecular weights in kg/mol
DEFAULT_MOLECULAR_WEIGHTS: dict[str, float] = {
    "N2": 0.0280134,
    "O2": 0.0319988,
    "NO": 0.0300061,
    "N": 0.0140067,
    "O": 0.0159994,
    "N+": 0.0140062,
    "O+": 0.0159989,
    "NO+": 0.0300056,
    "N2+": 0.0280129,
    "e-": 5.4858e-7,
}

# Default formation enthalpies in J/kg
DEFAULT_FORMATION_ENTHALPIES: dict[str, float] = {
    "N2": 0.0,
    "O2": 0.0,
    "NO": 2.996120e6,
    "N": 3.362161e7,
    "O": 1.542000e7,
    "N+": 1.340700e8,
    "O+": 9.756200e7,
    "NO+": 3.283800e7,
    "N2+": 5.370700e7,
    "e-": 0.0,
}

# Default characteristic vibrational temperatures in K
DEFAULT_THETA_V: dict[str, float] = {
    "N2": 3395.0,
    "O2": 2239.0,
    "NO": 2817.0,
    "N": 0.0,  # Atomic — no vibration
    "O": 0.0,
    "N+": 0.0,
    "O+": 0.0,
    "NO+": 3421.0,
    "N2+": 3175.0,
    "e-": 0.0,
}


class GasConfig(BaseModuleConfig):
    """Configuration for gas mixture properties.

    All species data is configurable; defaults are standard air values
    from Park (1993) and Gupta (1990).
    """

    species: list[str] = Field(
        default_factory=lambda: ["N2", "O2", "NO", "N", "O"],
        description="List of species names in the mixture.",
    )
    molecular_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "N2": 0.0280134,
            "O2": 0.0319988,
            "NO": 0.0300061,
            "N": 0.0140067,
            "O": 0.0159994,
        },
        description="Molecular weights in kg/mol for each species.",
    )
    formation_enthalpies: dict[str, float] = Field(
        default_factory=lambda: {
            "N2": 0.0,
            "O2": 0.0,
            "NO": 2.996120e6,
            "N": 3.362161e7,
            "O": 1.542000e7,
        },
        description="Formation enthalpies in J/kg for each species.",
    )
    theta_v: dict[str, float] = Field(
        default_factory=lambda: {
            "N2": 3395.0,
            "O2": 2239.0,
            "NO": 2817.0,
            "N": 0.0,
            "O": 0.0,
        },
        description="Characteristic vibrational temperatures in K.",
    )
    gamma: float = Field(
        default=1.4,
        gt=1.0,
        le=2.0,
        description="Ratio of specific heats (calorically perfect gas).",
    )
    universal_gas_constant: float = Field(
        default=UNIVERSAL_GAS_CONSTANT,
        gt=0.0,
        description="Universal gas constant in J/(mol·K).",
    )

    @model_validator(mode="after")
    def _validate_species_data(self) -> GasConfig:
        """Ensure all species have corresponding property entries."""
        for sp in self.species:
            if sp not in self.molecular_weights:
                msg = f"Species '{sp}' missing from molecular_weights"
                raise ValueError(msg)
            if sp not in self.formation_enthalpies:
                msg = f"Species '{sp}' missing from formation_enthalpies"
                raise ValueError(msg)
        return self

    @property
    def n_species(self) -> int:
        """Number of species in the mixture."""
        return len(self.species)

    def gas_constant(self, species: str) -> float:
        """Specific gas constant R_s = R_u / M_s in J/(kg·K)."""
        return self.universal_gas_constant / self.molecular_weights[species]

    def mixture_molecular_weight(self, mass_fractions: dict[str, float]) -> float:
        """Compute mixture molecular weight from mass fractions."""
        inv_m = sum(mass_fractions.get(sp, 0.0) / self.molecular_weights[sp] for sp in self.species)
        if inv_m <= 0.0:
            msg = "Invalid mass fractions: sum(Yi/Mi) <= 0"
            raise ValueError(msg)
        return 1.0 / inv_m
