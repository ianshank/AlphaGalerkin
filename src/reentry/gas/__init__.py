"""Gas thermodynamic and transport properties for high-temperature air.

Provides equation of state, specific heats, viscosity, conductivity,
and diffusion coefficients for multi-species gas mixtures relevant
to hypersonic reentry aerothermodynamics.
"""

from src.reentry.gas.eos import (
    CaloricallyPerfectEOS,
    EquationOfState,
    ThermallyPerfectEOS,
)
from src.reentry.gas.species import SpeciesData, get_species_data
from src.reentry.gas.transport import BlottnerTransport

__all__ = [
    "BlottnerTransport",
    "CaloricallyPerfectEOS",
    "EquationOfState",
    "SpeciesData",
    "ThermallyPerfectEOS",
    "get_species_data",
]
