"""Wall boundary condition configuration.

Defines surface thermal and catalytic models for
reentry vehicle heat shield surfaces.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from src.templates.config import BaseModuleConfig


class CatalyticModel(str, Enum):
    """Wall catalysis models."""

    NON_CATALYTIC = "non_catalytic"
    FULLY_CATALYTIC = "fully_catalytic"
    PARTIALLY_CATALYTIC = "partially_catalytic"


class WallThermalModel(str, Enum):
    """Wall thermal boundary condition models."""

    ISOTHERMAL = "isothermal"
    ADIABATIC = "adiabatic"
    RADIATIVE_EQUILIBRIUM = "radiative_equilibrium"


class WallConfig(BaseModuleConfig):
    """Configuration for wall boundary conditions."""

    thermal_model: WallThermalModel = Field(
        default=WallThermalModel.ISOTHERMAL,
        description="Wall thermal boundary condition model.",
    )
    wall_temperature_K: float = Field(  # noqa: N815
        default=300.0,
        gt=0.0,
        description="Wall temperature in K (isothermal only).",
    )
    emissivity: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="Surface emissivity for radiative equilibrium.",
    )
    catalytic_model: CatalyticModel = Field(
        default=CatalyticModel.NON_CATALYTIC,
        description="Wall catalysis model for species recombination.",
    )
    catalytic_efficiency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Catalytic efficiency gamma (partially catalytic only).",
    )
