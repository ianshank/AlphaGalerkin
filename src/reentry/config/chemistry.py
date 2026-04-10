"""Chemistry mechanism configuration for thermochemical nonequilibrium.

Supports finite-rate chemistry models:
- Park 1993 (standard 5-species air)
- Gupta 1990 (alternative rate constants)
- Custom (user-specified rate parameters)
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from src.templates.config import BaseModuleConfig


class ChemistryMechanism(str, Enum):
    """Supported chemical kinetics mechanisms."""

    PARK_1993 = "park_1993"
    GUPTA_1990 = "gupta_1990"
    CUSTOM = "custom"


class ChemistryConfig(BaseModuleConfig):
    """Configuration for finite-rate chemistry.

    Defines mechanism selection and solver parameters for
    chemical source term integration.
    """

    mechanism: ChemistryMechanism = Field(
        default=ChemistryMechanism.PARK_1993,
        description="Chemical kinetics mechanism to use.",
    )
    n_reactions: int = Field(
        default=17,
        ge=0,
        description="Number of reactions in the mechanism.",
    )
    # Operator splitting parameters
    chemistry_substeps: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of chemistry substeps per fluid timestep.",
    )
    stiff_ode_method: str = Field(
        default="bdf2",
        description="ODE integrator for stiff chemistry (bdf2, backward_euler).",
    )
    stiff_ode_rtol: float = Field(
        default=1e-6,
        gt=0.0,
        description="Relative tolerance for stiff ODE solver.",
    )
    stiff_ode_atol: float = Field(
        default=1e-10,
        gt=0.0,
        description="Absolute tolerance for stiff ODE solver.",
    )
    # Two-temperature model
    enable_two_temperature: bool = Field(
        default=True,
        description="Enable translational-vibrational two-temperature model.",
    )
    # Landau-Teller relaxation
    park_ttv_exponent: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Park T-Tv coupling exponent s in T_a = T^s * Tv^(1-s).",
    )
