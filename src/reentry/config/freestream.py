"""Freestream condition configuration for reentry simulations.

Defines the undisturbed flow state at a given altitude and velocity,
including temperature, density, pressure, and species composition.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig


class FreestreamConfig(BaseModuleConfig):
    """Freestream flow conditions.

    All values must be provided via config — no hardcoded defaults
    for mission-specific parameters.
    """

    mach: float = Field(
        ...,
        gt=0.0,
        description="Freestream Mach number.",
    )
    velocity_m_s: float = Field(
        ...,
        gt=0.0,
        description="Freestream velocity in m/s.",
    )
    density_kg_m3: float = Field(
        ...,
        gt=0.0,
        description="Freestream density in kg/m^3.",
    )
    temperature_K: float = Field(  # noqa: N815
        ...,
        gt=0.0,
        description="Freestream temperature in K.",
    )
    pressure_Pa: float | None = Field(  # noqa: N815
        default=None,
        ge=0.0,
        description="Freestream pressure in Pa. Computed from rho*R*T if not given.",
    )
    altitude_km: float = Field(
        default=0.0,
        ge=0.0,
        le=200.0,
        description="Altitude in km (for logging/provenance).",
    )
    angle_of_attack_deg: float = Field(
        default=0.0,
        ge=-90.0,
        le=90.0,
        description="Angle of attack in degrees.",
    )
    mass_fractions: dict[str, float] = Field(
        default_factory=lambda: {"N2": 0.767, "O2": 0.233},
        description="Freestream species mass fractions (must sum to 1.0).",
    )

    @model_validator(mode="after")
    def _validate_mass_fractions(self) -> FreestreamConfig:
        """Check mass fractions sum to 1.0 within tolerance."""
        total = sum(self.mass_fractions.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"Mass fractions must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self
