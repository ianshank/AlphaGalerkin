"""FIRE II trajectory point configuration.

Defines freestream conditions at specific times along
the FIRE II reentry trajectory (NASA TN D-3646).
"""

from __future__ import annotations

from pydantic import Field

from src.templates.config import BaseModuleConfig


class TrajectoryPoint(BaseModuleConfig):
    """A single trajectory point with freestream conditions."""

    time_s: float = Field(
        ...,
        description="Time along trajectory in seconds.",
    )
    altitude_km: float = Field(
        ...,
        gt=0.0,
        description="Altitude in km.",
    )
    velocity_m_s: float = Field(
        ...,
        gt=0.0,
        description="Vehicle velocity in m/s.",
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
    mach: float = Field(
        ...,
        gt=0.0,
        description="Freestream Mach number.",
    )
    expected_heat_flux_W_m2: float | None = Field(  # noqa: N815
        default=None,
        ge=0.0,
        description="Expected stagnation heat flux from flight data (W/m^2).",
    )


class TrajectoryConfig(BaseModuleConfig):
    """FIRE II trajectory configuration with multiple trajectory points."""

    vehicle_name: str = Field(
        default="FIRE_II",
        description="Vehicle identifier.",
    )
    nose_radius_m: float = Field(
        default=0.9347,
        gt=0.0,
        description="Vehicle nose radius in meters.",
    )
    cone_half_angle_deg: float = Field(
        default=33.0,
        gt=0.0,
        lt=90.0,
        description="Cone half-angle in degrees.",
    )
    points: list[TrajectoryPoint] = Field(
        default_factory=list,
        description="Trajectory points along the flight path.",
    )
