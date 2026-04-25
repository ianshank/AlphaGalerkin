"""Wind field configuration for fire-atmosphere coupling.

Defines wind model parameters including fire-modified winds
and mid-flame wind speed corrections.
"""

from __future__ import annotations

from pydantic import Field

from src.templates.config import BaseModuleConfig


class WindConfig(BaseModuleConfig):
    """Configuration for wind field modeling."""

    default_speed_m_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Default wind speed in m/s (used if no sensor data).",
    )
    default_direction_deg: float = Field(
        default=270.0,
        ge=0.0,
        lt=360.0,
        description="Default wind direction in degrees from north (clockwise).",
    )
    mid_flame_adjustment: float = Field(
        default=0.4,
        gt=0.0,
        le=1.0,
        description="Wind reduction factor for mid-flame height.",
    )
    enable_fire_induced_wind: bool = Field(
        default=False,
        description="Enable fire-induced wind modification (buoyancy-driven).",
    )
    fire_wind_coupling_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Coupling strength between fire and wind field.",
    )
    max_wind_speed_m_s: float = Field(
        default=50.0,
        gt=0.0,
        description="Maximum allowed wind speed (clamp above this).",
    )
