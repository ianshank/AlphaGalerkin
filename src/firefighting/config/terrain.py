"""Terrain configuration for topographic effects on fire spread.

Defines DEM source, slope model, and aspect correction parameters.
"""

from __future__ import annotations

from pydantic import Field

from src.templates.config import BaseModuleConfig


class TerrainConfig(BaseModuleConfig):
    """Configuration for terrain effects."""

    enable_slope_effects: bool = Field(
        default=True,
        description="Enable terrain slope effects on spread rate.",
    )
    slope_factor: float = Field(
        default=5.275,
        gt=0.0,
        description="Slope correction factor (Rothermel phi_s coefficient).",
    )
    max_slope_deg: float = Field(
        default=80.0,
        gt=0.0,
        le=90.0,
        description="Maximum slope angle in degrees (clamp above this).",
    )
    dem_resolution_m: float = Field(
        default=30.0,
        gt=0.0,
        description="DEM resolution in meters.",
    )
    flat_terrain: bool = Field(
        default=False,
        description="If True, ignore terrain and assume flat ground.",
    )
