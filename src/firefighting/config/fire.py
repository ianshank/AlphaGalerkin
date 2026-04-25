"""Fire physics configuration.

Defines fuel properties, ignition parameters, radiation/convection
coefficients, and spread rate model selection.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig


class FuelCategory(str, Enum):
    """LANDFIRE fuel model categories (Anderson 13 + Scott/Burgan 40)."""

    SHORT_GRASS = "short_grass"
    TIMBER_GRASS = "timber_grass"
    TALL_GRASS = "tall_grass"
    CHAPARRAL = "chaparral"
    BRUSH = "brush"
    DORMANT_BRUSH = "dormant_brush"
    SOUTHERN_ROUGH = "southern_rough"
    COMPACT_TIMBER_LITTER = "compact_timber_litter"
    HARDWOOD_LITTER = "hardwood_litter"
    TIMBER_UNDERSTORY = "timber_understory"
    LIGHT_LOGGING_SLASH = "light_logging_slash"
    MEDIUM_LOGGING_SLASH = "medium_logging_slash"
    HEAVY_LOGGING_SLASH = "heavy_logging_slash"
    CUSTOM = "custom"


class SpreadModel(str, Enum):
    """Fire spread rate models."""

    ROTHERMEL = "rothermel"
    LEVEL_SET = "level_set"
    ADVECTION_DIFFUSION = "advection_diffusion"


class FireConfig(BaseModuleConfig):
    """Configuration for fire physics parameters."""

    # Fuel properties
    fuel_category: FuelCategory = Field(
        default=FuelCategory.SHORT_GRASS,
        description="LANDFIRE fuel model category.",
    )
    fuel_load_kg_m2: float = Field(
        default=0.74,
        gt=0.0,
        description="Fuel loading in kg/m^2 (oven-dry weight).",
    )
    fuel_moisture_fraction: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Fuel moisture content as fraction of dry weight.",
    )
    fuel_depth_m: float = Field(
        default=0.3,
        gt=0.0,
        description="Fuel bed depth in meters.",
    )
    fuel_density_kg_m3: float = Field(
        default=512.0,
        gt=0.0,
        description="Fuel particle density in kg/m^3.",
    )
    surface_area_to_volume_1_m: float = Field(
        default=3281.0,
        gt=0.0,
        description="Fuel surface-area-to-volume ratio in 1/m.",
    )

    # Ignition parameters
    ignition_temperature_K: float = Field(  # noqa: N815
        default=573.15,
        gt=0.0,
        description="Ignition temperature in K (default 300C).",
    )
    heat_of_combustion_J_kg: float = Field(  # noqa: N815
        default=1.8e7,
        gt=0.0,
        description="Heat of combustion in J/kg.",
    )

    # Radiation parameters
    emissivity: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Flame emissivity.",
    )
    stefan_boltzmann: float = Field(
        default=5.670374e-8,
        gt=0.0,
        description="Stefan-Boltzmann constant in W/(m^2·K^4).",
    )
    radiation_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of heat release as radiation.",
    )

    # Spread model
    spread_model: SpreadModel = Field(
        default=SpreadModel.LEVEL_SET,
        description="Fire spread rate model.",
    )

    # Ambient conditions
    ambient_temperature_K: float = Field(  # noqa: N815
        default=300.0,
        gt=0.0,
        description="Ambient air temperature in K.",
    )
    air_heat_capacity_J_m3_K: float = Field(  # noqa: N815
        default=1200.0,
        gt=0.0,
        description="Volumetric heat capacity of air rho*cp in J/(m^3·K).",
    )
    max_temperature_K: float = Field(  # noqa: N815
        default=3000.0,
        gt=0.0,
        description="Maximum allowed temperature (clamp for stability).",
    )
    ignition_multiplier: float = Field(
        default=1.5,
        gt=1.0,
        description="Flame temperature = ignition_temperature * this factor.",
    )
    burn_threshold_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fuel consumption fraction to consider a cell burned.",
    )

    @model_validator(mode="after")
    def _validate_temperatures(self) -> FireConfig:
        if self.ignition_temperature_K <= self.ambient_temperature_K:
            msg = "ignition_temperature must be greater than ambient_temperature"
            raise ValueError(msg)
        return self
