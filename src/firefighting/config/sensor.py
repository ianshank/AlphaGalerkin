"""Sensor data ingestion configuration.

Defines parameters for thermal camera, wind sensor, and GPS
data processing with staleness tracking and confidence decay.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from src.templates.config import BaseModuleConfig


class ThermalCameraType(str, Enum):
    """Supported thermal camera types."""

    FLIR = "flir"
    DJI = "dji"
    GENERIC = "generic"


class SensorConfig(BaseModuleConfig):
    """Configuration for sensor data processing."""

    # Thermal camera
    thermal_camera_type: ThermalCameraType = Field(
        default=ThermalCameraType.GENERIC,
        description="Thermal camera model for decoding.",
    )
    thermal_resolution_x: int = Field(
        default=640,
        ge=32,
        description="Thermal image width in pixels.",
    )
    thermal_resolution_y: int = Field(
        default=480,
        ge=32,
        description="Thermal image height in pixels.",
    )
    thermal_fov_deg: float = Field(
        default=45.0,
        gt=0.0,
        lt=180.0,
        description="Thermal camera field of view in degrees.",
    )

    # Staleness tracking
    stale_threshold_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Seconds after which sensor data is considered stale.",
    )
    confidence_decay_rate: float = Field(
        default=0.1,
        gt=0.0,
        description="Exponential decay rate for confidence (per second past stale).",
    )
    min_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence floor (never drops below this).",
    )

    # Wind sensor
    wind_update_rate_hz: float = Field(
        default=10.0,
        gt=0.0,
        description="Expected wind sensor update rate in Hz.",
    )

    # GPS
    gps_accuracy_m: float = Field(
        default=2.0,
        gt=0.0,
        description="Expected GPS horizontal accuracy in meters.",
    )
