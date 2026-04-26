"""Drone sensor data ingestion and processing.

Handles thermal camera, wind sensor, and GPS data with
staleness tracking and confidence decay for graceful degradation.
"""

from src.firefighting.sensor.boundary_encoder import BoundaryEncoder
from src.firefighting.sensor.staleness import StalenessTracker
from src.firefighting.sensor.thermal import ThermalCameraDecoder, ThermalFrame
from src.firefighting.sensor.wind import WindFieldInterpolator, WindObservation

__all__ = [
    "BoundaryEncoder",
    "StalenessTracker",
    "ThermalCameraDecoder",
    "ThermalFrame",
    "WindFieldInterpolator",
    "WindObservation",
]
