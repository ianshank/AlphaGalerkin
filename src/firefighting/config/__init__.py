"""Pydantic configuration schemas for firefighting prediction.

All fuel models, sensor parameters, edge deployment constraints,
and solver settings are configurable — zero hardcoded values.
"""

from src.firefighting.config.edge import EdgeConfig
from src.firefighting.config.fire import FireConfig, FuelCategory
from src.firefighting.config.sensor import SensorConfig
from src.firefighting.config.solver import FireSolverConfig
from src.firefighting.config.terrain import TerrainConfig
from src.firefighting.config.wind import WindConfig

__all__ = [
    "EdgeConfig",
    "FireConfig",
    "FireSolverConfig",
    "FuelCategory",
    "SensorConfig",
    "TerrainConfig",
    "WindConfig",
]
