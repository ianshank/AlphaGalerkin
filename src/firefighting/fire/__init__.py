"""Fire physics module for wildfire spread prediction.

Provides fuel models, ignition/pyrolysis, radiative and convective
heat transfer, terrain effects, wind coupling, and level-set
perimeter tracking.
"""

from src.firefighting.fire.convection import ConvectiveHeatTransfer
from src.firefighting.fire.fuel import FuelModel, FuelState
from src.firefighting.fire.ignition import IgnitionModel
from src.firefighting.fire.perimeter import LevelSetPerimeter
from src.firefighting.fire.radiation import RadiativeHeatTransfer
from src.firefighting.fire.terrain import TerrainEffects
from src.firefighting.fire.wind import WindField

__all__ = [
    "ConvectiveHeatTransfer",
    "FuelModel",
    "FuelState",
    "IgnitionModel",
    "LevelSetPerimeter",
    "RadiativeHeatTransfer",
    "TerrainEffects",
    "WindField",
]
