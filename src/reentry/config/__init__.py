"""Pydantic configuration schemas for reentry aerodynamics.

All gas constants, reaction rates, freestream conditions, mesh parameters,
and solver settings are configurable — zero hardcoded values.
"""

from src.reentry.config.chemistry import ChemistryConfig, ChemistryMechanism
from src.reentry.config.freestream import FreestreamConfig
from src.reentry.config.gas import GasConfig, SpeciesName
from src.reentry.config.mesh import ReentryMeshConfig
from src.reentry.config.solver import FluxScheme, LimiterType, ReentrySolverConfig
from src.reentry.config.trajectory import TrajectoryConfig, TrajectoryPoint
from src.reentry.config.wall import CatalyticModel, WallConfig

__all__ = [
    "CatalyticModel",
    "ChemistryConfig",
    "ChemistryMechanism",
    "FluxScheme",
    "FreestreamConfig",
    "GasConfig",
    "LimiterType",
    "ReentryMeshConfig",
    "ReentrySolverConfig",
    "SpeciesName",
    "TrajectoryConfig",
    "TrajectoryPoint",
    "WallConfig",
]
