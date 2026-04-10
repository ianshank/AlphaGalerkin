"""Flow solver components for compressible Navier-Stokes.

Provides 1D/2D Euler and Navier-Stokes solvers, state conversions,
boundary conditions, shock detection, and conservation auditing.
"""

from src.reentry.solver.boundary import (
    BoundaryFace,
    FreestreamBC,
    SupersonicOutflowBC,
    SymmetryBC,
    WallBC,
)
from src.reentry.solver.cfl import CFLController
from src.reentry.solver.euler_1d import Euler1DResult, Euler1DSolver, ShockTubeIC
from src.reentry.solver.euler_2d import Euler2DResult, Euler2DSolver
from src.reentry.solver.navier_stokes import NavierStokes2DSolver
from src.reentry.solver.residual import ResidualMonitor
from src.reentry.solver.shock_detector import ShockDetector
from src.reentry.solver.state import ConservativeState, PrimitiveState

__all__ = [
    "BoundaryFace",
    "FreestreamBC",
    "WallBC",
    "SymmetryBC",
    "SupersonicOutflowBC",
    "CFLController",
    "Euler1DSolver",
    "Euler1DResult",
    "ShockTubeIC",
    "Euler2DSolver",
    "Euler2DResult",
    "NavierStokes2DSolver",
    "ResidualMonitor",
    "ShockDetector",
    "ConservativeState",
    "PrimitiveState",
]
