"""Physics modules for PDE families.

This package provides PDE physics modules that specify weak forms,
boundary conditions, manufactured solutions, and reward functions
for the AlphaGalerkin discretization environment.

Available modules (via the physics registry):
- ``poisson_2d``: Elliptic Poisson equation.
- ``heat_2d``: Parabolic heat equation.
- ``burgers_1d``: Nonlinear Burgers equation.
- ``wave_1d``: Hyperbolic wave equation.
- ``advdiff_2d``: Advection-diffusion equation.
- ``navier_stokes_2d``: Incompressible Navier-Stokes with SGS closures.
"""

from __future__ import annotations

# Import modules to trigger registration via decorators.
import src.alphagalerkin.physics.advection_diffusion as advection_diffusion  # noqa: F401
import src.alphagalerkin.physics.burgers as burgers  # noqa: F401
import src.alphagalerkin.physics.heat as heat  # noqa: F401
import src.alphagalerkin.physics.navier_stokes as navier_stokes  # noqa: F401
import src.alphagalerkin.physics.poisson as poisson  # noqa: F401
import src.alphagalerkin.physics.wave as wave  # noqa: F401
from src.alphagalerkin.physics.base import (
    BoundaryCondition,
    ManufacturedSolution,
    PhysicsModuleBase,
    SolveResult,
)
from src.alphagalerkin.physics.manufactured import (
    MMS_CATALOG,
    poisson_polynomial,
    poisson_sinsin,
)
from src.alphagalerkin.physics.registry import (
    PhysicsRegistry,
    register_physics,
)

__all__ = [
    # Base types
    "BoundaryCondition",
    "ManufacturedSolution",
    "PhysicsModuleBase",
    "SolveResult",
    # Manufactured solution catalog
    "MMS_CATALOG",
    "poisson_polynomial",
    "poisson_sinsin",
    # Registry
    "PhysicsRegistry",
    "register_physics",
]
