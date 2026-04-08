"""PDE solving as sequential decision-making for AlphaGalerkin.

This module extends the AlphaZero framework to solve PDEs by treating:
- Basis function selection as actions
- Mesh refinement as strategic decisions
- Error reduction as game rewards

Key components:
- PDEGame: Abstract interface for PDE-based games
- PDEState: State representation for PDE solving
- PDEOperator: Abstract PDE definition interface
- PDERegistry: Registration and discovery of PDE operators
"""

import contextlib

from src.pde.config import (
    BasisSelectionConfig,
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
)
from src.pde.game import PDEGame, PDEResult, PDEState
from src.pde.geometry import (
    CylinderFlowDomain,
    DomainGeometry,
    GeometryConfig,
    GeometryType,
    LShapedDomain,
    RectangularDomain,
    create_geometry,
)
from src.pde.operators import (
    AdvectionDiffusionOperator,
    BurgersOperator,
    LShapedPoissonOperator,
    PDEOperator,
    PoissonOperator,
)
from src.pde.registry import PDEOperatorRegistry, register_pde_operator

# Import register_games to trigger PDE game registration in GameRegistry
# at import time, matching the pattern used by Go and Chess in src/games/__init__.py.
with contextlib.suppress(Exception):  # pragma: no cover — optional dependency chain
    from src.pde import register_games as _register_games  # noqa: F401

__all__ = [
    # Config
    "PDEConfig",
    "PDEGameConfig",
    "BasisSelectionConfig",
    "MeshRefinementConfig",
    # Core
    "PDEGame",
    "PDEState",
    "PDEResult",
    # Geometry
    "DomainGeometry",
    "RectangularDomain",
    "LShapedDomain",
    "CylinderFlowDomain",
    "GeometryConfig",
    "GeometryType",
    "create_geometry",
    # Operators
    "PDEOperator",
    "PoissonOperator",
    "BurgersOperator",
    "AdvectionDiffusionOperator",
    "LShapedPoissonOperator",
    # Registry
    "PDEOperatorRegistry",
    "register_pde_operator",
]
