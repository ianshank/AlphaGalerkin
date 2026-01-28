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

from src.pde.config import (
    BasisSelectionConfig,
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
)
from src.pde.game import PDEGame, PDEResult, PDEState
from src.pde.operators import (
    AdvectionDiffusionOperator,
    BurgersOperator,
    PDEOperator,
    PoissonOperator,
)
from src.pde.registry import PDEOperatorRegistry, register_pde_operator

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
    # Operators
    "PDEOperator",
    "PoissonOperator",
    "BurgersOperator",
    "AdvectionDiffusionOperator",
    # Registry
    "PDEOperatorRegistry",
    "register_pde_operator",
]
