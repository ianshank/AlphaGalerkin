"""PDE Game implementations for AlphaGalerkin.

This module provides concrete implementations of PDEGame:
- BasisSelectionGame: Add basis functions to improve approximation
- MeshRefinementGame: Refine mesh elements to reduce error
- SwarmPlanningGame: MCTS-guided multi-agent swarm coordination
- CollocationGame: Place collocation points strategically
"""

from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.games.lshape_amr import LShapeAMRGame
from src.pde.games.mesh_refinement import MeshRefinementGame
from src.pde.games.swarm_planning import SwarmPlanningGame

__all__ = [
    "BasisSelectionGame",
    "LShapeAMRGame",
    "MeshRefinementGame",
    "SwarmPlanningGame",
]
