"""PDE Game implementations for AlphaGalerkin.

This module provides concrete implementations of PDEGame:
- BasisSelectionGame: Add basis functions to improve approximation
- MeshRefinementGame: Refine mesh elements to reduce error
- CollocationGame: Place collocation points strategically
"""

from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.games.mesh_refinement import MeshRefinementGame

__all__ = [
    "BasisSelectionGame",
    "MeshRefinementGame",
]
