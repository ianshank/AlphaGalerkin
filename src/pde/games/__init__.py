"""PDE Game implementations for AlphaGalerkin.

This module provides concrete implementations of PDEGame:
- BasisSelectionGame: Add basis functions to improve approximation
- MeshRefinementGame: Refine mesh elements to reduce error
- SwarmPlanningGame: MCTS-guided multi-agent swarm coordination
- CollocationGame: Place collocation points strategically
"""

from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.games.mesh_refinement import MeshRefinementGame
from src.pde.games.swarm_planning import SwarmPlanningGame

# NOTE: LShapeAMRGame is intentionally NOT eagerly re-exported here. It imports
# the MCTS engine (src.mcts.evaluator), and pulling that into every
# ``import src.pde.games`` rippled the MCTS/torch import graph into unrelated
# per-module coverage gates (scaling_law / research_loop) that run under CI's C
# coverage tracer, tipping them into a torch SIGSEGV. Import it directly from
# the submodule: ``from src.pde.games.lshape_amr import LShapeAMRGame``.

__all__ = [
    "BasisSelectionGame",
    "MeshRefinementGame",
    "SwarmPlanningGame",
]
