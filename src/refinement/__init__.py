"""Domain-free sequential-refinement engine.

The ``RefinementGame`` abstraction and its MCTS adapter are the domain-agnostic
core that ``src.pde`` (PDE basis/mesh refinement) implements. It carries no
PDE-specific types, so the same single-agent MCTS engine can drive any refinement
domain that reframes "reduce an error estimate under a budget" as a game.
"""

from src.refinement.adapter import RefinementGameAdapter
from src.refinement.config import RefinementGameConfig
from src.refinement.game import RefinementGame
from src.refinement.registry import RefinementGameRegistry, register_refinement_game
from src.refinement.state import RefinementLike, RefinementState
from src.refinement.substrate import RefinementSubstrate, SubstrateSolveResult
from src.refinement.substrate_registry import (
    RefinementSubstrateRegistry,
    register_refinement_substrate,
)

__all__ = [
    "RefinementGame",
    "RefinementGameAdapter",
    "RefinementGameConfig",
    "RefinementGameRegistry",
    "RefinementLike",
    "RefinementState",
    "RefinementSubstrate",
    "RefinementSubstrateRegistry",
    "SubstrateSolveResult",
    "register_refinement_game",
    "register_refinement_substrate",
]
