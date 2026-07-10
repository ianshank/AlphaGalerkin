"""Domain-free sequential-refinement engine.

The ``RefinementGame`` abstraction and its MCTS adapter are the domain-agnostic
core that ``src.pde`` (PDE basis/mesh refinement) and ``src.thermo`` (λ-window
sample scheduling) both implement. It carries no PDE-specific types, so the same
single-agent MCTS engine drives every refinement domain.
"""

from src.refinement.adapter import RefinementGameAdapter
from src.refinement.config import RefinementGameConfig
from src.refinement.game import RefinementGame
from src.refinement.registry import RefinementGameRegistry, register_refinement_game
from src.refinement.state import RefinementLike, RefinementState

__all__ = [
    "RefinementGame",
    "RefinementGameAdapter",
    "RefinementGameConfig",
    "RefinementGameRegistry",
    "RefinementLike",
    "RefinementState",
    "register_refinement_game",
]
