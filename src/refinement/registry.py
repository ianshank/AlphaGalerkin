"""Registry for domain-free refinement games.

Mirrors ``src.pde.register_games`` (which registers PDE games in the
``GameRegistry``) using the shared ``src.templates.registry`` factory, so any
domain can register a ``RefinementGame`` under a string key with the
``@register_refinement_game("name")`` decorator.
"""

from __future__ import annotations

from src.refinement.game import RefinementGame
from src.templates.registry import create_registry

RefinementGameRegistry, register_refinement_game = create_registry(
    "RefinementGame",
    RefinementGame,  # type: ignore[type-abstract]
)

__all__ = ["RefinementGameRegistry", "register_refinement_game"]
