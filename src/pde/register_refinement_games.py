"""Register substrate-backed ``RefinementGame`` implementations.

Importing this module triggers registration so callers can resolve games via
``RefinementGameRegistry`` without pulling heavy imports through package
``__init__.py`` (the documented SIGSEGV / coverage-tracer failure class)::

    import src.pde.register_refinement_games  # noqa: F401

    from src.refinement.registry import RefinementGameRegistry
    game_cls = RefinementGameRegistry().get_or_raise("substrate_refinement")
"""

from __future__ import annotations

import structlog

# Side-effect import: @register_refinement_game on SubstrateRefinementGame.
from src.pde.games.substrate_refinement import (
    GAME_REGISTRY_NAME,
    SubstrateRefinementGame,
)
from src.research.substrates.factory import ensure_substrate_registrants

logger = structlog.get_logger(__name__)

# Ensure substrate kinds resolve for config-driven construction inside the game.
ensure_substrate_registrants()

logger.info(
    "refinement_games_registered",
    games=(GAME_REGISTRY_NAME,),
)

__all__ = ["GAME_REGISTRY_NAME", "SubstrateRefinementGame"]
