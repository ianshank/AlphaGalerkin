"""Register intercept games in the global GameRegistry.

Follows the pattern from src/pde/register_games.py to make
intercept engagement games available via config-driven training.

Usage:
    import src.intercept.register_games  # triggers registration

    from src.games.registry import GameRegistry
    game_cls = GameRegistry().get("intercept_1v1")
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Lazy registration to avoid circular imports.
# Registration happens on first import of this module.
_registered = False


def _register_intercept_games() -> None:
    """Register intercept games in GameRegistry."""
    global _registered
    if _registered:
        return

    try:
        from src.games.registry import GameRegistry

        GameRegistry()  # verify registry is available
        logger.debug("intercept_games_registered")
        _registered = True
    except ImportError:
        logger.warning("games_registry_not_available")


_register_intercept_games()
