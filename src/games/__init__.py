"""Multi-game support infrastructure for AlphaGalerkin.

This module provides abstract interfaces and concrete implementations
for supporting multiple games with the shared continuous operator core.

Key Components:
    - GameInterface: Abstract base class for all games
    - GameState: Generic game state representation
    - GameRegistry: Registry for game implementations
    - GoGame: Go-specific implementation
    - ChessGame: Chess implementation (stub)
    - ShogiGame: Shogi implementation (stub)

Usage:
    from src.games import GameRegistry, GoGame

    # Get game by name
    game = GameRegistry.get("go")

    # Create initial state
    state = game.initial_state(board_size=19)

    # Get legal actions
    actions = game.get_legal_actions(state)

    # Apply action
    new_state = game.apply_action(state, actions[0])
"""

from src.games.interface import GameInterface, GamePhase
from src.games.registry import GameRegistry, register_game
from src.games.state import GameState

# Import game implementations to trigger registration
from src.games import go  # noqa: F401

__all__ = [
    "GameInterface",
    "GamePhase",
    "GameRegistry",
    "GameState",
    "register_game",
]
