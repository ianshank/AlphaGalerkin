"""Multi-game support infrastructure for AlphaGalerkin.

This module provides abstract interfaces and concrete implementations
for supporting multiple games with the shared continuous operator core.

Key Components:
    - GameInterface: Abstract base class for all games
    - GameState: Generic game state representation
    - GameRegistry: Registry for game implementations
    - GoGame: Go game implementation (full Chinese rules, superko)
    - ChessGame: Chess implementation (full rules, castling, en passant, promotions)
    - ShogiGame: Shogi implementation (stub - not yet implemented)

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

# Optional torch-dependent imports
try:
    # Import game implementations to trigger registration
    from src.games import (
        chess,  # noqa: F401
        go,  # noqa: F401
    )
    from src.games.interface import GameInterface, GamePhase
    from src.games.registry import GameRegistry, register_game
    from src.games.state import GameState

    _HAS_TORCH = True
except ImportError:
    # Torch not available - provide stubs
    GameInterface = None
    GamePhase = None
    GameRegistry = None
    GameState = None
    register_game = None
    _HAS_TORCH = False

__all__ = [
    "GameInterface",
    "GamePhase",
    "GameRegistry",
    "GameState",
    "register_game",
]
