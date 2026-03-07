"""Stateful wrapper for stateless GameInterface to be used with MCTS.

Adapts the stateless GameInterface API (which passes GameState explicitly)
to the object-oriented API expected by MCTS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.games.state import GameState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.games.interface import GameInterface

class StatefulGameWrapper:
    """Wraps a stateless GameInterface and GameState to satisfy MCTS GameInterface."""

    def __init__(self, game: GameInterface, state: GameState) -> None:
        """Initialize the wrapper.

        Args:
            game: Stateless game interface.
            state: Current game state.

        """
        self.game = game
        self.state = state

    def get_state(self) -> NDArray[np.float32]:
        """Get tensor representation as numpy array."""
        return self.game.to_tensor(self.state).cpu().numpy()

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        return self.game.get_legal_actions(self.state)

    def apply_action(self, action: int) -> None:
        """Apply action to game state."""
        self.state = self.game.apply_action(self.state, action)

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.game.is_terminal(self.state)

    def get_winner(self) -> int:
        """Get winner: 1 for black, -1 for white, 0 for draw."""
        winner = self.game.get_winner(self.state)
        return winner if winner is not None else 0

    def clone(self) -> StatefulGameWrapper:
        """Create a deep copy of the wrapper."""
        # The game interface is stateless and thread-safe.
        # The state objects are largely immutable and deepcopied on apply_action.
        import copy
        return StatefulGameWrapper(self.game, copy.deepcopy(self.state))
