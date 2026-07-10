"""Adapter bridging a ``RefinementGame`` to the MCTS ``GameInterface``.

Domain-free counterpart to ``src.pde.mcts_adapter.PDEGameAdapter``: it translates
a ``RefinementGame`` over ``RefinementState`` into the board-game-style protocol
MCTS expects, and exposes the correct single-agent backup mode so callers wire
the search correctly.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.mcts.search import SearchMode
    from src.refinement.game import RefinementGame
    from src.refinement.state import RefinementState


class RefinementGameAdapter:
    """Wrap a ``RefinementGame`` for use with ``MCTS.search()``.

    A refinement problem is **single-agent**, so callers should construct MCTS
    with ``search_mode=adapter.search_mode`` (``SearchMode.SINGLE_AGENT``) — the
    default ``ZERO_SUM`` would invert the value at odd depths and make the search
    minimise the objective (the F0 bug this whole surface exists to prevent).
    """

    def __init__(self, game: RefinementGame) -> None:
        """Initialise the adapter from a concrete refinement game."""
        self.game = game
        self.state: RefinementState = game.get_initial_state()
        self.error_history: list[float] = [self.state.error_estimate]
        self._prev_state: RefinementState | None = None

    @property
    def action_space_size(self) -> int:
        """Total action count of the wrapped game (for evaluator construction)."""
        return self.game.action_space_size

    @property
    def search_mode(self) -> SearchMode:
        """The correct MCTS backup mode for a single-agent refinement game."""
        from src.mcts.search import SearchMode

        return SearchMode.SINGLE_AGENT

    # ------------------------------------------------------------------ #
    # GameInterface protocol                                              #
    # ------------------------------------------------------------------ #

    def get_state(self) -> NDArray[np.float32]:
        """Encode the current state as a numpy float32 tensor."""
        tensor = self.game.to_tensor(self.state)
        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float32)
        arr: NDArray[np.float32] = tensor.detach().cpu().numpy().astype(np.float32)
        return arr

    def get_legal_actions(self) -> list[int]:
        """Return the legal action indices in the current state."""
        return self.game.get_valid_actions(self.state)

    def apply_action(self, action: int) -> None:
        """Apply ``action``, advancing the internal state."""
        self._prev_state = self.state
        self.state = self.game.apply_action(self.state, action)
        self.error_history.append(self.state.error_estimate)

    def get_last_reward(self) -> float:
        """Per-edge reward for the last action (``SupportsStepReward``)."""
        if self._prev_state is None:
            return 0.0
        return float(self.game.get_reward(self.state, self._prev_state))

    def is_terminal(self) -> bool:
        """Return True when the refinement episode has terminated."""
        return self.game.is_terminal(self.state)

    def get_winner(self) -> int:
        """Map the terminal outcome to ``{-1, 0, 1}``."""
        return self.game.get_winner(self.state)

    def clone(self) -> RefinementGameAdapter:
        """Return a sibling-safe copy for MCTS simulation."""
        cloned = RefinementGameAdapter.__new__(RefinementGameAdapter)
        cloned.game = self.game.clone()
        cloned.state = self.state.clone()
        cloned.error_history = list(self.error_history)
        cloned._prev_state = (
            copy.deepcopy(self._prev_state) if self._prev_state is not None else None
        )
        return cloned

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset to the initial state."""
        self.state = self.game.get_initial_state()
        self.error_history = [self.state.error_estimate]
        self._prev_state = None

    @property
    def current_error(self) -> float:
        """Current error estimate."""
        return self.state.error_estimate

    @property
    def error_reduction(self) -> float:
        """Fractional error reduction from the initial state."""
        if not self.error_history or self.error_history[0] == 0:
            return 0.0
        return 1.0 - (self.error_history[-1] / self.error_history[0])
