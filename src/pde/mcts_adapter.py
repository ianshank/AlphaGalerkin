"""Adapter connecting PDE games to the MCTS search engine.

Bridges the ``PDEGame`` interface (which models PDE-solving as a
sequential decision game) with the ``MCTS.GameInterface`` protocol
(which expects board-game-style ``get_state``/``apply_action``/
``is_terminal``/``get_winner``/``clone``).

Usage::

    from src.pde.mcts_adapter import PDEGameAdapter
    from src.pde.games.basis_selection import BasisSelectionGame
    from src.pde.operators import PoissonOperator
    from src.pde.config import PDEConfig, PDEGameConfig, PDEType
    from src.mcts.search import MCTS
    from src.mcts.evaluator import RandomEvaluator

    pde_config = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
    game_config = PDEGameConfig(
        name="basis_game",
        pde_config=pde_config,
        game_mode="basis_selection",
    )
    operator = PoissonOperator(pde_config)
    pde_game = BasisSelectionGame(operator, game_config)
    adapter = PDEGameAdapter(pde_game)

    evaluator = RandomEvaluator(action_size=pde_game.action_space_size)
    mcts = MCTS(evaluator=evaluator, n_simulations=50)
    policy = mcts.search(adapter)
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.pde.game import PDEGame, PDEState

logger = structlog.get_logger(__name__)


class PDEGameAdapter:
    """Adapter that wraps a ``PDEGame`` for use with ``MCTS.search()``.

    Satisfies the ``GameInterface`` protocol defined in
    ``src/mcts/search.py`` by translating PDE-specific methods into
    the board-game-style interface that MCTS expects.

    Error reduction is mapped to a ``[-1, 1]`` outcome for the
    ``get_winner`` method: convergence maps to +1, budget exhaustion
    with poor error maps to -1, and intermediate outcomes interpolate.

    Attributes:
        pde_game: The underlying PDE game instance.
        state: Current PDE state (mutated by ``apply_action``).
        error_history: Tracked error values across steps.

    """

    def __init__(self, pde_game: PDEGame) -> None:
        """Initialize adapter.

        Args:
            pde_game: A concrete ``PDEGame`` implementation (e.g.
                ``BasisSelectionGame``, ``MeshRefinementGame``).

        """
        self.pde_game = pde_game
        self.state: PDEState = pde_game.get_initial_state()
        self.error_history: list[float] = [self.state.error_estimate]

        logger.debug(
            "pde_mcts_adapter_created",
            game_type=type(pde_game).__name__,
            initial_error=self.state.error_estimate,
            action_space_size=pde_game.action_space_size,
        )

    # ------------------------------------------------------------------ #
    # GameInterface protocol methods                                      #
    # ------------------------------------------------------------------ #

    def get_state(self) -> NDArray[np.float32]:
        """Get the current state as a numpy tensor.

        Delegates to ``pde_game.to_tensor()`` and converts the
        resulting PyTorch tensor to numpy for MCTS compatibility.

        Returns:
            State tensor (channels, height, width) or (channels, n).

        """
        tensor = self.pde_game.to_tensor(self.state)
        arr: NDArray[np.float32] = tensor.detach().cpu().numpy().astype(np.float32)
        return arr

    def get_legal_actions(self) -> list[int]:
        """Return indices of legal actions in the current state.

        Returns:
            Sorted list of valid action indices.

        """
        return self.pde_game.get_valid_actions(self.state)

    def apply_action(self, action: int) -> None:
        """Apply an action, mutating the internal state.

        Args:
            action: Action index to apply.

        Raises:
            ValueError: If the action is not legal.

        """
        prev_state = self.state
        self.state = self.pde_game.apply_action(self.state, action)
        self.error_history.append(self.state.error_estimate)

        logger.debug(
            "pde_action_applied",
            action=action,
            error_before=prev_state.error_estimate,
            error_after=self.state.error_estimate,
            step=self.state.step,
        )

    def is_terminal(self) -> bool:
        """Check whether the PDE game has terminated.

        Returns:
            True when the game has converged or budget is exhausted.

        """
        return self.pde_game.is_terminal(self.state)

    def get_winner(self) -> int:
        """Map the PDE outcome to {-1, 0, 1}.

        Convention:
        - +1: Converged (error < tolerance) — *success*
        - -1: Budget exhausted with error > 2x tolerance — *failure*
        -  0: Ambiguous / partial convergence

        Returns:
            Outcome in {-1, 0, 1}.

        """
        if not self.error_history:
            return 0

        initial_error = self.error_history[0]
        final_error = self.error_history[-1]

        config = self.pde_game.config
        if final_error < config.error_tolerance:
            return 1  # Converged successfully

        # Measure relative error reduction
        if initial_error > 0:
            reduction_ratio = final_error / initial_error
        else:
            reduction_ratio = 1.0

        if reduction_ratio < config.winner_good_reduction_threshold:
            return 1
        elif reduction_ratio > config.winner_poor_reduction_threshold:
            return -1
        else:
            return 0

    def clone(self) -> PDEGameAdapter:
        """Create a deep copy for MCTS simulation.

        Returns:
            Independent copy of this adapter.

        """
        cloned = PDEGameAdapter.__new__(PDEGameAdapter)
        cloned.pde_game = self.pde_game  # Game rules are shared (stateless)
        cloned.state = copy.deepcopy(self.state)
        cloned.error_history = list(self.error_history)
        return cloned

    # ------------------------------------------------------------------ #
    # Additional helpers                                                  #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the adapter to the initial state."""
        self.state = self.pde_game.get_initial_state()
        self.error_history = [self.state.error_estimate]

    @property
    def current_error(self) -> float:
        """Current error estimate."""
        return self.state.error_estimate

    @property
    def error_reduction(self) -> float:
        """Total error reduction from initial state."""
        if not self.error_history or self.error_history[0] == 0:
            return 0.0
        return 1.0 - (self.error_history[-1] / self.error_history[0])
