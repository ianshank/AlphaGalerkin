"""GameInterface wrapper for PDE games.

Bridges PDE games (BasisSelectionGame, MeshRefinementGame) to the
GameInterface protocol used by the trainer and self-play infrastructure.
This enables PDE games to be registered in GameRegistry and used via
config (e.g. ``training.game=pde_basis``).

The wrapper converts between PDEState and GameState, and delegates
all game logic to the underlying PDEGame implementation.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from torch import Tensor

from src.games.interface import GameInterface, GamePhase, GameResult
from src.games.state import ActionMask, GameState

if TYPE_CHECKING:
    from src.pde.config import PDEConfig, PDEGameConfig
    from src.pde.game import PDEGame, PDEState

logger = structlog.get_logger(__name__)


class PDEGameInterface(GameInterface):
    """Wraps a PDEGame to satisfy the GameInterface contract.

    Maps PDE-specific concepts to the board game abstraction:
    - PDEState → GameState (error field stored as board, metadata carries PDE info)
    - Action space → PDE basis/refinement actions
    - Terminal condition → convergence or budget exhaustion
    - Winner → +1 if converged, -1 if failed, 0 if ambiguous

    The ``grid_size`` parameter controls the spatial resolution of the
    tensor encoding passed to the neural network.
    """

    name = "pde_basis"
    description = "PDE basis selection via MCTS-guided Galerkin method"

    min_board_size = 4
    max_board_size = 64
    default_board_size = 16

    def __init__(
        self,
        pde_game: PDEGame,
        grid_size: int = 16,
    ) -> None:
        """Initialize PDE game interface.

        Args:
            pde_game: Concrete PDEGame instance (BasisSelectionGame, etc.).
            grid_size: Resolution for the tensor encoding grid.

        """
        self.pde_game = pde_game
        self.grid_size = grid_size
        self._action_space_size = pde_game.action_space_size
        self._state_channels = pde_game.state_channels

    @property
    def action_space_size(self) -> int:
        """Total number of PDE actions (basis functions or refinement choices)."""
        return self._action_space_size

    @property
    def state_channels(self) -> int:
        """Number of input channels for the neural network."""
        return self._state_channels

    @property
    def n_players(self) -> int:
        """PDE games are single-player (agent vs. PDE)."""
        return 1

    def initial_state(self, board_size: int | None = None) -> GameState:
        """Create initial PDE game state.

        Args:
            board_size: Ignored for PDE games (grid_size used instead).

        Returns:
            GameState wrapping the initial PDEState.

        """
        pde_state = self.pde_game.get_initial_state()
        return self._pde_to_game_state(pde_state)

    def get_legal_actions(self, state: GameState) -> list[int]:
        """Get valid PDE actions from current state."""
        pde_state = self._game_to_pde_state(state)
        return self.pde_game.get_valid_actions(pde_state)

    def get_action_mask(self, state: GameState) -> ActionMask:
        """Get action mask for legal PDE actions."""
        pde_state = self._game_to_pde_state(state)
        mask_tensor = self.pde_game.get_action_mask(pde_state)
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask_tensor, dtype=bool)
        return ActionMask(mask=mask_np, action_space_size=self._action_space_size)

    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply PDE action and return new state."""
        pde_state = self._game_to_pde_state(state)
        new_pde_state = self.pde_game.apply_action(pde_state, action)
        return self._pde_to_game_state(
            new_pde_state,
            prev_move_history=list(state.move_history) + [action],
        )

    def is_terminal(self, state: GameState) -> bool:
        """Check if PDE game has terminated (converged or budget exhausted)."""
        pde_state = self._game_to_pde_state(state)
        return self.pde_game.is_terminal(pde_state)

    def get_result(self, state: GameState) -> GameResult:
        """Get PDE game result."""
        pde_state = self._game_to_pde_state(state)
        winner = self._compute_winner(pde_state, state)
        error = pde_state.error_estimate
        return GameResult(
            winner=winner if winner != 0 else None,
            score_black=float(1.0 - error),  # Higher score = lower error
            score_white=float(error),
            reason="converged" if winner == 1 else "budget_exhausted",
            move_count=state.move_number,
        )

    def get_winner(self, state: GameState) -> int | None:
        """Get winner: 1 if converged, -1 if failed, None if ambiguous."""
        pde_state = self._game_to_pde_state(state)
        winner = self._compute_winner(pde_state, state)
        return winner if winner != 0 else None

    def to_tensor(self, state: GameState) -> Tensor:
        """Convert PDE state to neural network input tensor."""
        pde_state = self._game_to_pde_state(state)
        return self.pde_game.to_tensor(pde_state)

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        """PDE games have no geometric symmetries by default.

        Override in subclasses for domain-specific symmetries
        (e.g., periodic BCs allow translation symmetry).
        """
        return [(state, policy)]

    def get_phase(self, state: GameState) -> GamePhase:
        """Get current PDE solving phase."""
        if self.is_terminal(state):
            return GamePhase.TERMINAL
        pde_state = self._game_to_pde_state(state)
        tolerance = getattr(self.pde_game.config, "tolerance", 0.01)
        if pde_state.error_estimate > tolerance * 10:
            return GamePhase.OPENING
        elif pde_state.error_estimate > tolerance:
            return GamePhase.MIDGAME
        return GamePhase.ENDGAME

    # ------------------------------------------------------------------
    # State conversion helpers
    # ------------------------------------------------------------------

    def _pde_to_game_state(
        self,
        pde_state: PDEState,
        prev_move_history: list[int] | None = None,
    ) -> GameState:
        """Convert PDEState to GameState.

        Stores the PDE tensor encoding as the board and preserves
        the full PDEState in metadata for lossless round-tripping.
        """
        tensor = self.pde_game.to_tensor(pde_state)
        board = tensor.detach().cpu().numpy()

        return GameState(
            board=board,
            current_player=1,  # Single-player game
            move_number=pde_state.step,
            move_history=prev_move_history or [],
            metadata={
                "_pde_state": pde_state,
                "error_estimate": pde_state.error_estimate,
                "step": pde_state.step,
            },
        )

    @staticmethod
    def _game_to_pde_state(state: GameState) -> PDEState:
        """Extract PDEState from GameState metadata."""
        pde_state = state.metadata.get("_pde_state")
        if pde_state is None:
            raise ValueError(
                "GameState does not contain a PDEState in metadata. "
                "Was this state created by PDEGameInterface?"
            )
        return pde_state

    def _compute_winner(self, pde_state: PDEState, state: GameState) -> int:
        """Compute winner from PDE state.

        +1: Converged (error < tolerance)
        -1: Budget exhausted with poor error
         0: Ambiguous / partial convergence
        """
        tolerance = getattr(self.pde_game.config, "tolerance", 0.01)

        if pde_state.error_estimate < tolerance:
            return 1

        # Check error reduction from initial state
        initial_error = state.metadata.get("_initial_error", pde_state.error_estimate)
        if initial_error > 0 and pde_state.error_estimate / initial_error < 0.1:
            return 1  # 90%+ reduction
        elif initial_error > 0 and pde_state.error_estimate / initial_error > 0.5:
            return -1  # Less than 50% reduction
        return 0
