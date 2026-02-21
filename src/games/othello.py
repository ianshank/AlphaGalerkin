"""Othello (Reversi) game implementation for AlphaGalerkin.

This module provides a complete Othello implementation following the
GameInterface contract, with variable board sizes (6×6 to 16×16) to
demonstrate resolution-independent zero-shot transfer.

Features:
    - Standard Othello rules (disc flipping in all 8 directions)
    - Variable board sizes (6, 8, 10, 12, etc.)
    - Pass when no legal moves (game ends when both pass)
    - Neural network tensor encoding (3 planes per player perspective)
    - 8-fold symmetry for data augmentation
    - Configurable via GameInterface contract

Resolution Independence:
    Train on 6×6 → evaluate on 8×8, 10×10, 12×12 for
    cross-resolution transfer benchmarks.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import structlog
import torch
from torch import Tensor

from src.games.interface import GameInterface, GamePhase, GameResult
from src.games.registry import register_game
from src.games.state import ActionMask, GameState

logger = structlog.get_logger(__name__)

# Piece constants
EMPTY = 0
BLACK = 1
WHITE = -1

# Direction vectors for disc flipping (8 directions)
_DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# Number of planes in neural network encoding
_STATE_CHANNELS = 3


@register_game("othello")
class OthelloGame(GameInterface):
    """Othello (Reversi) game implementation.

    Implements standard Othello rules with variable board sizes:
    - Discs are placed to flip opponent pieces in straight lines
    - If no legal move exists, player must pass
    - Game ends when neither player can move
    - Winner has the most discs

    The board is always even-sized (6, 8, 10, 12, ...) with the
    standard 2×2 center starting configuration.

    Attributes:
        name: Game identifier for registry.
        description: Human-readable game description.
        min_board_size: Minimum supported board size.
        max_board_size: Maximum supported board size.
        default_board_size: Default board size.

    """

    name: ClassVar[str] = "othello"
    description: ClassVar[str] = "Othello (Reversi) with variable board sizes"
    min_board_size: ClassVar[int] = 4
    max_board_size: ClassVar[int] = 16
    default_board_size: ClassVar[int] = 8

    def __init__(self) -> None:
        """Initialize Othello game."""
        self._board_size: int = self.default_board_size

    @property
    def action_space_size(self) -> int:
        """Get total action space size.

        Returns:
            board_size² + 1 (last action is pass).

        """
        return self._board_size * self._board_size + 1

    @property
    def state_channels(self) -> int:
        """Get number of input channels for neural network.

        Uses 3 planes:
        - Plane 0: Current player's discs
        - Plane 1: Opponent's discs
        - Plane 2: Current player indicator (all 1s if black, all 0s if white)

        Returns:
            3 feature planes.

        """
        return _STATE_CHANNELS

    def initial_state(self, board_size: int | None = None) -> GameState:
        """Create initial Othello board with center discs.

        The initial position places 2 black and 2 white discs in the
        center of the board in a diagonal pattern.

        Args:
            board_size: Board size (must be even, default: 8).

        Returns:
            Initial GameState.

        Raises:
            ValueError: If board_size is odd or out of range.

        """
        board_size = board_size or self.default_board_size
        self._board_size = board_size

        if board_size % 2 != 0:
            raise ValueError(f"Othello board size must be even, got {board_size}")
        if not (self.min_board_size <= board_size <= self.max_board_size):
            raise ValueError(
                f"Board size {board_size} out of range "
                f"[{self.min_board_size}, {self.max_board_size}]"
            )

        board = np.zeros((board_size, board_size), dtype=np.int8)

        # Place initial center discs
        mid = board_size // 2
        board[mid - 1, mid - 1] = WHITE
        board[mid - 1, mid] = BLACK
        board[mid, mid - 1] = BLACK
        board[mid, mid] = WHITE

        return GameState(
            board=board,
            current_player=BLACK,
            move_number=0,
            move_history=[],
            metadata={
                "board_size": board_size,
                "consecutive_passes": 0,
            },
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        """Get list of legal actions.

        A move is legal if placing a disc there flips at least one
        opponent disc. If no placement is legal, only pass is available.

        Args:
            state: Current game state.

        Returns:
            List of legal action indices.

        """
        board = state.board
        board_size = state.board_size
        player = state.current_player
        legal = []

        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == EMPTY:
                    if self._would_flip(board, row, col, player):
                        legal.append(row * board_size + col)

        # If no placement moves, pass is the only legal action
        if not legal:
            legal.append(board_size * board_size)  # pass action
        return legal

    def get_action_mask(self, state: GameState) -> ActionMask:
        """Get action mask for legal moves.

        Args:
            state: Current game state.

        Returns:
            ActionMask with legal moves marked True.

        """
        board_size = state.board_size
        action_size = board_size * board_size + 1
        mask = np.zeros(action_size, dtype=bool)

        for action in self.get_legal_actions(state):
            mask[action] = True

        return ActionMask(mask=mask, action_space_size=action_size)

    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action and return new state.

        Places a disc and flips all opponent discs in valid lines.
        Pass action leaves the board unchanged.

        Args:
            state: Current game state.
            action: Action index (row * board_size + col, or board_size² for pass).

        Returns:
            New GameState after the action.

        Raises:
            ValueError: If action is illegal.

        """
        board_size = state.board_size
        pass_action = board_size * board_size

        # Handle pass
        if action == pass_action:
            consecutive = state.metadata.get("consecutive_passes", 0) + 1
            return state.with_move(
                action=action,
                new_board=state.board.copy(),
                consecutive_passes=consecutive,
            )

        row = action // board_size
        col = action % board_size
        player = state.current_player
        board = state.board.copy()

        if board[row, col] != EMPTY:
            raise ValueError(f"Illegal move: position ({row}, {col}) is occupied")

        # Place disc
        board[row, col] = player

        # Flip discs in all directions
        flipped = self._flip_discs(board, row, col, player)
        if not flipped:
            raise ValueError(f"Illegal move: ({row}, {col}) flips no opponent discs")

        return state.with_move(
            action=action,
            new_board=board,
            consecutive_passes=0,
        )

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended.

        Game ends when both players must pass consecutively, or when
        the board is full.

        Args:
            state: Current game state.

        Returns:
            True if game is over.

        """
        # Two consecutive passes
        if state.metadata.get("consecutive_passes", 0) >= 2:
            return True

        # Board is full
        board_size = state.board_size
        if np.count_nonzero(state.board) == board_size * board_size:
            return True

        return False

    def get_result(self, state: GameState) -> GameResult:
        """Get game result by counting discs.

        Args:
            state: Terminal game state.

        Returns:
            GameResult with disc counts as scores.

        """
        board = state.board
        black_count = float(np.sum(board == BLACK))
        white_count = float(np.sum(board == WHITE))

        if black_count > white_count:
            winner = BLACK
        elif white_count > black_count:
            winner = WHITE
        else:
            winner = None

        return GameResult(
            winner=winner,
            score_black=black_count,
            score_white=white_count,
            reason="disc_count",
            move_count=state.move_number,
        )

    def get_winner(self, state: GameState) -> int | None:
        """Get winner from terminal state.

        Args:
            state: Terminal game state.

        Returns:
            1 for black, -1 for white, None for draw.

        """
        if not self.is_terminal(state):
            return None
        return self.get_result(state).winner

    def to_tensor(self, state: GameState) -> Tensor:
        """Convert state to neural network input tensor.

        Uses 3 planes:
        - Plane 0: Current player's discs (binary)
        - Plane 1: Opponent's discs (binary)
        - Plane 2: Current player indicator (all 1s for black, 0s for white)

        Args:
            state: Game state to encode.

        Returns:
            Tensor of shape (3, board_size, board_size).

        """
        board_size = state.board_size
        tensor = torch.zeros(_STATE_CHANNELS, board_size, board_size)
        board = state.board
        player = state.current_player

        tensor[0] = torch.from_numpy((board == player).astype(np.float32))
        tensor[1] = torch.from_numpy((board == -player).astype(np.float32))
        tensor[2] = 1.0 if player == BLACK else 0.0

        return tensor

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        """Get 8-fold symmetries (rotations and reflections).

        Othello boards have full D4 symmetry (4 rotations × 2 reflections).

        Args:
            state: Game state.
            policy: Policy distribution over actions.

        Returns:
            List of 8 (state, policy) pairs.

        """
        board_size = state.board_size
        symmetries: list[tuple[GameState, np.ndarray | Tensor]] = []

        if isinstance(policy, Tensor):
            policy_np = policy.cpu().numpy()
            is_tensor = True
        else:
            policy_np = policy
            is_tensor = False

        pass_prob = policy_np[-1]
        board_policy = policy_np[: board_size * board_size].reshape(board_size, board_size)
        board = state.board

        for rotation in range(4):
            for reflection in [False, True]:
                transformed_board = np.rot90(board, rotation)
                transformed_policy = np.rot90(board_policy, rotation)

                if reflection:
                    transformed_board = np.fliplr(transformed_board)
                    transformed_policy = np.fliplr(transformed_policy)

                new_state = GameState(
                    board=transformed_board.copy(),
                    current_player=state.current_player,
                    move_number=state.move_number,
                    move_history=[],
                    metadata=state.metadata.copy(),
                )

                new_policy = np.concatenate([transformed_policy.flatten(), [pass_prob]])

                if is_tensor:
                    new_policy = torch.from_numpy(new_policy)

                symmetries.append((new_state, new_policy))

        return symmetries

    def get_phase(self, state: GameState) -> GamePhase:
        """Get current game phase based on disc count.

        Args:
            state: Current game state.

        Returns:
            Current GamePhase.

        """
        if self.is_terminal(state):
            return GamePhase.TERMINAL

        board_size = state.board_size
        total_cells = board_size * board_size
        filled = np.count_nonzero(state.board)
        progress = filled / total_cells

        if progress < 0.25:
            return GamePhase.OPENING
        elif progress < 0.75:
            return GamePhase.MIDGAME
        else:
            return GamePhase.ENDGAME

    # --- Private helper methods ---

    def _would_flip(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        player: int,
    ) -> bool:
        """Check if placing a disc at (row, col) would flip any opponent discs.

        Args:
            board: Current board state.
            row: Row to place disc.
            col: Column to place disc.
            player: Player placing the disc.

        Returns:
            True if at least one disc would be flipped.

        """
        board_size = board.shape[0]
        opponent = -player

        for dr, dc in _DIRECTIONS:
            r, c = row + dr, col + dc
            found_opponent = False

            while 0 <= r < board_size and 0 <= c < board_size:
                if board[r, c] == opponent:
                    found_opponent = True
                elif board[r, c] == player:
                    if found_opponent:
                        return True
                    break
                else:
                    break
                r += dr
                c += dc

        return False

    def _flip_discs(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        player: int,
    ) -> list[tuple[int, int]]:
        """Flip opponent discs after placing at (row, col).

        Modifies board in-place and returns list of flipped positions.

        Args:
            board: Board state (modified in-place).
            row: Row where disc was placed.
            col: Column where disc was placed.
            player: Player who placed the disc.

        Returns:
            List of (row, col) positions that were flipped.

        """
        board_size = board.shape[0]
        opponent = -player
        all_flipped: list[tuple[int, int]] = []

        for dr, dc in _DIRECTIONS:
            flipped_in_dir: list[tuple[int, int]] = []
            r, c = row + dr, col + dc

            while 0 <= r < board_size and 0 <= c < board_size:
                if board[r, c] == opponent:
                    flipped_in_dir.append((r, c))
                elif board[r, c] == player:
                    # Found bracketing disc — flip all in between
                    all_flipped.extend(flipped_in_dir)
                    break
                else:
                    break
                r += dr
                c += dc

        for r, c in all_flipped:
            board[r, c] = player

        return all_flipped
