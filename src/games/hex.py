"""Hex game implementation for AlphaGalerkin.

This module provides a complete Hex implementation following the
GameInterface contract, with variable board sizes (5×5 to 19×19)
to demonstrate resolution-independent zero-shot transfer.

Hex is the purest test of resolution independence because:
- Identical mechanics at every board size
- No komi or score adjustments needed
- Connection-based winning (not territory counting)
- First player advantage is proven but game is unsolved for N >= 10

Features:
    - Standard Hex rules (connect opposite sides)
    - Variable board sizes (5 to 19)
    - No draws possible (Hex is a determined game)
    - Union-Find for efficient connectivity checking
    - Neural network tensor encoding (3 planes)
    - 2-fold symmetry (180° rotation + color swap)

Resolution Independence:
    Train on 7×7 → evaluate on 11×11, 13×13, 19×19.
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
BLACK = 1  # Black connects top-bottom
WHITE = -1  # White connects left-right

# Hex has 6 neighbors (hexagonal adjacency on a rhombus grid)
_HEX_NEIGHBORS = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
]

# Number of planes in neural network encoding
_STATE_CHANNELS = 3


class _UnionFind:
    """Union-Find (Disjoint Set Union) for efficient connectivity checking.

    Used to determine if a player has connected their two opposite edges.
    Virtual nodes represent the board edges for O(alpha(n)) win checking.
    """

    def __init__(self, size: int) -> None:
        """Initialize Union-Find with given number of elements.

        Args:
            size: Number of elements (including virtual edge nodes).

        """
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        """Find root with path compression.

        Args:
            x: Element to find root of.

        Returns:
            Root element.

        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union two sets by rank.

        Args:
            x: First element.
            y: Second element.

        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set.

        Args:
            x: First element.
            y: Second element.

        Returns:
            True if connected.

        """
        return self.find(x) == self.find(y)

    def copy(self) -> _UnionFind:
        """Create a deep copy.

        Returns:
            New _UnionFind with identical state.

        """
        uf = _UnionFind(len(self.parent))
        uf.parent = self.parent.copy()
        uf.rank = self.rank.copy()
        return uf


@register_game("hex")
class HexGame(GameInterface):
    """Hex game implementation.

    Hex is played on an N×N rhombus grid where:
    - Black (player 1) connects top edge to bottom edge
    - White (player -1) connects left edge to right edge
    - Players alternately place stones on empty cells
    - No passing — a player must place a stone each turn
    - No draws are possible (Hex is determined)

    The board is represented as an N×N array where the hex adjacency
    is encoded as 6 neighbors per cell (see _HEX_NEIGHBORS).

    Internally uses Union-Find with virtual edge nodes for O(α(n))
    connectivity checking after each move.

    Attributes:
        name: Game identifier.
        description: Human-readable game description.

    """

    name: ClassVar[str] = "hex"
    description: ClassVar[str] = "Hex connection game with variable board sizes"
    min_board_size: ClassVar[int] = 3
    max_board_size: ClassVar[int] = 19
    default_board_size: ClassVar[int] = 11

    def __init__(self) -> None:
        """Initialize Hex game."""
        self._board_size: int = self.default_board_size

    @property
    def action_space_size(self) -> int:
        """Get total action space size.

        No pass move in Hex — action space is exactly board_size².

        Returns:
            board_size² actions.

        """
        return self._board_size * self._board_size

    @property
    def state_channels(self) -> int:
        """Get number of input channels for neural network.

        Uses 3 planes:
        - Plane 0: Current player's stones
        - Plane 1: Opponent's stones
        - Plane 2: Current player indicator

        Returns:
            3 feature planes.

        """
        return _STATE_CHANNELS

    def initial_state(self, board_size: int | None = None) -> GameState:
        """Create initial empty Hex board.

        Args:
            board_size: Board size (default: 11).

        Returns:
            Initial GameState with empty board and Union-Find structures.

        Raises:
            ValueError: If board_size is out of range.

        """
        board_size = board_size or self.default_board_size
        self._board_size = board_size

        if not (self.min_board_size <= board_size <= self.max_board_size):
            raise ValueError(
                f"Board size {board_size} out of range "
                f"[{self.min_board_size}, {self.max_board_size}]"
            )

        board = np.zeros((board_size, board_size), dtype=np.int8)

        # Union-Find for each player
        # Nodes: board_size * board_size cells + 2 virtual edge nodes
        # Virtual nodes are at indices N*N (edge A) and N*N+1 (edge B)
        n_cells = board_size * board_size
        uf_black = _UnionFind(n_cells + 2)
        uf_white = _UnionFind(n_cells + 2)

        # Connect virtual nodes to edge cells
        for i in range(board_size):
            # Black: top row → virtual_A, bottom row → virtual_B
            uf_black.union(i, n_cells)  # top row
            uf_black.union((board_size - 1) * board_size + i, n_cells + 1)  # bottom row
            # White: left col → virtual_A, right col → virtual_B
            uf_white.union(i * board_size, n_cells)  # left col
            uf_white.union(i * board_size + (board_size - 1), n_cells + 1)  # right col

        return GameState(
            board=board,
            current_player=BLACK,
            move_number=0,
            move_history=[],
            metadata={
                "board_size": board_size,
                "uf_black": uf_black,
                "uf_white": uf_white,
                "winner": None,
            },
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        """Get list of legal actions (all empty cells).

        Args:
            state: Current game state.

        Returns:
            List of legal action indices.

        """
        if state.metadata.get("winner") is not None:
            return []

        board = state.board
        board_size = state.board_size
        legal = []

        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == EMPTY:
                    legal.append(row * board_size + col)

        return legal

    def get_action_mask(self, state: GameState) -> ActionMask:
        """Get action mask for legal moves.

        Args:
            state: Current game state.

        Returns:
            ActionMask with legal moves marked True.

        """
        board_size = state.board_size
        action_size = board_size * board_size
        mask = np.zeros(action_size, dtype=bool)

        for action in self.get_legal_actions(state):
            mask[action] = True

        return ActionMask(mask=mask, action_space_size=action_size)

    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action and return new state.

        Places a stone and updates the Union-Find connectivity structure.

        Args:
            state: Current game state.
            action: Action index (row * board_size + col).

        Returns:
            New GameState after placing the stone.

        Raises:
            ValueError: If action is illegal.

        """
        board_size = state.board_size
        row = action // board_size
        col = action % board_size
        player = state.current_player
        board = state.board.copy()

        if board[row, col] != EMPTY:
            raise ValueError(f"Illegal move: position ({row}, {col}) is occupied")

        board[row, col] = player

        # Copy Union-Find structures
        uf_black: _UnionFind = state.metadata["uf_black"].copy()
        uf_white: _UnionFind = state.metadata["uf_white"].copy()

        # Update connectivity for the placed stone
        cell_idx = row * board_size + col
        uf = uf_black if player == BLACK else uf_white

        for dr, dc in _HEX_NEIGHBORS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if board[nr, nc] == player:
                    neighbor_idx = nr * board_size + nc
                    uf.union(cell_idx, neighbor_idx)

        # Check for winner
        n_cells = board_size * board_size
        winner = None
        if uf_black.connected(n_cells, n_cells + 1):
            winner = BLACK
        elif uf_white.connected(n_cells, n_cells + 1):
            winner = WHITE

        return state.with_move(
            action=action,
            new_board=board,
            uf_black=uf_black,
            uf_white=uf_white,
            winner=winner,
        )

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended (a player has connected their edges).

        Args:
            state: Current game state.

        Returns:
            True if a player has won.

        """
        return state.metadata.get("winner") is not None

    def get_result(self, state: GameState) -> GameResult:
        """Get game result.

        In Hex there are no draws — the winner has a connected path.

        Args:
            state: Terminal game state.

        Returns:
            GameResult with winner.

        """
        winner = state.metadata.get("winner")
        return GameResult(
            winner=winner,
            score_black=1.0 if winner == BLACK else 0.0,
            score_white=1.0 if winner == WHITE else 0.0,
            reason="connection",
            move_count=state.move_number,
        )

    def get_winner(self, state: GameState) -> int | None:
        """Get winner from current state.

        Args:
            state: Game state.

        Returns:
            1 for black, -1 for white, None if no winner yet.

        """
        return state.metadata.get("winner")

    def to_tensor(self, state: GameState) -> Tensor:
        """Convert state to neural network input tensor.

        Uses 3 planes:
        - Plane 0: Current player's stones (binary)
        - Plane 1: Opponent's stones (binary)
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
        """Get symmetries for data augmentation.

        Hex on a rhombus grid has a 180° rotational symmetry combined
        with a color/edge swap. This gives 2 positions per state:
        1. Original
        2. 180° rotation with player swap (Black ↔ White, top-bottom ↔ left-right)

        For simplicity, we return only the identity (the 180° symmetry
        requires reconstructing Union-Find state, which is expensive).

        Args:
            state: Game state.
            policy: Policy distribution over actions.

        Returns:
            List containing the original (state, policy) pair.

        """
        # Identity symmetry only — Hex's 180° rotation requires
        # swapping Black/White connectivity semantics
        return [(state, policy)]

    def get_phase(self, state: GameState) -> GamePhase:
        """Get current game phase based on board fill.

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

        if progress < 0.15:
            return GamePhase.OPENING
        elif progress < 0.6:
            return GamePhase.MIDGAME
        else:
            return GamePhase.ENDGAME

    def action_to_string(self, action: int, board_size: int | None = None) -> str:
        """Convert action index to human-readable string.

        Args:
            action: Action index.
            board_size: Board size for coordinate conversion.

        Returns:
            Coordinate string (e.g., "A1", "C5").

        """
        board_size = board_size or self._board_size
        row = action // board_size
        col = action % board_size

        col_letter = chr(ord("A") + col)
        return f"{col_letter}{row + 1}"

    def string_to_action(self, move_str: str, board_size: int | None = None) -> int:
        """Convert human-readable string to action index.

        Args:
            move_str: Move string (e.g., "A1", "C5").
            board_size: Board size for coordinate conversion.

        Returns:
            Action index.

        """
        board_size = board_size or self._board_size
        col = ord(move_str[0].upper()) - ord("A")
        row = int(move_str[1:]) - 1
        return row * board_size + col
