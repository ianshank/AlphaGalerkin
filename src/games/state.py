"""Generic game state representation.

This module provides a flexible game state class that can represent
the state of any board game while maintaining compatibility with
the continuous operator architecture.

Design Principles:
    - Generic: Works for any board game
    - Immutable: States are not modified in place
    - Hashable: Can be used as dictionary keys for transposition tables
    - Serializable: Can be saved/loaded for analysis
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=False)
class GameState:
    """Generic game state representation.

    Represents the complete state of a game at any point in time.
    Designed to be game-agnostic while supporting the features needed
    for MCTS and neural network evaluation.

    Attributes:
        board: Board state as numpy array (game-specific encoding).
        current_player: Current player to move (1 or -1 typically).
        move_number: Number of moves played.
        move_history: List of previous moves.
        metadata: Game-specific additional data.

    """

    board: np.ndarray
    current_player: int = 1
    move_number: int = 0
    move_history: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # For hash computation
    _hash: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate state after initialization."""
        if self.board is None:
            raise ValueError("Board cannot be None")

        # Ensure board is numpy array
        if not isinstance(self.board, np.ndarray):
            self.board = np.array(self.board)

        # Ensure move_history is a list
        if self.move_history is None:
            self.move_history = []

        # Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}

    def copy(self) -> GameState:
        """Create a deep copy of the state.

        Returns:
            New GameState with copied data.

        """
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            move_number=self.move_number,
            move_history=self.move_history.copy(),
            metadata=self.metadata.copy(),
        )

    def with_move(
        self,
        action: int,
        new_board: np.ndarray,
        next_player: int | None = None,
        **metadata_updates: Any,
    ) -> GameState:
        """Create a new state after applying a move.

        Args:
            action: The action taken.
            new_board: The resulting board state.
            next_player: Next player to move (switches if None).
            **metadata_updates: Updates to metadata.

        Returns:
            New GameState representing the position after the move.

        """
        new_metadata = self.metadata.copy()
        new_metadata.update(metadata_updates)

        return GameState(
            board=new_board,
            current_player=next_player if next_player is not None else -self.current_player,
            move_number=self.move_number + 1,
            move_history=[*self.move_history, action],
            metadata=new_metadata,
        )

    @property
    def board_size(self) -> int:
        """Get board size (assuming square board).

        Returns:
            Board dimension.

        """
        return self.board.shape[-1]

    @property
    def last_move(self) -> int | None:
        """Get the last move played.

        Returns:
            Last move action or None if no moves.

        """
        return self.move_history[-1] if self.move_history else None

    def __hash__(self) -> int:
        """Compute hash for transposition table lookups.

        Returns:
            Hash of the game state.

        """
        if self._hash is None:
            # Hash based on board state and current player
            board_bytes = self.board.tobytes()
            player_bytes = bytes([self.current_player + 2])  # +2 to make positive
            combined = board_bytes + player_bytes
            self._hash = int(hashlib.md5(combined).hexdigest(), 16)

        return self._hash

    def __eq__(self, other: object) -> bool:
        """Check equality with another state.

        Args:
            other: Other object to compare.

        Returns:
            True if states are equal.

        """
        if not isinstance(other, GameState):
            return False

        return (
            np.array_equal(self.board, other.board) and self.current_player == other.current_player
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.

        """
        return {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "move_number": self.move_number,
            "move_history": self.move_history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameState:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            GameState instance.

        """
        return cls(
            board=np.array(data["board"]),
            current_player=data["current_player"],
            move_number=data.get("move_number", 0),
            move_history=data.get("move_history", []),
            metadata=data.get("metadata", {}),
        )

    def flip_perspective(self) -> GameState:
        """Create state from opponent's perspective.

        Useful for training with symmetric data augmentation.

        Returns:
            New state with flipped perspective.

        """
        return GameState(
            board=-self.board,  # Flip piece colors
            current_player=-self.current_player,
            move_number=self.move_number,
            move_history=self.move_history.copy(),
            metadata=self.metadata.copy(),
        )


@dataclass
class ActionMask:
    """Represents legal actions as a boolean mask.

    Attributes:
        mask: Boolean array where True indicates legal action.
        action_space_size: Total size of action space.

    """

    mask: np.ndarray
    action_space_size: int

    def __post_init__(self) -> None:
        """Validate mask dimensions."""
        if len(self.mask) != self.action_space_size:
            raise ValueError(
                f"Mask size {len(self.mask)} != action_space_size {self.action_space_size}"
            )

    @property
    def legal_actions(self) -> list[int]:
        """Get list of legal action indices.

        Returns:
            List of legal action indices.

        """
        return list(np.where(self.mask)[0])

    @property
    def num_legal(self) -> int:
        """Get number of legal actions.

        Returns:
            Count of legal actions.

        """
        return int(self.mask.sum())

    def is_legal(self, action: int) -> bool:
        """Check if action is legal.

        Args:
            action: Action index to check.

        Returns:
            True if action is legal.

        """
        if 0 <= action < self.action_space_size:
            return bool(self.mask[action])
        return False


def create_empty_state(
    board_size: int,
    n_planes: int = 1,
    dtype: np.dtype = np.float32,
) -> GameState:
    """Create an empty game state.

    Args:
        board_size: Size of the board.
        n_planes: Number of board planes.
        dtype: Data type for board array.

    Returns:
        Empty GameState.

    """
    if n_planes > 1:
        board = np.zeros((n_planes, board_size, board_size), dtype=dtype)
    else:
        board = np.zeros((board_size, board_size), dtype=dtype)

    return GameState(
        board=board,
        current_player=1,
        move_number=0,
        move_history=[],
        metadata={},
    )
