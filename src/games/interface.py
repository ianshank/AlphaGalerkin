"""Abstract game interface for multi-game support.

This module defines the abstract interface that all game implementations
must follow. This enables the continuous operator core to work with
any board game that implements this interface.

Design Principles:
    - Abstract: All game-specific logic is encapsulated
    - Consistent: Same interface for all games
    - Efficient: Optimized for MCTS tree traversal
    - Symmetric: Support for data augmentation via symmetries
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from src.games.state import ActionMask, GameState

if TYPE_CHECKING:
    from collections.abc import Sequence


class GamePhase(str, Enum):
    """Phases of a game."""

    SETUP = "setup"  # Initial setup (e.g., piece placement)
    OPENING = "opening"  # Opening phase
    MIDGAME = "midgame"  # Middle game
    ENDGAME = "endgame"  # End game phase
    TERMINAL = "terminal"  # Game over


@dataclass
class GameResult:
    """Result of a completed game."""

    winner: int | None  # 1, -1, or None for draw
    score_black: float
    score_white: float
    reason: str  # e.g., "resignation", "timeout", "score", "checkmate"
    move_count: int


class GameInterface(ABC):
    """Abstract base class for all game implementations.

    This interface defines the contract that all game implementations must
    follow. It enables the AlphaGalerkin architecture to work with any
    game without modification to the core neural network or MCTS code.

    Implementing a new game requires:
    1. Implementing all abstract methods
    2. Registering with GameRegistry
    3. Providing a tensor encoding for neural network input

    """

    # Class-level attributes (override in subclasses)
    name: str = "unknown"
    description: str = "Unknown game"

    # Configuration
    min_board_size: int = 1
    max_board_size: int = 100
    default_board_size: int = 19

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Get total size of action space.

        Returns:
            Number of possible actions (including pass if applicable).

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_channels(self) -> int:
        """Get number of input channels for neural network.

        Returns:
            Number of feature planes in tensor encoding.

        """
        raise NotImplementedError

    @property
    def n_players(self) -> int:
        """Get number of players.

        Returns:
            Number of players (default 2).

        """
        return 2

    @abstractmethod
    def initial_state(self, board_size: int | None = None) -> GameState:
        """Create initial game state.

        Args:
            board_size: Board size (uses default if None).

        Returns:
            Initial GameState for a new game.

        """
        raise NotImplementedError

    @abstractmethod
    def get_legal_actions(self, state: GameState) -> list[int]:
        """Get list of legal actions from current state.

        Args:
            state: Current game state.

        Returns:
            List of legal action indices.

        """
        raise NotImplementedError

    @abstractmethod
    def get_action_mask(self, state: GameState) -> ActionMask:
        """Get action mask indicating legal actions.

        Args:
            state: Current game state.

        Returns:
            ActionMask with legal actions marked True.

        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action and return new state.

        Args:
            state: Current game state.
            action: Action index to apply.

        Returns:
            New GameState after applying action.

        Raises:
            ValueError: If action is illegal.

        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended.

        Args:
            state: Current game state.

        Returns:
            True if game is over.

        """
        raise NotImplementedError

    @abstractmethod
    def get_result(self, state: GameState) -> GameResult:
        """Get game result from terminal state.

        Args:
            state: Terminal game state.

        Returns:
            GameResult with winner and scores.

        """
        raise NotImplementedError

    @abstractmethod
    def get_winner(self, state: GameState) -> int | None:
        """Get winner from terminal state.

        Args:
            state: Terminal game state.

        Returns:
            1 for black win, -1 for white win, None for draw.

        """
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self, state: GameState) -> Tensor:
        """Convert state to neural network input tensor.

        Args:
            state: Game state to encode.

        Returns:
            Tensor of shape (channels, height, width).

        """
        raise NotImplementedError

    @abstractmethod
    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        """Get symmetric transformations of state and policy.

        Used for data augmentation during training.

        Args:
            state: Game state.
            policy: Policy distribution over actions.

        Returns:
            List of (transformed_state, transformed_policy) tuples.

        """
        raise NotImplementedError

    def get_phase(self, state: GameState) -> GamePhase:
        """Get current game phase.

        Args:
            state: Current game state.

        Returns:
            Current GamePhase.

        """
        if self.is_terminal(state):
            return GamePhase.TERMINAL

        # Default heuristic based on move count
        total_moves = state.board_size ** 2
        progress = state.move_number / total_moves

        if progress < 0.1:
            return GamePhase.OPENING
        elif progress < 0.7:
            return GamePhase.MIDGAME
        else:
            return GamePhase.ENDGAME

    def action_to_string(self, action: int, board_size: int | None = None) -> str:
        """Convert action index to human-readable string.

        Args:
            action: Action index.
            board_size: Board size for coordinate conversion.

        Returns:
            Human-readable action string (e.g., "D4", "pass").

        """
        board_size = board_size or self.default_board_size

        if action == board_size * board_size:
            return "pass"

        row = action // board_size
        col = action % board_size

        # Use letter for column (skip 'I')
        col_letter = chr(ord('A') + col)
        if col_letter >= 'I':
            col_letter = chr(ord(col_letter) + 1)

        return f"{col_letter}{board_size - row}"

    def string_to_action(self, move_str: str, board_size: int | None = None) -> int:
        """Convert human-readable move to action index.

        Args:
            move_str: Move string (e.g., "D4", "pass").
            board_size: Board size for coordinate conversion.

        Returns:
            Action index.

        """
        board_size = board_size or self.default_board_size

        if move_str.lower() == "pass":
            return board_size * board_size

        # Parse coordinate
        col_letter = move_str[0].upper()
        row_num = int(move_str[1:])

        # Convert letter to column index (skip 'I')
        col = ord(col_letter) - ord('A')
        if col_letter > 'I':
            col -= 1

        row = board_size - row_num

        return row * board_size + col

    def validate_action(self, state: GameState, action: int) -> bool:
        """Check if action is valid.

        Args:
            state: Current game state.
            action: Action to validate.

        Returns:
            True if action is valid.

        """
        if action < 0 or action >= self.action_space_size:
            return False

        return action in self.get_legal_actions(state)

    def get_observation_shape(self, board_size: int | None = None) -> tuple[int, int, int]:
        """Get shape of observation tensor.

        Args:
            board_size: Board size (uses default if None).

        Returns:
            Tuple of (channels, height, width).

        """
        board_size = board_size or self.default_board_size
        return (self.state_channels, board_size, board_size)

    def batch_to_tensor(
        self,
        states: Sequence[GameState],
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Convert batch of states to tensor.

        Args:
            states: Sequence of game states.
            device: Target device for tensor.

        Returns:
            Batched tensor of shape (batch, channels, height, width).

        """
        tensors = [self.to_tensor(state) for state in states]
        return torch.stack(tensors).to(device)

    def get_canonical_form(self, state: GameState) -> GameState:
        """Get canonical form of state (from current player's perspective).

        By default, returns the state unchanged. Override for games
        where this transformation is meaningful.

        Args:
            state: Game state.

        Returns:
            Canonical GameState.

        """
        if state.current_player == 1:
            return state
        return state.flip_perspective()

    def clone(self) -> GameInterface:
        """Create a copy of the game interface.

        Returns:
            New instance of the game.

        """
        return type(self)()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class GameConfig:
    """Configuration for a game instance.

    Attributes:
        game_name: Name of the game.
        board_size: Board size to use.
        komi: Komi for Go (ignored for other games).
        time_control: Time control settings.

    """

    def __init__(
        self,
        game_name: str,
        board_size: int | None = None,
        komi: float = 7.5,
        time_control: dict[str, Any] | None = None,
    ) -> None:
        """Initialize game configuration.

        Args:
            game_name: Name of the game.
            board_size: Board size (uses game default if None).
            komi: Komi value for Go.
            time_control: Time control settings.

        """
        self.game_name = game_name
        self.board_size = board_size
        self.komi = komi
        self.time_control = time_control or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_name": self.game_name,
            "board_size": self.board_size,
            "komi": self.komi,
            "time_control": self.time_control,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameConfig:
        """Create from dictionary."""
        return cls(
            game_name=data["game_name"],
            board_size=data.get("board_size"),
            komi=data.get("komi", 7.5),
            time_control=data.get("time_control"),
        )
