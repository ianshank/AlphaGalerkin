"""Go game implementation for AlphaGalerkin.

This module provides a complete Go implementation following the
GameInterface contract, enabling the continuous operator architecture
to be used for Go.

Features:
    - Standard Go rules (Chinese scoring)
    - Superko rule
    - Variable board sizes (9x9, 13x13, 19x19, etc.)
    - Neural network tensor encoding
    - 8-fold symmetry for data augmentation
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.games.interface import GameInterface, GamePhase, GameResult
from src.games.registry import register_game
from src.games.state import ActionMask, GameState

# Constants
EMPTY = 0
BLACK = 1
WHITE = -1


@register_game("go")
class GoGame(GameInterface):
    """Go game implementation.

    Implements the standard Go rules with:
    - Chinese scoring (area scoring)
    - Superko rule (positional superko)
    - Standard komi (7.5 for 19x19, adjustable)

    Attributes:
        komi: Points given to white to compensate for black's first move.
        superko: Whether to enforce superko rule.

    """

    name = "go"
    description = "The ancient board game of Go"
    min_board_size = 5
    max_board_size = 25
    default_board_size = 19

    def __init__(
        self,
        komi: float = 7.5,
        superko: bool = True,
    ) -> None:
        """Initialize Go game.

        Args:
            komi: Komi value (points to white).
            superko: Enable superko rule.

        """
        self.komi = komi
        self.superko = superko
        self._board_size: int = self.default_board_size

    @property
    def action_space_size(self) -> int:
        """Get total action space size.

        Returns:
            board_size^2 + 1 (for pass move).

        """
        return self._board_size * self._board_size + 1

    @property
    def state_channels(self) -> int:
        """Get number of input channels for neural network.

        Uses 17 planes like AlphaGo Zero:
        - Planes 0-7: Black stones for last 8 positions
        - Planes 8-15: White stones for last 8 positions
        - Plane 16: Current player (all 1s if black, all 0s if white)

        Returns:
            17 feature planes.

        """
        return 17

    def initial_state(self, board_size: int | None = None) -> GameState:
        """Create initial empty Go board.

        Args:
            board_size: Board size (default: 19).

        Returns:
            Initial GameState with empty board.

        """
        board_size = board_size or self.default_board_size
        self._board_size = board_size

        board = np.zeros((board_size, board_size), dtype=np.int8)

        return GameState(
            board=board,
            current_player=BLACK,
            move_number=0,
            move_history=[],
            metadata={
                "komi": self.komi,
                "board_size": board_size,
                "captured_black": 0,
                "captured_white": 0,
                "position_hashes": set(),  # For superko
                "consecutive_passes": 0,
            },
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        """Get list of legal moves.

        Args:
            state: Current game state.

        Returns:
            List of legal action indices.

        """
        legal = []
        board_size = state.board_size
        board = state.board
        current_player = state.current_player

        # Check each intersection
        for i in range(board_size * board_size):
            row = i // board_size
            col = i % board_size

            if board[row, col] == EMPTY:
                # Check if move is legal (not suicide, not superko)
                if self._is_legal_move(state, row, col, current_player):
                    legal.append(i)

        # Pass is always legal
        legal.append(board_size * board_size)

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
        """Apply move and return new state.

        Args:
            state: Current game state.
            action: Move index to apply.

        Returns:
            New GameState after move.

        Raises:
            ValueError: If move is illegal.

        """
        board_size = state.board_size
        new_board = state.board.copy()
        current_player = state.current_player
        metadata = state.metadata.copy()

        # Handle pass
        if action == board_size * board_size:
            metadata["consecutive_passes"] = metadata.get("consecutive_passes", 0) + 1
            return state.with_move(
                action=action,
                new_board=new_board,
                consecutive_passes=metadata["consecutive_passes"],
            )

        # Reset consecutive passes
        metadata["consecutive_passes"] = 0

        # Apply stone placement
        row = action // board_size
        col = action % board_size

        if new_board[row, col] != EMPTY:
            raise ValueError(f"Illegal move: position ({row}, {col}) is occupied")

        new_board[row, col] = current_player

        # Capture opponent stones
        opponent = -current_player
        captured = 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if new_board[nr, nc] == opponent:
                    group = self._get_group(new_board, nr, nc)
                    if not self._has_liberties(new_board, group):
                        captured += len(group)
                        for gr, gc in group:
                            new_board[gr, gc] = EMPTY

        # Update capture counts
        if current_player == BLACK:
            metadata["captured_white"] = metadata.get("captured_white", 0) + captured
        else:
            metadata["captured_black"] = metadata.get("captured_black", 0) + captured

        # Update position hash set for superko
        position_hash = hash(new_board.tobytes())
        position_hashes = metadata.get("position_hashes", set()).copy()
        position_hashes.add(position_hash)
        metadata["position_hashes"] = position_hashes

        return state.with_move(
            action=action,
            new_board=new_board,
            **metadata,
        )

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended.

        Game ends after two consecutive passes.

        Args:
            state: Current game state.

        Returns:
            True if game is over.

        """
        return state.metadata.get("consecutive_passes", 0) >= 2

    def get_result(self, state: GameState) -> GameResult:
        """Get game result from terminal state.

        Uses Chinese scoring (area scoring).

        Args:
            state: Terminal game state.

        Returns:
            GameResult with winner and scores.

        """
        board = state.board
        board_size = state.board_size
        komi = state.metadata.get("komi", self.komi)

        # Count territory and stones
        black_score = 0
        white_score = komi

        counted = np.zeros_like(board, dtype=bool)

        for row in range(board_size):
            for col in range(board_size):
                if counted[row, col]:
                    continue

                if board[row, col] == BLACK:
                    black_score += 1
                    counted[row, col] = True
                elif board[row, col] == WHITE:
                    white_score += 1
                    counted[row, col] = True
                else:
                    # Empty - determine territory owner
                    territory, owner = self._get_territory(board, row, col, counted)
                    if owner == BLACK:
                        black_score += territory
                    elif owner == WHITE:
                        white_score += territory

        # Determine winner
        if black_score > white_score:
            winner = BLACK
        elif white_score > black_score:
            winner = WHITE
        else:
            winner = None  # Draw (rare)

        return GameResult(
            winner=winner,
            score_black=black_score,
            score_white=white_score,
            reason="score",
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

        Uses 17 planes:
        - Planes 0-7: Current player's stones (history)
        - Planes 8-15: Opponent's stones (history)
        - Plane 16: Color to play (all 1s if black, all 0s if white)

        Args:
            state: Game state to encode.

        Returns:
            Tensor of shape (17, board_size, board_size).

        """
        board_size = state.board_size
        tensor = torch.zeros(17, board_size, board_size)

        # Current board position for planes 0 and 8
        board = state.board

        if state.current_player == BLACK:
            tensor[0] = torch.from_numpy((board == BLACK).astype(np.float32))
            tensor[8] = torch.from_numpy((board == WHITE).astype(np.float32))
        else:
            tensor[0] = torch.from_numpy((board == WHITE).astype(np.float32))
            tensor[8] = torch.from_numpy((board == BLACK).astype(np.float32))

        # Fill remaining history planes with current position
        # (simplified - full implementation would track history)
        for i in range(1, 8):
            tensor[i] = tensor[0]
            tensor[8 + i] = tensor[8]

        # Color to play plane
        if state.current_player == BLACK:
            tensor[16] = 1.0
        else:
            tensor[16] = 0.0

        return tensor

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        """Get 8-fold symmetries (rotations and reflections).

        Args:
            state: Game state.
            policy: Policy distribution over actions.

        Returns:
            List of 8 (state, policy) pairs representing symmetries.

        """
        board_size = state.board_size
        symmetries = []

        if isinstance(policy, Tensor):
            policy_np = policy.cpu().numpy()
            is_tensor = True
        else:
            policy_np = policy
            is_tensor = False

        # Separate pass move probability
        pass_prob = policy_np[-1] if len(policy_np) > board_size * board_size else 0
        board_policy = policy_np[: board_size * board_size].reshape(board_size, board_size)

        board = state.board

        for rotation in range(4):
            for reflection in [False, True]:
                # Transform board
                transformed_board = np.rot90(board, rotation)
                transformed_policy = np.rot90(board_policy, rotation)

                if reflection:
                    transformed_board = np.fliplr(transformed_board)
                    transformed_policy = np.fliplr(transformed_policy)

                # Create new state
                new_state = GameState(
                    board=transformed_board.copy(),
                    current_player=state.current_player,
                    move_number=state.move_number,
                    move_history=[],  # History not meaningful after transform
                    metadata=state.metadata.copy(),
                )

                # Reconstruct policy vector
                new_policy = np.concatenate(
                    [
                        transformed_policy.flatten(),
                        [pass_prob],
                    ]
                )

                if is_tensor:
                    new_policy = torch.from_numpy(new_policy)

                symmetries.append((new_state, new_policy))

        return symmetries

    def get_phase(self, state: GameState) -> GamePhase:
        """Get current game phase.

        Args:
            state: Current game state.

        Returns:
            Current GamePhase.

        """
        if self.is_terminal(state):
            return GamePhase.TERMINAL

        board_size = state.board_size
        total_moves = board_size * board_size

        # Heuristic based on move count and typical game length
        if state.move_number < total_moves * 0.1:
            return GamePhase.OPENING
        elif state.move_number < total_moves * 0.7:
            return GamePhase.MIDGAME
        else:
            return GamePhase.ENDGAME

    def _is_legal_move(
        self,
        state: GameState,
        row: int,
        col: int,
        player: int,
    ) -> bool:
        """Check if a move is legal.

        Args:
            state: Current state.
            row: Row to place stone.
            col: Column to place stone.
            player: Player making the move.

        Returns:
            True if move is legal.

        """
        board = state.board.copy()
        board_size = state.board_size

        # Place stone temporarily
        board[row, col] = player

        # Check for captures
        opponent = -player
        has_capture = False

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if board[nr, nc] == opponent:
                    group = self._get_group(board, nr, nc)
                    if not self._has_liberties(board, group):
                        has_capture = True
                        for gr, gc in group:
                            board[gr, gc] = EMPTY

        # Check for suicide
        own_group = self._get_group(board, row, col)
        if not has_capture and not self._has_liberties(board, own_group):
            return False  # Suicide

        # Check for superko
        if self.superko:
            position_hash = hash(board.tobytes())
            if position_hash in state.metadata.get("position_hashes", set()):
                return False  # Superko violation

        return True

    def _get_group(
        self,
        board: np.ndarray,
        row: int,
        col: int,
    ) -> set[tuple[int, int]]:
        """Get all stones in a group.

        Args:
            board: Board state.
            row: Starting row.
            col: Starting column.

        Returns:
            Set of (row, col) positions in the group.

        """
        color = board[row, col]
        if color == EMPTY:
            return set()

        board_size = board.shape[0]
        group = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            if not (0 <= r < board_size and 0 <= c < board_size):
                continue
            if board[r, c] != color:
                continue

            group.add((r, c))
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

        return group

    def _has_liberties(
        self,
        board: np.ndarray,
        group: set[tuple[int, int]],
    ) -> bool:
        """Check if a group has any liberties.

        Args:
            board: Board state.
            group: Set of positions in the group.

        Returns:
            True if group has at least one liberty.

        """
        board_size = board.shape[0]

        for row, col in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if board[nr, nc] == EMPTY:
                        return True

        return False

    def _get_territory(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        counted: np.ndarray,
    ) -> tuple[int, int]:
        """Flood-fill to determine territory ownership.

        Args:
            board: Board state.
            row: Starting row.
            col: Starting column.
            counted: Array tracking counted positions.

        Returns:
            Tuple of (territory_size, owner).

        """
        board_size = board.shape[0]
        territory = set()
        stack = [(row, col)]
        borders = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in territory:
                continue
            if not (0 <= r < board_size and 0 <= c < board_size):
                continue
            if counted[r, c]:
                continue

            if board[r, c] == EMPTY:
                territory.add((r, c))
                counted[r, c] = True
                stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
            else:
                borders.add(board[r, c])

        # Determine owner
        if len(borders) == 1:
            owner = borders.pop()
        else:
            owner = EMPTY  # Neutral territory

        return len(territory), owner
