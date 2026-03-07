"""Chess game implementation for AlphaGalerkin.

This module provides a complete Chess implementation following the
GameInterface contract, enabling the continuous operator architecture
to be used for Chess and demonstrating multi-game support.

Features:
    - Full standard chess rules
    - Castling (kingside and queenside)
    - En passant captures
    - Pawn promotion (to Queen, Rook, Bishop, Knight)
    - Check and checkmate detection
    - Stalemate detection
    - 50-move rule
    - Threefold repetition
    - Insufficient material detection
    - AlphaZero-style neural network encoding (119 planes)
    - Horizontal symmetry for data augmentation

Action encoding follows AlphaZero's scheme:
    - 8x8 board with moves encoded as (from_square, move_type)
    - Action space: 8 * 8 * 73 = 4672 possible actions
    - Move types: queen moves (56), knight moves (8), underpromotions (9)
"""

from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from src.games.interface import GameInterface, GameResult
from src.games.registry import register_game
from src.games.state import ActionMask, GameState


# Piece encodings
class Piece(IntEnum):
    """Chess piece types."""

    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


# Color multipliers (positive for white, negative for black)
WHITE = 1
BLACK = -1

# Board dimensions
BOARD_SIZE = 8

# Move type encoding (following AlphaZero)
# Queen moves: 7 directions × 7 distances = 49 + 7 = 56 (N, NE, E, SE, S, SW, W, NW)
# Knight moves: 8 possible L-shapes
# Underpromotions: 3 pieces × 3 directions = 9 (to knight, bishop, rook on capture/straight)
NUM_MOVE_TYPES = 73
ACTION_SPACE_SIZE = BOARD_SIZE * BOARD_SIZE * NUM_MOVE_TYPES  # 4672

# Direction vectors (row_delta, col_delta)
DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "NE": (-1, 1),
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
}

# Knight move offsets
KNIGHT_MOVES: list[tuple[int, int]] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]

# Starting position (0=empty, positive=white, negative=black)
STARTING_POSITION: list[list[int]] = [
    [-4, -2, -3, -5, -6, -3, -2, -4],  # Black pieces (row 0)
    [-1, -1, -1, -1, -1, -1, -1, -1],  # Black pawns (row 1)
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [1, 1, 1, 1, 1, 1, 1, 1],  # White pawns (row 6)
    [4, 2, 3, 5, 6, 3, 2, 4],  # White pieces (row 7)
]


def _in_bounds(row: int, col: int) -> bool:
    """Check if position is on the board."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _get_piece_type(piece: int) -> int:
    """Get piece type (absolute value)."""
    return abs(piece)


def _get_piece_color(piece: int) -> int:
    """Get piece color (1=white, -1=black, 0=empty)."""
    if piece > 0:
        return WHITE
    elif piece < 0:
        return BLACK
    return 0


@register_game("chess")
class ChessGame(GameInterface):
    """Chess game implementation.

    Implements standard chess rules with all special moves
    (castling, en passant, promotion) and draw conditions.

    Action encoding:
        action = from_square * 73 + move_type
        - from_square: 0-63 (row * 8 + col)
        - move_type: 0-72 (see _encode_move / _decode_move)

    State encoding (119 planes for neural network):
        - Planes 0-5: Own piece bitboards (P, N, B, R, Q, K)
        - Planes 6-11: Opponent piece bitboards
        - Planes 12-17: Own piece history T-1
        - Planes 18-23: Opponent piece history T-1
        - ... (8 history positions total)
        - Planes 104-111: Remaining castling rights (4 binary planes)
        - Plane 112: En passant square (or zeros)
        - Plane 113: Halfmove clock (normalized)
        - Plane 114-118: Total move count encoding

    """

    name: ClassVar[str] = "chess"
    description: ClassVar[str] = "Standard chess"
    min_board_size: ClassVar[int] = 8
    max_board_size: ClassVar[int] = 8
    default_board_size: ClassVar[int] = 8

    def __init__(self) -> None:
        """Initialize Chess game."""
        self._board_size = BOARD_SIZE

    @property
    def action_space_size(self) -> int:
        """Get total action space size.

        Returns:
            4672 (8×8×73 possible move encodings).

        """
        return ACTION_SPACE_SIZE

    @property
    def state_channels(self) -> int:
        """Get number of input channels for neural network.

        Uses 119 planes following AlphaZero's encoding:
        - 12 planes per timestep (6 piece types × 2 colors)
        - 8 timesteps of history (96 planes total)
        - Plus auxiliary features (castling, en passant, etc.)

        Returns:
            119 feature planes.

        """
        return 119

    def initial_state(self, board_size: int | None = None) -> GameState:
        """Create initial chess position.

        Args:
            board_size: Ignored for chess (always 8x8).

        Returns:
            Initial GameState with starting position.

        """
        board = np.array(STARTING_POSITION, dtype=np.int8)

        return GameState(
            board=board,
            current_player=WHITE,
            move_number=0,
            move_history=[],
            metadata={
                "castling_rights": {
                    "K": True,  # White kingside
                    "Q": True,  # White queenside
                    "k": True,  # Black kingside
                    "q": True,  # Black queenside
                },
                "en_passant_square": None,  # (row, col) if available
                "halfmove_clock": 0,  # For 50-move rule
                "position_history": [],  # For threefold repetition
                "board_history": [],  # For neural network encoding
            },
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        """Get list of all legal moves.

        Args:
            state: Current game state.

        Returns:
            List of legal action indices.

        """
        legal_moves = []
        board = state.board
        current_player = state.current_player

        # Find all pieces of current player
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row, col]
                if _get_piece_color(piece) != current_player:
                    continue

                piece_type = _get_piece_type(piece)

                # Generate moves based on piece type
                if piece_type == Piece.PAWN:
                    moves = self._get_pawn_moves(state, row, col, current_player)
                elif piece_type == Piece.KNIGHT:
                    moves = self._get_knight_moves(state, row, col, current_player)
                elif piece_type == Piece.BISHOP:
                    moves = self._get_sliding_moves(
                        state, row, col, current_player, ["NE", "SE", "SW", "NW"]
                    )
                elif piece_type == Piece.ROOK:
                    moves = self._get_sliding_moves(
                        state, row, col, current_player, ["N", "E", "S", "W"]
                    )
                elif piece_type == Piece.QUEEN:
                    moves = self._get_sliding_moves(
                        state, row, col, current_player, list(DIRECTIONS.keys())
                    )
                elif piece_type == Piece.KING:
                    moves = self._get_king_moves(state, row, col, current_player)
                else:
                    continue

                # Filter out moves that leave king in check
                for move in moves:
                    action = self._encode_move(row, col, move)
                    if action is not None and self._is_legal_after_check(state, row, col, move):
                        legal_moves.append(action)

        return legal_moves

    def _get_pawn_moves(
        self,
        state: GameState,
        row: int,
        col: int,
        player: int,
    ) -> list[tuple[int, int, int | None]]:
        """Get pawn moves: (to_row, to_col, promotion_piece or None)."""
        moves: list[tuple[int, int, int | None]] = []
        board = state.board
        direction = -1 if player == WHITE else 1
        start_row = 6 if player == WHITE else 1
        promotion_row = 0 if player == WHITE else 7

        # Single push
        to_row = row + direction
        if _in_bounds(to_row, col) and board[to_row, col] == 0:
            if to_row == promotion_row:
                # Promotion
                for promo_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                    moves.append((to_row, col, promo_piece))
            else:
                moves.append((to_row, col, None))

                # Double push from starting position
                if row == start_row:
                    to_row_2 = row + 2 * direction
                    if board[to_row_2, col] == 0:
                        moves.append((to_row_2, col, None))

        # Captures (including en passant)
        for dc in [-1, 1]:
            to_col = col + dc
            if not _in_bounds(to_row, to_col):
                continue

            target = board[to_row, to_col]
            is_enemy = target != 0 and _get_piece_color(target) != player
            is_en_passant = state.metadata.get("en_passant_square") == (to_row, to_col)

            if is_enemy or is_en_passant:
                if to_row == promotion_row:
                    for promo_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                        moves.append((to_row, to_col, promo_piece))
                else:
                    moves.append((to_row, to_col, None))

        return moves

    def _get_knight_moves(
        self,
        state: GameState,
        row: int,
        col: int,
        player: int,
    ) -> list[tuple[int, int, int | None]]:
        """Get knight moves."""
        moves: list[tuple[int, int, int | None]] = []
        board = state.board

        for dr, dc in KNIGHT_MOVES:
            to_row, to_col = row + dr, col + dc
            if not _in_bounds(to_row, to_col):
                continue

            target = board[to_row, to_col]
            if target == 0 or _get_piece_color(target) != player:
                moves.append((to_row, to_col, None))

        return moves

    def _get_sliding_moves(
        self,
        state: GameState,
        row: int,
        col: int,
        player: int,
        directions: list[str],
    ) -> list[tuple[int, int, int | None]]:
        """Get sliding piece moves (bishop, rook, queen)."""
        moves: list[tuple[int, int, int | None]] = []
        board = state.board

        for dir_name in directions:
            dr, dc = DIRECTIONS[dir_name]
            for dist in range(1, 8):
                to_row, to_col = row + dr * dist, col + dc * dist
                if not _in_bounds(to_row, to_col):
                    break

                target = board[to_row, to_col]
                if target == 0:
                    moves.append((to_row, to_col, None))
                elif _get_piece_color(target) != player:
                    moves.append((to_row, to_col, None))
                    break
                else:
                    break  # Own piece blocking

        return moves

    def _get_king_moves(
        self,
        state: GameState,
        row: int,
        col: int,
        player: int,
    ) -> list[tuple[int, int, int | None]]:
        """Get king moves including castling."""
        moves: list[tuple[int, int, int | None]] = []
        board = state.board

        # Normal king moves (one square in any direction)
        for dr, dc in DIRECTIONS.values():
            to_row, to_col = row + dr, col + dc
            if not _in_bounds(to_row, to_col):
                continue

            target = board[to_row, to_col]
            if target == 0 or _get_piece_color(target) != player:
                moves.append((to_row, to_col, None))

        # Castling
        if not self._is_in_check(state, player):
            castling = state.metadata.get("castling_rights", {})

            # Kingside castling
            k_key = "K" if player == WHITE else "k"
            if castling.get(k_key) and self._can_castle_kingside(state, player):
                to_col = col + 2  # King moves 2 squares right
                moves.append((row, to_col, None))

            # Queenside castling
            q_key = "Q" if player == WHITE else "q"
            if castling.get(q_key) and self._can_castle_queenside(state, player):
                to_col = col - 2  # King moves 2 squares left
                moves.append((row, to_col, None))

        return moves

    def _can_castle_kingside(self, state: GameState, player: int) -> bool:
        """Check if kingside castling is legal."""
        board = state.board
        row = 7 if player == WHITE else 0

        # Check squares between king and rook are empty
        if board[row, 5] != 0 or board[row, 6] != 0:
            return False

        # Check king doesn't pass through check
        for col in [5, 6]:
            test_board = board.copy()
            test_board[row, 4] = 0
            test_board[row, col] = Piece.KING * player
            test_state = GameState(
                board=test_board,
                current_player=player,
                move_number=state.move_number,
                move_history=state.move_history.copy(),
                metadata=state.metadata.copy(),
            )
            if self._is_in_check(test_state, player):
                return False

        return True

    def _can_castle_queenside(self, state: GameState, player: int) -> bool:
        """Check if queenside castling is legal."""
        board = state.board
        row = 7 if player == WHITE else 0

        # Check squares between king and rook are empty
        if board[row, 1] != 0 or board[row, 2] != 0 or board[row, 3] != 0:
            return False

        # Check king doesn't pass through check
        for col in [2, 3]:
            test_board = board.copy()
            test_board[row, 4] = 0
            test_board[row, col] = Piece.KING * player
            test_state = GameState(
                board=test_board,
                current_player=player,
                move_number=state.move_number,
                move_history=state.move_history.copy(),
                metadata=state.metadata.copy(),
            )
            if self._is_in_check(test_state, player):
                return False

        return True

    def _is_in_check(self, state: GameState, player: int) -> bool:
        """Check if the given player's king is in check."""
        board = state.board
        king_pos = None

        # Find the king
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row, col] == Piece.KING * player:
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if king_pos is None:
            return True  # No king = in check (shouldn't happen in valid game)

        king_row, king_col = king_pos
        opponent = -player

        # Check for attacks from each piece type
        # Knight attacks
        for dr, dc in KNIGHT_MOVES:
            r, c = king_row + dr, king_col + dc
            if _in_bounds(r, c) and board[r, c] == Piece.KNIGHT * opponent:
                return True

        # Sliding attacks (bishop/queen diagonals, rook/queen orthogonals)
        for dir_name, (dr, dc) in DIRECTIONS.items():
            for dist in range(1, 8):
                r, c = king_row + dr * dist, king_col + dc * dist
                if not _in_bounds(r, c):
                    break

                piece = board[r, c]
                if piece == 0:
                    continue

                if _get_piece_color(piece) == opponent:
                    piece_type = _get_piece_type(piece)
                    is_diagonal = dir_name in ["NE", "SE", "SW", "NW"]
                    is_orthogonal = dir_name in ["N", "E", "S", "W"]

                    if piece_type == Piece.QUEEN:
                        return True
                    if piece_type == Piece.BISHOP and is_diagonal:
                        return True
                    if piece_type == Piece.ROOK and is_orthogonal:
                        return True
                break  # Blocked by any piece

        # Pawn attacks
        pawn_dir = -1 if player == WHITE else 1
        for dc in [-1, 1]:
            r, c = king_row + pawn_dir, king_col + dc
            if _in_bounds(r, c) and board[r, c] == Piece.PAWN * opponent:
                return True

        # King attacks (for preventing kings from being adjacent)
        for dr, dc in DIRECTIONS.values():
            r, c = king_row + dr, king_col + dc
            if _in_bounds(r, c) and board[r, c] == Piece.KING * opponent:
                return True

        return False

    def _is_legal_after_check(
        self,
        state: GameState,
        from_row: int,
        from_col: int,
        move: tuple[int, int, int | None],
    ) -> bool:
        """Check if making this move leaves the king safe."""
        to_row, to_col, promotion = move
        board = state.board.copy()
        player = state.current_player

        # Make the move on a copy
        piece = board[from_row, from_col]
        board[from_row, from_col] = 0

        # Handle en passant capture
        if _get_piece_type(piece) == Piece.PAWN:
            ep_square = state.metadata.get("en_passant_square")
            if ep_square == (to_row, to_col):
                # Remove captured pawn
                capture_row = from_row  # Pawn is on the same row as attacker
                board[capture_row, to_col] = 0

        # Handle promotion
        if promotion:
            board[to_row, to_col] = promotion * player
        else:
            board[to_row, to_col] = piece

        # Handle castling rook movement
        if _get_piece_type(piece) == Piece.KING and abs(to_col - from_col) == 2:
            if to_col > from_col:  # Kingside
                board[from_row, 7] = 0
                board[from_row, 5] = Piece.ROOK * player
            else:  # Queenside
                board[from_row, 0] = 0
                board[from_row, 3] = Piece.ROOK * player

        # Check if king is in check
        test_state = GameState(
            board=board,
            current_player=player,
            move_number=state.move_number,
            move_history=state.move_history.copy(),
            metadata=state.metadata.copy(),
        )
        return not self._is_in_check(test_state, player)

    def _encode_move(
        self,
        from_row: int,
        from_col: int,
        move: tuple[int, int, int | None],
    ) -> int | None:
        """Encode a move as an action index.

        Action encoding: from_square * 73 + move_type
        Move types (0-72):
            0-55: Queen-like moves (7 directions × 8 max squares, but 56 used)
            56-63: Knight moves (8 L-shapes)
            64-72: Underpromotions (9 types)
        """
        to_row, to_col, promotion = move
        from_square = from_row * BOARD_SIZE + from_col

        dr = to_row - from_row
        dc = to_col - from_col

        # Knight moves
        if (dr, dc) in KNIGHT_MOVES:
            knight_idx = KNIGHT_MOVES.index((dr, dc))
            move_type = 56 + knight_idx
            return from_square * NUM_MOVE_TYPES + move_type

        # Underpromotions (knight, bishop, rook - queen is default)
        if promotion is not None and promotion != Piece.QUEEN:
            # Direction: straight (0), left capture (-1), right capture (+1)
            if dc == 0:
                dir_idx = 0
            elif dc == -1:
                dir_idx = 1
            else:
                dir_idx = 2

            # Piece type index: knight=0, bishop=1, rook=2
            piece_idx = {Piece.KNIGHT: 0, Piece.BISHOP: 1, Piece.ROOK: 2}[promotion]
            move_type = 64 + piece_idx * 3 + dir_idx
            return from_square * NUM_MOVE_TYPES + move_type

        # Queen-like moves (sliding + king moves + queen promotions)
        if dr == 0 and dc == 0:
            return None  # Invalid move

        # Determine direction index
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        direction = None
        if dr < 0 and dc == 0:
            direction = "N"
        elif dr < 0 and dc > 0:
            direction = "NE"
        elif dr == 0 and dc > 0:
            direction = "E"
        elif dr > 0 and dc > 0:
            direction = "SE"
        elif dr > 0 and dc == 0:
            direction = "S"
        elif dr > 0 and dc < 0:
            direction = "SW"
        elif dr == 0 and dc < 0:
            direction = "W"
        elif dr < 0 and dc < 0:
            direction = "NW"

        if direction is None:
            return None

        dir_idx = directions.index(direction)
        distance = max(abs(dr), abs(dc))
        if distance > 7:
            return None

        # Move type = direction * 7 + (distance - 1)
        move_type = dir_idx * 7 + (distance - 1)
        return from_square * NUM_MOVE_TYPES + move_type

    def _decode_move(self, action: int) -> tuple[int, int, int, int, int | None]:
        """Decode action index to (from_row, from_col, to_row, to_col, promotion).

        Returns tuple of (from_row, from_col, to_row, to_col, promotion_piece).
        """
        from_square = action // NUM_MOVE_TYPES
        move_type = action % NUM_MOVE_TYPES
        from_row = from_square // BOARD_SIZE
        from_col = from_square % BOARD_SIZE

        promotion = None

        if move_type < 56:
            # Queen-like move
            dir_idx = move_type // 7
            distance = (move_type % 7) + 1
            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            dr, dc = DIRECTIONS[directions[dir_idx]]
            to_row = from_row + dr * distance
            to_col = from_col + dc * distance

        elif move_type < 64:
            # Knight move
            knight_idx = move_type - 56
            dr, dc = KNIGHT_MOVES[knight_idx]
            to_row = from_row + dr
            to_col = from_col + dc

        else:
            # Underpromotion
            promo_idx = move_type - 64
            piece_idx = promo_idx // 3
            dir_idx = promo_idx % 3

            promotion = [Piece.KNIGHT, Piece.BISHOP, Piece.ROOK][piece_idx]
            dc = [-1, 0, 1][dir_idx]  # Left, straight, right
            dr = -1 if from_row > 0 else 1  # Determine pawn direction
            to_row = from_row + dr
            to_col = from_col + dc

        return from_row, from_col, to_row, to_col, promotion

    def get_action_mask(self, state: GameState) -> ActionMask:
        """Get action mask for legal moves.

        Args:
            state: Current game state.

        Returns:
            ActionMask with legal moves marked True.

        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        legal_actions = self.get_legal_actions(state)

        for action in legal_actions:
            mask[action] = True

        return ActionMask(mask=mask, action_space_size=self.action_space_size)

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
        from_row, from_col, to_row, to_col, promotion = self._decode_move(action)

        if not _in_bounds(to_row, to_col):
            raise ValueError(f"Invalid move: target out of bounds ({to_row}, {to_col})")

        board = state.board.copy()
        piece = board[from_row, from_col]
        player = state.current_player
        metadata = state.metadata.copy()
        piece_type = _get_piece_type(piece)

        # Track if this is a capture or pawn move (for 50-move rule)
        is_capture = board[to_row, to_col] != 0
        is_pawn_move = piece_type == Piece.PAWN

        # Handle en passant capture
        if piece_type == Piece.PAWN:
            ep_square = metadata.get("en_passant_square")
            if ep_square == (to_row, to_col):
                # Remove captured pawn
                capture_row = from_row
                board[capture_row, to_col] = 0
                is_capture = True

        # Clear source square
        board[from_row, from_col] = 0

        # Place piece (with promotion if applicable)
        if promotion:
            board[to_row, to_col] = promotion * player
        else:
            board[to_row, to_col] = piece

        # Handle castling
        if piece_type == Piece.KING and abs(to_col - from_col) == 2:
            if to_col > from_col:  # Kingside
                board[from_row, 7] = 0
                board[from_row, 5] = Piece.ROOK * player
            else:  # Queenside
                board[from_row, 0] = 0
                board[from_row, 3] = Piece.ROOK * player

        # Update castling rights
        castling = metadata.get("castling_rights", {}).copy()
        if piece_type == Piece.KING:
            if player == WHITE:
                castling["K"] = False
                castling["Q"] = False
            else:
                castling["k"] = False
                castling["q"] = False
        elif piece_type == Piece.ROOK:
            if player == WHITE:
                if from_col == 7:
                    castling["K"] = False
                elif from_col == 0:
                    castling["Q"] = False
            else:
                if from_col == 7:
                    castling["k"] = False
                elif from_col == 0:
                    castling["q"] = False
        metadata["castling_rights"] = castling

        # Update en passant square
        if piece_type == Piece.PAWN and abs(to_row - from_row) == 2:
            metadata["en_passant_square"] = ((from_row + to_row) // 2, from_col)
        else:
            metadata["en_passant_square"] = None

        # Update halfmove clock
        if is_capture or is_pawn_move:
            metadata["halfmove_clock"] = 0
        else:
            metadata["halfmove_clock"] = metadata.get("halfmove_clock", 0) + 1

        # Update position history for threefold repetition
        position_hash = self._hash_position(board, castling, metadata.get("en_passant_square"))
        pos_history = metadata.get("position_history", []).copy()
        pos_history.append(position_hash)
        metadata["position_history"] = pos_history

        # Update board history for neural network encoding
        board_history = metadata.get("board_history", []).copy()
        board_history.append(state.board.copy())
        if len(board_history) > 8:
            board_history = board_history[-8:]
        metadata["board_history"] = board_history

        return state.with_move(
            action=action,
            new_board=board,
            **metadata,
        )

    def _hash_position(
        self,
        board: np.ndarray,
        castling: dict,
        ep_square: tuple[int, int] | None,
    ) -> str:
        """Create a hash for position comparison (threefold repetition)."""
        data = board.tobytes()
        data += str(sorted(castling.items())).encode()
        if ep_square:
            data += str(ep_square).encode()
        return hashlib.md5(data).hexdigest()

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended.

        Args:
            state: Current game state.

        Returns:
            True if game is over.

        """
        # No legal moves = checkmate or stalemate
        if len(self.get_legal_actions(state)) == 0:
            return True

        # 50-move rule
        if state.metadata.get("halfmove_clock", 0) >= 100:  # 50 moves = 100 half-moves
            return True

        # Threefold repetition
        pos_history = state.metadata.get("position_history", [])
        if pos_history:
            current_pos = pos_history[-1] if pos_history else None
            if current_pos and pos_history.count(current_pos) >= 3:
                return True

        # Insufficient material
        return bool(self._is_insufficient_material(state))

    def _is_insufficient_material(self, state: GameState) -> bool:
        """Check for insufficient material to mate."""
        board = state.board
        pieces = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row, col]
                if piece != 0:
                    pieces.append((_get_piece_type(piece), _get_piece_color(piece), (row, col)))

        # Extract piece counts
        white_pieces = [p for p, c, _ in pieces if c == WHITE]
        black_pieces = [p for p, c, _ in pieces if c == BLACK]

        # Only kings
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True

        # King + minor vs king
        if len(white_pieces) == 1 and len(black_pieces) == 2:
            if Piece.KNIGHT in black_pieces or Piece.BISHOP in black_pieces:
                return True
        if len(black_pieces) == 1 and len(white_pieces) == 2:
            if Piece.KNIGHT in white_pieces or Piece.BISHOP in white_pieces:
                return True

        # King + bishop vs king + bishop (same color bishops)
        # Simplified: don't check bishop colors
        if (
            len(white_pieces) == 2
            and len(black_pieces) == 2
            and Piece.BISHOP in white_pieces
            and Piece.BISHOP in black_pieces
        ):
            # Could check if bishops are same color, but simplified
            pass

        return False

    def get_result(self, state: GameState) -> GameResult:
        """Get game result from terminal state.

        Args:
            state: Terminal game state.

        Returns:
            GameResult with winner and reason.

        """
        move_count = state.move_number

        if not self.is_terminal(state):
            return GameResult(
                winner=None,
                reason="game_ongoing",
                score_white=0.0,
                score_black=0.0,
                move_count=move_count,
            )

        legal_moves = self.get_legal_actions(state)
        player = state.current_player

        # No legal moves
        if len(legal_moves) == 0:
            if self._is_in_check(state, player):
                # Checkmate - opponent wins
                return GameResult(
                    winner=-player,
                    reason="checkmate",
                    score_white=0.0 if player == WHITE else 1.0,
                    score_black=0.0 if player == BLACK else 1.0,
                    move_count=move_count,
                )
            else:
                # Stalemate
                return GameResult(
                    winner=0,
                    reason="stalemate",
                    score_white=0.5,
                    score_black=0.5,
                    move_count=move_count,
                )

        # 50-move rule
        if state.metadata.get("halfmove_clock", 0) >= 100:
            return GameResult(
                winner=0,
                reason="fifty_move_rule",
                score_white=0.5,
                score_black=0.5,
                move_count=move_count,
            )

        # Threefold repetition
        pos_history = state.metadata.get("position_history", [])
        if pos_history:
            current_pos = pos_history[-1]
            if pos_history.count(current_pos) >= 3:
                return GameResult(
                    winner=0,
                    reason="threefold_repetition",
                    score_white=0.5,
                    score_black=0.5,
                    move_count=move_count,
                )

        # Insufficient material
        if self._is_insufficient_material(state):
            return GameResult(
                winner=0,
                reason="insufficient_material",
                score_white=0.5,
                score_black=0.5,
                move_count=move_count,
            )

        return GameResult(
            winner=None,
            reason="unknown",
            score_white=0.0,
            score_black=0.0,
            move_count=move_count,
        )

    def get_winner(self, state: GameState) -> int | None:
        """Get winner from terminal state.

        Args:
            state: Game state.

        Returns:
            1 for white win, -1 for black win, None for draw or ongoing.

        """
        result = self.get_result(state)
        return result.winner

    def to_tensor(self, state: GameState) -> Tensor:
        """Convert state to neural network input tensor.

        Creates 119-plane encoding following AlphaZero:
        - 12 planes per timestep (6 piece types × 2 colors)
        - 8 history timesteps
        - Plus auxiliary features

        Args:
            state: Game state to encode.

        Returns:
            Tensor of shape (119, 8, 8).

        """
        planes = np.zeros((self.state_channels, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board = state.board
        player = state.current_player

        # Helper to add piece planes
        def add_piece_planes(b: np.ndarray, offset: int, color: int) -> None:
            for piece_type in range(1, 7):
                mask = (b == piece_type * color).astype(np.float32)
                planes[offset + piece_type - 1] = mask

        # Current position (planes 0-11)
        add_piece_planes(board, 0, player)  # Own pieces
        add_piece_planes(board, 6, -player)  # Opponent pieces

        # History positions (planes 12-95)
        board_history = state.metadata.get("board_history", [])
        for t, hist_board in enumerate(reversed(board_history[:7])):
            base_offset = 12 + t * 12
            add_piece_planes(hist_board, base_offset, player)
            add_piece_planes(hist_board, base_offset + 6, -player)

        # Castling rights (planes 96-99)
        castling = state.metadata.get("castling_rights", {})
        planes[96] = float(castling.get("K", False))
        planes[97] = float(castling.get("Q", False))
        planes[98] = float(castling.get("k", False))
        planes[99] = float(castling.get("q", False))

        # No-progress count / halfmove clock (plane 100)
        halfmove = state.metadata.get("halfmove_clock", 0)
        planes[100] = halfmove / 100.0

        # En passant square (plane 101)
        ep_square = state.metadata.get("en_passant_square")
        if ep_square:
            planes[101, ep_square[0], ep_square[1]] = 1.0

        # Current player (plane 102) - 1 if white, 0 if black
        planes[102] = 1.0 if player == WHITE else 0.0

        # Total move count (planes 103-118) - binary encoding
        move_num = state.move_number
        for i in range(16):
            planes[103 + i] = (move_num >> i) & 1

        return torch.from_numpy(planes)

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        """Get symmetric positions for data augmentation.

        Chess has only horizontal symmetry (unlike Go's 8-fold symmetry).

        Args:
            state: Game state.
            policy: Policy vector for this state.

        Returns:
            List of (state, policy) pairs including original.

        """
        result: list[tuple[GameState, np.ndarray | Tensor]] = [(state, policy)]

        # Horizontal flip
        flipped_board = np.flip(state.board, axis=1).copy()
        flipped_metadata = state.metadata.copy()

        # Flip castling rights (kingside <-> queenside)
        castling = flipped_metadata.get("castling_rights", {})
        flipped_castling = {
            "K": castling.get("Q", False),
            "Q": castling.get("K", False),
            "k": castling.get("q", False),
            "q": castling.get("k", False),
        }
        flipped_metadata["castling_rights"] = flipped_castling

        # Flip en passant square
        ep = flipped_metadata.get("en_passant_square")
        if ep:
            flipped_metadata["en_passant_square"] = (ep[0], BOARD_SIZE - 1 - ep[1])

        flipped_state = GameState(
            board=flipped_board,
            current_player=state.current_player,
            move_number=state.move_number,
            move_history=state.move_history.copy(),
            metadata=flipped_metadata,
        )

        # Flip policy
        policy_np = policy.cpu().numpy() if isinstance(policy, Tensor) else policy

        flipped_policy = self._flip_policy(policy_np)

        if isinstance(policy, Tensor):
            flipped_policy = torch.from_numpy(flipped_policy)

        result.append((flipped_state, flipped_policy))

        return result

    def _flip_policy(self, policy: np.ndarray) -> np.ndarray:
        """Flip policy horizontally."""
        flipped = np.zeros_like(policy)

        for action in range(len(policy)):
            from_row, from_col, to_row, to_col, promotion = self._decode_move(action)

            # Flip columns
            flipped_from_col = BOARD_SIZE - 1 - from_col
            flipped_to_col = BOARD_SIZE - 1 - to_col

            # Re-encode with flipped coordinates
            move = (to_row, flipped_to_col, promotion)
            flipped_action = self._encode_move(from_row, flipped_from_col, move)

            if flipped_action is not None and 0 <= flipped_action < len(policy):
                flipped[flipped_action] = policy[action]

        return flipped

    def action_to_string(self, action: int, state: GameState | None = None) -> str:
        """Convert action to algebraic notation.

        Args:
            action: Action index.
            state: Optional state for disambiguation.

        Returns:
            Move in algebraic notation (e.g., "e2e4", "e1g1" for castling).

        """
        from_row, from_col, to_row, to_col, promotion = self._decode_move(action)

        files = "abcdefgh"
        ranks = "87654321"

        from_sq = files[from_col] + ranks[from_row]
        to_sq = files[to_col] + ranks[to_row]
        move_str = from_sq + to_sq

        if promotion:
            promo_char = {Piece.QUEEN: "q", Piece.ROOK: "r", Piece.BISHOP: "b", Piece.KNIGHT: "n"}
            move_str += promo_char[promotion]

        return move_str

    def string_to_action(self, move_str: str, state: GameState) -> int | None:
        """Convert algebraic notation to action.

        Args:
            move_str: Move in algebraic notation (e.g., "e2e4").
            state: Current state for validation.

        Returns:
            Action index or None if invalid.

        """
        if len(move_str) < 4:
            return None

        files = "abcdefgh"
        ranks = "87654321"

        try:
            from_col = files.index(move_str[0])
            from_row = ranks.index(move_str[1])
            to_col = files.index(move_str[2])
            to_row = ranks.index(move_str[3])
        except ValueError:
            return None

        promotion = None
        if len(move_str) >= 5:
            promo_map = {"q": Piece.QUEEN, "r": Piece.ROOK, "b": Piece.BISHOP, "n": Piece.KNIGHT}
            promotion = promo_map.get(move_str[4].lower())

        move = (to_row, to_col, promotion)
        action = self._encode_move(from_row, from_col, move)

        # Validate action is legal
        if action is not None and action in self.get_legal_actions(state):
            return action

        return None
