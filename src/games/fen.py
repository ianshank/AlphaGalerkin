"""FEN (Forsyth-Edwards Notation) serialization for chess positions.

Provides bidirectional conversion between AlphaGalerkin's internal
GameState representation and standard FEN strings, enabling
communication with UCI engines and chess GUIs.

FEN Format: <placement> <active> <castling> <en_passant> <halfmove> <fullmove>
Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.games.chess import BLACK, BOARD_SIZE, WHITE, Piece
from src.games.state import GameState

if TYPE_CHECKING:
    pass

# Standard starting position FEN
STARTING_FEN: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Number of fields in a valid FEN string
FEN_FIELD_COUNT: int = 6

# Piece to FEN character mapping (white pieces uppercase, black lowercase)
_PIECE_TO_FEN_WHITE: dict[int, str] = {
    Piece.PAWN: "P",
    Piece.KNIGHT: "N",
    Piece.BISHOP: "B",
    Piece.ROOK: "R",
    Piece.QUEEN: "Q",
    Piece.KING: "K",
}

_PIECE_TO_FEN_BLACK: dict[int, str] = {
    Piece.PAWN: "p",
    Piece.KNIGHT: "n",
    Piece.BISHOP: "b",
    Piece.ROOK: "r",
    Piece.QUEEN: "q",
    Piece.KING: "k",
}

# FEN character to (piece_type, color) mapping
_FEN_TO_PIECE: dict[str, tuple[int, int]] = {
    "P": (Piece.PAWN, WHITE),
    "N": (Piece.KNIGHT, WHITE),
    "B": (Piece.BISHOP, WHITE),
    "R": (Piece.ROOK, WHITE),
    "Q": (Piece.QUEEN, WHITE),
    "K": (Piece.KING, WHITE),
    "p": (Piece.PAWN, BLACK),
    "n": (Piece.KNIGHT, BLACK),
    "b": (Piece.BISHOP, BLACK),
    "r": (Piece.ROOK, BLACK),
    "q": (Piece.QUEEN, BLACK),
    "k": (Piece.KING, BLACK),
}

# File letters for algebraic notation
_FILES: str = "abcdefgh"

# Rank numbers (row 0 = rank 8, row 7 = rank 1)
_RANKS: str = "87654321"


class FENError(ValueError):
    """Error raised for invalid FEN strings."""


def state_to_fen(state: GameState) -> str:
    """Serialize a chess GameState to FEN notation.

    Args:
        state: Chess game state with board array and metadata.

    Returns:
        FEN string representing the position.

    Raises:
        FENError: If state lacks required metadata fields.

    """
    board = state.board
    metadata = state.metadata

    # 1. Piece placement (from rank 8 to rank 1)
    placement = _board_to_placement(board)

    # 2. Active color
    active = "w" if state.current_player == WHITE else "b"

    # 3. Castling availability
    castling = _castling_to_fen(metadata.get("castling_rights", {}))

    # 4. En passant target square
    ep_square = metadata.get("en_passant_square")
    en_passant = _ep_to_fen(ep_square)

    # 5. Halfmove clock
    halfmove = str(metadata.get("halfmove_clock", 0))

    # 6. Fullmove number
    fullmove = str(state.move_number // 2 + 1)

    return f"{placement} {active} {castling} {en_passant} {halfmove} {fullmove}"


def fen_to_state(fen: str) -> GameState:
    """Parse a FEN string into a chess GameState.

    Args:
        fen: FEN string to parse.

    Returns:
        GameState with board, current_player, and chess metadata.

    Raises:
        FENError: If the FEN string is malformed or contains invalid data.

    """
    fields = fen.strip().split()
    if len(fields) != FEN_FIELD_COUNT:
        raise FENError(f"FEN must have {FEN_FIELD_COUNT} fields, got {len(fields)}: {fen!r}")

    placement, active, castling, en_passant, halfmove, fullmove = fields

    # 1. Parse piece placement
    board = _placement_to_board(placement)

    # 2. Parse active color
    if active == "w":
        current_player = WHITE
    elif active == "b":
        current_player = BLACK
    else:
        raise FENError(f"Invalid active color: {active!r}")

    # 3. Parse castling rights
    castling_rights = _fen_to_castling(castling)

    # 4. Parse en passant square
    ep_square = _fen_to_ep(en_passant)

    # 5. Parse halfmove clock
    try:
        halfmove_clock = int(halfmove)
    except ValueError as e:
        raise FENError(f"Invalid halfmove clock: {halfmove!r}") from e

    if halfmove_clock < 0:
        raise FENError(f"Halfmove clock cannot be negative: {halfmove_clock}")

    # 6. Parse fullmove number
    try:
        fullmove_number = int(fullmove)
    except ValueError as e:
        raise FENError(f"Invalid fullmove number: {fullmove!r}") from e

    if fullmove_number < 1:
        raise FENError(f"Fullmove number must be >= 1: {fullmove_number}")

    # Compute move_number from fullmove and active color
    # fullmove increments after black's move
    move_number = (fullmove_number - 1) * 2
    if current_player == BLACK:
        move_number += 1

    metadata: dict[str, Any] = {
        "castling_rights": castling_rights,
        "en_passant_square": ep_square,
        "halfmove_clock": halfmove_clock,
        "position_history": [],
        "board_history": [],
    }

    return GameState(
        board=board,
        current_player=current_player,
        move_number=move_number,
        move_history=[],
        metadata=metadata,
    )


def _board_to_placement(board: np.ndarray) -> str:
    """Convert 8x8 board array to FEN placement string.

    The board uses signed integers: positive for white, negative for black.
    The absolute value is the Piece enum value.

    Args:
        board: 8x8 numpy array of piece values.

    Returns:
        FEN placement string (e.g. "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").

    """
    ranks = []
    for row in range(BOARD_SIZE):
        rank_str = ""
        empty_count = 0

        for col in range(BOARD_SIZE):
            piece_val = int(board[row, col])

            if piece_val == Piece.EMPTY:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0

                piece_type = abs(piece_val)
                if piece_val > 0:
                    rank_str += _PIECE_TO_FEN_WHITE[piece_type]
                else:
                    rank_str += _PIECE_TO_FEN_BLACK[piece_type]

        if empty_count > 0:
            rank_str += str(empty_count)

        ranks.append(rank_str)

    return "/".join(ranks)


def _placement_to_board(placement: str) -> np.ndarray:
    """Parse FEN placement string into 8x8 board array.

    Args:
        placement: FEN placement field (e.g. "rnbqkbnr/pppppppp/...").

    Returns:
        8x8 numpy int8 array.

    Raises:
        FENError: If the placement string is malformed.

    """
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    rank_strs = placement.split("/")

    if len(rank_strs) != BOARD_SIZE:
        raise FENError(f"FEN placement must have {BOARD_SIZE} ranks, got {len(rank_strs)}")

    for row, rank_str in enumerate(rank_strs):
        col = 0
        for char in rank_str:
            if char.isdigit():
                skip = int(char)
                if skip < 1 or skip > BOARD_SIZE:
                    raise FENError(f"Invalid empty square count: {char}")
                col += skip
            elif char in _FEN_TO_PIECE:
                if col >= BOARD_SIZE:
                    raise FENError(f"Too many squares in rank {BOARD_SIZE - row}: {rank_str!r}")
                piece_type, color = _FEN_TO_PIECE[char]
                board[row, col] = piece_type * color
                col += 1
            else:
                raise FENError(f"Invalid character in placement: {char!r}")

        if col != BOARD_SIZE:
            raise FENError(f"Rank {BOARD_SIZE - row} has {col} squares, expected {BOARD_SIZE}")

    return board


def _castling_to_fen(castling_rights: dict[str, bool]) -> str:
    """Convert castling rights dict to FEN castling field.

    Args:
        castling_rights: Dict with keys "K", "Q", "k", "q" mapping to booleans.

    Returns:
        FEN castling string (e.g. "KQkq", "Kq", "-").

    """
    result = ""
    for right in ("K", "Q", "k", "q"):
        if castling_rights.get(right, False):
            result += right

    return result if result else "-"


def _fen_to_castling(castling_str: str) -> dict[str, bool]:
    """Parse FEN castling field into rights dict.

    Args:
        castling_str: FEN castling field (e.g. "KQkq", "-").

    Returns:
        Dict with "K", "Q", "k", "q" keys.

    Raises:
        FENError: If the castling string contains invalid characters.

    """
    valid_chars = set("KQkq-")
    if not all(c in valid_chars for c in castling_str):
        invalid = [c for c in castling_str if c not in valid_chars]
        raise FENError(f"Invalid castling characters: {invalid}")

    if castling_str == "-":
        return {"K": False, "Q": False, "k": False, "q": False}

    return {
        "K": "K" in castling_str,
        "Q": "Q" in castling_str,
        "k": "k" in castling_str,
        "q": "q" in castling_str,
    }


def _ep_to_fen(ep_square: tuple[int, int] | None) -> str:
    """Convert en passant square tuple to FEN notation.

    Args:
        ep_square: (row, col) tuple or None.

    Returns:
        Algebraic square string (e.g. "e3") or "-".

    """
    if ep_square is None:
        return "-"

    row, col = ep_square
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return _FILES[col] + _RANKS[row]

    return "-"


def _fen_to_ep(ep_str: str) -> tuple[int, int] | None:
    """Parse FEN en passant field into (row, col) tuple.

    Args:
        ep_str: Algebraic square (e.g. "e3") or "-".

    Returns:
        (row, col) tuple or None.

    Raises:
        FENError: If the en passant string is invalid.

    """
    if ep_str == "-":
        return None

    if len(ep_str) != 2:
        raise FENError(f"Invalid en passant square: {ep_str!r}")

    file_char, rank_char = ep_str[0], ep_str[1]

    if file_char not in _FILES:
        raise FENError(f"Invalid en passant file: {file_char!r}")

    if rank_char not in _RANKS:
        raise FENError(f"Invalid en passant rank: {rank_char!r}")

    col = _FILES.index(file_char)
    row = _RANKS.index(rank_char)

    return (row, col)
