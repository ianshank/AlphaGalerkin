"""Tests for FEN serialization/deserialization.

Tests roundtrip conversion, known positions, edge cases,
and invalid input handling for the FEN converter.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.chess import BLACK, BOARD_SIZE, WHITE, ChessGame, Piece
from src.games.fen import (
    STARTING_FEN,
    FENError,
    fen_to_state,
    state_to_fen,
)
from src.games.state import GameState


class TestStartingPosition:
    """Tests for the standard starting position."""

    @pytest.fixture
    def game(self) -> ChessGame:
        return ChessGame()

    def test_starting_fen_constant(self) -> None:
        assert STARTING_FEN == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_state_to_fen_initial(self, game: ChessGame) -> None:
        state = game.initial_state()
        fen = state_to_fen(state)
        assert fen == STARTING_FEN

    def test_fen_to_state_initial(self) -> None:
        state = fen_to_state(STARTING_FEN)
        assert state.current_player == WHITE
        assert state.move_number == 0
        assert state.metadata["castling_rights"] == {
            "K": True,
            "Q": True,
            "k": True,
            "q": True,
        }
        assert state.metadata["en_passant_square"] is None
        assert state.metadata["halfmove_clock"] == 0

    def test_initial_board_pieces(self) -> None:
        state = fen_to_state(STARTING_FEN)
        board = state.board

        # White pieces on ranks 1-2 (rows 7-6)
        assert board[7, 0] == Piece.ROOK * WHITE
        assert board[7, 1] == Piece.KNIGHT * WHITE
        assert board[7, 4] == Piece.KING * WHITE
        assert board[6, 3] == Piece.PAWN * WHITE

        # Black pieces on ranks 7-8 (rows 1-0)
        assert board[0, 0] == Piece.ROOK * BLACK
        assert board[0, 4] == Piece.KING * BLACK
        assert board[1, 4] == Piece.PAWN * BLACK

    def test_roundtrip_initial(self, game: ChessGame) -> None:
        state = game.initial_state()
        fen = state_to_fen(state)
        restored = fen_to_state(fen)
        assert np.array_equal(state.board, restored.board)
        assert state.current_player == restored.current_player


class TestKnownPositions:
    """Tests for well-known FEN positions."""

    @pytest.mark.parametrize(
        "fen,description",
        [
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "After 1.e4",
            ),
            (
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
                "Sicilian Defense",
            ),
            (
                "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
                "After 1.e4 Nc6",
            ),
            (
                "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
                "Alekhine Defense",
            ),
            (
                "8/8/8/8/8/8/8/4K2k w - - 0 1",
                "King vs King endgame",
            ),
            (
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
                "Both sides can castle",
            ),
            (
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1",
                "Partial castling rights",
            ),
            (
                "8/8/8/8/8/8/8/4K2k b - - 99 50",
                "Near 50-move rule",
            ),
        ],
    )
    def test_roundtrip_known_position(self, fen: str, description: str) -> None:
        state = fen_to_state(fen)
        restored_fen = state_to_fen(state)
        assert restored_fen == fen, f"Roundtrip failed for {description}"


class TestCastlingRights:
    """Tests for castling rights serialization."""

    def test_all_castling(self) -> None:
        state = fen_to_state("8/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        rights = state.metadata["castling_rights"]
        assert rights == {"K": True, "Q": True, "k": True, "q": True}

    def test_no_castling(self) -> None:
        state = fen_to_state("8/8/8/8/8/8/8/R3K2R w - - 0 1")
        rights = state.metadata["castling_rights"]
        assert rights == {"K": False, "Q": False, "k": False, "q": False}

    def test_partial_castling(self) -> None:
        state = fen_to_state("8/8/8/8/8/8/8/R3K2R w Kq - 0 1")
        rights = state.metadata["castling_rights"]
        assert rights["K"] is True
        assert rights["Q"] is False
        assert rights["k"] is False
        assert rights["q"] is True

    def test_castling_roundtrip(self) -> None:
        # Standard FEN castling order is always KQkq
        for castling_str in ["KQkq", "Kq", "Qk", "-", "K", "q"]:
            fen = f"8/8/8/8/8/8/8/4K2k w {castling_str} - 0 1"
            state = fen_to_state(fen)
            assert state_to_fen(state) == fen

    def test_castling_nonstandard_order_normalized(self) -> None:
        # Non-standard order like "kQ" should be normalized to "Qk" on roundtrip
        state = fen_to_state("8/8/8/8/8/8/8/4K2k w kQ - 0 1")
        fen = state_to_fen(state)
        assert "Qk" in fen  # Normalized to KQkq ordering


class TestEnPassant:
    """Tests for en passant square serialization."""

    def test_no_en_passant(self) -> None:
        state = fen_to_state(STARTING_FEN)
        assert state.metadata["en_passant_square"] is None

    def test_en_passant_e3(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        state = fen_to_state(fen)
        ep = state.metadata["en_passant_square"]
        assert ep is not None
        # e3 = file e (col 4), rank 3 (row 5)
        assert ep == (5, 4)

    def test_en_passant_c6(self) -> None:
        fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
        state = fen_to_state(fen)
        ep = state.metadata["en_passant_square"]
        assert ep is not None
        # c6 = file c (col 2), rank 6 (row 2)
        assert ep == (2, 2)


class TestMoveNumber:
    """Tests for move number and halfmove clock."""

    def test_initial_move_number(self) -> None:
        state = fen_to_state(STARTING_FEN)
        assert state.move_number == 0

    def test_black_to_move_number(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        state = fen_to_state(fen)
        assert state.move_number == 1  # (1-1)*2 + 1 = 1

    def test_fullmove_10_white(self) -> None:
        fen = "8/8/8/8/8/8/8/4K2k w - - 0 10"
        state = fen_to_state(fen)
        assert state.move_number == 18  # (10-1)*2 = 18

    def test_fullmove_10_black(self) -> None:
        fen = "8/8/8/8/8/8/8/4K2k b - - 0 10"
        state = fen_to_state(fen)
        assert state.move_number == 19  # (10-1)*2 + 1 = 19

    def test_halfmove_clock(self) -> None:
        fen = "8/8/8/8/8/8/8/4K2k w - - 42 10"
        state = fen_to_state(fen)
        assert state.metadata["halfmove_clock"] == 42


class TestInvalidFEN:
    """Tests for error handling of malformed FEN strings."""

    def test_too_few_fields(self) -> None:
        with pytest.raises(FENError, match="must have 6 fields"):
            fen_to_state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")

    def test_too_many_fields(self) -> None:
        with pytest.raises(FENError, match="must have 6 fields"):
            fen_to_state("8/8/8/8/8/8/8/4K2k w - - 0 1 extra")

    def test_invalid_active_color(self) -> None:
        with pytest.raises(FENError, match="Invalid active color"):
            fen_to_state("8/8/8/8/8/8/8/4K2k x - - 0 1")

    def test_invalid_piece_char(self) -> None:
        with pytest.raises(FENError, match="Invalid character"):
            fen_to_state("8/8/8/8/8/8/8/4X2k w - - 0 1")

    def test_too_many_squares_in_rank(self) -> None:
        with pytest.raises(FENError):
            fen_to_state("rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_wrong_number_of_ranks(self) -> None:
        with pytest.raises(FENError, match="must have 8 ranks"):
            fen_to_state("8/8/8/8/8/8/4K2k w - - 0 1")

    def test_invalid_castling_char(self) -> None:
        with pytest.raises(FENError, match="Invalid castling"):
            fen_to_state("8/8/8/8/8/8/8/4K2k w XYZ - 0 1")

    def test_invalid_en_passant(self) -> None:
        with pytest.raises(FENError, match="Invalid en passant"):
            fen_to_state("8/8/8/8/8/8/8/4K2k w - z9 0 1")

    def test_negative_halfmove(self) -> None:
        with pytest.raises(FENError, match="cannot be negative"):
            fen_to_state("8/8/8/8/8/8/8/4K2k w - - -1 1")

    def test_zero_fullmove(self) -> None:
        with pytest.raises(FENError, match="must be >= 1"):
            fen_to_state("8/8/8/8/8/8/8/4K2k w - - 0 0")

    def test_non_numeric_halfmove(self) -> None:
        with pytest.raises(FENError, match="Invalid halfmove"):
            fen_to_state("8/8/8/8/8/8/8/4K2k w - - abc 1")


class TestBoardEncoding:
    """Tests for board array encoding correctness."""

    def test_empty_board_except_kings(self) -> None:
        state = fen_to_state("8/8/8/8/8/8/8/4K2k w - - 0 1")
        board = state.board
        # Should have exactly 2 non-zero squares
        assert np.count_nonzero(board) == 2
        assert board[7, 4] == Piece.KING * WHITE
        assert board[7, 7] == Piece.KING * BLACK

    def test_all_piece_types(self) -> None:
        # FEN with all piece types
        fen = "rnbqk3/pppp4/8/8/8/8/PPPP4/RNBQK3 w - - 0 1"
        state = fen_to_state(fen)
        board = state.board

        # White pieces (row 7)
        assert board[7, 0] == Piece.ROOK * WHITE
        assert board[7, 1] == Piece.KNIGHT * WHITE
        assert board[7, 2] == Piece.BISHOP * WHITE
        assert board[7, 3] == Piece.QUEEN * WHITE
        assert board[7, 4] == Piece.KING * WHITE

        # Black pieces (row 0)
        assert board[0, 0] == Piece.ROOK * BLACK
        assert board[0, 1] == Piece.KNIGHT * BLACK
        assert board[0, 2] == Piece.BISHOP * BLACK
        assert board[0, 3] == Piece.QUEEN * BLACK
        assert board[0, 4] == Piece.KING * BLACK


class TestStateToFen:
    """Tests for state_to_fen with manually constructed states."""

    def test_empty_board_fen(self) -> None:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[7, 4] = Piece.KING * WHITE
        board[0, 4] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=0,
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
            },
        )

        fen = state_to_fen(state)
        assert fen == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    def test_state_with_en_passant(self) -> None:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[7, 4] = Piece.KING * WHITE
        board[0, 4] = Piece.KING * BLACK
        board[3, 4] = Piece.PAWN * WHITE  # e5

        state = GameState(
            board=board,
            current_player=BLACK,
            move_number=1,
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": (5, 4),  # e3
                "halfmove_clock": 0,
            },
        )

        fen = state_to_fen(state)
        assert "e3" in fen
        assert fen.startswith("4k3/8/8/4P3/8/8/8/4K3 b")
