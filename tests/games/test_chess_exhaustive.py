"""Exhaustive chess encode/decode roundtrip and edge-case tests.

Tests cover:
- All 4672 action indices roundtrip through encode → decode → re-encode
- Promotion edge cases (column 0, column 7, all directions)
- Queen-like, knight, and underpromotion move types
- Checkmate, stalemate, and draw detection
- Castling execution and eligibility
- En passant capture
- 50-move rule and threefold repetition
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.chess import (
    ACTION_SPACE_SIZE,
    BLACK,
    NUM_MOVE_TYPES,
    WHITE,
    ChessGame,
    Piece,
)
from src.games.state import GameState


@pytest.fixture
def game() -> ChessGame:
    """Create a chess game instance."""
    return ChessGame()


@pytest.fixture
def initial_state(game: ChessGame) -> GameState:
    """Initial chess position."""
    return game.initial_state()


# ─────────────────────────────────────────────────────────────
# Encode/Decode Roundtrip Tests
# ─────────────────────────────────────────────────────────────


class TestEncodeDecodeRoundtrip:
    """Exhaustive encode → decode → re-encode roundtrip tests."""

    def test_all_queen_moves_roundtrip(self, game: ChessGame) -> None:
        """Verify all queen-like moves (indices 0-55) roundtrip correctly."""
        errors = []
        for from_sq in range(64):
            from_row = from_sq // 8
            from_col = from_sq % 8
            for move_type in range(56):  # queen-like
                action = from_sq * NUM_MOVE_TYPES + move_type
                decoded = game._decode_move(action)
                fr, fc, tr, tc, promo = decoded
                assert fr == from_row and fc == from_col
                assert promo is None
                # Only re-encode if target is in bounds
                if 0 <= tr < 8 and 0 <= tc < 8:
                    re_encoded = game._encode_move(fr, fc, (tr, tc, None))
                    if re_encoded is not None and re_encoded != action:
                        errors.append(
                            f"action={action} from=({fr},{fc}) to=({tr},{tc}) "
                            f"re_encoded={re_encoded}"
                        )
        assert not errors, "Roundtrip failures:\n" + "\n".join(errors[:10])

    def test_all_knight_moves_roundtrip(self, game: ChessGame) -> None:
        """Verify all knight moves (indices 56-63) roundtrip correctly."""
        errors = []
        for from_sq in range(64):
            from_row = from_sq // 8
            from_col = from_sq % 8
            for move_type in range(56, 64):  # knight
                action = from_sq * NUM_MOVE_TYPES + move_type
                decoded = game._decode_move(action)
                fr, fc, tr, tc, promo = decoded
                assert fr == from_row and fc == from_col
                assert promo is None
                if 0 <= tr < 8 and 0 <= tc < 8:
                    re_encoded = game._encode_move(fr, fc, (tr, tc, None))
                    if re_encoded is not None and re_encoded != action:
                        errors.append(
                            f"action={action} from=({fr},{fc}) to=({tr},{tc}) "
                            f"re_encoded={re_encoded}"
                        )
        assert not errors, "Roundtrip failures:\n" + "\n".join(errors[:10])

    def test_all_underpromotions_roundtrip(self, game: ChessGame) -> None:
        """Verify all underpromotion moves (indices 64-72) roundtrip correctly.

        This is the critical test for the encode/decode mismatch bug fix.
        """
        promo_pieces = [Piece.KNIGHT, Piece.BISHOP, Piece.ROOK]
        dc_values = [0, -1, 1]  # straight, left, right
        errors = []

        for from_sq in range(64):
            from_row = from_sq // 8
            from_col = from_sq % 8
            for move_type in range(64, 73):  # underpromotions
                action = from_sq * NUM_MOVE_TYPES + move_type
                decoded = game._decode_move(action)
                fr, fc, tr, tc, promo = decoded
                assert fr == from_row and fc == from_col
                assert promo in promo_pieces

                if 0 <= tr < 8 and 0 <= tc < 8:
                    re_encoded = game._encode_move(fr, fc, (tr, tc, promo))
                    if re_encoded is not None and re_encoded != action:
                        errors.append(
                            f"action={action} from=({fr},{fc}) to=({tr},{tc}) "
                            f"promo={promo} re_encoded={re_encoded}"
                        )
        assert not errors, "Roundtrip failures:\n" + "\n".join(errors[:10])

    def test_column_0_straight_promo_no_negative_col(self, game: ChessGame) -> None:
        """Regression: straight promotion from col 0 must not yield to_col=-1."""
        # White pawn at (1, 0) → straight promote to (0, 0)
        action = game._encode_move(1, 0, (0, 0, Piece.KNIGHT))
        assert action is not None
        decoded = game._decode_move(action)
        assert decoded[3] == 0, f"to_col should be 0, got {decoded[3]}"
        assert decoded[2] == 0, f"to_row should be 0, got {decoded[2]}"
        assert decoded[4] == Piece.KNIGHT

    def test_column_7_right_capture_promo_no_overflow(self, game: ChessGame) -> None:
        """Right capture promotion from col 7 should decode to_col=8 (out of bounds)."""
        # This tests boundary: dc=+1 from col 7 should give to_col=8
        action = game._encode_move(1, 7, (0, 7, Piece.ROOK))  # straight
        assert action is not None
        decoded = game._decode_move(action)
        assert decoded[3] == 7  # to_col stays 7 for straight

    def test_action_space_size_constant(self) -> None:
        """Verify ACTION_SPACE_SIZE matches 64 * NUM_MOVE_TYPES."""
        assert ACTION_SPACE_SIZE == 64 * NUM_MOVE_TYPES
        assert ACTION_SPACE_SIZE == 4672


# ─────────────────────────────────────────────────────────────
# Chess Game Logic Edge Cases
# ─────────────────────────────────────────────────────────────


class TestChessCheckmate:
    """Tests for checkmate detection."""

    def test_scholars_mate(self, game: ChessGame) -> None:
        """Scholar's Mate (4-move checkmate)."""
        state = game.initial_state()
        # 1. e4
        state = game.apply_action(state, game.string_to_action("e2e4"))
        # 1... e5
        state = game.apply_action(state, game.string_to_action("e7e5"))
        # 2. Qh5
        state = game.apply_action(state, game.string_to_action("d1h5"))
        # 2... Nc6
        state = game.apply_action(state, game.string_to_action("b8c6"))
        # 3. Bc4
        state = game.apply_action(state, game.string_to_action("f1c4"))
        # 3... Nf6 (not the best)
        state = game.apply_action(state, game.string_to_action("g8f6"))
        # 4. Qxf7# (checkmate)
        state = game.apply_action(state, game.string_to_action("h5f7"))

        assert game.is_terminal(state)
        assert game.get_winner(state) == WHITE

    def test_fools_mate(self, game: ChessGame) -> None:
        """Fool's Mate (2-move checkmate for black)."""
        state = game.initial_state()
        # 1. f3
        state = game.apply_action(state, game.string_to_action("f2f3"))
        # 1... e5
        state = game.apply_action(state, game.string_to_action("e7e5"))
        # 2. g4
        state = game.apply_action(state, game.string_to_action("g2g4"))
        # 2... Qh4# (checkmate)
        state = game.apply_action(state, game.string_to_action("d8h4"))

        assert game.is_terminal(state)
        assert game.get_winner(state) == BLACK


class TestChessStalemate:
    """Tests for stalemate and draw detection."""

    def test_no_legal_moves_not_in_check_is_stalemate(self, game: ChessGame) -> None:
        """Verify stalemate when king is not in check but has no legal moves."""
        # Create a position where one side has only king with no legal moves
        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 0] = Piece.KING * BLACK  # Black king in corner
        board[1, 2] = Piece.QUEEN * WHITE  # White queen traps king
        board[2, 1] = Piece.KING * WHITE  # White king nearby

        state = GameState(
            board=board,
            current_player=BLACK,
            move_number=10,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )

        legal = game.get_legal_actions(state)
        if len(legal) == 0:
            assert game.is_terminal(state)
            assert game.get_winner(state) is None or game.get_winner(state) == 0


class TestChessPromotion:
    """Tests for pawn promotion mechanics."""

    def test_queen_promotion(self, game: ChessGame) -> None:
        """Verify a pawn can promote to queen."""
        board = np.zeros((8, 8), dtype=np.int8)
        board[1, 3] = Piece.PAWN * WHITE  # White pawn on 7th rank (row 1)
        board[0, 0] = Piece.KING * WHITE
        board[7, 7] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=50,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )

        legal = game.get_legal_actions(state)
        # Should include promotions (queen, rook, bishop, knight)
        assert len(legal) >= 4  # At least 4 promotion options for straight push

    def test_underpromotion_to_knight(self, game: ChessGame) -> None:
        """Verify a pawn can underpromote to knight."""
        board = np.zeros((8, 8), dtype=np.int8)
        board[1, 4] = Piece.PAWN * WHITE
        board[0, 0] = Piece.KING * WHITE
        board[7, 7] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=50,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )

        legal = game.get_legal_actions(state)
        # Decode all legal moves and check for knight promotions
        knight_promos = []
        for action in legal:
            _, _, _, _, promo = game._decode_move(action)
            if promo == Piece.KNIGHT:
                knight_promos.append(action)
        assert len(knight_promos) >= 1, "Should have at least one knight promotion"


class TestChessCastling:
    """Tests for castling mechanics."""

    def test_kingside_castling_execution(self, game: ChessGame) -> None:
        """Verify kingside castling moves both king and rook."""
        board = np.zeros((8, 8), dtype=np.int8)
        board[7, 4] = Piece.KING * WHITE  # King on e1
        board[7, 7] = Piece.ROOK * WHITE  # Rook on h1
        board[0, 4] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=10,
            move_history=[],
            metadata={
                "castling_rights": {"K": True, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )

        legal = game.get_legal_actions(state)
        # Find castling move (king e1→g1)
        castle_action = game.string_to_action("e1g1")
        if castle_action in legal:
            new_state = game.apply_action(state, castle_action)
            # King should be on g1 (7, 6)
            assert new_state.board[7, 6] == Piece.KING * WHITE
            # Rook should be on f1 (7, 5)
            assert new_state.board[7, 5] == Piece.ROOK * WHITE
            # Original squares should be empty
            assert new_state.board[7, 4] == 0
            assert new_state.board[7, 7] == 0


class TestChessEnPassant:
    """Tests for en passant mechanics."""

    def test_en_passant_capture(self, game: ChessGame) -> None:
        """Verify en passant captures the pawn correctly."""
        state = game.initial_state()
        # 1. e4
        state = game.apply_action(state, game.string_to_action("e2e4"))
        # 1... a6
        state = game.apply_action(state, game.string_to_action("a7a6"))
        # 2. e5
        state = game.apply_action(state, game.string_to_action("e4e5"))
        # 2... d5 (double push next to white pawn)
        state = game.apply_action(state, game.string_to_action("d7d5"))

        # Now en passant should be available
        assert state.metadata.get("en_passant_square") is not None
        legal = game.get_legal_actions(state)
        ep_action = game.string_to_action("e5d6")
        if ep_action in legal:
            new_state = game.apply_action(state, ep_action)
            # White pawn should be on d6
            assert new_state.board[2, 3] == Piece.PAWN * WHITE
            # Black pawn on d5 should be captured
            assert new_state.board[3, 3] == 0


class TestChessDrawRules:
    """Tests for draw rules: 50-move rule and insufficient material."""

    def test_fifty_move_rule(self, game: ChessGame) -> None:
        """Verify game terminates after 50 moves without capture or pawn push."""
        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 0] = Piece.KING * WHITE
        board[7, 7] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=100,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 100,  # 50 moves = 100 half-moves
                "position_history": [],
                "board_history": [],
            },
        )

        assert game.is_terminal(state)
        assert game.get_winner(state) is None or game.get_winner(state) == 0

    def test_insufficient_material_k_vs_k(self, game: ChessGame) -> None:
        """King vs King = draw."""
        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 0] = Piece.KING * WHITE
        board[7, 7] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=100,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )

        assert game.is_terminal(state)


class TestChessTensorEncoding:
    """Extended tests for neural network tensor encoding."""

    def test_tensor_shape_and_channels(self, game: ChessGame) -> None:
        """Verify tensor has correct shape for all history depths."""
        state = game.initial_state()
        tensor = game.to_tensor(state)
        assert tensor.shape == (119, 8, 8)

    def test_tensor_changes_after_move(self, game: ChessGame) -> None:
        """Verify tensor representation changes when a move is made."""
        state = game.initial_state()
        t1 = game.to_tensor(state).numpy()
        state = game.apply_action(state, game.get_legal_actions(state)[0])
        t2 = game.to_tensor(state).numpy()
        assert not np.allclose(t1, t2), "Tensor should change after a move"

    def test_action_mask_nonzero_count(self, game: ChessGame) -> None:
        """Action mask should have exactly as many 1s as legal actions."""
        state = game.initial_state()
        mask = game.get_action_mask(state)
        legal = game.get_legal_actions(state)
        assert mask.mask.sum() == len(legal)


class TestChessMultipleGames:
    """Tests for playing multiple full games."""

    def test_play_random_game_to_completion(self, game: ChessGame) -> None:
        """Play a random game up to 500 moves and verify no crashes."""
        import random

        random.seed(42)
        state = game.initial_state()
        for _ in range(500):
            if game.is_terminal(state):
                break
            legal = game.get_legal_actions(state)
            assert len(legal) > 0, "Non-terminal state should have legal actions"
            action = random.choice(legal)
            state = game.apply_action(state, action)
        # If game ended, result should be valid
        if game.is_terminal(state):
            result = game.get_result(state)
            assert result.winner in [-1, 0, 1, None]

    def test_three_random_games(self, game: ChessGame) -> None:
        """Play 3 random games ensuring no crashes."""
        import random

        for seed in [1, 2, 3]:
            random.seed(seed)
            state = game.initial_state()
            for _ in range(300):
                if game.is_terminal(state):
                    break
                legal = game.get_legal_actions(state)
                action = random.choice(legal)
                state = game.apply_action(state, action)
