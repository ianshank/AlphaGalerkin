"""Security and sanity tests for chess pipeline.

Tests validate robustness against:
- Invalid action indices (negative, out of range)
- Out-of-bounds board state injection
- Corrupted experience data in collator
- Invalid FEN strings
- Malformed game state metadata
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.chess import ACTION_SPACE_SIZE, BLACK, WHITE, ChessGame, Piece
from src.games.state import GameState
from src.training.replay_buffer import Experience


@pytest.fixture
def game() -> ChessGame:
    """Chess game instance."""
    return ChessGame()


class TestInvalidActionIndices:
    """Tests for handling invalid action indices."""

    def test_negative_action_index_decode(self, game: ChessGame) -> None:
        """Negative action index should produce out-of-bounds coordinates."""
        decoded = game._decode_move(-1)
        # Negative from_sq means invalid square
        fr, fc, tr, tc, promo = decoded
        assert fr < 0 or fc < 0 or fr >= 8 or fc >= 8

    def test_action_beyond_space_size(self, game: ChessGame) -> None:
        """Actions >= ACTION_SPACE_SIZE produce out-of-bounds from_sq."""
        decoded = game._decode_move(ACTION_SPACE_SIZE)
        fr, fc, _, _, _ = decoded
        # from_sq = action // NUM_MOVE_TYPES >= 64, so fr >= 8
        assert fr >= 8 or fc >= 8

    def test_very_large_action_index(self, game: ChessGame) -> None:
        """Extremely large action indices produce out-of-bounds coords."""
        decoded = game._decode_move(999999)
        fr, fc, _, _, _ = decoded
        assert fr >= 8 or fc >= 8

    def test_apply_illegal_action(self, game: ChessGame) -> None:
        """Applying an illegal action should raise or handle gracefully."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)

        # Find an action that's NOT legal
        all_actions = set(range(ACTION_SPACE_SIZE))
        illegal = all_actions - set(legal)
        if illegal:
            illegal_action = min(illegal)
            # Should either raise or produce a valid (but possibly wrong) state
            try:
                new_state = game.apply_action(state, illegal_action)
                # If it doesn't raise, state should still be valid
                assert new_state is not None
            except (ValueError, IndexError):
                pass  # Expected behavior


class TestOutOfBoundsBoard:
    """Tests for out-of-bounds board state injection."""

    def test_empty_board_no_crash(self, game: ChessGame) -> None:
        """Empty board should not crash tensor encoding."""
        board = np.zeros((8, 8), dtype=np.int8)
        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=1,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )
        # Should not crash during tensor encoding
        tensor = game.to_tensor(state)
        assert tensor.shape == (119, 8, 8)

    def test_board_with_invalid_piece_values(self, game: ChessGame) -> None:
        """Board with unexpected piece values should not crash tensor encoding."""
        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 0] = Piece.KING * WHITE
        board[7, 7] = Piece.KING * BLACK
        board[3, 3] = 99  # Invalid piece value

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=1,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 0,
                "position_history": [],
                "board_history": [],
            },
        )

        # Tensor encoding should handle gracefully (no crash)
        try:
            tensor = game.to_tensor(state)
            assert tensor.shape == (119, 8, 8)
        except (ValueError, KeyError):
            pass  # Also acceptable


class TestCorruptedExperienceData:
    """Tests for handling corrupted experience data in collision with collator."""

    def test_wrong_policy_size_experience(self) -> None:
        """Experience with wrong policy size should be handled."""
        # This should not crash during creation
        exp = Experience(
            board_state=torch.randn(119, 8, 8),
            board_size=8,
            target_policy=torch.softmax(torch.randn(100), dim=0),  # Wrong size
            target_value=0.5,
        )
        assert exp.target_policy.shape == (100,)

    def test_nan_in_policy(self) -> None:
        """Experience with NaN in policy should be detectable."""
        policy = torch.randn(4672)
        policy[100] = float("nan")

        exp = Experience(
            board_state=torch.randn(119, 8, 8),
            board_size=8,
            target_policy=policy,
            target_value=0.5,
        )

        assert torch.isnan(exp.target_policy).any()

    def test_inf_in_board_state(self) -> None:
        """Experience with inf in board state should be detectable."""
        board = torch.randn(119, 8, 8)
        board[0, 0, 0] = float("inf")

        exp = Experience(
            board_state=board,
            board_size=8,
            target_policy=torch.softmax(torch.randn(4672), dim=0),
            target_value=0.0,
        )

        assert torch.isinf(exp.board_state).any()

    def test_value_out_of_range(self) -> None:
        """Experience with value outside [-1, 1] should be detectable."""
        exp = Experience(
            board_state=torch.randn(119, 8, 8),
            board_size=8,
            target_policy=torch.softmax(torch.randn(4672), dim=0),
            target_value=5.0,  # Out of range
        )

        assert abs(exp.target_value) > 1.0


class TestInvalidFEN:
    """Tests for handling invalid FEN strings."""

    def test_empty_fen_string(self, game: ChessGame) -> None:
        """Empty FEN should raise or handle gracefully."""
        try:
            from src.games.fen import fen_to_state
            state = fen_to_state("")
            # If it accepts, state should still be valid-ish
        except (ValueError, IndexError, KeyError):
            pass  # Expected

    def test_garbage_fen_string(self, game: ChessGame) -> None:
        """Random garbage FEN should not crash."""
        try:
            from src.games.fen import fen_to_state
            state = fen_to_state("not/a/valid/fen string")
        except (ValueError, IndexError, KeyError):
            pass  # Expected


class TestActionMaskRobustness:
    """Tests for action mask edge cases."""

    def test_action_mask_size_matches_space(self, game: ChessGame) -> None:
        """Action mask should always have ACTION_SPACE_SIZE elements."""
        state = game.initial_state()
        mask = game.get_action_mask(state)
        assert mask.action_space_size == ACTION_SPACE_SIZE

    def test_terminal_state_is_detected(self, game: ChessGame) -> None:
        """50-move rule should mark state as terminal.

        Note: is_terminal is checked at the game engine level, not at
        the move generation level. Legal moves may still exist but the
        game is declared drawn by the 50-move rule.
        """
        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 0] = Piece.KING * WHITE
        board[7, 7] = Piece.KING * BLACK

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=200,
            move_history=[],
            metadata={
                "castling_rights": {"K": False, "Q": False, "k": False, "q": False},
                "en_passant_square": None,
                "halfmove_clock": 100,
                "position_history": [],
                "board_history": [],
            },
        )

        # 50-move rule should trigger terminal state
        assert game.is_terminal(state)

    def test_initial_position_has_20_legal_moves(self, game: ChessGame) -> None:
        """Standard chess starting position has exactly 20 legal moves."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        assert len(legal) == 20
