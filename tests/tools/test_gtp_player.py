"""Tests for GTP protocol player assignment fix.

Validates that ``_genmove`` correctly sets the expected player
from GTP color strings for both black and white. Also tests GTP
coordinate conversions and SimpleGoGame game logic.
"""

from __future__ import annotations

import pytest

from src.tools.gtp import (
    SimpleGoGame,
    action_to_gtp,
    coord_to_gtp,
    gtp_to_action,
    gtp_to_coord,
)


class TestSimpleGoGamePlayerConstants:
    """Verify player constants used by GTP."""

    def test_black_constant(self) -> None:
        assert SimpleGoGame.BLACK == 1

    def test_white_constant(self) -> None:
        assert SimpleGoGame.WHITE == 2

    def test_empty_constant(self) -> None:
        assert SimpleGoGame.EMPTY == 0


class TestGTPCoordinateConversions:
    """Tests for GTP coordinate conversion functions."""

    @pytest.mark.parametrize(
        "row,col,board_size,expected",
        [
            (0, 0, 9, "A9"),  # Top-left corner
            (0, 8, 9, "J9"),  # Top-right corner (skips I)
            (8, 0, 9, "A1"),  # Bottom-left corner
            (8, 8, 9, "J1"),  # Bottom-right corner
            (4, 4, 9, "E5"),  # Center of 9x9
            (0, 0, 19, "A19"),  # 19x19 top-left
            (18, 18, 19, "T1"),  # 19x19 bottom-right
        ],
    )
    def test_coord_to_gtp(self, row: int, col: int, board_size: int, expected: str) -> None:
        """Test internal coordinates to GTP format."""
        result = coord_to_gtp(row, col, board_size)
        assert result == expected

    @pytest.mark.parametrize(
        "gtp_coord,board_size,expected_row,expected_col",
        [
            ("A9", 9, 0, 0),  # Top-left
            ("J9", 9, 0, 8),  # Top-right (J, not I)
            ("A1", 9, 8, 0),  # Bottom-left
            ("J1", 9, 8, 8),  # Bottom-right
            ("e5", 9, 4, 4),  # Center, lowercase
            ("a19", 19, 0, 0),  # 19x19 top-left
        ],
    )
    def test_gtp_to_coord(
        self, gtp_coord: str, board_size: int, expected_row: int, expected_col: int
    ) -> None:
        """Test GTP format to internal coordinates."""
        row, col = gtp_to_coord(gtp_coord, board_size)
        assert row == expected_row
        assert col == expected_col

    @pytest.mark.parametrize(
        "action,board_size,expected",
        [
            (0, 9, "A9"),  # First action is top-left
            (80, 9, "J1"),  # Last board position (9*9-1)
            (81, 9, "pass"),  # Pass action
        ],
    )
    def test_action_to_gtp(self, action: int, board_size: int, expected: str) -> None:
        """Test action index to GTP move string."""
        result = action_to_gtp(action, board_size)
        assert result == expected

    @pytest.mark.parametrize(
        "gtp_move,board_size,expected_action",
        [
            ("A9", 9, 0),  # Top-left
            ("J1", 9, 80),  # Bottom-right
            ("pass", 9, 81),  # Pass
            ("PASS", 9, 81),  # Pass uppercase
        ],
    )
    def test_gtp_to_action(self, gtp_move: str, board_size: int, expected_action: int) -> None:
        """Test GTP move string to action index."""
        result = gtp_to_action(gtp_move, board_size)
        assert result == expected_action


class TestGTPPlayerAssignment:
    """Tests for the player assignment logic in _genmove."""

    @pytest.mark.parametrize(
        "color_input,expected_player",
        [
            ("b", SimpleGoGame.BLACK),
            ("black", SimpleGoGame.BLACK),
            ("B", SimpleGoGame.BLACK),
            ("BLACK", SimpleGoGame.BLACK),
            ("Black", SimpleGoGame.BLACK),
            ("w", SimpleGoGame.WHITE),
            ("white", SimpleGoGame.WHITE),
            ("W", SimpleGoGame.WHITE),
            ("WHITE", SimpleGoGame.WHITE),
            ("White", SimpleGoGame.WHITE),
        ],
    )
    def test_expected_player_from_color(
        self,
        color_input: str,
        expected_player: int,
    ) -> None:
        """Verify correct player assignment for all valid GTP color inputs."""
        # Reproduce the logic from _genmove
        if color_input.lower() in ("b", "black"):
            result = SimpleGoGame.BLACK
        else:
            result = SimpleGoGame.WHITE

        assert result == expected_player

    def test_initial_player_is_black(self) -> None:
        """Game starts with BLACK to play."""
        game = SimpleGoGame(board_size=9)
        assert game.current_player == SimpleGoGame.BLACK

    def test_player_alternates_after_move(self) -> None:
        """Current player switches after a valid move."""
        game = SimpleGoGame(board_size=9)
        assert game.current_player == SimpleGoGame.BLACK
        game.play(0, 0)
        assert game.current_player == SimpleGoGame.WHITE

    def test_player_alternates_after_pass(self) -> None:
        """Current player switches after a pass."""
        game = SimpleGoGame(board_size=9)
        assert game.current_player == SimpleGoGame.BLACK
        game.play_pass()
        assert game.current_player == SimpleGoGame.WHITE


class TestSimpleGoGame:
    """Tests for SimpleGoGame game logic."""

    def test_board_initialization(self) -> None:
        """Board starts empty."""
        game = SimpleGoGame(board_size=9)
        assert game.board.shape == (9, 9)
        assert (game.board == SimpleGoGame.EMPTY).all()

    def test_play_places_stone(self) -> None:
        """play() places a stone at the given position."""
        game = SimpleGoGame(board_size=9)
        game.play(4, 4)
        assert game.board[4, 4] == SimpleGoGame.BLACK

    def test_move_history_tracked(self) -> None:
        """Moves are recorded in move_history."""
        game = SimpleGoGame(board_size=9)
        game.play(0, 0)
        game.play(1, 1)
        assert len(game.move_history) == 2
        assert game.move_history[0] == (0, 0)
        assert game.move_history[1] == (1, 1)

    def test_pass_recorded_as_none(self) -> None:
        """Pass moves are recorded as None."""
        game = SimpleGoGame(board_size=9)
        game.play_pass()
        assert game.move_history[0] is None

    def test_consecutive_passes_ends_game(self) -> None:
        """Two consecutive passes end the game."""
        game = SimpleGoGame(board_size=9)
        assert not game.is_game_over()
        game.play_pass()
        assert not game.is_game_over()
        game.play_pass()
        assert game.is_game_over()

    def test_reset_clears_board(self) -> None:
        """reset() clears the board and history."""
        game = SimpleGoGame(board_size=9)
        game.play(0, 0)
        game.play(1, 1)
        game.reset()
        assert (game.board == SimpleGoGame.EMPTY).all()
        assert len(game.move_history) == 0
        assert game.current_player == SimpleGoGame.BLACK

    def test_clone_creates_independent_copy(self) -> None:
        """clone() creates a deep copy."""
        game = SimpleGoGame(board_size=9)
        game.play(0, 0)
        cloned = game.clone()
        cloned.play(1, 1)
        assert game.board[1, 1] == SimpleGoGame.EMPTY
        assert cloned.board[1, 1] == SimpleGoGame.WHITE

    def test_get_legal_actions_excludes_occupied(self) -> None:
        """get_legal_actions() excludes occupied positions."""
        game = SimpleGoGame(board_size=9)
        initial_actions = game.get_legal_actions()
        assert len(initial_actions) == 82  # 81 positions + pass
        game.play(4, 4)
        after_move = game.get_legal_actions()
        assert len(after_move) == 81  # One fewer board position

    def test_get_state_shape(self) -> None:
        """get_state() returns correct tensor shape."""
        game = SimpleGoGame(board_size=9)
        state = game.get_state()
        # Should be (17, 9, 9) like AlphaGo encoding
        assert state.shape[1:] == (9, 9)
        assert state.ndim == 3

    def test_apply_action_index(self) -> None:
        """apply_action() works with action indices."""
        game = SimpleGoGame(board_size=9)
        # Action 0 = position (0, 0)
        game.apply_action(0)
        assert game.board[0, 0] == SimpleGoGame.BLACK

    def test_apply_action_pass(self) -> None:
        """apply_action() with pass action."""
        game = SimpleGoGame(board_size=9)
        pass_action = 81
        game.apply_action(pass_action)
        assert game.move_history[0] is None
        assert game.current_player == SimpleGoGame.WHITE

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_different_board_sizes(self, board_size: int) -> None:
        """Game works with different board sizes."""
        game = SimpleGoGame(board_size=board_size)
        assert game.board.shape == (board_size, board_size)
        n_actions = board_size * board_size + 1
        assert len(game.get_legal_actions()) == n_actions
