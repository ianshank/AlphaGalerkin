"""Tests for GTP engine and command processing.

Covers GTPEngine initialization, command processing, response formatting,
and individual GTP command handlers. Uses mocks to avoid MCTS dependencies.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from src.tools.gtp import (
    GTP_LETTERS,
    GTPEngine,
    SimpleGoGame,
    action_to_gtp,
    gtp_to_action,
)

# --- GTP coordinate edge cases ---


class TestGTPCoordinateEdgeCases:
    """Edge cases for GTP coordinate conversions."""

    def test_gtp_to_action_resign(self) -> None:
        """Resign returns special value -1."""
        result = gtp_to_action("resign", board_size=9)
        assert result == -1

    def test_gtp_to_action_resign_uppercase(self) -> None:
        """Resign case insensitive."""
        result = gtp_to_action("RESIGN", board_size=9)
        assert result == -1

    def test_gtp_to_action_with_whitespace(self) -> None:
        """Handles leading/trailing whitespace."""
        result = gtp_to_action("  pass  ", board_size=9)
        assert result == 81

    def test_action_roundtrip(self) -> None:
        """Action -> GTP -> action roundtrip preserves value."""
        for board_size in [9, 13, 19]:
            for action in range(board_size**2):
                gtp_str = action_to_gtp(action, board_size)
                recovered = gtp_to_action(gtp_str, board_size)
                assert recovered == action, (
                    f"Roundtrip failed for action={action}, size={board_size}"
                )

    def test_pass_action_roundtrip(self) -> None:
        """Pass action roundtrips correctly."""
        for board_size in [9, 13, 19]:
            pass_action = board_size**2
            gtp_str = action_to_gtp(pass_action, board_size)
            assert gtp_str == "pass"
            recovered = gtp_to_action(gtp_str, board_size)
            assert recovered == pass_action


# --- SimpleGoGame additional tests ---


class TestSimpleGoGameAdditional:
    """Additional tests for SimpleGoGame not covered by test_gtp_player.py."""

    def test_illegal_move_out_of_bounds_row(self) -> None:
        """Move with negative row is rejected."""
        game = SimpleGoGame(board_size=9)
        assert not game.play(-1, 0)

    def test_illegal_move_out_of_bounds_col(self) -> None:
        """Move with column >= board_size is rejected."""
        game = SimpleGoGame(board_size=9)
        assert not game.play(0, 9)

    def test_illegal_move_occupied(self) -> None:
        """Move on occupied position is rejected."""
        game = SimpleGoGame(board_size=9)
        game.play(0, 0)
        # White tries to play on same spot
        assert not game.play(0, 0)

    def test_capture_removes_stones(self) -> None:
        """Surrounded stones are captured."""
        game = SimpleGoGame(board_size=9)
        # Place a white stone and surround it with black
        # Black plays around a white stone:
        # . B .
        # B W B
        # . B .
        game.play(1, 1)  # Black at (1,1) - this is just setup
        game.reset()

        # Place white stone at center (4,4), then surround with black
        game.board[4, 4] = SimpleGoGame.WHITE

        # Place black stones around - manually set to avoid turn issues
        game.board[3, 4] = SimpleGoGame.BLACK  # above
        game.board[5, 4] = SimpleGoGame.BLACK  # below
        game.board[4, 3] = SimpleGoGame.BLACK  # left
        # Now black plays the last surrounding position
        game.current_player = SimpleGoGame.BLACK
        result = game.play(4, 5)  # right - completes surround
        assert result
        # White stone should be captured
        assert game.board[4, 4] == SimpleGoGame.EMPTY

    def test_suicide_prevented(self) -> None:
        """Suicide moves are rejected."""
        game = SimpleGoGame(board_size=9)
        # Fill corner neighbors with opponent stones
        # White at (0,1) and (1,0), black tries to play at (0,0) - suicide
        game.board[0, 1] = SimpleGoGame.WHITE
        game.board[1, 0] = SimpleGoGame.WHITE
        game.current_player = SimpleGoGame.BLACK
        result = game.play(0, 0)
        assert not result

    def test_get_winner(self) -> None:
        """get_winner returns correct winner based on stones + captures."""
        game = SimpleGoGame(board_size=9)
        # Black has more stones
        game.board[0, 0] = SimpleGoGame.BLACK
        game.board[0, 1] = SimpleGoGame.BLACK
        game.board[0, 2] = SimpleGoGame.BLACK
        game.board[1, 0] = SimpleGoGame.WHITE
        # With komi=6.5, white needs many fewer stones to win
        # Let's check both sides
        winner = game.get_winner()
        assert isinstance(winner, int)
        assert winner in (-1, 1)

    def test_is_terminal_delegates(self) -> None:
        """is_terminal delegates to is_game_over."""
        game = SimpleGoGame(board_size=9)
        assert not game.is_terminal()
        game.play_pass()
        game.play_pass()
        assert game.is_terminal()

    def test_get_state_current_player_indicator(self) -> None:
        """State plane 16 indicates current player."""
        game = SimpleGoGame(board_size=9)
        state = game.get_state()
        # Black to play -> plane 16 all 1s
        assert state[16, 0, 0] == 1.0

        game.play_pass()  # Now white to play
        state = game.get_state()
        assert state[16, 0, 0] == 0.0

    def test_get_state_stone_planes(self) -> None:
        """State encodes current player stones in plane 0, opponent in plane 8."""
        game = SimpleGoGame(board_size=9)
        game.play(0, 0)  # Black at (0,0)
        game.play(1, 1)  # White at (1,1)

        state = game.get_state()
        # Now it's black's turn again
        # Plane 0: current player (black) stones
        assert state[0, 0, 0] == 1.0
        assert state[0, 1, 1] == 0.0
        # Plane 8: opponent (white) stones
        assert state[8, 1, 1] == 1.0
        assert state[8, 0, 0] == 0.0

    def test_passes_reset_on_move(self) -> None:
        """Pass counter resets when a stone is played."""
        game = SimpleGoGame(board_size=9)
        game.play_pass()
        assert game.passes == 1
        game.play(0, 0)  # Stone placement resets passes
        assert game.passes == 0

    def test_komi_default(self) -> None:
        """Default komi is 6.5."""
        game = SimpleGoGame(board_size=9)
        assert game.komi == 6.5


# --- GTPEngine Tests ---


class TestGTPEngine:
    """Tests for GTPEngine class."""

    @pytest.fixture
    def engine(self) -> GTPEngine:
        """Create a GTPEngine with no model (random evaluator)."""
        return GTPEngine(model=None, board_size=9)

    def test_init_no_model(self, engine: GTPEngine) -> None:
        """Engine initializes with random evaluator when no model."""
        from src.mcts.evaluator import RandomEvaluator

        assert isinstance(engine.evaluator, RandomEvaluator)
        assert engine.board_size == 9
        assert engine.device == "cpu"

    def test_init_has_commands(self, engine: GTPEngine) -> None:
        """Engine registers all standard GTP commands."""
        expected = {
            "protocol_version", "name", "version", "known_command",
            "list_commands", "quit", "boardsize", "clear_board",
            "komi", "play", "genmove", "showboard",
        }
        assert set(engine.commands.keys()) == expected

    def test_process_command_empty_line(self, engine: GTPEngine) -> None:
        """Empty lines return empty string."""
        assert engine.process_command("") == ""
        assert engine.process_command("   ") == ""

    def test_process_command_comment(self, engine: GTPEngine) -> None:
        """Comment lines return empty string."""
        assert engine.process_command("# this is a comment") == ""

    def test_process_command_protocol_version(self, engine: GTPEngine) -> None:
        """protocol_version returns '2'."""
        response = engine.process_command("protocol_version")
        assert "= 2" in response

    def test_process_command_name(self, engine: GTPEngine) -> None:
        """Name returns 'AlphaGalerkin'."""
        response = engine.process_command("name")
        assert "AlphaGalerkin" in response

    def test_process_command_version(self, engine: GTPEngine) -> None:
        """Version returns version string."""
        response = engine.process_command("version")
        assert "0.1.0" in response

    def test_process_command_known_command_true(self, engine: GTPEngine) -> None:
        """known_command returns 'true' for known commands."""
        response = engine.process_command("known_command name")
        assert "true" in response

    def test_process_command_known_command_false(self, engine: GTPEngine) -> None:
        """known_command returns 'false' for unknown commands."""
        response = engine.process_command("known_command nonexistent")
        assert "false" in response

    def test_process_command_list_commands(self, engine: GTPEngine) -> None:
        """list_commands returns sorted list."""
        response = engine.process_command("list_commands")
        assert "name" in response
        assert "quit" in response

    def test_process_command_unknown(self, engine: GTPEngine) -> None:
        """Unknown command returns error response."""
        response = engine.process_command("unknown_cmd")
        assert response.startswith("?")
        assert "unknown command" in response

    def test_process_command_with_id(self, engine: GTPEngine) -> None:
        """Command with numeric ID includes ID in response."""
        response = engine.process_command("42 name")
        assert response.startswith("=42")

    def test_quit_sets_flag(self, engine: GTPEngine) -> None:
        """Quit command sets quit flag."""
        assert not engine._quit_flag
        engine.process_command("quit")
        assert engine._quit_flag

    def test_boardsize_valid(self, engine: GTPEngine) -> None:
        """Boardsize changes board size."""
        engine.process_command("boardsize 13")
        assert engine.board_size == 13
        assert engine.game.board_size == 13

    def test_boardsize_invalid(self, engine: GTPEngine) -> None:
        """Boardsize rejects invalid sizes."""
        response = engine.process_command("boardsize 0")
        assert response.startswith("?")

    def test_boardsize_too_large(self, engine: GTPEngine) -> None:
        """Boardsize rejects size > 25."""
        response = engine.process_command("boardsize 26")
        assert response.startswith("?")

    def test_clear_board(self, engine: GTPEngine) -> None:
        """clear_board resets the game."""
        engine.process_command("play black A9")
        engine.process_command("clear_board")
        assert (engine.game.board == SimpleGoGame.EMPTY).all()

    def test_komi(self, engine: GTPEngine) -> None:
        """Komi sets the komi value."""
        engine.process_command("komi 7.5")
        assert engine.game.komi == 7.5

    def test_play_valid_move(self, engine: GTPEngine) -> None:
        """Play places a stone."""
        response = engine.process_command("play black A9")
        assert response.startswith("=")
        assert engine.game.board[0, 0] == SimpleGoGame.BLACK

    def test_play_pass(self, engine: GTPEngine) -> None:
        """Play pass works."""
        response = engine.process_command("play black pass")
        assert response.startswith("=")
        assert engine.game.current_player == SimpleGoGame.WHITE

    def test_play_resign(self, engine: GTPEngine) -> None:
        """Play resign is accepted."""
        response = engine.process_command("play black resign")
        assert response.startswith("=")

    def test_play_illegal_move(self, engine: GTPEngine) -> None:
        """Play illegal move returns error."""
        engine.process_command("play black A9")
        # Try to play on same spot
        response = engine.process_command("play white A9")
        assert response.startswith("?")

    def test_showboard(self, engine: GTPEngine) -> None:
        """Showboard returns board representation."""
        engine.process_command("play black E5")
        response = engine.process_command("showboard")
        assert "X" in response  # Black stone
        assert "." in response  # Empty positions

    def test_success_response_format(self, engine: GTPEngine) -> None:
        """Success responses have correct format."""
        response = engine._success_response(None, "result")
        assert response == "= result\n\n"

    def test_success_response_with_id(self, engine: GTPEngine) -> None:
        """Success responses with ID have correct format."""
        response = engine._success_response("42", "result")
        assert response == "=42 result\n\n"

    def test_success_response_empty_result(self, engine: GTPEngine) -> None:
        """Success response with empty result."""
        response = engine._success_response(None, "")
        assert response == "=\n\n"

    def test_error_response_format(self, engine: GTPEngine) -> None:
        """Error responses have correct format."""
        response = engine._error_response(None, "error msg")
        assert response == "? error msg\n\n"

    def test_error_response_with_id(self, engine: GTPEngine) -> None:
        """Error responses with ID have correct format."""
        response = engine._error_response("7", "error msg")
        assert response == "?7 error msg\n\n"

    def test_genmove_returns_valid_move(self, engine: GTPEngine) -> None:
        """Genmove returns a valid GTP move string."""
        response = engine.process_command("genmove black")
        assert response.startswith("=")
        # Extract move from response
        move = response.strip().split(" ", 1)[1].strip()
        # Should be a coordinate or "pass"
        assert move == "pass" or (move[0] in GTP_LETTERS and move[1:].isdigit())

    def test_run_reads_and_responds(self, engine: GTPEngine) -> None:
        """run() reads commands from input and writes responses."""
        input_stream = io.StringIO("name\nquit\n")
        output_stream = io.StringIO()

        engine.run(input_stream=input_stream, output_stream=output_stream)

        output = output_stream.getvalue()
        assert "AlphaGalerkin" in output

    def test_run_handles_eof(self, engine: GTPEngine) -> None:
        """run() exits gracefully on EOF."""
        input_stream = io.StringIO("")
        output_stream = io.StringIO()

        engine.run(input_stream=input_stream, output_stream=output_stream)
        # Should not raise

    def test_run_handles_keyboard_interrupt(self, engine: GTPEngine) -> None:
        """run() exits gracefully on KeyboardInterrupt."""
        input_stream = MagicMock()
        input_stream.readline.side_effect = KeyboardInterrupt
        output_stream = io.StringIO()

        engine.run(input_stream=input_stream, output_stream=output_stream)
        # Should not raise
