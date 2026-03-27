"""Comprehensive tests for the GTP protocol engine in src/tools/gtp.py.

Covers GTPEngine command dispatch, response formatting, game state management,
coordinate conversions, and error handling without requiring real models or GPU.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tools.gtp import (
    GTP_LETTERS,
    GTPEngine,
    SimpleGoGame,
    action_to_gtp,
    coord_to_gtp,
    gtp_to_action,
    gtp_to_coord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mcts() -> MagicMock:
    """Return a mock MCTS object with the required interface."""
    mcts = MagicMock()
    mcts.get_action.return_value = 0
    mcts.advance.return_value = None
    mcts.reset.return_value = None
    return mcts


@pytest.fixture()
def mock_random_evaluator() -> MagicMock:
    """Return a mock RandomEvaluator."""
    evaluator = MagicMock()
    return evaluator


@pytest.fixture()
def engine(mock_mcts: MagicMock) -> GTPEngine:
    """Return a GTPEngine with mocked MCTS (no model, board_size=9)."""
    with (
        patch("src.tools.gtp.RandomEvaluator") as mock_re,
        patch("src.tools.gtp.MCTS") as mock_mcts_cls,
    ):
        mock_mcts_cls.return_value = mock_mcts
        mock_re.return_value = MagicMock()
        eng = GTPEngine(model=None, board_size=9, device="cpu")
    eng.mcts = mock_mcts
    return eng


# ---------------------------------------------------------------------------
# GTP_LETTERS constant
# ---------------------------------------------------------------------------


class TestGTPLetters:
    """Verify the GTP_LETTERS constant skips 'I'."""

    def test_no_i_in_letters(self) -> None:
        assert "I" not in GTP_LETTERS

    def test_first_letter_is_a(self) -> None:
        assert GTP_LETTERS[0] == "A"

    def test_length_sufficient_for_19x19(self) -> None:
        assert len(GTP_LETTERS) >= 19

    def test_j_follows_h(self) -> None:
        h_idx = GTP_LETTERS.index("H")
        j_idx = GTP_LETTERS.index("J")
        assert j_idx == h_idx + 1


# ---------------------------------------------------------------------------
# coord_to_gtp / gtp_to_coord round-trip
# ---------------------------------------------------------------------------


class TestCoordRoundTrip:
    """Round-trip tests for coord_to_gtp and gtp_to_coord."""

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_round_trip_corners(self, board_size: int) -> None:
        corners = [
            (0, 0),
            (0, board_size - 1),
            (board_size - 1, 0),
            (board_size - 1, board_size - 1),
        ]
        for row, col in corners:
            gtp = coord_to_gtp(row, col, board_size)
            r, c = gtp_to_coord(gtp, board_size)
            assert (r, c) == (row, col), (
                f"Round-trip failed for ({row},{col}) on {board_size}x{board_size}"
            )

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_round_trip_all_positions(self, board_size: int) -> None:
        for row in range(board_size):
            for col in range(board_size):
                gtp = coord_to_gtp(row, col, board_size)
                r, c = gtp_to_coord(gtp, board_size)
                assert (r, c) == (row, col)

    def test_gtp_row_numbering(self) -> None:
        # Row 0 (top) should map to GTP row board_size (highest number)
        gtp = coord_to_gtp(0, 0, 9)
        assert gtp.endswith("9")

    def test_gtp_row_bottom(self) -> None:
        # Bottom row maps to GTP row 1
        gtp = coord_to_gtp(8, 0, 9)
        assert gtp.endswith("1")

    def test_gtp_to_coord_lowercase(self) -> None:
        row, col = gtp_to_coord("a9", 9)
        assert (row, col) == (0, 0)

    def test_gtp_to_coord_mixed_case(self) -> None:
        row, col = gtp_to_coord("E5", 9)
        assert (row, col) == gtp_to_coord("e5", 9)


# ---------------------------------------------------------------------------
# action_to_gtp / gtp_to_action
# ---------------------------------------------------------------------------


class TestActionConversions:
    """Tests for action index to/from GTP string conversion."""

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_pass_action_round_trip(self, board_size: int) -> None:
        pass_action = board_size**2
        gtp_str = action_to_gtp(pass_action, board_size)
        assert gtp_str == "pass"
        action = gtp_to_action("pass", board_size)
        assert action == pass_action

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_all_board_positions_round_trip(self, board_size: int) -> None:
        for action in range(board_size**2):
            gtp_str = action_to_gtp(action, board_size)
            back = gtp_to_action(gtp_str, board_size)
            assert back == action

    def test_resign_returns_minus_one(self) -> None:
        assert gtp_to_action("resign", 9) == -1

    def test_resign_uppercase(self) -> None:
        assert gtp_to_action("RESIGN", 9) == -1

    def test_gtp_to_action_strips_whitespace(self) -> None:
        assert gtp_to_action("  pass  ", 9) == 81

    @pytest.mark.parametrize(
        "action,board_size,expected_col_letter",
        [
            (0, 9, "A"),
            (1, 9, "B"),
            (7, 9, "H"),
            (8, 9, "J"),  # Skips I
        ],
    )
    def test_action_column_letters(
        self, action: int, board_size: int, expected_col_letter: str
    ) -> None:
        gtp_str = action_to_gtp(action, board_size)
        assert gtp_str[0] == expected_col_letter


# ---------------------------------------------------------------------------
# SimpleGoGame advanced coverage
# ---------------------------------------------------------------------------


class TestSimpleGoGameCaptures:
    """Tests for stone capture mechanics."""

    def test_capture_single_stone(self) -> None:
        game = SimpleGoGame(board_size=9)
        # Surround a white stone at (1,1) with black stones
        # Black plays (1,1)
        game.play(1, 1)  # Black at (1,1)
        game.play(0, 0)  # White somewhere safe
        # Now we need to capture. Set up a surrounded white stone manually.
        game2 = SimpleGoGame(board_size=5)
        # White stone at center (2,2); surround with black
        game2.board[2, 2] = SimpleGoGame.WHITE
        game2.board[1, 2] = SimpleGoGame.BLACK
        game2.board[3, 2] = SimpleGoGame.BLACK
        game2.board[2, 1] = SimpleGoGame.BLACK
        game2.current_player = SimpleGoGame.BLACK
        # Playing (2,3) should capture the white stone
        result = game2.play(2, 3)
        assert result is True
        assert game2.board[2, 2] == SimpleGoGame.EMPTY
        assert game2.captures[SimpleGoGame.BLACK] == 1

    def test_illegal_move_occupied_position(self) -> None:
        game = SimpleGoGame(board_size=9)
        game.play(4, 4)
        result = game.play(4, 4)
        assert result is False

    def test_illegal_move_out_of_bounds_row(self) -> None:
        game = SimpleGoGame(board_size=9)
        result = game.play(-1, 0)
        assert result is False

    def test_illegal_move_out_of_bounds_col(self) -> None:
        game = SimpleGoGame(board_size=9)
        result = game.play(0, 9)
        assert result is False

    def test_illegal_move_out_of_bounds_negative_col(self) -> None:
        game = SimpleGoGame(board_size=9)
        result = game.play(0, -1)
        assert result is False

    def test_suicide_rejected(self) -> None:
        """A move that would leave a stone with no liberty is rejected."""
        game = SimpleGoGame(board_size=5)
        # Surround (0,0) with opponent stones so placing there is suicide
        game.board[0, 1] = SimpleGoGame.WHITE
        game.board[1, 0] = SimpleGoGame.WHITE
        game.current_player = SimpleGoGame.BLACK
        result = game.play(0, 0)
        assert result is False
        assert game.board[0, 0] == SimpleGoGame.EMPTY


class TestSimpleGoGameGetWinner:
    """Tests for get_winner method."""

    def test_black_wins_more_stones(self) -> None:
        game = SimpleGoGame(board_size=5)
        # Fill board heavily with black stones
        for col in range(5):
            game.board[0, col] = SimpleGoGame.BLACK
            game.board[1, col] = SimpleGoGame.BLACK
        # Only one white stone
        game.board[4, 4] = SimpleGoGame.WHITE
        game.current_player = SimpleGoGame.BLACK
        # Black has 10 stones, white has 1 + 6.5 komi = 7.5
        # Black wins
        winner = game.get_winner()
        assert winner == 1

    def test_white_wins_with_komi(self) -> None:
        game = SimpleGoGame(board_size=5)
        # Equal stones, but komi gives white the win
        game.board[0, 0] = SimpleGoGame.BLACK
        game.board[0, 1] = SimpleGoGame.WHITE
        game.current_player = SimpleGoGame.BLACK
        # black=1, white=1+6.5=7.5 -> white wins -> current player (black) returns -1
        winner = game.get_winner()
        assert winner == -1

    def test_is_terminal_delegates_to_is_game_over(self) -> None:
        game = SimpleGoGame(board_size=9)
        assert game.is_terminal() is False
        game.play_pass()
        game.play_pass()
        assert game.is_terminal() is True


class TestSimpleGoGameState:
    """Tests for get_state method."""

    def test_state_current_player_plane_black(self) -> None:
        game = SimpleGoGame(board_size=5)
        state = game.get_state()
        # Plane 16: 1.0 for black
        assert state[16, 0, 0] == 1.0

    def test_state_current_player_plane_white(self) -> None:
        game = SimpleGoGame(board_size=5)
        game.play_pass()  # Switch to white
        state = game.get_state()
        assert state[16, 0, 0] == 0.0

    def test_state_stones_reflected_correctly(self) -> None:
        game = SimpleGoGame(board_size=5)
        game.play(2, 2)  # Black at center
        state = game.get_state()
        # After black plays, it is white's turn
        # Plane 8 (opponent = black) should have a 1 at (2,2)
        assert state[8, 2, 2] == 1.0
        assert state[0, 2, 2] == 0.0

    def test_state_dtype_float32(self) -> None:
        game = SimpleGoGame(board_size=9)
        state = game.get_state()
        assert state.dtype == np.float32


class TestSimpleGoGameIsLegalMove:
    """Tests for _is_legal_move via get_legal_actions."""

    def test_occupied_position_not_legal(self) -> None:
        game = SimpleGoGame(board_size=9)
        game.board[4, 4] = SimpleGoGame.BLACK
        legal = game.get_legal_actions()
        assert 4 * 9 + 4 not in legal

    def test_pass_always_legal(self) -> None:
        game = SimpleGoGame(board_size=9)
        legal = game.get_legal_actions()
        assert game.board_size**2 in legal

    def test_capture_enables_otherwise_surrounded_position(self) -> None:
        """A position that looks like suicide but captures an opponent group is legal."""
        game = SimpleGoGame(board_size=5)
        # Place black stones surrounding (2,2), leaving (2,3) for the capturing move
        # White has one stone at (2,2) with no liberty if (2,3) is played
        game.board[2, 2] = SimpleGoGame.WHITE
        game.board[1, 2] = SimpleGoGame.BLACK
        game.board[3, 2] = SimpleGoGame.BLACK
        game.board[2, 1] = SimpleGoGame.BLACK
        game.current_player = SimpleGoGame.BLACK
        # (2,3) would normally have no liberty but captures (2,2) first
        assert game._is_legal_move(2, 3) is True


# ---------------------------------------------------------------------------
# GTPEngine – initialisation
# ---------------------------------------------------------------------------


class TestGTPEngineInit:
    """Tests for GTPEngine construction."""

    def test_init_without_model_uses_random_evaluator(self) -> None:
        with (
            patch("src.tools.gtp.RandomEvaluator") as mock_re,
            patch("src.tools.gtp.MCTS"),
        ):
            eng = GTPEngine(model=None, board_size=9, device="cpu")
        mock_re.assert_called_once_with(9**2 + 1)
        assert eng.model is None

    def test_init_with_model_uses_fnet_evaluator(self) -> None:
        fake_model = MagicMock()
        with (
            patch("src.tools.gtp.FNetEvaluator") as mock_fe,
            patch("src.tools.gtp.MCTS"),
        ):
            eng = GTPEngine(model=fake_model, board_size=9, device="cpu")
        mock_fe.assert_called_once_with(fake_model, device="cpu")
        assert eng.model is fake_model

    def test_quit_flag_initially_false(self, engine: GTPEngine) -> None:
        assert engine._quit_flag is False

    def test_commands_dict_contains_expected_keys(self, engine: GTPEngine) -> None:
        expected = {
            "protocol_version",
            "name",
            "version",
            "known_command",
            "list_commands",
            "quit",
            "boardsize",
            "clear_board",
            "komi",
            "play",
            "genmove",
            "showboard",
        }
        assert expected == set(engine.commands.keys())


# ---------------------------------------------------------------------------
# GTPEngine – response formatting
# ---------------------------------------------------------------------------


class TestResponseFormatting:
    """Tests for _success_response and _error_response."""

    def test_success_with_id_and_result(self, engine: GTPEngine) -> None:
        resp = engine._success_response("1", "AlphaGalerkin")
        assert resp == "=1 AlphaGalerkin\n\n"

    def test_success_without_id(self, engine: GTPEngine) -> None:
        resp = engine._success_response(None, "AlphaGalerkin")
        assert resp == "= AlphaGalerkin\n\n"

    def test_success_empty_result_with_id(self, engine: GTPEngine) -> None:
        resp = engine._success_response("5", "")
        assert resp == "=5\n\n"

    def test_success_empty_result_without_id(self, engine: GTPEngine) -> None:
        resp = engine._success_response(None, "")
        assert resp == "=\n\n"

    def test_error_with_id(self, engine: GTPEngine) -> None:
        resp = engine._error_response("3", "illegal move")
        assert resp == "?3 illegal move\n\n"

    def test_error_without_id(self, engine: GTPEngine) -> None:
        resp = engine._error_response(None, "unknown command: foo")
        assert resp == "? unknown command: foo\n\n"


# ---------------------------------------------------------------------------
# GTPEngine – process_command parsing
# ---------------------------------------------------------------------------


class TestProcessCommandParsing:
    """Tests for command parsing in process_command."""

    def test_empty_line_returns_empty_string(self, engine: GTPEngine) -> None:
        assert engine.process_command("") == ""

    def test_whitespace_only_returns_empty_string(self, engine: GTPEngine) -> None:
        assert engine.process_command("   ") == ""

    def test_comment_line_returns_empty_string(self, engine: GTPEngine) -> None:
        assert engine.process_command("# this is a comment") == ""

    def test_unknown_command_returns_error(self, engine: GTPEngine) -> None:
        resp = engine.process_command("nonexistent_command")
        assert resp.startswith("?")
        assert "unknown command" in resp

    def test_command_id_is_parsed(self, engine: GTPEngine) -> None:
        resp = engine.process_command("42 name")
        assert "=42" in resp

    def test_command_id_reflected_in_error(self, engine: GTPEngine) -> None:
        resp = engine.process_command("99 bad_command")
        assert "?99" in resp

    def test_command_case_insensitive(self, engine: GTPEngine) -> None:
        resp = engine.process_command("NAME")
        assert "AlphaGalerkin" in resp

    def test_line_with_only_id_returns_empty(self, engine: GTPEngine) -> None:
        # A line that has only a digit token parses as id with no command
        resp = engine.process_command("42")
        assert resp == ""

    def test_exception_in_handler_returns_error(self, engine: GTPEngine) -> None:
        # boardsize with invalid argument raises ValueError
        resp = engine.process_command("boardsize notanumber")
        assert resp.startswith("?")


# ---------------------------------------------------------------------------
# GTPEngine – individual command handlers
# ---------------------------------------------------------------------------


class TestGTPCommandHandlers:
    """Tests for each GTP command handler."""

    def test_protocol_version(self, engine: GTPEngine) -> None:
        resp = engine.process_command("protocol_version")
        assert "= 2" in resp

    def test_name(self, engine: GTPEngine) -> None:
        resp = engine.process_command("name")
        assert "AlphaGalerkin" in resp

    def test_version(self, engine: GTPEngine) -> None:
        resp = engine.process_command("version")
        assert "0.1.0" in resp

    def test_known_command_true(self, engine: GTPEngine) -> None:
        resp = engine.process_command("known_command name")
        assert "true" in resp

    def test_known_command_false(self, engine: GTPEngine) -> None:
        resp = engine.process_command("known_command unknown_cmd")
        assert "false" in resp

    def test_known_command_case_insensitive(self, engine: GTPEngine) -> None:
        resp = engine.process_command("known_command NAME")
        assert "true" in resp

    def test_list_commands_contains_all(self, engine: GTPEngine) -> None:
        resp = engine.process_command("list_commands")
        for cmd in engine.commands:
            assert cmd in resp

    def test_quit_sets_flag(self, engine: GTPEngine) -> None:
        engine.process_command("quit")
        assert engine._quit_flag is True

    def test_quit_response_format(self, engine: GTPEngine) -> None:
        resp = engine.process_command("quit")
        assert resp == "=\n\n"

    def test_clear_board(self, engine: GTPEngine) -> None:
        engine.game.play(0, 0)
        engine.process_command("clear_board")
        assert (engine.game.board == SimpleGoGame.EMPTY).all()
        engine.mcts.reset.assert_called()

    def test_komi_updates_game(self, engine: GTPEngine) -> None:
        engine.process_command("komi 7.5")
        assert engine.game.komi == pytest.approx(7.5)

    @pytest.mark.parametrize("komi_value", ["0.5", "6.5", "7.5", "0.0"])
    def test_komi_various_values(self, engine: GTPEngine, komi_value: str) -> None:
        engine.process_command(f"komi {komi_value}")
        assert engine.game.komi == pytest.approx(float(komi_value))

    def test_boardsize_valid(self, engine: GTPEngine) -> None:
        resp = engine.process_command("boardsize 13")
        assert resp.startswith("=")
        assert engine.board_size == 13
        assert engine.game.board_size == 13

    @pytest.mark.parametrize("bad_size", ["1", "26", "0"])
    def test_boardsize_invalid(self, engine: GTPEngine, bad_size: str) -> None:
        resp = engine.process_command(f"boardsize {bad_size}")
        assert resp.startswith("?")

    def test_boardsize_resets_mcts(self, engine: GTPEngine) -> None:
        engine.process_command("boardsize 13")
        engine.mcts.reset.assert_called()

    def test_boardsize_updates_random_evaluator(self) -> None:
        """When boardsize changes, RandomEvaluator is recreated with new action count."""
        from src.mcts.evaluator import RandomEvaluator

        with patch("src.tools.gtp.MCTS") as mock_mcts_cls:
            mock_mcts_cls.return_value = MagicMock()
            eng = GTPEngine(model=None, board_size=9, device="cpu")
        # Confirm evaluator is a real RandomEvaluator instance before boardsize change
        assert isinstance(eng.evaluator, RandomEvaluator)
        eng.process_command("boardsize 13")
        # After boardsize 13, evaluator should be re-created with 13^2+1 = 170 actions
        assert isinstance(eng.evaluator, RandomEvaluator)
        assert eng.evaluator.n_actions == 13**2 + 1

    def test_play_black_move(self, engine: GTPEngine) -> None:
        resp = engine.process_command("play black A9")
        assert resp.startswith("=")
        assert engine.game.board[0, 0] == SimpleGoGame.BLACK

    def test_play_white_move(self, engine: GTPEngine) -> None:
        engine.game.play(0, 0)  # Advance to white's turn
        engine.game.current_player = SimpleGoGame.WHITE
        resp = engine.process_command("play white B9")
        assert resp.startswith("=")
        assert engine.game.board[0, 1] == SimpleGoGame.WHITE

    def test_play_pass_black(self, engine: GTPEngine) -> None:
        resp = engine.process_command("play black pass")
        assert resp.startswith("=")
        assert engine.game.move_history[-1] is None

    def test_play_pass_white(self, engine: GTPEngine) -> None:
        engine.game.current_player = SimpleGoGame.WHITE
        resp = engine.process_command("play white pass")
        assert resp.startswith("=")

    def test_play_resign(self, engine: GTPEngine) -> None:
        resp = engine.process_command("play black resign")
        assert resp.startswith("=")

    def test_play_illegal_move_returns_error(self, engine: GTPEngine) -> None:
        engine.game.board[0, 0] = SimpleGoGame.BLACK
        resp = engine.process_command("play white A9")
        assert resp.startswith("?")
        assert "illegal move" in resp

    def test_play_advances_mcts(self, engine: GTPEngine) -> None:
        engine.process_command("play black A9")
        engine.mcts.advance.assert_called()

    def test_play_resign_does_not_advance_mcts(self, engine: GTPEngine) -> None:
        # resign has action -1, so advance should not be called
        engine.mcts.advance.reset_mock()
        engine.process_command("play black resign")
        engine.mcts.advance.assert_not_called()

    def test_showboard_contains_column_labels(self, engine: GTPEngine) -> None:
        resp = engine.process_command("showboard")
        assert "A" in resp
        assert "B" in resp

    def test_showboard_contains_row_numbers(self, engine: GTPEngine) -> None:
        resp = engine.process_command("showboard")
        assert "9" in resp
        assert "1" in resp

    def test_showboard_empty_board_all_dots(self, engine: GTPEngine) -> None:
        resp = engine.process_command("showboard")
        # Empty board positions shown as dots
        assert "." in resp

    def test_showboard_reflects_black_stone(self, engine: GTPEngine) -> None:
        engine.game.board[0, 0] = SimpleGoGame.BLACK
        resp = engine.process_command("showboard")
        assert "X" in resp

    def test_showboard_reflects_white_stone(self, engine: GTPEngine) -> None:
        engine.game.board[0, 0] = SimpleGoGame.WHITE
        resp = engine.process_command("showboard")
        assert "O" in resp


# ---------------------------------------------------------------------------
# GTPEngine – genmove
# ---------------------------------------------------------------------------


class TestGenmove:
    """Tests for _genmove command."""

    def test_genmove_black_returns_valid_move(self, engine: GTPEngine) -> None:
        engine.mcts.get_action.return_value = 0  # A9
        resp = engine.process_command("genmove black")
        assert resp.startswith("=")
        assert "A9" in resp

    def test_genmove_white_returns_valid_move(self, engine: GTPEngine) -> None:
        engine.game.current_player = SimpleGoGame.WHITE
        engine.mcts.get_action.return_value = 0
        resp = engine.process_command("genmove white")
        assert resp.startswith("=")

    def test_genmove_pass_action(self, engine: GTPEngine) -> None:
        engine.mcts.get_action.return_value = engine.board_size**2
        resp = engine.process_command("genmove black")
        assert "pass" in resp

    def test_genmove_applies_move_to_board(self, engine: GTPEngine) -> None:
        engine.mcts.get_action.return_value = 0  # row=0, col=0
        engine.process_command("genmove black")
        assert engine.game.board[0, 0] == SimpleGoGame.BLACK

    def test_genmove_applies_pass_to_game(self, engine: GTPEngine) -> None:
        engine.mcts.get_action.return_value = engine.board_size**2
        engine.process_command("genmove black")
        assert engine.game.move_history[-1] is None

    def test_genmove_calls_mcts_advance(self, engine: GTPEngine) -> None:
        engine.mcts.get_action.return_value = 5
        engine.process_command("genmove black")
        engine.mcts.advance.assert_called_with(5)

    def test_genmove_player_mismatch_logged(
        self, engine: GTPEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        # White's turn expected but game says black
        engine.game.current_player = SimpleGoGame.BLACK
        engine.mcts.get_action.return_value = 0
        # Should still succeed (just logs warning)
        resp = engine.process_command("genmove white")
        assert resp.startswith("=")

    @pytest.mark.parametrize("color", ["b", "B", "black", "BLACK", "Black"])
    def test_genmove_black_color_variants(
        self, engine: GTPEngine, color: str
    ) -> None:
        engine.mcts.get_action.return_value = 0
        resp = engine.process_command(f"genmove {color}")
        assert resp.startswith("=")

    @pytest.mark.parametrize("color", ["w", "W", "white", "WHITE", "White"])
    def test_genmove_white_color_variants(
        self, engine: GTPEngine, color: str
    ) -> None:
        engine.game.current_player = SimpleGoGame.WHITE
        engine.mcts.get_action.return_value = 0
        resp = engine.process_command(f"genmove {color}")
        assert resp.startswith("=")


# ---------------------------------------------------------------------------
# GTPEngine – run() loop
# ---------------------------------------------------------------------------


class TestGTPEngineRun:
    """Tests for the GTPEngine.run() I/O loop."""

    def test_run_processes_single_command(self, engine: GTPEngine) -> None:
        input_stream = io.StringIO("name\n")
        output_stream = io.StringIO()
        engine.run(input_stream, output_stream)
        assert "AlphaGalerkin" in output_stream.getvalue()

    def test_run_stops_on_eof(self, engine: GTPEngine) -> None:
        input_stream = io.StringIO("")
        output_stream = io.StringIO()
        engine.run(input_stream, output_stream)  # Should not hang

    def test_run_stops_on_quit(self, engine: GTPEngine) -> None:
        input_stream = io.StringIO("quit\nname\n")
        output_stream = io.StringIO()
        engine.run(input_stream, output_stream)
        # 'name' should not be processed after quit
        output = output_stream.getvalue()
        assert "AlphaGalerkin" not in output

    def test_run_processes_multiple_commands(self, engine: GTPEngine) -> None:
        input_stream = io.StringIO("name\nversion\n")
        output_stream = io.StringIO()
        engine.run(input_stream, output_stream)
        output = output_stream.getvalue()
        assert "AlphaGalerkin" in output
        assert "0.1.0" in output

    def test_run_flushes_output(self, engine: GTPEngine) -> None:
        """Verify flush is called on every response."""
        input_stream = io.StringIO("name\n")
        output_stream = MagicMock()
        output_stream.readline = io.StringIO("name\n").readline
        engine.run(input_stream, output_stream)
        output_stream.flush.assert_called()

    def test_run_handles_keyboard_interrupt(self, engine: GTPEngine) -> None:
        """KeyboardInterrupt breaks out of the loop cleanly."""

        class _InterruptStream:
            def readline(self) -> str:
                raise KeyboardInterrupt

        engine.run(_InterruptStream(), io.StringIO())  # Should not raise


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() CLI entry point."""

    def test_main_no_model_creates_engine_and_runs(self) -> None:
        with (
            patch("sys.argv", ["gtp"]),
            patch("src.tools.gtp.GTPEngine") as mock_engine_cls,
        ):
            mock_instance = MagicMock()
            mock_engine_cls.return_value = mock_instance
            from src.tools.gtp import main

            main()
        mock_engine_cls.assert_called_once()
        mock_instance.run.assert_called_once()

    def test_main_model_not_found_exits(self, tmp_path: pytest.TempPathFactory) -> None:
        nonexistent = "/nonexistent/path/model.pt"
        with (
            patch("sys.argv", ["gtp", "--model", nonexistent]),
            patch("src.tools.gtp.GTPEngine"),
            pytest.raises(SystemExit) as exc_info,
        ):
            from src.tools.gtp import main

            main()
        assert exc_info.value.code == 1

    def test_main_model_load_failure_exits(self, tmp_path: pytest.TempPathFactory) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        with (
            patch("sys.argv", ["gtp", "--model", str(model_file)]),
            patch(
                "src.training.checkpoint.create_model_from_checkpoint",
                side_effect=RuntimeError("load failed"),
            ),
            patch("src.tools.gtp.GTPEngine"),
            pytest.raises(SystemExit) as exc_info,
        ):
            from src.tools.gtp import main

            main()
        assert exc_info.value.code == 1

    def test_main_with_board_size_arg(self) -> None:
        with (
            patch("sys.argv", ["gtp", "--board-size", "9"]),
            patch("src.tools.gtp.GTPEngine") as mock_engine_cls,
        ):
            mock_instance = MagicMock()
            mock_engine_cls.return_value = mock_instance
            from src.tools.gtp import main

            main()
        call_kwargs = mock_engine_cls.call_args
        board_size_passed = call_kwargs.kwargs.get(
            "board_size",
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None,
        )
        assert board_size_passed == 9 or 9 in str(call_kwargs)

    def test_main_model_loaded_successfully(self, tmp_path: pytest.TempPathFactory) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        fake_model = MagicMock()
        with (
            patch("sys.argv", ["gtp", "--model", str(model_file)]),
            patch(
                "src.training.checkpoint.create_model_from_checkpoint",
                return_value=(fake_model, {"config": "data"}),
            ),
            patch("src.tools.gtp.GTPEngine") as mock_engine_cls,
        ):
            mock_instance = MagicMock()
            mock_engine_cls.return_value = mock_instance
            from src.tools.gtp import main

            main()
        mock_engine_cls.assert_called_once()
        mock_instance.run.assert_called_once()

    def test_main_model_loaded_no_config(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test branch where create_model_from_checkpoint returns empty config."""
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        fake_model = MagicMock()
        with (
            patch("sys.argv", ["gtp", "--model", str(model_file)]),
            patch(
                "src.training.checkpoint.create_model_from_checkpoint",
                return_value=(fake_model, {}),
            ),
            patch("src.tools.gtp.GTPEngine") as mock_engine_cls,
        ):
            mock_instance = MagicMock()
            mock_engine_cls.return_value = mock_instance
            from src.tools.gtp import main

            main()
        mock_engine_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: full command sequence
# ---------------------------------------------------------------------------


class TestGTPCommandSequence:
    """Integration tests simulating realistic GTP sessions."""

    def test_full_game_setup_sequence(self, engine: GTPEngine) -> None:
        engine.mcts.get_action.return_value = 0
        cmds = [
            ("boardsize 9", "="),
            ("clear_board", "="),
            ("komi 6.5", "="),
            ("play black A9", "="),
            ("genmove white", "="),
        ]
        for cmd, expected_prefix in cmds:
            resp = engine.process_command(cmd)
            assert resp.startswith(expected_prefix), f"Failed for: {cmd!r} -> {resp!r}"

    def test_multiple_plays_then_showboard(self, engine: GTPEngine) -> None:
        engine.process_command("play black A9")
        engine.process_command("play white B9")
        resp = engine.process_command("showboard")
        assert "X" in resp
        assert "O" in resp

    def test_two_passes_game_over(self, engine: GTPEngine) -> None:
        engine.process_command("play black pass")
        engine.process_command("play white pass")
        assert engine.game.is_game_over()

    def test_clear_board_after_moves(self, engine: GTPEngine) -> None:
        engine.process_command("play black A9")
        engine.process_command("play white B9")
        engine.process_command("clear_board")
        assert (engine.game.board == SimpleGoGame.EMPTY).all()
        assert engine.game.current_player == SimpleGoGame.BLACK

    def test_numbered_commands_sequence(self, engine: GTPEngine) -> None:
        resp1 = engine.process_command("1 name")
        resp2 = engine.process_command("2 version")
        assert "=1 AlphaGalerkin" in resp1
        assert "=2 0.1.0" in resp2
