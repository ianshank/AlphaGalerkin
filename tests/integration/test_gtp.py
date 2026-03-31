"""Integration tests for GTP interface."""

from __future__ import annotations

import pytest

from src.tools.gtp import (
    GTPEngine,
    SimpleGoGame,
    action_to_gtp,
    coord_to_gtp,
    gtp_to_action,
    gtp_to_coord,
)


class TestCoordinateConversion:
    """Tests for GTP coordinate conversion."""

    def test_coord_to_gtp(self) -> None:
        """Test internal to GTP coordinate conversion."""
        # Top-left corner (internal 0,0) -> A19 (for 19x19)
        assert coord_to_gtp(0, 0, 19) == "A19"

        # Bottom-right corner (internal 18,18) -> T1 (for 19x19)
        assert coord_to_gtp(18, 18, 19) == "T1"

        # D4 position
        assert coord_to_gtp(15, 3, 19) == "D4"

    def test_gtp_to_coord(self) -> None:
        """Test GTP to internal coordinate conversion."""
        # A19 -> (0, 0)
        assert gtp_to_coord("A19", 19) == (0, 0)

        # T1 -> (18, 18)
        assert gtp_to_coord("T1", 19) == (18, 18)

        # D4 -> (15, 3)
        assert gtp_to_coord("D4", 19) == (15, 3)

    def test_roundtrip(self) -> None:
        """Test coordinate conversion roundtrip."""
        for row in range(19):
            for col in range(19):
                gtp = coord_to_gtp(row, col, 19)
                r, c = gtp_to_coord(gtp, 19)
                assert (r, c) == (row, col)

    def test_action_conversion(self) -> None:
        """Test action index conversion."""
        # Position 0 -> A19
        assert action_to_gtp(0, 19) == "A19"

        # Pass move
        assert action_to_gtp(361, 19) == "pass"

        # Roundtrip
        for action in [0, 1, 180, 360, 361]:
            gtp = action_to_gtp(action, 19)
            recovered = gtp_to_action(gtp, 19)
            assert recovered == action


class TestSimpleGoGame:
    """Tests for simple Go game implementation."""

    def test_initial_state(self) -> None:
        """Test initial game state."""
        game = SimpleGoGame(9)

        assert game.board_size == 9
        assert game.current_player == SimpleGoGame.BLACK
        assert (game.board == 0).all()

    def test_play_stone(self) -> None:
        """Test playing a stone."""
        game = SimpleGoGame(9)

        assert game.play(4, 4)  # Play at center
        assert game.board[4, 4] == SimpleGoGame.BLACK
        assert game.current_player == SimpleGoGame.WHITE

    def test_illegal_move_occupied(self) -> None:
        """Test that occupied position is illegal."""
        game = SimpleGoGame(9)

        game.play(4, 4)
        assert not game.play(4, 4)  # Can't play on occupied

    def test_pass(self) -> None:
        """Test pass move."""
        game = SimpleGoGame(9)

        game.play_pass()
        assert game.current_player == SimpleGoGame.WHITE
        assert game.passes == 1

    def test_game_over(self) -> None:
        """Test game over detection."""
        game = SimpleGoGame(9)

        assert not game.is_game_over()

        game.play_pass()
        assert not game.is_game_over()

        game.play_pass()
        assert game.is_game_over()

    def test_get_state(self) -> None:
        """Test state tensor generation."""
        game = SimpleGoGame(9)
        game.play(4, 4)  # Black plays center

        state = game.get_state()

        assert state.shape == (17, 9, 9)
        # Plane 8 should have white's perspective of black's stone
        # (current player is now white)
        assert state[8, 4, 4] == 1.0  # Opponent's stone

    def test_clone(self) -> None:
        """Test game cloning."""
        game = SimpleGoGame(9)
        game.play(4, 4)
        game.play(3, 3)

        clone = game.clone()

        # Modify original
        game.play(5, 5)

        # Clone should be unchanged
        assert clone.board[5, 5] == SimpleGoGame.EMPTY

    def test_suicide_not_in_legal_actions(self) -> None:
        """Test that suicide moves are not included in legal actions.

        Set up a position where playing in a corner would be suicide:
           0 1 2
        0  . X .
        1  X . .
        2  . . .

        Playing at (0,0) would be suicide for Black.
        """
        game = SimpleGoGame(9)

        # White surrounds corner (0,0)
        game.board[0, 1] = SimpleGoGame.WHITE  # Right of corner
        game.board[1, 0] = SimpleGoGame.WHITE  # Below corner
        game.current_player = SimpleGoGame.BLACK

        legal_actions = game.get_legal_actions()
        corner_action = 0  # Action for (0,0)

        # Corner should NOT be in legal actions (suicide)
        assert corner_action not in legal_actions

        # But other empty positions should be legal
        assert 2 in legal_actions  # (0,2) is empty and legal
        assert game.board_size**2 in legal_actions  # Pass is always legal

    def test_capture_move_is_legal(self) -> None:
        """Test that capturing moves are legal even if position looks like suicide.

        Set up:
           0 1 2
        0  . W B
        1  B B .
        2  . . .

        White at (0,1), Black at (0,2), (1,0), (1,1).
        White's only liberty is (0,0).
        Black plays at (0,0):
        - Would normally look like suicide (surrounded by stones)
        - But captures White at (0,1) first, giving Black liberty
        - So this move should be legal.
        """
        game = SimpleGoGame(9)

        # Set up: White in atari (one liberty), Black can capture
        game.board[0, 1] = SimpleGoGame.WHITE  # White stone with one liberty at (0,0)
        game.board[0, 2] = SimpleGoGame.BLACK  # Black surrounds White
        game.board[1, 0] = SimpleGoGame.BLACK  # Black surrounds corner
        game.board[1, 1] = SimpleGoGame.BLACK  # Black surrounds White (removes (1,1) liberty)
        game.current_player = SimpleGoGame.BLACK

        legal_actions = game.get_legal_actions()
        capture_action = 0  # Action for (0,0) - captures White

        # This move captures White first, so it's NOT suicide
        assert capture_action in legal_actions

        # Verify the capture actually works
        assert game.play(0, 0)
        assert game.board[0, 1] == SimpleGoGame.EMPTY  # White was captured

    def test_all_legal_actions_are_playable(self) -> None:
        """Test that every action from get_legal_actions() succeeds with play()."""
        game = SimpleGoGame(5)  # Small board for speed

        # Play some random moves to create an interesting position
        moves = [(0, 0), (1, 1), (0, 1), (2, 2), (1, 0), (3, 3)]
        for row, col in moves:
            game.play(row, col)

        # Get legal actions
        legal = game.get_legal_actions()
        pass_action = game.board_size**2

        # Every non-pass legal action should succeed
        for action in legal:
            if action == pass_action:
                continue
            test_game = game.clone()
            row = action // game.board_size
            col = action % game.board_size
            result = test_game.play(row, col)
            assert (
                result
            ), f"Action {action} at ({row},{col}) was in legal_actions but play() returned False"


class TestGTPEngine:
    """Tests for GTP engine."""

    @pytest.fixture
    def engine(self) -> GTPEngine:
        """Create GTP engine with random play."""
        return GTPEngine(model=None, board_size=9)

    def test_protocol_version(self, engine: GTPEngine) -> None:
        """Test protocol version command."""
        response = engine.process_command("protocol_version")
        assert "= 2" in response

    def test_name(self, engine: GTPEngine) -> None:
        """Test name command."""
        response = engine.process_command("name")
        assert "AlphaGalerkin" in response

    def test_boardsize(self, engine: GTPEngine) -> None:
        """Test boardsize command."""
        response = engine.process_command("boardsize 19")
        assert response.startswith("=")
        assert engine.board_size == 19

    def test_clear_board(self, engine: GTPEngine) -> None:
        """Test clear_board command."""
        engine.process_command("play black D4")
        engine.process_command("clear_board")

        assert (engine.game.board == 0).all()

    def test_play(self, engine: GTPEngine) -> None:
        """Test play command."""
        response = engine.process_command("play black D4")
        assert response.startswith("=")

        # Verify stone was placed
        row, col = gtp_to_coord("D4", 9)
        assert engine.game.board[row, col] == SimpleGoGame.BLACK

    def test_genmove(self, engine: GTPEngine) -> None:
        """Test genmove command."""
        response = engine.process_command("genmove black")

        # Should return a valid move
        assert response.startswith("=")
        # Move should be either a coordinate or "pass"
        move = response.strip().split()[-1]
        assert move == "pass" or (len(move) >= 2 and move[0].isalpha())

    def test_showboard(self, engine: GTPEngine) -> None:
        """Test showboard command."""
        engine.process_command("play black D4")
        response = engine.process_command("showboard")

        assert "X" in response  # Black stone marker

    def test_known_command(self, engine: GTPEngine) -> None:
        """Test known_command."""
        assert "true" in engine.process_command("known_command play")
        assert "false" in engine.process_command("known_command invalid_cmd")

    def test_command_with_id(self, engine: GTPEngine) -> None:
        """Test command with ID."""
        response = engine.process_command("123 name")
        assert "=123" in response

    def test_error_response(self, engine: GTPEngine) -> None:
        """Test error response format."""
        response = engine.process_command("invalid_command")
        assert response.startswith("?")

    def test_full_game_sequence(self, engine: GTPEngine) -> None:
        """Test a sequence of moves."""
        commands = [
            "boardsize 9",
            "clear_board",
            "play black D5",
            "play white E5",
            "play black D4",
            "play white E4",
            "genmove black",
        ]

        for cmd in commands:
            response = engine.process_command(cmd)
            assert response.startswith("="), f"Command '{cmd}' failed: {response}"
