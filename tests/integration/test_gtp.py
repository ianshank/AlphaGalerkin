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
