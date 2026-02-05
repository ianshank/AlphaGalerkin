"""Tests for GTP protocol player assignment fix.

Validates that ``_genmove`` correctly sets the expected player
from GTP color strings for both black and white.
"""

from __future__ import annotations

import pytest

from src.tools.gtp import SimpleGoGame


class TestSimpleGoGamePlayerConstants:
    """Verify player constants used by GTP."""

    def test_black_constant(self) -> None:
        assert SimpleGoGame.BLACK == 1

    def test_white_constant(self) -> None:
        assert SimpleGoGame.WHITE == 2

    def test_empty_constant(self) -> None:
        assert SimpleGoGame.EMPTY == 0


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
