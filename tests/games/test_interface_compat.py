"""Tests for backwards-compatibility of ``string_to_action`` (PR #53 reviews).

Covers:
- gemini ``src/games/interface.py:304``: extract ``board_size`` from a
  ``GameState`` instead of falling back to the default.
- Copilot ``src/games/interface.py:289`` and ``src/games/chess.py:1175``:
  the old ``board_size=`` keyword must still work, with a
  ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.games.chess import ChessGame
from src.games.go import GoGame
from src.games.state import GameState


def _make_go_state(board_size: int) -> GameState:
    """Build an empty Go state of the requested size."""
    board = np.zeros((board_size, board_size), dtype=np.int8)
    return GameState(
        board=board,
        current_player=1,
        move_number=0,
        move_history=[],
        metadata={"game_type": "go", "board_size": board_size},
    )


class TestStringToActionStateExtraction:
    """gemini-code-assist: state.board_size must be honoured."""

    def test_state_overrides_default_board_size(self) -> None:
        game = GoGame()  # default_board_size=19
        state_9 = _make_go_state(9)

        # "A1" on 9×9 → row=8, col=0 → action 72
        action = game.string_to_action("A1", state_9)
        assert action == 8 * 9 + 0

    def test_int_size_still_works(self) -> None:
        game = GoGame()
        action = game.string_to_action("A1", 9)
        assert action == 8 * 9 + 0

    def test_default_when_neither_provided(self) -> None:
        game = GoGame()
        action = game.string_to_action("pass")
        assert action == 19 * 19


class TestStringToActionDeprecatedKwarg:
    """Copilot: legacy ``board_size=`` kwarg must still work."""

    def test_legacy_board_size_kwarg_emits_deprecation(self) -> None:
        game = GoGame()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", DeprecationWarning)
            action = game.string_to_action("A1", board_size=9)
        assert action == 8 * 9 + 0
        assert any(
            issubclass(r.category, DeprecationWarning) and "board_size" in str(r.message)
            for r in records
        )

    def test_unexpected_kwarg_raises(self) -> None:
        game = GoGame()
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            game.string_to_action("A1", bogus=42)  # type: ignore[call-arg]

    def test_chess_legacy_kwarg(self) -> None:
        game = ChessGame()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", DeprecationWarning)
            # board_size is ignored for chess but must not raise
            action = game.string_to_action("e2e4", board_size=8)
        # e2e4 maps to a valid action without state legality validation
        assert action is not None
        assert any(issubclass(r.category, DeprecationWarning) for r in records)

    def test_chess_unexpected_kwarg_raises(self) -> None:
        game = ChessGame()
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            game.string_to_action("e2e4", bogus=1)  # type: ignore[call-arg]
