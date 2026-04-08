"""Additional coverage tests for games/state.py.

Covers: __post_init__ edge cases, copy, with_move, hash, eq, to/from_dict,
flip_perspective, ActionMask, create_empty_state.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.state import ActionMask, GameState, create_empty_state


class TestGameStateInit:
    """Test GameState initialization edge cases."""

    def test_board_none_raises(self) -> None:
        with pytest.raises(ValueError, match="Board cannot be None"):
            GameState(board=None)  # type: ignore[arg-type]

    def test_board_list_converted(self) -> None:
        gs = GameState(board=[[1, 0], [0, -1]])
        assert isinstance(gs.board, np.ndarray)

    def test_move_history_none_converted(self) -> None:
        gs = GameState(board=np.zeros((3, 3)), move_history=None)  # type: ignore[arg-type]
        assert gs.move_history == []

    def test_metadata_none_converted(self) -> None:
        gs = GameState(board=np.zeros((3, 3)), metadata=None)  # type: ignore[arg-type]
        assert gs.metadata == {}


class TestGameStateMethods:
    """Test GameState methods."""

    def test_copy_is_deep(self) -> None:
        gs = GameState(board=np.ones((3, 3)), move_history=[1, 2])
        gs2 = gs.copy()
        gs2.board[0, 0] = 99
        assert gs.board[0, 0] == 1.0

    def test_with_move(self) -> None:
        gs = GameState(board=np.zeros((3, 3)), current_player=1)
        new_board = np.ones((3, 3))
        gs2 = gs.with_move(action=5, new_board=new_board)
        assert gs2.current_player == -1
        assert gs2.move_number == 1
        assert gs2.move_history == [5]

    def test_with_move_explicit_player(self) -> None:
        gs = GameState(board=np.zeros((3, 3)), current_player=1)
        gs2 = gs.with_move(action=0, new_board=np.zeros((3, 3)), next_player=1)
        assert gs2.current_player == 1

    def test_with_move_metadata_updates(self) -> None:
        gs = GameState(board=np.zeros((3, 3)), metadata={"a": 1})
        gs2 = gs.with_move(action=0, new_board=np.zeros((3, 3)), b=2)
        assert gs2.metadata["a"] == 1
        assert gs2.metadata["b"] == 2

    def test_board_size(self) -> None:
        gs = GameState(board=np.zeros((5, 5)))
        assert gs.board_size == 5

    def test_last_move_empty(self) -> None:
        gs = GameState(board=np.zeros((3, 3)))
        assert gs.last_move is None

    def test_last_move(self) -> None:
        gs = GameState(board=np.zeros((3, 3)), move_history=[1, 2, 3])
        assert gs.last_move == 3


class TestGameStateHashEq:
    """Test hash and equality."""

    def test_hash_consistent(self) -> None:
        gs = GameState(board=np.array([[1, 0], [0, -1]]), current_player=1)
        assert hash(gs) == hash(gs)

    def test_eq_same(self) -> None:
        board = np.array([[1, 0], [0, -1]])
        gs1 = GameState(board=board.copy(), current_player=1)
        gs2 = GameState(board=board.copy(), current_player=1)
        assert gs1 == gs2

    def test_eq_different_player(self) -> None:
        board = np.zeros((2, 2))
        gs1 = GameState(board=board.copy(), current_player=1)
        gs2 = GameState(board=board.copy(), current_player=-1)
        assert gs1 != gs2

    def test_eq_different_type(self) -> None:
        gs = GameState(board=np.zeros((2, 2)))
        assert gs != "not a state"


class TestGameStateSerialize:
    """Test serialization."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        gs = GameState(
            board=np.array([[1.0, 0.0], [0.0, -1.0]]),
            current_player=-1,
            move_number=5,
            move_history=[0, 1, 2, 3, 4],
            metadata={"key": "value"},
        )
        d = gs.to_dict()
        gs2 = GameState.from_dict(d)
        assert np.array_equal(gs.board, gs2.board)
        assert gs2.current_player == -1
        assert gs2.move_number == 5
        assert gs2.move_history == [0, 1, 2, 3, 4]

    def test_flip_perspective(self) -> None:
        gs = GameState(
            board=np.array([[1.0, -1.0], [0.0, 1.0]]),
            current_player=1,
        )
        flipped = gs.flip_perspective()
        assert flipped.current_player == -1
        assert flipped.board[0, 0] == -1.0


class TestActionMask:
    """Test ActionMask."""

    def test_creation(self) -> None:
        mask = ActionMask(mask=np.array([True, False, True]), action_space_size=3)
        assert mask.num_legal == 2
        assert mask.legal_actions == [0, 2]

    def test_is_legal(self) -> None:
        mask = ActionMask(mask=np.array([True, False, True]), action_space_size=3)
        assert mask.is_legal(0) is True
        assert mask.is_legal(1) is False
        assert mask.is_legal(5) is False  # out of bounds

    def test_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Mask size"):
            ActionMask(mask=np.array([True, False]), action_space_size=3)


class TestCreateEmptyState:
    """Test create_empty_state factory."""

    def test_single_plane(self) -> None:
        gs = create_empty_state(board_size=9)
        assert gs.board.shape == (9, 9)

    def test_multi_plane(self) -> None:
        gs = create_empty_state(board_size=9, n_planes=17)
        assert gs.board.shape == (17, 9, 9)
