"""Tests for the analysis <-> Go-engine / model bridge."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.go_adapter import (
    BLACK_MARK,
    EMPTY_MARK,
    WHITE_MARK,
    action_to_move,
    build_checkpoint_model_evaluator,
    make_model_evaluator,
    move_to_action,
    reconstruct_board,
)


class TestCoordinateMapping:
    """The reviewer's (x, y) convention must round-trip with engine actions."""

    @given(
        board_size=st.integers(min_value=1, max_value=19),
        data=st.data(),
    )
    def test_move_action_round_trip(self, board_size: int, data: st.DataObject) -> None:
        x = data.draw(st.integers(min_value=0, max_value=board_size - 1))
        y = data.draw(st.integers(min_value=0, max_value=board_size - 1))

        action = move_to_action(x, y, board_size)
        assert action == y * board_size + x
        assert action_to_move(action, board_size) == (x, y)

    def test_action_indexes_row_major(self) -> None:
        # action = y * size + x  ->  top-left (0,0) is 0; (1,0) is 1; (0,1) is size.
        assert move_to_action(0, 0, 9) == 0
        assert move_to_action(1, 0, 9) == 1
        assert move_to_action(0, 1, 9) == 9


class TestReconstructBoard:
    """Capture/ko-correct reconstruction, not a naive replay."""

    def test_basic_placement(self) -> None:
        moves = [("B", 3, 3), ("W", 15, 3)]
        board = reconstruct_board(moves, 19)

        assert board[3][3] == BLACK_MARK  # board[y][x]
        assert board[3][15] == WHITE_MARK
        assert board[0][0] == EMPTY_MARK

    def test_capture_removes_surrounded_stone(self) -> None:
        # White stone at the centre of a 3x3 board, surrounded by black -> captured.
        moves = [
            ("W", 1, 1),  # centre
            ("B", 0, 1),
            ("B", 2, 1),
            ("B", 1, 0),
            ("B", 1, 2),  # this move removes white's last liberty
        ]
        board = reconstruct_board(moves, 3)

        assert board[1][1] == EMPTY_MARK, "captured white stone must be removed"
        assert board[1][0] == BLACK_MARK
        assert board[1][2] == BLACK_MARK
        assert board[0][1] == BLACK_MARK
        assert board[2][1] == BLACK_MARK

    def test_naive_replay_would_keep_captured_stone(self) -> None:
        # Guard against regressing to the old naive replay: the captured point
        # must be empty even though a stone was explicitly placed there.
        moves = [
            ("W", 1, 1),
            ("B", 0, 1),
            ("B", 2, 1),
            ("B", 1, 0),
            ("B", 1, 2),
        ]
        board = reconstruct_board(moves, 3)
        white_count = sum(row.count(WHITE_MARK) for row in board)
        assert white_count == 0

    def test_pass_and_out_of_bounds_ignored(self) -> None:
        moves = [("B", 3, 3), ("W", -1, -1), ("B", 99, 99)]
        board = reconstruct_board(moves, 19)
        assert board[3][3] == BLACK_MARK
        # No exception and no stray stones from the sentinels.
        assert sum(row.count(BLACK_MARK) for row in board) == 1

    def test_illegal_occupied_move_skipped(self) -> None:
        # Second move targets an occupied point; reconstruction skips it.
        moves = [("B", 3, 3), ("W", 3, 3)]
        board = reconstruct_board(moves, 19)
        assert board[3][3] == BLACK_MARK


class _FakeMCTSEvaluator:
    """Minimal stand-in for FNetEvaluator exposing .evaluate -> (value, policy)."""

    def __init__(self, value: float, board_size: int, hot_action: int) -> None:
        self.value = value
        self.n_actions = board_size * board_size + 1
        self.hot_action = hot_action
        self.seen_state: np.ndarray | None = None
        self.seen_legal: list[int] | None = None

    def evaluate(self, state: np.ndarray, legal_actions: list[int]) -> SimpleNamespace:
        self.seen_state = state
        self.seen_legal = legal_actions
        policy = np.full(self.n_actions, 0.01, dtype=np.float32)
        policy[self.hot_action] = 0.9
        return SimpleNamespace(value=self.value, policy=policy)


class TestMakeModelEvaluator:
    """Adapter from an MCTS evaluator to the PositionEvaluator callable contract."""

    def test_returns_value_and_policy(self) -> None:
        board_size = 5
        fake = _FakeMCTSEvaluator(value=0.4, board_size=board_size, hot_action=12)
        evaluator = make_model_evaluator(fake)

        board_state = [[EMPTY_MARK] * board_size for _ in range(board_size)]
        board_state[0][0] = BLACK_MARK

        value, policy = evaluator(board_state)

        assert value == pytest.approx(0.4)
        assert policy.shape == (board_size * board_size + 1,)
        assert int(np.argmax(policy)) == 12

    def test_encodes_board_and_legal_actions(self) -> None:
        board_size = 5
        fake = _FakeMCTSEvaluator(value=0.0, board_size=board_size, hot_action=0)
        evaluator = make_model_evaluator(fake)

        board_state = [[EMPTY_MARK] * board_size for _ in range(board_size)]
        board_state[2][2] = BLACK_MARK  # occupied point not legal

        evaluator(board_state)

        assert fake.seen_state is not None
        # GoGame encodes 17 feature planes of shape (size, size).
        assert fake.seen_state.shape == (17, board_size, board_size)
        # Occupied centre (action = 2*5 + 2 = 12) is excluded from legal actions.
        assert fake.seen_legal is not None
        assert 12 not in fake.seen_legal
        # Pass is always legal.
        assert board_size * board_size in fake.seen_legal


class TestBuildCheckpointModelEvaluator:
    """End-to-end wiring is delegated to create_model_from_checkpoint + FNetEvaluator."""

    def test_missing_checkpoint_raises(self, tmp_path) -> None:
        with pytest.raises(Exception):
            build_checkpoint_model_evaluator(str(tmp_path / "nope.pt"), device="cpu")
