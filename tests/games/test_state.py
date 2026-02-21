"""Tests for GameState and related state classes.

Covers:
- GameState initialization and validation
- GameState copy, with_move, flip_perspective
- GameState properties (board_size, last_move)
- GameState hashing and equality
- GameState serialization (to_dict, from_dict)
- ActionMask creation, validation, properties
- create_empty_state helper function
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.state import ActionMask, GameState, create_empty_state


class TestGameStateInit:
    """Tests for GameState initialization and __post_init__ validation."""

    def test_basic_creation(self) -> None:
        """Test creating a GameState with standard arguments."""
        board = np.zeros((9, 9), dtype=np.float32)
        state = GameState(board=board, current_player=1, move_number=0)
        assert state.current_player == 1
        assert state.move_number == 0
        assert state.board.shape == (9, 9)

    def test_board_cannot_be_none(self) -> None:
        """Test that board=None raises ValueError (line 52)."""
        with pytest.raises(ValueError, match="Board cannot be None"):
            GameState(board=None, current_player=1)  # type: ignore[arg-type]

    def test_board_converted_to_ndarray(self) -> None:
        """Test that a non-ndarray board is converted (line 56)."""
        board_list = [[0, 1], [1, 0]]
        state = GameState(board=board_list, current_player=1)  # type: ignore[arg-type]
        assert isinstance(state.board, np.ndarray)
        assert state.board.shape == (2, 2)

    def test_move_history_none_becomes_empty_list(self) -> None:
        """Test that move_history=None is replaced with [] (line 60)."""
        board = np.zeros((3, 3))
        state = GameState(board=board, current_player=1, move_history=None)  # type: ignore[arg-type]
        assert state.move_history == []
        assert isinstance(state.move_history, list)

    def test_metadata_none_becomes_empty_dict(self) -> None:
        """Test that metadata=None is replaced with {} (line 64)."""
        board = np.zeros((3, 3))
        state = GameState(board=board, current_player=1, metadata=None)  # type: ignore[arg-type]
        assert state.metadata == {}
        assert isinstance(state.metadata, dict)

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        board = np.zeros((5, 5))
        state = GameState(board=board)
        assert state.current_player == 1
        assert state.move_number == 0
        assert state.move_history == []
        assert state.metadata == {}


class TestGameStateProperties:
    """Tests for GameState properties."""

    def test_board_size_square(self) -> None:
        """Test board_size property for a square board."""
        board = np.zeros((9, 9))
        state = GameState(board=board)
        assert state.board_size == 9

    def test_board_size_with_planes(self) -> None:
        """Test board_size for a multi-plane board."""
        board = np.zeros((3, 13, 13))
        state = GameState(board=board)
        assert state.board_size == 13

    def test_last_move_with_history(self) -> None:
        """Test last_move when move_history is non-empty."""
        board = np.zeros((5, 5))
        state = GameState(board=board, move_history=[10, 20, 30])
        assert state.last_move == 30

    def test_last_move_empty_history(self) -> None:
        """Test last_move returns None when no moves (line 129)."""
        board = np.zeros((5, 5))
        state = GameState(board=board, move_history=[])
        assert state.last_move is None


class TestGameStateCopy:
    """Tests for GameState.copy method."""

    def test_copy_creates_independent_state(self) -> None:
        """Test copy creates a deep copy."""
        board = np.array([[1, 0], [0, -1]])
        state = GameState(
            board=board,
            current_player=1,
            move_number=5,
            move_history=[1, 2, 3],
            metadata={"key": "value"},
        )
        copied = state.copy()

        assert np.array_equal(copied.board, state.board)
        assert copied.current_player == state.current_player
        assert copied.move_number == state.move_number
        assert copied.move_history == state.move_history
        assert copied.metadata == state.metadata

        # Ensure independence
        copied.board[0, 0] = 99
        assert state.board[0, 0] == 1

        copied.move_history.append(99)
        assert 99 not in state.move_history


class TestGameStateWithMove:
    """Tests for GameState.with_move method."""

    def test_with_move_creates_new_state(self) -> None:
        """Test with_move creates a new state with incremented move number."""
        board = np.zeros((3, 3))
        state = GameState(board=board, current_player=1, move_number=0)

        new_board = np.zeros((3, 3))
        new_board[1, 1] = 1
        new_state = state.with_move(action=4, new_board=new_board)

        assert new_state.move_number == 1
        assert new_state.current_player == -1  # switched
        assert new_state.move_history == [4]
        assert new_state.board[1, 1] == 1

    def test_with_move_explicit_next_player(self) -> None:
        """Test with_move with explicit next_player parameter."""
        board = np.zeros((3, 3))
        state = GameState(board=board, current_player=1, move_number=0)

        new_board = board.copy()
        new_state = state.with_move(action=0, new_board=new_board, next_player=1)
        assert new_state.current_player == 1  # not switched

    def test_with_move_metadata_updates(self) -> None:
        """Test with_move with metadata updates."""
        board = np.zeros((3, 3))
        state = GameState(
            board=board, current_player=1, metadata={"a": 1, "b": 2}
        )

        new_board = board.copy()
        new_state = state.with_move(action=0, new_board=new_board, b=99, c=3)
        assert new_state.metadata["a"] == 1
        assert new_state.metadata["b"] == 99
        assert new_state.metadata["c"] == 3

    def test_with_move_appends_to_history(self) -> None:
        """Test with_move appends action to move history."""
        board = np.zeros((3, 3))
        state = GameState(board=board, move_history=[10, 20])

        new_board = board.copy()
        new_state = state.with_move(action=30, new_board=new_board)
        assert new_state.move_history == [10, 20, 30]


class TestGameStateHash:
    """Tests for GameState hashing."""

    def test_hash_is_deterministic(self) -> None:
        """Test same state always produces same hash."""
        board = np.array([[1, 0], [0, -1]])
        state1 = GameState(board=board.copy(), current_player=1)
        state2 = GameState(board=board.copy(), current_player=1)

        assert hash(state1) == hash(state2)

    def test_different_boards_different_hash(self) -> None:
        """Test different board states produce different hashes."""
        state1 = GameState(board=np.array([[1, 0], [0, 0]]), current_player=1)
        state2 = GameState(board=np.array([[0, 1], [0, 0]]), current_player=1)

        assert hash(state1) != hash(state2)

    def test_different_players_different_hash(self) -> None:
        """Test same board with different player produces different hash."""
        board = np.array([[1, 0], [0, -1]])
        state1 = GameState(board=board.copy(), current_player=1)
        state2 = GameState(board=board.copy(), current_player=-1)

        assert hash(state1) != hash(state2)

    def test_hash_cached(self) -> None:
        """Test hash is cached after first computation."""
        board = np.array([[1, 0], [0, -1]])
        state = GameState(board=board, current_player=1)

        assert state._hash is None
        h = hash(state)
        # _hash stores the full MD5 int; hash() may truncate on some platforms
        assert state._hash is not None
        # Second call should return same value
        assert hash(state) == h

    def test_state_usable_as_dict_key(self) -> None:
        """Test GameState can be used as dictionary key (transposition table)."""
        board = np.array([[1, 0], [0, -1]])
        state = GameState(board=board.copy(), current_player=1)
        d = {state: "value"}
        assert d[state] == "value"


class TestGameStateEquality:
    """Tests for GameState equality."""

    def test_equal_states(self) -> None:
        """Test two equal states compare as equal."""
        board = np.array([[1, 0], [0, -1]])
        state1 = GameState(board=board.copy(), current_player=1)
        state2 = GameState(board=board.copy(), current_player=1)

        assert state1 == state2

    def test_different_board_not_equal(self) -> None:
        """Test states with different boards are not equal."""
        state1 = GameState(board=np.zeros((3, 3)), current_player=1)
        state2 = GameState(board=np.ones((3, 3)), current_player=1)

        assert state1 != state2

    def test_different_player_not_equal(self) -> None:
        """Test states with different current player are not equal."""
        board = np.zeros((3, 3))
        state1 = GameState(board=board.copy(), current_player=1)
        state2 = GameState(board=board.copy(), current_player=-1)

        assert state1 != state2

    def test_not_equal_to_non_gamestate(self) -> None:
        """Test comparing GameState with non-GameState returns False (line 158)."""
        state = GameState(board=np.zeros((3, 3)), current_player=1)

        assert state != "not a state"
        assert state != 42
        assert state != None  # noqa: E711
        assert state != [1, 2, 3]


class TestGameStateSerialization:
    """Tests for GameState to_dict/from_dict."""

    def test_to_dict(self) -> None:
        """Test to_dict produces correct dictionary (line 171)."""
        board = np.array([[1, 0], [0, -1]])
        state = GameState(
            board=board,
            current_player=1,
            move_number=5,
            move_history=[1, 2],
            metadata={"key": "val"},
        )
        d = state.to_dict()

        assert d["board"] == [[1, 0], [0, -1]]
        assert d["current_player"] == 1
        assert d["move_number"] == 5
        assert d["move_history"] == [1, 2]
        assert d["metadata"] == {"key": "val"}

    def test_from_dict(self) -> None:
        """Test from_dict creates correct GameState (line 190)."""
        d = {
            "board": [[1, 0], [0, -1]],
            "current_player": -1,
            "move_number": 3,
            "move_history": [5, 10],
            "metadata": {"foo": "bar"},
        }
        state = GameState.from_dict(d)

        assert np.array_equal(state.board, np.array([[1, 0], [0, -1]]))
        assert state.current_player == -1
        assert state.move_number == 3
        assert state.move_history == [5, 10]
        assert state.metadata == {"foo": "bar"}

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal data (defaults used)."""
        d = {
            "board": [[0, 0], [0, 0]],
            "current_player": 1,
        }
        state = GameState.from_dict(d)

        assert state.move_number == 0
        assert state.move_history == []
        assert state.metadata == {}

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict -> from_dict roundtrip preserves data."""
        board = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
        state = GameState(
            board=board,
            current_player=-1,
            move_number=9,
            move_history=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            metadata={"captures": 3},
        )

        d = state.to_dict()
        restored = GameState.from_dict(d)

        assert np.array_equal(restored.board, state.board)
        assert restored.current_player == state.current_player
        assert restored.move_number == state.move_number
        assert restored.move_history == state.move_history
        assert restored.metadata == state.metadata


class TestGameStateFlipPerspective:
    """Tests for GameState.flip_perspective."""

    def test_flip_perspective(self) -> None:
        """Test flip_perspective negates board and player (line 207)."""
        board = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
        state = GameState(
            board=board,
            current_player=1,
            move_number=6,
            move_history=[0, 1, 2, 3, 4, 5],
            metadata={"key": "val"},
        )

        flipped = state.flip_perspective()

        assert np.array_equal(flipped.board, -board)
        assert flipped.current_player == -1
        assert flipped.move_number == 6
        assert flipped.move_history == [0, 1, 2, 3, 4, 5]
        assert flipped.metadata == {"key": "val"}

    def test_flip_perspective_independence(self) -> None:
        """Test flip_perspective creates independent copies of mutable fields."""
        board = np.array([[1, 0], [0, -1]])
        state = GameState(
            board=board,
            current_player=1,
            move_history=[1, 2],
            metadata={"k": "v"},
        )

        flipped = state.flip_perspective()

        flipped.move_history.append(99)
        flipped.metadata["new"] = "data"

        assert 99 not in state.move_history
        assert "new" not in state.metadata

    def test_double_flip_returns_to_original(self) -> None:
        """Test flipping twice returns to the original board."""
        board = np.array([[1, -1], [-1, 1]])
        state = GameState(board=board, current_player=1)

        double_flipped = state.flip_perspective().flip_perspective()
        assert np.array_equal(double_flipped.board, state.board)
        assert double_flipped.current_player == state.current_player


class TestActionMask:
    """Tests for ActionMask dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating an ActionMask."""
        mask = np.array([True, False, True, False, True])
        am = ActionMask(mask=mask, action_space_size=5)
        assert am.action_space_size == 5

    def test_size_mismatch_raises_error(self) -> None:
        """Test that mask/size mismatch raises ValueError (line 232)."""
        mask = np.array([True, False, True])
        with pytest.raises(ValueError, match="Mask size 3 != action_space_size 5"):
            ActionMask(mask=mask, action_space_size=5)

    def test_legal_actions_property(self) -> None:
        """Test legal_actions returns correct indices (line 244)."""
        mask = np.array([False, True, False, True, False, True])
        am = ActionMask(mask=mask, action_space_size=6)
        assert am.legal_actions == [1, 3, 5]

    def test_legal_actions_none(self) -> None:
        """Test legal_actions with all-False mask."""
        mask = np.array([False, False, False])
        am = ActionMask(mask=mask, action_space_size=3)
        assert am.legal_actions == []

    def test_legal_actions_all(self) -> None:
        """Test legal_actions with all-True mask."""
        mask = np.array([True, True, True])
        am = ActionMask(mask=mask, action_space_size=3)
        assert am.legal_actions == [0, 1, 2]

    def test_num_legal_property(self) -> None:
        """Test num_legal returns correct count."""
        mask = np.array([True, False, True, True, False])
        am = ActionMask(mask=mask, action_space_size=5)
        assert am.num_legal == 3

    def test_num_legal_zero(self) -> None:
        """Test num_legal with no legal actions."""
        mask = np.array([False, False])
        am = ActionMask(mask=mask, action_space_size=2)
        assert am.num_legal == 0

    def test_is_legal_valid_action(self) -> None:
        """Test is_legal for valid actions."""
        mask = np.array([True, False, True])
        am = ActionMask(mask=mask, action_space_size=3)
        assert am.is_legal(0) is True
        assert am.is_legal(1) is False
        assert am.is_legal(2) is True

    def test_is_legal_out_of_bounds(self) -> None:
        """Test is_legal returns False for out-of-bounds (line 268)."""
        mask = np.array([True, True])
        am = ActionMask(mask=mask, action_space_size=2)
        assert am.is_legal(-1) is False
        assert am.is_legal(2) is False
        assert am.is_legal(100) is False


class TestCreateEmptyState:
    """Tests for create_empty_state helper function."""

    def test_single_plane(self) -> None:
        """Test creating empty state with 1 plane (2D board)."""
        state = create_empty_state(board_size=9, n_planes=1)
        assert state.board.shape == (9, 9)
        assert np.all(state.board == 0)
        assert state.current_player == 1
        assert state.move_number == 0
        assert state.move_history == []
        assert state.metadata == {}

    def test_multiple_planes(self) -> None:
        """Test creating empty state with multiple planes (lines 287-292)."""
        state = create_empty_state(board_size=13, n_planes=3)
        assert state.board.shape == (3, 13, 13)
        assert np.all(state.board == 0)

    def test_custom_dtype(self) -> None:
        """Test creating empty state with custom dtype."""
        state = create_empty_state(board_size=5, dtype=np.int8)
        assert state.board.dtype == np.int8

    def test_default_dtype_float32(self) -> None:
        """Test default dtype is float32."""
        state = create_empty_state(board_size=5)
        assert state.board.dtype == np.float32

    def test_default_n_planes_is_one(self) -> None:
        """Test default n_planes is 1 (2D board)."""
        state = create_empty_state(board_size=7)
        assert state.board.ndim == 2

    def test_multiple_planes_ndim(self) -> None:
        """Test multi-plane state has 3 dimensions."""
        state = create_empty_state(board_size=7, n_planes=5)
        assert state.board.ndim == 3
        assert state.board.shape[0] == 5
