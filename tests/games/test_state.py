"""Tests for GameState and ActionMask.

Tests cover:
- GameState creation with valid data
- Hash computation (consistent, deterministic)
- Serialization/deserialization roundtrip
- with_move creates new state
- flip_perspective
- ActionMask creation and application
- create_empty_state helper
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.state import ActionMask, GameState, create_empty_state

# --- Fixtures ---


@pytest.fixture(params=[5, 9, 13, 19])
def board_size(request: pytest.FixtureRequest) -> int:
    """Parametrized board sizes for testing."""
    return request.param


@pytest.fixture
def empty_board(board_size: int) -> np.ndarray:
    """Create an empty board of given size."""
    return np.zeros((board_size, board_size), dtype=np.float32)


@pytest.fixture
def populated_board() -> np.ndarray:
    """Create a board with some pieces placed."""
    rng = np.random.default_rng(seed=42)
    board = np.zeros((9, 9), dtype=np.float32)
    # Place a few stones deterministically
    board[4, 4] = 1  # Black at center
    board[3, 4] = -1  # White above center
    board[4, 3] = -1  # White left of center
    board[2, 2] = 1  # Black at (2,2)
    return board


@pytest.fixture
def sample_state(populated_board: np.ndarray) -> GameState:
    """Create a sample game state with some moves played."""
    return GameState(
        board=populated_board,
        current_player=1,
        move_number=4,
        move_history=[40, 31, 39, 20],
        metadata={"captured_black": 0, "captured_white": 0},
    )


@pytest.fixture(params=[1, -1])
def player(request: pytest.FixtureRequest) -> int:
    """Parametrized player values."""
    return request.param


# --- GameState Creation Tests ---


class TestGameStateCreation:
    """Tests for GameState initialization and validation."""

    def test_create_with_numpy_board(self, empty_board: np.ndarray) -> None:
        """Test creation with a numpy array board."""
        state = GameState(board=empty_board)

        assert isinstance(state.board, np.ndarray)
        assert state.current_player == 1
        assert state.move_number == 0
        assert state.move_history == []
        assert state.metadata == {}

    def test_create_with_list_board(self) -> None:
        """Test creation with a plain list auto-converts to numpy."""
        board_list = [[0, 0], [0, 0]]
        state = GameState(board=board_list)

        assert isinstance(state.board, np.ndarray)
        assert state.board.shape == (2, 2)

    def test_create_with_player(self, empty_board: np.ndarray, player: int) -> None:
        """Test creation with specified player."""
        state = GameState(board=empty_board, current_player=player)
        assert state.current_player == player

    def test_create_with_metadata(self, empty_board: np.ndarray) -> None:
        """Test creation with metadata dict."""
        metadata = {"komi": 7.5, "superko": True}
        state = GameState(board=empty_board, metadata=metadata)

        assert state.metadata == metadata

    def test_create_with_move_history(self, empty_board: np.ndarray) -> None:
        """Test creation with move history."""
        history = [10, 20, 30]
        state = GameState(
            board=empty_board,
            move_number=3,
            move_history=history,
        )

        assert state.move_history == history
        assert state.move_number == 3

    def test_board_none_raises(self) -> None:
        """Test that None board raises ValueError."""
        with pytest.raises(ValueError, match="Board cannot be None"):
            GameState(board=None)

    def test_none_move_history_defaults_to_list(self, empty_board: np.ndarray) -> None:
        """Test that None move_history is replaced with empty list."""
        state = GameState(board=empty_board, move_history=None)
        assert state.move_history == []

    def test_none_metadata_defaults_to_dict(self, empty_board: np.ndarray) -> None:
        """Test that None metadata is replaced with empty dict."""
        state = GameState(board=empty_board, metadata=None)
        assert state.metadata == {}

    def test_multiplane_board(self) -> None:
        """Test creation with multi-plane board (channels, H, W)."""
        board = np.zeros((17, 9, 9), dtype=np.float32)
        state = GameState(board=board)

        assert state.board.shape == (17, 9, 9)
        assert state.board_size == 9  # Last dimension


# --- GameState Properties ---


class TestGameStateProperties:
    """Tests for GameState property accessors."""

    def test_board_size(self, board_size: int) -> None:
        """Test board_size property returns last dimension."""
        board = np.zeros((board_size, board_size), dtype=np.float32)
        state = GameState(board=board)
        assert state.board_size == board_size

    def test_last_move_with_history(self, sample_state: GameState) -> None:
        """Test last_move returns the last action in history."""
        assert sample_state.last_move == 20

    def test_last_move_no_history(self, empty_board: np.ndarray) -> None:
        """Test last_move returns None when no moves played."""
        state = GameState(board=empty_board)
        assert state.last_move is None


# --- Hash Tests ---


class TestGameStateHash:
    """Tests for GameState hash computation."""

    def test_hash_deterministic(self, sample_state: GameState) -> None:
        """Test hash returns same value on repeated calls."""
        h1 = hash(sample_state)
        h2 = hash(sample_state)
        assert h1 == h2

    def test_hash_consistent_for_equal_states(self, populated_board: np.ndarray) -> None:
        """Test that equal states produce equal hashes."""
        state_a = GameState(board=populated_board.copy(), current_player=1)
        state_b = GameState(board=populated_board.copy(), current_player=1)

        assert hash(state_a) == hash(state_b)

    def test_hash_differs_for_different_boards(self) -> None:
        """Test that different boards produce different hashes."""
        board_a = np.zeros((9, 9), dtype=np.float32)
        board_b = np.zeros((9, 9), dtype=np.float32)
        board_b[0, 0] = 1

        state_a = GameState(board=board_a, current_player=1)
        state_b = GameState(board=board_b, current_player=1)

        assert hash(state_a) != hash(state_b)

    def test_hash_differs_for_different_players(self) -> None:
        """Test that different current_player produces different hashes."""
        board = np.zeros((9, 9), dtype=np.float32)

        state_a = GameState(board=board.copy(), current_player=1)
        state_b = GameState(board=board.copy(), current_player=-1)

        assert hash(state_a) != hash(state_b)

    def test_hash_usable_as_dict_key(self, sample_state: GameState) -> None:
        """Test state can be used as dictionary key."""
        table = {sample_state: "value"}
        assert table[sample_state] == "value"

    def test_hash_usable_in_set(self, populated_board: np.ndarray) -> None:
        """Test states can be added to sets correctly."""
        state_a = GameState(board=populated_board.copy(), current_player=1)
        state_b = GameState(board=populated_board.copy(), current_player=1)
        state_c = GameState(board=np.zeros((9, 9), dtype=np.float32), current_player=1)

        state_set = {state_a, state_b, state_c}
        # state_a and state_b are equal so the set should have 2 items
        assert len(state_set) == 2


# --- Equality Tests ---


class TestGameStateEquality:
    """Tests for GameState equality comparison."""

    def test_equal_states(self, populated_board: np.ndarray) -> None:
        """Test equality for states with same board and player."""
        state_a = GameState(board=populated_board.copy(), current_player=1)
        state_b = GameState(board=populated_board.copy(), current_player=1)
        assert state_a == state_b

    def test_not_equal_different_board(self) -> None:
        """Test inequality for different boards."""
        board_a = np.zeros((9, 9), dtype=np.float32)
        board_b = np.ones((9, 9), dtype=np.float32)

        state_a = GameState(board=board_a, current_player=1)
        state_b = GameState(board=board_b, current_player=1)
        assert state_a != state_b

    def test_not_equal_different_player(self) -> None:
        """Test inequality for different current_player."""
        board = np.zeros((9, 9), dtype=np.float32)

        state_a = GameState(board=board.copy(), current_player=1)
        state_b = GameState(board=board.copy(), current_player=-1)
        assert state_a != state_b

    def test_not_equal_to_non_gamestate(self) -> None:
        """Test inequality with non-GameState object."""
        state = GameState(board=np.zeros((9, 9)))
        assert state != "not a state"
        assert state != 42
        assert state != None  # noqa: E711


# --- Copy Tests ---


class TestGameStateCopy:
    """Tests for GameState deep copy."""

    def test_copy_returns_new_instance(self, sample_state: GameState) -> None:
        """Test copy returns a different object."""
        copy = sample_state.copy()
        assert copy is not sample_state

    def test_copy_has_independent_board(self, sample_state: GameState) -> None:
        """Test copy board is independent from original."""
        copy = sample_state.copy()
        copy.board[0, 0] = 99
        assert sample_state.board[0, 0] != 99

    def test_copy_has_independent_move_history(self, sample_state: GameState) -> None:
        """Test copy move_history is independent from original."""
        copy = sample_state.copy()
        copy.move_history.append(999)
        assert 999 not in sample_state.move_history

    def test_copy_preserves_values(self, sample_state: GameState) -> None:
        """Test copy preserves all field values."""
        copy = sample_state.copy()

        assert np.array_equal(copy.board, sample_state.board)
        assert copy.current_player == sample_state.current_player
        assert copy.move_number == sample_state.move_number
        assert copy.move_history == sample_state.move_history
        assert copy.metadata == sample_state.metadata


# --- with_move Tests ---


class TestGameStateWithMove:
    """Tests for GameState.with_move method."""

    def test_with_move_creates_new_state(self, sample_state: GameState) -> None:
        """Test with_move returns a new GameState."""
        new_board = sample_state.board.copy()
        new_board[0, 0] = 1
        new_state = sample_state.with_move(action=0, new_board=new_board)

        assert new_state is not sample_state

    def test_with_move_increments_move_number(self, sample_state: GameState) -> None:
        """Test move_number is incremented."""
        original_move_number = sample_state.move_number
        new_state = sample_state.with_move(
            action=0,
            new_board=sample_state.board.copy(),
        )
        assert new_state.move_number == original_move_number + 1

    def test_with_move_appends_to_history(self, sample_state: GameState) -> None:
        """Test action is appended to move_history."""
        action = 55
        new_state = sample_state.with_move(
            action=action,
            new_board=sample_state.board.copy(),
        )
        assert new_state.move_history[-1] == action
        assert len(new_state.move_history) == len(sample_state.move_history) + 1

    def test_with_move_switches_player_by_default(self, sample_state: GameState) -> None:
        """Test current_player is negated when next_player is None."""
        original_player = sample_state.current_player
        new_state = sample_state.with_move(
            action=0,
            new_board=sample_state.board.copy(),
        )
        assert new_state.current_player == -original_player

    def test_with_move_explicit_next_player(self, sample_state: GameState) -> None:
        """Test explicit next_player overrides default switching."""
        new_state = sample_state.with_move(
            action=0,
            new_board=sample_state.board.copy(),
            next_player=sample_state.current_player,  # Same player
        )
        assert new_state.current_player == sample_state.current_player

    def test_with_move_updates_metadata(self, sample_state: GameState) -> None:
        """Test metadata_updates are merged into metadata."""
        new_state = sample_state.with_move(
            action=0,
            new_board=sample_state.board.copy(),
            consecutive_passes=1,
            capture_count=3,
        )
        assert new_state.metadata["consecutive_passes"] == 1
        assert new_state.metadata["capture_count"] == 3

    def test_with_move_preserves_original(self, sample_state: GameState) -> None:
        """Test original state is not modified."""
        original_history_len = len(sample_state.move_history)
        original_move_number = sample_state.move_number

        _ = sample_state.with_move(
            action=0,
            new_board=sample_state.board.copy(),
        )

        assert len(sample_state.move_history) == original_history_len
        assert sample_state.move_number == original_move_number


# --- Serialization Tests ---


class TestGameStateSerialization:
    """Tests for GameState to_dict/from_dict roundtrip."""

    def test_to_dict_structure(self, sample_state: GameState) -> None:
        """Test to_dict returns expected keys."""
        d = sample_state.to_dict()

        assert "board" in d
        assert "current_player" in d
        assert "move_number" in d
        assert "move_history" in d
        assert "metadata" in d

    def test_to_dict_board_is_list(self, sample_state: GameState) -> None:
        """Test board is converted to nested list."""
        d = sample_state.to_dict()
        assert isinstance(d["board"], list)

    def test_roundtrip_preserves_board(self, sample_state: GameState) -> None:
        """Test serialization roundtrip preserves board data."""
        d = sample_state.to_dict()
        restored = GameState.from_dict(d)

        np.testing.assert_array_almost_equal(restored.board, sample_state.board)

    def test_roundtrip_preserves_player(self, sample_state: GameState) -> None:
        """Test serialization roundtrip preserves current_player."""
        d = sample_state.to_dict()
        restored = GameState.from_dict(d)

        assert restored.current_player == sample_state.current_player

    def test_roundtrip_preserves_move_number(self, sample_state: GameState) -> None:
        """Test serialization roundtrip preserves move_number."""
        d = sample_state.to_dict()
        restored = GameState.from_dict(d)

        assert restored.move_number == sample_state.move_number

    def test_roundtrip_preserves_move_history(self, sample_state: GameState) -> None:
        """Test serialization roundtrip preserves move_history."""
        d = sample_state.to_dict()
        restored = GameState.from_dict(d)

        assert restored.move_history == sample_state.move_history

    def test_roundtrip_preserves_metadata(self, sample_state: GameState) -> None:
        """Test serialization roundtrip preserves metadata."""
        d = sample_state.to_dict()
        restored = GameState.from_dict(d)

        assert restored.metadata == sample_state.metadata

    def test_from_dict_defaults(self) -> None:
        """Test from_dict provides defaults for optional fields."""
        minimal = {
            "board": [[0, 0], [0, 0]],
            "current_player": 1,
        }
        state = GameState.from_dict(minimal)

        assert state.move_number == 0
        assert state.move_history == []
        assert state.metadata == {}

    @pytest.mark.parametrize("board_size", [5, 9, 13, 19])
    def test_roundtrip_various_sizes(self, board_size: int) -> None:
        """Test roundtrip works for various board sizes."""
        board = np.random.default_rng(seed=board_size).standard_normal(
            (board_size, board_size)
        ).astype(np.float32)
        state = GameState(board=board, current_player=-1)

        d = state.to_dict()
        restored = GameState.from_dict(d)

        np.testing.assert_array_almost_equal(restored.board, state.board, decimal=5)
        assert restored.current_player == state.current_player


# --- flip_perspective Tests ---


class TestFlipPerspective:
    """Tests for GameState.flip_perspective method."""

    def test_flip_negates_board(self, sample_state: GameState) -> None:
        """Test board values are negated."""
        flipped = sample_state.flip_perspective()
        np.testing.assert_array_equal(flipped.board, -sample_state.board)

    def test_flip_negates_player(self, sample_state: GameState) -> None:
        """Test current_player is negated."""
        flipped = sample_state.flip_perspective()
        assert flipped.current_player == -sample_state.current_player

    def test_flip_preserves_move_number(self, sample_state: GameState) -> None:
        """Test move_number is preserved."""
        flipped = sample_state.flip_perspective()
        assert flipped.move_number == sample_state.move_number

    def test_flip_preserves_move_history(self, sample_state: GameState) -> None:
        """Test move_history is preserved."""
        flipped = sample_state.flip_perspective()
        assert flipped.move_history == sample_state.move_history

    def test_double_flip_roundtrip(self, sample_state: GameState) -> None:
        """Test flipping twice returns to original board and player."""
        double_flipped = sample_state.flip_perspective().flip_perspective()
        np.testing.assert_array_equal(double_flipped.board, sample_state.board)
        assert double_flipped.current_player == sample_state.current_player

    def test_flip_creates_independent_copy(self, sample_state: GameState) -> None:
        """Test flipped state has independent board array."""
        flipped = sample_state.flip_perspective()
        flipped.board[0, 0] = 99
        assert sample_state.board[0, 0] != 99


# --- ActionMask Tests ---


class TestActionMask:
    """Tests for ActionMask creation and methods."""

    @pytest.fixture(params=[82, 362, 4672])
    def action_space(self, request: pytest.FixtureRequest) -> int:
        """Parametrized action space sizes (Go 9x9, Go 19x19, Chess)."""
        return request.param

    def test_create_valid_mask(self, action_space: int) -> None:
        """Test creation with valid mask and action_space_size."""
        mask = np.zeros(action_space, dtype=bool)
        mask[0] = True
        mask[action_space - 1] = True

        am = ActionMask(mask=mask, action_space_size=action_space)

        assert am.action_space_size == action_space
        assert len(am.mask) == action_space

    def test_size_mismatch_raises(self) -> None:
        """Test ValueError when mask size does not match action_space_size."""
        mask = np.zeros(10, dtype=bool)

        with pytest.raises(ValueError, match="Mask size"):
            ActionMask(mask=mask, action_space_size=20)

    def test_legal_actions_returns_correct_indices(self) -> None:
        """Test legal_actions property returns indices of True values."""
        mask = np.zeros(82, dtype=bool)
        legal_indices = [0, 10, 40, 81]
        for idx in legal_indices:
            mask[idx] = True

        am = ActionMask(mask=mask, action_space_size=82)
        assert am.legal_actions == legal_indices

    def test_legal_actions_empty_mask(self) -> None:
        """Test legal_actions returns empty list when no actions legal."""
        mask = np.zeros(82, dtype=bool)
        am = ActionMask(mask=mask, action_space_size=82)
        assert am.legal_actions == []

    def test_num_legal(self) -> None:
        """Test num_legal returns count of True values."""
        mask = np.zeros(82, dtype=bool)
        mask[:20] = True

        am = ActionMask(mask=mask, action_space_size=82)
        assert am.num_legal == 20

    def test_is_legal_true_for_legal_action(self) -> None:
        """Test is_legal returns True for legal actions."""
        mask = np.zeros(82, dtype=bool)
        mask[40] = True

        am = ActionMask(mask=mask, action_space_size=82)
        assert am.is_legal(40) is True

    def test_is_legal_false_for_illegal_action(self) -> None:
        """Test is_legal returns False for illegal actions."""
        mask = np.zeros(82, dtype=bool)

        am = ActionMask(mask=mask, action_space_size=82)
        assert am.is_legal(40) is False

    def test_is_legal_false_for_out_of_range(self) -> None:
        """Test is_legal returns False for out-of-range action indices."""
        mask = np.ones(82, dtype=bool)
        am = ActionMask(mask=mask, action_space_size=82)

        assert am.is_legal(-1) is False
        assert am.is_legal(82) is False
        assert am.is_legal(100) is False

    def test_all_legal_mask(self) -> None:
        """Test mask where all actions are legal."""
        mask = np.ones(82, dtype=bool)
        am = ActionMask(mask=mask, action_space_size=82)

        assert am.num_legal == 82
        assert len(am.legal_actions) == 82


# --- create_empty_state Tests ---


class TestCreateEmptyState:
    """Tests for the create_empty_state helper function."""

    def test_single_plane(self, board_size: int) -> None:
        """Test creating empty state with single plane."""
        state = create_empty_state(board_size=board_size)

        assert state.board.shape == (board_size, board_size)
        assert np.all(state.board == 0)
        assert state.current_player == 1
        assert state.move_number == 0

    @pytest.mark.parametrize("n_planes", [1, 4, 17, 119])
    def test_multi_plane(self, n_planes: int) -> None:
        """Test creating empty state with multiple planes."""
        state = create_empty_state(board_size=9, n_planes=n_planes)

        if n_planes > 1:
            assert state.board.shape == (n_planes, 9, 9)
        else:
            assert state.board.shape == (9, 9)

    def test_custom_dtype(self) -> None:
        """Test creating empty state with custom dtype."""
        state = create_empty_state(board_size=9, dtype=np.int8)
        assert state.board.dtype == np.int8

    def test_default_dtype_is_float32(self) -> None:
        """Test default dtype is float32."""
        state = create_empty_state(board_size=9)
        assert state.board.dtype == np.float32
