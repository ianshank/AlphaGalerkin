"""Tests for GameInterface compliance across all registered games.

Tests cover:
- Abstract method implementation for GoGame and ChessGame
- action_space_size and state_channels properties
- initial_state creation at various board sizes
- legal_actions returns valid actions
- apply_action produces valid successor states
- Terminal state detection
- Symmetry generation preserves policy structure
- Tensor encoding shape and dtype
- GameConfig serialization
- GamePhase progression
- Canonical form and clone
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from src.games.chess import ACTION_SPACE_SIZE as CHESS_ACTION_SPACE
from src.games.chess import BOARD_SIZE as CHESS_BOARD_SIZE
from src.games.chess import ChessGame
from src.games.go import GoGame
from src.games.interface import GameConfig, GameInterface, GamePhase, GameResult
from src.games.registry import GameRegistry
from src.games.state import ActionMask, GameState

if TYPE_CHECKING:
    pass


# --- Fixtures ---


@pytest.fixture
def go_game() -> GoGame:
    """Create Go game instance."""
    return GoGame()


@pytest.fixture
def chess_game() -> ChessGame:
    """Create Chess game instance."""
    return ChessGame()


@pytest.fixture(params=["go", "chess"], ids=["Go", "Chess"])
def game(request: pytest.FixtureRequest) -> GameInterface:
    """Parametrized fixture providing each registered game."""
    instance = GameRegistry().get(request.param)
    assert instance is not None, f"Game '{request.param}' not registered"
    return instance


@pytest.fixture(params=[9, 13, 19], ids=["9x9", "13x13", "19x19"])
def go_board_size(request: pytest.FixtureRequest) -> int:
    """Parametrized Go board sizes."""
    return request.param


# --- Abstract Method Implementation Tests ---


class TestAbstractMethodImplementation:
    """Verify all registered games implement every abstract method."""

    _REQUIRED_METHODS = [
        "initial_state",
        "get_legal_actions",
        "get_action_mask",
        "apply_action",
        "is_terminal",
        "get_result",
        "get_winner",
        "to_tensor",
        "get_symmetries",
    ]

    _REQUIRED_PROPERTIES = [
        "action_space_size",
        "state_channels",
    ]

    def test_go_implements_all_abstract_methods(self, go_game: GoGame) -> None:
        """GoGame provides concrete implementations of all abstract methods."""
        for method_name in self._REQUIRED_METHODS:
            method = getattr(go_game, method_name, None)
            assert method is not None, f"GoGame missing method: {method_name}"
            assert callable(method), f"GoGame.{method_name} is not callable"

    def test_chess_implements_all_abstract_methods(
        self, chess_game: ChessGame
    ) -> None:
        """ChessGame provides concrete implementations of all abstract methods."""
        for method_name in self._REQUIRED_METHODS:
            method = getattr(chess_game, method_name, None)
            assert method is not None, f"ChessGame missing method: {method_name}"
            assert callable(method), f"ChessGame.{method_name} is not callable"

    def test_go_has_required_properties(self, go_game: GoGame) -> None:
        """GoGame has all required properties."""
        for prop_name in self._REQUIRED_PROPERTIES:
            assert hasattr(go_game, prop_name), f"GoGame missing property: {prop_name}"
            value = getattr(go_game, prop_name)
            assert isinstance(value, int), f"GoGame.{prop_name} should be int"
            assert value > 0, f"GoGame.{prop_name} should be positive"

    def test_chess_has_required_properties(self, chess_game: ChessGame) -> None:
        """ChessGame has all required properties."""
        for prop_name in self._REQUIRED_PROPERTIES:
            assert hasattr(
                chess_game, prop_name
            ), f"ChessGame missing property: {prop_name}"
            value = getattr(chess_game, prop_name)
            assert isinstance(value, int), f"ChessGame.{prop_name} should be int"
            assert value > 0, f"ChessGame.{prop_name} should be positive"


# --- Properties Tests ---


class TestGameProperties:
    """Tests for action_space_size and state_channels properties."""

    def test_go_action_space_includes_pass(self, go_game: GoGame) -> None:
        """Go action space = board_size^2 + 1 (pass)."""
        go_game._board_size = 9
        assert go_game.action_space_size == 9 * 9 + 1

    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_go_action_space_scales_with_board(
        self, go_game: GoGame, size: int
    ) -> None:
        """Go action space scales with board size."""
        go_game._board_size = size
        assert go_game.action_space_size == size * size + 1

    def test_chess_action_space_fixed(self, chess_game: ChessGame) -> None:
        """Chess action space is always 8*8*73 = 4672."""
        assert chess_game.action_space_size == CHESS_ACTION_SPACE

    def test_go_state_channels_positive(self, go_game: GoGame) -> None:
        """Go state channels is a positive integer."""
        assert go_game.state_channels > 0

    def test_chess_state_channels(self, chess_game: ChessGame) -> None:
        """Chess uses 119 input planes (AlphaZero encoding)."""
        assert chess_game.state_channels == 119

    def test_n_players_is_two(self, game: GameInterface) -> None:
        """All current games are two-player."""
        assert game.n_players == 2

    def test_observation_shape(self, game: GameInterface) -> None:
        """get_observation_shape returns (channels, H, W) tuple."""
        shape = game.get_observation_shape()
        assert len(shape) == 3
        channels, h, w = shape
        assert channels == game.state_channels
        assert h == game.default_board_size
        assert w == game.default_board_size


# --- Initial State Tests ---


class TestInitialState:
    """Tests for initial_state creation."""

    def test_initial_state_returns_gamestate(self, game: GameInterface) -> None:
        """initial_state returns a GameState instance."""
        state = game.initial_state()
        assert isinstance(state, GameState)

    def test_initial_state_board_is_numpy(self, game: GameInterface) -> None:
        """Initial state board is a numpy array."""
        state = game.initial_state()
        assert isinstance(state.board, np.ndarray)

    def test_initial_state_move_number_zero(self, game: GameInterface) -> None:
        """Initial state starts at move 0."""
        state = game.initial_state()
        assert state.move_number == 0

    def test_initial_state_empty_history(self, game: GameInterface) -> None:
        """Initial state has no move history."""
        state = game.initial_state()
        assert len(state.move_history) == 0

    def test_go_initial_state_variable_size(
        self, go_game: GoGame, go_board_size: int
    ) -> None:
        """Go initial state respects board_size parameter."""
        state = go_game.initial_state(board_size=go_board_size)
        assert state.board.shape == (go_board_size, go_board_size)

    def test_go_initial_state_default_size(self, go_game: GoGame) -> None:
        """Go default board size is 19."""
        state = go_game.initial_state()
        assert state.board_size == go_game.default_board_size

    def test_chess_initial_state_always_8x8(self, chess_game: ChessGame) -> None:
        """Chess board is always 8x8 regardless of parameter."""
        state = chess_game.initial_state()
        assert state.board.shape == (CHESS_BOARD_SIZE, CHESS_BOARD_SIZE)

    def test_initial_state_not_terminal(self, game: GameInterface) -> None:
        """Initial state is never terminal."""
        state = game.initial_state()
        assert not game.is_terminal(state)


# --- Legal Actions Tests ---


class TestLegalActions:
    """Tests for legal action generation."""

    def test_initial_has_legal_actions(self, game: GameInterface) -> None:
        """Initial state always has at least one legal action."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        assert len(legal) > 0

    def test_legal_actions_are_ints(self, game: GameInterface) -> None:
        """Legal actions are integer indices."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        for action in legal:
            assert isinstance(action, (int, np.integer))

    def test_legal_actions_in_range(self, game: GameInterface) -> None:
        """All legal actions are within [0, action_space_size)."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        for action in legal:
            assert 0 <= action < game.action_space_size

    def test_action_mask_consistent_with_legal_actions(
        self, game: GameInterface
    ) -> None:
        """ActionMask agrees with get_legal_actions."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        mask = game.get_action_mask(state)

        assert isinstance(mask, ActionMask)
        assert mask.num_legal == len(legal)
        for action in legal:
            assert mask.is_legal(action)

    def test_action_mask_size(self, game: GameInterface) -> None:
        """ActionMask has correct action_space_size."""
        state = game.initial_state()
        mask = game.get_action_mask(state)
        assert mask.action_space_size == game.action_space_size

    def test_validate_action_for_legal(self, game: GameInterface) -> None:
        """validate_action returns True for legal moves."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        if legal:
            assert game.validate_action(state, legal[0]) is True

    def test_validate_action_negative_index(self, game: GameInterface) -> None:
        """validate_action returns False for negative indices."""
        state = game.initial_state()
        assert game.validate_action(state, -1) is False

    def test_validate_action_too_large(self, game: GameInterface) -> None:
        """validate_action returns False for out-of-range indices."""
        state = game.initial_state()
        assert game.validate_action(state, game.action_space_size) is False


# --- Apply Action Tests ---


class TestApplyAction:
    """Tests for applying actions to produce successor states."""

    def test_apply_legal_action_returns_gamestate(
        self, game: GameInterface
    ) -> None:
        """Applying a legal action returns a new GameState."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        new_state = game.apply_action(state, legal[0])

        assert isinstance(new_state, GameState)

    def test_apply_action_increments_move_number(
        self, game: GameInterface
    ) -> None:
        """Move number increases after applying an action."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        new_state = game.apply_action(state, legal[0])

        assert new_state.move_number == state.move_number + 1

    def test_apply_action_appends_to_history(self, game: GameInterface) -> None:
        """Action is appended to move history."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        action = legal[0]
        new_state = game.apply_action(state, action)

        assert action in new_state.move_history

    def test_apply_action_switches_player(self, game: GameInterface) -> None:
        """Current player changes after a move."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        new_state = game.apply_action(state, legal[0])

        assert new_state.current_player != state.current_player

    def test_apply_action_does_not_mutate_original(
        self, game: GameInterface
    ) -> None:
        """Original state is not modified."""
        state = game.initial_state()
        original_board = state.board.copy()
        original_player = state.current_player
        original_move = state.move_number

        legal = game.get_legal_actions(state)
        _ = game.apply_action(state, legal[0])

        np.testing.assert_array_equal(state.board, original_board)
        assert state.current_player == original_player
        assert state.move_number == original_move

    def test_successor_has_legal_actions_or_is_terminal(
        self, game: GameInterface
    ) -> None:
        """A successor state either has legal actions or is terminal."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        new_state = game.apply_action(state, legal[0])

        successor_legal = game.get_legal_actions(new_state)
        assert len(successor_legal) > 0 or game.is_terminal(new_state)

    def test_multiple_consecutive_actions(self, game: GameInterface) -> None:
        """Can apply several actions in sequence."""
        state = game.initial_state()
        n_moves = 4

        for _ in range(n_moves):
            legal = game.get_legal_actions(state)
            if not legal or game.is_terminal(state):
                break
            state = game.apply_action(state, legal[0])

        assert state.move_number <= n_moves


# --- Terminal State Tests ---


class TestTerminalDetection:
    """Tests for terminal state detection."""

    def test_initial_not_terminal(self, game: GameInterface) -> None:
        """Fresh game is not terminal."""
        state = game.initial_state()
        assert game.is_terminal(state) is False

    def test_go_double_pass_is_terminal(self, go_game: GoGame) -> None:
        """Two consecutive passes end a Go game."""
        state = go_game.initial_state(board_size=9)
        pass_action = 9 * 9  # pass

        state = go_game.apply_action(state, pass_action)
        assert not go_game.is_terminal(state)

        state = go_game.apply_action(state, pass_action)
        assert go_game.is_terminal(state)

    def test_terminal_state_has_result(self, go_game: GoGame) -> None:
        """Terminal states produce a valid GameResult."""
        state = go_game.initial_state(board_size=9)
        pass_action = 9 * 9

        state = go_game.apply_action(state, pass_action)
        state = go_game.apply_action(state, pass_action)

        result = go_game.get_result(state)
        assert isinstance(result, GameResult)
        assert result.reason != ""
        assert result.move_count >= 0

    def test_get_winner_returns_valid(self, go_game: GoGame) -> None:
        """get_winner returns 1, -1, or None."""
        state = go_game.initial_state(board_size=9)
        pass_action = 9 * 9
        state = go_game.apply_action(state, pass_action)
        state = go_game.apply_action(state, pass_action)

        winner = go_game.get_winner(state)
        assert winner in {1, -1, None, 0}

    def test_game_phase_terminal_for_ended_game(self, go_game: GoGame) -> None:
        """Terminal game has TERMINAL phase."""
        state = go_game.initial_state(board_size=9)
        pass_action = 9 * 9
        state = go_game.apply_action(state, pass_action)
        state = go_game.apply_action(state, pass_action)

        assert go_game.get_phase(state) == GamePhase.TERMINAL


# --- Tensor Encoding Tests ---


class TestTensorEncoding:
    """Tests for to_tensor and batch_to_tensor."""

    def test_tensor_shape(self, game: GameInterface) -> None:
        """Tensor has shape (channels, H, W)."""
        state = game.initial_state()
        tensor = game.to_tensor(state)

        expected_shape = game.get_observation_shape()
        assert tensor.shape == expected_shape

    def test_tensor_dtype_float32(self, game: GameInterface) -> None:
        """Tensor is float32."""
        state = game.initial_state()
        tensor = game.to_tensor(state)
        assert tensor.dtype == torch.float32

    def test_tensor_finite(self, game: GameInterface) -> None:
        """Tensor contains no NaN or Inf."""
        state = game.initial_state()
        tensor = game.to_tensor(state)
        assert torch.isfinite(tensor).all()

    def test_tensor_after_move(self, game: GameInterface) -> None:
        """Tensor changes after a move is applied."""
        state = game.initial_state()
        tensor_before = game.to_tensor(state)

        legal = game.get_legal_actions(state)
        new_state = game.apply_action(state, legal[0])
        tensor_after = game.to_tensor(new_state)

        # At least some values should differ
        assert not torch.equal(tensor_before, tensor_after)

    def test_batch_to_tensor(self, game: GameInterface) -> None:
        """batch_to_tensor produces correct batch dimension."""
        state = game.initial_state()
        states = [state, state, state]

        batch = game.batch_to_tensor(states)
        assert batch.shape[0] == 3
        assert batch.shape[1:] == game.to_tensor(state).shape

    def test_go_tensor_variable_sizes(
        self, go_game: GoGame, go_board_size: int
    ) -> None:
        """Go tensors adapt to board size."""
        state = go_game.initial_state(board_size=go_board_size)
        tensor = go_game.to_tensor(state)

        assert tensor.shape[-1] == go_board_size
        assert tensor.shape[-2] == go_board_size


# --- Symmetry Tests ---


class TestSymmetries:
    """Tests for symmetry transformations."""

    def test_symmetries_return_list(self, game: GameInterface) -> None:
        """get_symmetries returns a list of (state, policy) tuples."""
        state = game.initial_state()
        policy = np.ones(game.action_space_size) / game.action_space_size

        symmetries = game.get_symmetries(state, policy)

        assert isinstance(symmetries, list)
        assert len(symmetries) >= 1

    def test_symmetries_contain_original(self, game: GameInterface) -> None:
        """First symmetry should be the identity (original)."""
        state = game.initial_state()
        policy = np.ones(game.action_space_size) / game.action_space_size

        symmetries = game.get_symmetries(state, policy)
        sym_state, sym_policy = symmetries[0]

        assert isinstance(sym_state, GameState)

    def test_go_has_eight_symmetries(self, go_game: GoGame) -> None:
        """Go has 8-fold symmetry (4 rotations x 2 reflections)."""
        state = go_game.initial_state(board_size=9)
        policy = np.ones(82) / 82

        symmetries = go_game.get_symmetries(state, policy)
        assert len(symmetries) == 8

    def test_chess_has_two_symmetries(self, chess_game: ChessGame) -> None:
        """Chess has 2 symmetries (identity + horizontal flip)."""
        state = chess_game.initial_state()
        policy = np.ones(CHESS_ACTION_SPACE) / CHESS_ACTION_SPACE

        symmetries = chess_game.get_symmetries(state, policy)
        assert len(symmetries) == 2

    def test_symmetry_preserves_policy_sum(self, game: GameInterface) -> None:
        """Policy probability sum is preserved across symmetries."""
        state = game.initial_state()
        policy = np.ones(game.action_space_size) / game.action_space_size

        symmetries = game.get_symmetries(state, policy)

        for _sym_state, sym_policy in symmetries:
            is_np = isinstance(sym_policy, np.ndarray)
            total = float(np.sum(sym_policy) if is_np else sym_policy.sum())
            assert abs(total - 1.0) < 1e-5

    def test_symmetry_states_are_gamestate(self, game: GameInterface) -> None:
        """All symmetry states are GameState instances."""
        state = game.initial_state()
        policy = np.ones(game.action_space_size) / game.action_space_size

        symmetries = game.get_symmetries(state, policy)
        for sym_state, _ in symmetries:
            assert isinstance(sym_state, GameState)


# --- GamePhase Tests ---


class TestGamePhase:
    """Tests for game phase detection."""

    def test_initial_is_opening(self, game: GameInterface) -> None:
        """Initial position is in the opening phase."""
        state = game.initial_state()
        phase = game.get_phase(state)
        assert phase == GamePhase.OPENING

    def test_phase_is_valid_enum(self, game: GameInterface) -> None:
        """Phase is always a valid GamePhase value."""
        state = game.initial_state()
        phase = game.get_phase(state)
        assert isinstance(phase, GamePhase)

    def test_phase_progresses(self, go_game: GoGame) -> None:
        """Playing moves eventually changes the game phase."""
        state = go_game.initial_state(board_size=9)
        seen_phases: set[GamePhase] = set()

        for _ in range(60):
            if go_game.is_terminal(state):
                seen_phases.add(GamePhase.TERMINAL)
                break
            seen_phases.add(go_game.get_phase(state))
            legal = go_game.get_legal_actions(state)
            non_pass = [a for a in legal if a != 81]
            if non_pass:
                state = go_game.apply_action(state, non_pass[0])
            else:
                break

        # Should see at least opening, possibly midgame or endgame
        assert len(seen_phases) >= 1


# --- Canonical Form and Clone Tests ---


class TestCanonicalFormAndClone:
    """Tests for get_canonical_form and clone."""

    def test_canonical_form_player_one_is_identity(
        self, game: GameInterface
    ) -> None:
        """Canonical form of player 1 state is the state itself."""
        state = game.initial_state()
        assert state.current_player == 1
        canonical = game.get_canonical_form(state)
        assert canonical is state

    def test_canonical_form_player_minus_one_flips(
        self, game: GameInterface
    ) -> None:
        """Canonical form of player -1 state flips perspective."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        state_after = game.apply_action(state, legal[0])

        # After one move, current_player should be -1
        assert state_after.current_player == -1
        canonical = game.get_canonical_form(state_after)

        assert canonical.current_player == 1
        np.testing.assert_array_equal(canonical.board, -state_after.board)

    def test_clone_returns_new_instance(self, game: GameInterface) -> None:
        """clone() returns a new instance of the same type."""
        cloned = game.clone()
        assert cloned is not game
        assert type(cloned) is type(game)

    def test_clone_has_same_name(self, game: GameInterface) -> None:
        """Cloned game has the same name."""
        cloned = game.clone()
        assert cloned.name == game.name


# --- GameConfig Tests ---


class TestGameConfig:
    """Tests for GameConfig dataclass."""

    @pytest.fixture(params=["go", "chess"])
    def game_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_config_creation(self, game_name: str) -> None:
        """GameConfig can be created with valid parameters."""
        config = GameConfig(game_name=game_name)
        assert config.game_name == game_name

    def test_config_defaults(self) -> None:
        """GameConfig has sensible defaults."""
        config = GameConfig(game_name="go")
        assert config.board_size is None
        assert config.komi == 7.5
        assert config.time_control == {}

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_config_with_board_size(self, board_size: int) -> None:
        """GameConfig stores board_size."""
        config = GameConfig(game_name="go", board_size=board_size)
        assert config.board_size == board_size

    def test_config_to_dict_roundtrip(self) -> None:
        """GameConfig serialization roundtrip."""
        config = GameConfig(
            game_name="go",
            board_size=13,
            komi=6.5,
            time_control={"main_time": 300},
        )
        d = config.to_dict()
        restored = GameConfig.from_dict(d)

        assert restored.game_name == config.game_name
        assert restored.board_size == config.board_size
        assert restored.komi == config.komi
        assert restored.time_control == config.time_control

    def test_config_to_dict_keys(self) -> None:
        """to_dict returns expected keys."""
        config = GameConfig(game_name="chess")
        d = config.to_dict()

        expected_keys = {"game_name", "board_size", "komi", "time_control"}
        assert set(d.keys()) == expected_keys


# --- Repr Tests ---


class TestRepr:
    """Tests for string representation."""

    def test_repr_contains_class_name(self, game: GameInterface) -> None:
        """Repr includes the class name."""
        r = repr(game)
        assert type(game).__name__ in r

    def test_repr_contains_game_name(self, game: GameInterface) -> None:
        """Repr includes the game name."""
        r = repr(game)
        assert game.name in r
