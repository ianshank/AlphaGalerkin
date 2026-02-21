"""Tests for GameInterface abstract base class and GameConfig.

Covers:
- Abstract method raise NotImplementedError paths
- GamePhase and GameResult dataclass
- Concrete method behavior via GoGame (inherits base implementations)
- get_phase with opening/midgame/endgame/terminal branches
- action_to_string / string_to_action with columns >= 'I'
- batch_to_tensor
- get_canonical_form for non-player-1
- validate_action
- get_observation_shape
- clone, __repr__
- GameConfig init, to_dict, from_dict
"""

from __future__ import annotations

from abc import ABC

import numpy as np
import pytest
import torch

from src.games.interface import GameConfig, GameInterface, GamePhase, GameResult
from src.games.state import ActionMask, GameState


# ---------------------------------------------------------------------------
# Minimal concrete subclass to test abstract method NotImplementedError stubs
# ---------------------------------------------------------------------------


class _StubGame(GameInterface):
    """Minimal concrete game for testing abstract method stubs.

    Each abstract method simply calls super() to hit the raise NotImplementedError
    lines in the base class.
    """

    name = "stub"
    description = "Stub game for tests"
    default_board_size = 5

    @property
    def action_space_size(self) -> int:  # type: ignore[override]
        return super().action_space_size  # triggers line 84

    @property
    def state_channels(self) -> int:  # type: ignore[override]
        return super().state_channels  # triggers line 95

    def initial_state(self, board_size: int | None = None) -> GameState:
        return super().initial_state(board_size)  # triggers line 118

    def get_legal_actions(self, state: GameState) -> list[int]:
        return super().get_legal_actions(state)  # triggers line 131

    def get_action_mask(self, state: GameState) -> ActionMask:
        return super().get_action_mask(state)  # triggers line 144

    def apply_action(self, state: GameState, action: int) -> GameState:
        return super().apply_action(state, action)  # triggers line 161

    def is_terminal(self, state: GameState) -> bool:
        return super().is_terminal(state)  # triggers line 174

    def get_result(self, state: GameState) -> GameResult:
        return super().get_result(state)  # triggers line 187

    def get_winner(self, state: GameState) -> int | None:
        return super().get_winner(state)  # triggers line 200

    def to_tensor(self, state: GameState) -> torch.Tensor:
        return super().to_tensor(state)  # triggers line 213

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | torch.Tensor,
    ) -> list[tuple[GameState, np.ndarray | torch.Tensor]]:
        return super().get_symmetries(state, policy)  # triggers line 233


class TestAbstractMethodNotImplemented:
    """Tests that abstract methods raise NotImplementedError (lines 84,95,118,131,144,161,174,187,200,213,233)."""

    @pytest.fixture
    def stub(self) -> _StubGame:
        return _StubGame()

    @pytest.fixture
    def dummy_state(self) -> GameState:
        return GameState(board=np.zeros((5, 5)), current_player=1)

    def test_action_space_size_not_implemented(self, stub: _StubGame) -> None:
        with pytest.raises(NotImplementedError):
            _ = stub.action_space_size

    def test_state_channels_not_implemented(self, stub: _StubGame) -> None:
        with pytest.raises(NotImplementedError):
            _ = stub.state_channels

    def test_initial_state_not_implemented(self, stub: _StubGame) -> None:
        with pytest.raises(NotImplementedError):
            stub.initial_state()

    def test_get_legal_actions_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.get_legal_actions(dummy_state)

    def test_get_action_mask_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.get_action_mask(dummy_state)

    def test_apply_action_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.apply_action(dummy_state, 0)

    def test_is_terminal_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.is_terminal(dummy_state)

    def test_get_result_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.get_result(dummy_state)

    def test_get_winner_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.get_winner(dummy_state)

    def test_to_tensor_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.to_tensor(dummy_state)

    def test_get_symmetries_not_implemented(
        self, stub: _StubGame, dummy_state: GameState
    ) -> None:
        with pytest.raises(NotImplementedError):
            stub.get_symmetries(dummy_state, np.zeros(5))


# ---------------------------------------------------------------------------
# Tests for GamePhase enum
# ---------------------------------------------------------------------------


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_phase_values(self) -> None:
        assert GamePhase.SETUP == "setup"
        assert GamePhase.OPENING == "opening"
        assert GamePhase.MIDGAME == "midgame"
        assert GamePhase.ENDGAME == "endgame"
        assert GamePhase.TERMINAL == "terminal"

    def test_phase_is_string_enum(self) -> None:
        assert isinstance(GamePhase.OPENING, str)


# ---------------------------------------------------------------------------
# Tests for GameResult dataclass
# ---------------------------------------------------------------------------


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_create_result(self) -> None:
        result = GameResult(
            winner=1,
            score_black=10.0,
            score_white=5.0,
            reason="score",
            move_count=50,
        )
        assert result.winner == 1
        assert result.score_black == 10.0
        assert result.score_white == 5.0
        assert result.reason == "score"
        assert result.move_count == 50

    def test_draw_result(self) -> None:
        result = GameResult(
            winner=None,
            score_black=5.0,
            score_white=5.0,
            reason="stalemate",
            move_count=100,
        )
        assert result.winner is None


# ---------------------------------------------------------------------------
# Tests for concrete methods using GoGame (which inherits base implementations)
# ---------------------------------------------------------------------------


class TestGamePhaseDetection:
    """Tests for get_phase method (lines 246, 254-257)."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_phase_opening(self, game: GameInterface) -> None:
        """Test opening phase for low move count (line 253)."""
        state = game.initial_state(board_size=9)
        # move_number=0, total_moves=81, progress=0.0 < 0.1 -> OPENING
        assert game.get_phase(state) == GamePhase.OPENING

    def test_phase_midgame(self, game: GameInterface) -> None:
        """Test midgame phase detection (lines 254-255)."""
        state = game.initial_state(board_size=9)
        # board_size=9, total_moves=81
        # For midgame: 0.1 <= progress < 0.7 -> move 9 to 56
        midgame_state = GameState(
            board=state.board.copy(),
            current_player=1,
            move_number=20,  # 20/81 ≈ 0.247 -> midgame
            move_history=list(range(20)),
            metadata=state.metadata.copy(),
        )
        assert game.get_phase(midgame_state) == GamePhase.MIDGAME

    def test_phase_endgame(self, game: GameInterface) -> None:
        """Test endgame phase detection (lines 256-257)."""
        state = game.initial_state(board_size=9)
        # board_size=9, total_moves=81
        # For endgame: progress >= 0.7 -> move 57+
        endgame_state = GameState(
            board=state.board.copy(),
            current_player=1,
            move_number=60,  # 60/81 ≈ 0.74 -> endgame
            move_history=list(range(60)),
            metadata=state.metadata.copy(),
        )
        assert game.get_phase(endgame_state) == GamePhase.ENDGAME

    def test_phase_terminal(self, game: GameInterface) -> None:
        """Test terminal phase detection (line 246)."""
        state = game.initial_state(board_size=9)
        # Make game terminal by having two consecutive passes
        terminal_state = GameState(
            board=state.board.copy(),
            current_player=1,
            move_number=5,
            move_history=[81, 81],  # two passes (action 81 = 9*9 = pass)
            metadata={
                **state.metadata,
                "consecutive_passes": 2,
            },
        )
        assert game.get_phase(terminal_state) == GamePhase.TERMINAL


class TestActionToString:
    """Tests for action_to_string method (lines 270-283, especially 281)."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_pass_action(self, game: GameInterface) -> None:
        """Test pass action produces 'pass' string."""
        board_size = 9
        pass_action = board_size * board_size  # 81
        result = game.action_to_string(pass_action, board_size=board_size)
        assert result == "pass"

    def test_corner_a1(self, game: GameInterface) -> None:
        """Test bottom-left corner action (A1)."""
        board_size = 9
        # A1 = row=8 (board_size-1), col=0
        action = 8 * 9 + 0  # row=8, col=0
        result = game.action_to_string(action, board_size=board_size)
        assert result == "A1"

    def test_column_after_i_skipped(self, game: GameInterface) -> None:
        """Test column >= 'I' is skipped (line 281).

        Column index 8 should map to 'J' (skipping 'I').
        """
        board_size = 19
        # col=8 -> letter 'I' is skipped -> should be 'J'
        # row=0 -> board_size - 0 = 19
        action = 0 * 19 + 8  # row=0, col=8
        result = game.action_to_string(action, board_size=board_size)
        assert result == "J19"

    def test_column_h_before_skip(self, game: GameInterface) -> None:
        """Test column H (index 7) is not affected by skip."""
        board_size = 19
        action = 0 * 19 + 7  # row=0, col=7
        result = game.action_to_string(action, board_size=board_size)
        assert result == "H19"

    def test_uses_default_board_size(self, game: GameInterface) -> None:
        """Test action_to_string uses default board_size when None."""
        # GoGame default is 19
        pass_action = 19 * 19
        result = game.action_to_string(pass_action)
        assert result == "pass"


class TestStringToAction:
    """Tests for string_to_action method (lines 285-312, especially 308)."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_pass_string(self, game: GameInterface) -> None:
        """Test 'pass' string maps to pass action."""
        board_size = 9
        result = game.string_to_action("pass", board_size=board_size)
        assert result == board_size * board_size  # 81

    def test_pass_case_insensitive(self, game: GameInterface) -> None:
        """Test 'PASS' and 'Pass' are also recognized."""
        board_size = 9
        assert game.string_to_action("PASS", board_size=board_size) == 81
        assert game.string_to_action("Pass", board_size=board_size) == 81

    def test_coordinate_a1(self, game: GameInterface) -> None:
        """Test A1 maps correctly."""
        board_size = 9
        action = game.string_to_action("A1", board_size=board_size)
        # A1 -> col=0, row = 9-1 = 8
        expected = 8 * 9 + 0
        assert action == expected

    def test_column_after_i_skipped(self, game: GameInterface) -> None:
        """Test column > 'I' is adjusted (line 308).

        'J' should map to column index 8 (since 'I' is skipped).
        """
        board_size = 19
        action = game.string_to_action("J19", board_size=board_size)
        # J -> ord('J')=74, ord('A')=65, col=74-65=9, then col-=1 -> 8
        # row = 19-19 = 0
        expected = 0 * 19 + 8
        assert action == expected

    def test_roundtrip_all_columns(self, game: GameInterface) -> None:
        """Test action_to_string/string_to_action roundtrip for all columns."""
        board_size = 19
        for col in range(board_size):
            action = 0 * board_size + col  # row=0
            string = game.action_to_string(action, board_size=board_size)
            recovered = game.string_to_action(string, board_size=board_size)
            assert recovered == action, f"Roundtrip failed for col={col}: {string}"

    def test_uses_default_board_size(self, game: GameInterface) -> None:
        """Test string_to_action uses default board_size when None."""
        result = game.string_to_action("pass")
        assert result == 19 * 19  # default board_size=19


class TestValidateAction:
    """Tests for validate_action method."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_valid_action(self, game: GameInterface) -> None:
        """Test valid action returns True."""
        state = game.initial_state(board_size=9)
        legal = game.get_legal_actions(state)
        assert game.validate_action(state, legal[0]) is True

    def test_negative_action(self, game: GameInterface) -> None:
        """Test negative action returns False."""
        state = game.initial_state(board_size=9)
        assert game.validate_action(state, -1) is False

    def test_action_beyond_space(self, game: GameInterface) -> None:
        """Test action >= action_space_size returns False."""
        state = game.initial_state(board_size=9)
        assert game.validate_action(state, game.action_space_size) is False


class TestGetObservationShape:
    """Tests for get_observation_shape method."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_observation_shape_default(self, game: GameInterface) -> None:
        """Test observation shape with default board size."""
        shape = game.get_observation_shape()
        assert shape == (17, 19, 19)

    def test_observation_shape_custom(self, game: GameInterface) -> None:
        """Test observation shape with custom board size."""
        shape = game.get_observation_shape(board_size=9)
        assert shape == (17, 9, 9)


class TestBatchToTensor:
    """Tests for batch_to_tensor method (lines 358-359)."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_batch_to_tensor_shape(self, game: GameInterface) -> None:
        """Test batch_to_tensor produces correct shape."""
        state1 = game.initial_state(board_size=9)
        state2 = game.initial_state(board_size=9)

        batch = game.batch_to_tensor([state1, state2])
        assert batch.shape == (2, 17, 9, 9)
        assert batch.device == torch.device("cpu")

    def test_batch_to_tensor_single(self, game: GameInterface) -> None:
        """Test batch_to_tensor with single state."""
        state = game.initial_state(board_size=9)
        batch = game.batch_to_tensor([state])
        assert batch.shape == (1, 17, 9, 9)

    def test_batch_to_tensor_device(self, game: GameInterface) -> None:
        """Test batch_to_tensor with explicit device."""
        state = game.initial_state(board_size=9)
        batch = game.batch_to_tensor([state], device="cpu")
        assert batch.device == torch.device("cpu")


class TestGetCanonicalForm:
    """Tests for get_canonical_form method (line 376)."""

    @pytest.fixture
    def game(self) -> GameInterface:
        from src.games.go import GoGame

        return GoGame()

    def test_canonical_form_player_1(self, game: GameInterface) -> None:
        """Test canonical form for player 1 returns state unchanged."""
        state = game.initial_state(board_size=9)
        assert state.current_player == 1
        canonical = game.get_canonical_form(state)
        assert canonical is state  # same object

    def test_canonical_form_player_minus_1(self, game: GameInterface) -> None:
        """Test canonical form for player -1 flips perspective (line 376)."""
        state = game.initial_state(board_size=9)
        # Make a state for player -1
        legal = game.get_legal_actions(state)
        state2 = game.apply_action(state, legal[0])
        assert state2.current_player == -1

        canonical = game.get_canonical_form(state2)
        assert canonical.current_player == 1
        # Board should be negated
        assert np.array_equal(canonical.board, -state2.board)


class TestClone:
    """Tests for clone method."""

    def test_clone_creates_new_instance(self) -> None:
        from src.games.go import GoGame

        game = GoGame()
        cloned = game.clone()
        assert isinstance(cloned, GoGame)
        assert cloned is not game

    def test_clone_type_preserved(self) -> None:
        from src.games.othello import OthelloGame

        game = OthelloGame()
        cloned = game.clone()
        assert isinstance(cloned, OthelloGame)
        assert type(cloned) is OthelloGame


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_go(self) -> None:
        from src.games.go import GoGame

        game = GoGame()
        r = repr(game)
        assert "GoGame" in r
        assert "go" in r

    def test_repr_format(self) -> None:
        from src.games.go import GoGame

        game = GoGame()
        assert repr(game) == "GoGame(name='go')"


class TestNPlayers:
    """Tests for n_players property."""

    def test_default_n_players(self) -> None:
        from src.games.go import GoGame

        game = GoGame()
        assert game.n_players == 2


# ---------------------------------------------------------------------------
# Tests for GameConfig (lines 419-422, 426, 436)
# ---------------------------------------------------------------------------


class TestGameConfig:
    """Tests for GameConfig class."""

    def test_init_defaults(self) -> None:
        """Test GameConfig with default values (lines 419-422)."""
        config = GameConfig(game_name="go")
        assert config.game_name == "go"
        assert config.board_size is None
        assert config.komi == 7.5
        assert config.time_control == {}

    def test_init_custom(self) -> None:
        """Test GameConfig with custom values."""
        tc = {"main": 300, "byo_yomi": 30}
        config = GameConfig(
            game_name="go",
            board_size=13,
            komi=6.5,
            time_control=tc,
        )
        assert config.game_name == "go"
        assert config.board_size == 13
        assert config.komi == 6.5
        assert config.time_control == tc

    def test_init_time_control_none_becomes_empty_dict(self) -> None:
        """Test time_control=None becomes {} (line 422)."""
        config = GameConfig(game_name="go", time_control=None)
        assert config.time_control == {}

    def test_to_dict(self) -> None:
        """Test to_dict produces correct dictionary (line 426)."""
        config = GameConfig(
            game_name="chess",
            board_size=8,
            komi=0.0,
            time_control={"main": 600},
        )
        d = config.to_dict()
        assert d == {
            "game_name": "chess",
            "board_size": 8,
            "komi": 0.0,
            "time_control": {"main": 600},
        }

    def test_from_dict(self) -> None:
        """Test from_dict creates correct GameConfig (line 436)."""
        d = {
            "game_name": "go",
            "board_size": 19,
            "komi": 7.5,
            "time_control": {"main": 300},
        }
        config = GameConfig.from_dict(d)
        assert config.game_name == "go"
        assert config.board_size == 19
        assert config.komi == 7.5
        assert config.time_control == {"main": 300}

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with only required fields (uses defaults)."""
        d = {"game_name": "hex"}
        config = GameConfig.from_dict(d)
        assert config.game_name == "hex"
        assert config.board_size is None
        assert config.komi == 7.5
        # from_dict passes data.get("time_control") which is None,
        # and __init__ converts None to {}
        assert config.time_control == {}

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict -> from_dict roundtrip."""
        original = GameConfig(
            game_name="othello",
            board_size=10,
            komi=0.0,
            time_control={"limit": 60},
        )
        restored = GameConfig.from_dict(original.to_dict())
        assert restored.game_name == original.game_name
        assert restored.board_size == original.board_size
        assert restored.komi == original.komi
        assert restored.time_control == original.time_control
