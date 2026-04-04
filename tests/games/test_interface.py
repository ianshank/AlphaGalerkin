"""Tests for GameInterface abstract base class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import Tensor

from src.games.interface import GameConfig, GameInterface, GamePhase, GameResult
from src.games.state import ActionMask, GameState

if TYPE_CHECKING:
    pass


class ConcreteGame(GameInterface):
    """Minimal concrete implementation for testing."""

    name = "test_game"
    description = "Test game for unit tests"
    min_board_size = 3
    max_board_size = 19
    default_board_size = 9

    def __init__(self) -> None:
        self._board_size = self.default_board_size

    @property
    def action_space_size(self) -> int:
        return self._board_size * self._board_size + 1

    @property
    def state_channels(self) -> int:
        return 4

    def initial_state(self, board_size: int | None = None) -> GameState:
        size = board_size or self.default_board_size
        self._board_size = size
        return GameState(
            board=np.zeros((size, size), dtype=np.int8),
            current_player=1,
            move_number=0,
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        size = state.board.shape[0]
        actions = []
        for i in range(size * size):
            r, c = divmod(i, size)
            if state.board[r, c] == 0:
                actions.append(i)
        actions.append(size * size)  # pass
        return actions

    def get_action_mask(self, state: GameState) -> ActionMask:
        size = state.board.shape[0]
        mask = np.zeros(size * size + 1, dtype=bool)
        for a in self.get_legal_actions(state):
            mask[a] = True
        return mask

    def apply_action(self, state: GameState, action: int) -> GameState:
        size = state.board.shape[0]
        new_board = state.board.copy()
        if action < size * size:
            r, c = divmod(action, size)
            new_board[r, c] = state.current_player
        return GameState(
            board=new_board,
            current_player=-state.current_player,
            move_number=state.move_number + 1,
            move_history=state.move_history + [action],
        )

    def is_terminal(self, state: GameState) -> bool:
        return state.move_number >= 10

    def get_result(self, state: GameState) -> GameResult:
        return GameResult(
            winner=1,
            score_black=1.0,
            score_white=0.0,
            reason="test",
            move_count=state.move_number,
        )

    def get_winner(self, state: GameState) -> int | None:
        return 1

    def to_tensor(self, state: GameState) -> Tensor:
        size = state.board.shape[0]
        tensor = torch.zeros(self.state_channels, size, size)
        tensor[0] = torch.from_numpy((state.board == 1).astype(np.float32))
        tensor[1] = torch.from_numpy((state.board == -1).astype(np.float32))
        tensor[2] = float(state.current_player == 1)
        return tensor

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        return [(state, policy)]


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_phase_values(self) -> None:
        assert GamePhase.SETUP == "setup"
        assert GamePhase.OPENING == "opening"
        assert GamePhase.MIDGAME == "midgame"
        assert GamePhase.ENDGAME == "endgame"
        assert GamePhase.TERMINAL == "terminal"

    def test_phase_is_string(self) -> None:
        assert isinstance(GamePhase.OPENING, str)


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_create_result(self) -> None:
        result = GameResult(
            winner=1,
            score_black=10.0,
            score_white=7.5,
            reason="score",
            move_count=200,
        )
        assert result.winner == 1
        assert result.score_black == 10.0
        assert result.score_white == 7.5
        assert result.reason == "score"
        assert result.move_count == 200

    def test_draw_result(self) -> None:
        result = GameResult(
            winner=None,
            score_black=5.0,
            score_white=5.0,
            reason="score",
            move_count=100,
        )
        assert result.winner is None

    def test_various_reasons(self) -> None:
        for reason in ["resignation", "timeout", "score", "checkmate"]:
            result = GameResult(
                winner=1, score_black=1.0, score_white=0.0,
                reason=reason, move_count=50,
            )
            assert result.reason == reason


class TestGameInterfaceAbstract:
    """Tests for abstract method requirements."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            GameInterface()  # type: ignore[abstract]

    def test_must_implement_action_space_size(self) -> None:
        class Incomplete(GameInterface):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        game = ConcreteGame()
        assert game.name == "test_game"


class TestGameInterfaceConcrete:
    """Tests for concrete helper methods on GameInterface."""

    @pytest.fixture
    def game(self) -> ConcreteGame:
        return ConcreteGame()

    def test_n_players_default(self, game: ConcreteGame) -> None:
        assert game.n_players == 2

    def test_initial_state(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        assert state.board.shape == (5, 5)
        assert state.current_player == 1
        assert state.move_number == 0

    def test_initial_state_default_size(self, game: ConcreteGame) -> None:
        state = game.initial_state()
        assert state.board.shape == (9, 9)

    def test_action_space_size(self, game: ConcreteGame) -> None:
        game._board_size = 5
        assert game.action_space_size == 26  # 5*5 + 1

    def test_state_channels(self, game: ConcreteGame) -> None:
        assert game.state_channels == 4

    def test_get_legal_actions(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        actions = game.get_legal_actions(state)
        assert len(actions) == 26  # 25 empty + pass

    def test_get_action_mask(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        mask = game.get_action_mask(state)
        assert mask.shape == (26,)
        assert mask.sum() == 26

    def test_apply_action(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        new_state = game.apply_action(state, 12)  # center of 5x5
        assert new_state.board[2, 2] == 1
        assert new_state.current_player == -1
        assert new_state.move_number == 1

    def test_is_terminal(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        assert not game.is_terminal(state)

        # Play 10 moves to make terminal
        for i in range(10):
            actions = game.get_legal_actions(state)
            state = game.apply_action(state, actions[0])
        assert game.is_terminal(state)

    def test_get_result(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        result = game.get_result(state)
        assert isinstance(result, GameResult)
        assert result.winner == 1

    def test_to_tensor(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        state = game.apply_action(state, 12)
        tensor = game.to_tensor(state)
        assert tensor.shape == (4, 5, 5)
        assert tensor.dtype == torch.float32

    def test_get_symmetries(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        policy = np.ones(26) / 26
        syms = game.get_symmetries(state, policy)
        assert len(syms) == 1

    def test_get_phase_default(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        phase = game.get_phase(state)
        assert phase == GamePhase.OPENING

    def test_get_phase_terminal(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        for i in range(10):
            actions = game.get_legal_actions(state)
            state = game.apply_action(state, actions[0])
        phase = game.get_phase(state)
        assert phase == GamePhase.TERMINAL

    def test_action_to_string(self, game: ConcreteGame) -> None:
        game._board_size = 9
        s = game.action_to_string(0, 9)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_action_to_string_pass(self, game: ConcreteGame) -> None:
        game._board_size = 9
        s = game.action_to_string(81, 9)
        assert s == "pass"

    def test_string_to_action(self, game: ConcreteGame) -> None:
        game._board_size = 9
        action = game.string_to_action("pass", 9)
        assert action == 81

    def test_validate_action_valid(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        assert game.validate_action(state, 0) is True

    def test_validate_action_invalid(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        assert game.validate_action(state, -1) is False
        assert game.validate_action(state, 9999) is False

    def test_get_observation_shape(self, game: ConcreteGame) -> None:
        shape = game.get_observation_shape(board_size=9)
        assert shape == (4, 9, 9)

    def test_get_observation_shape_default(self, game: ConcreteGame) -> None:
        shape = game.get_observation_shape()
        assert shape == (4, 9, 9)

    def test_batch_to_tensor(self, game: ConcreteGame) -> None:
        states = [game.initial_state(board_size=5) for _ in range(3)]
        batch = game.batch_to_tensor(states, device="cpu")
        assert batch.shape == (3, 4, 5, 5)

    def test_get_canonical_form_player1(self, game: ConcreteGame) -> None:
        state = game.initial_state(board_size=5)
        canonical = game.get_canonical_form(state)
        assert canonical.current_player == 1

    def test_clone(self, game: ConcreteGame) -> None:
        clone = game.clone()
        assert isinstance(clone, ConcreteGame)
        assert clone is not game

    def test_repr(self, game: ConcreteGame) -> None:
        r = repr(game)
        assert "test_game" in r


class TestGameConfig:
    """Tests for GameConfig."""

    def test_create_config(self) -> None:
        config = GameConfig(game_name="go", board_size=9)
        assert config.game_name == "go"
        assert config.board_size == 9
        assert config.komi == 7.5

    def test_default_values(self) -> None:
        config = GameConfig(game_name="go")
        assert config.board_size is None
        assert config.komi == 7.5
        assert config.time_control == {}

    def test_to_dict(self) -> None:
        config = GameConfig(game_name="go", board_size=9, komi=6.5)
        d = config.to_dict()
        assert d["game_name"] == "go"
        assert d["board_size"] == 9
        assert d["komi"] == 6.5

    def test_from_dict(self) -> None:
        data = {"game_name": "go", "board_size": 13, "komi": 5.5}
        config = GameConfig.from_dict(data)
        assert config.game_name == "go"
        assert config.board_size == 13
        assert config.komi == 5.5

    def test_roundtrip(self) -> None:
        original = GameConfig(game_name="go", board_size=9, komi=6.5)
        restored = GameConfig.from_dict(original.to_dict())
        assert restored.game_name == original.game_name
        assert restored.board_size == original.board_size
        assert restored.komi == original.komi
