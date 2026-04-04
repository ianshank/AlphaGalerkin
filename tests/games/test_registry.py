"""Tests for GameRegistry and registration utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from src.games.interface import GameInterface, GameResult
from src.games.registry import (
    GameRegistry,
    get_game,
    list_available_games,
    register_game,
)
from src.games.state import ActionMask, GameState


class DummyGame(GameInterface):
    """Dummy game for registry testing."""

    name = "dummy"
    description = "Dummy game"
    default_board_size = 5

    @property
    def action_space_size(self) -> int:
        return 26

    @property
    def state_channels(self) -> int:
        return 2

    def initial_state(self, board_size: int | None = None) -> GameState:
        size = board_size or self.default_board_size
        return GameState(board=np.zeros((size, size), dtype=np.int8))

    def get_legal_actions(self, state: GameState) -> list[int]:
        return list(range(26))

    def get_action_mask(self, state: GameState) -> ActionMask:
        return np.ones(26, dtype=bool)

    def apply_action(self, state: GameState, action: int) -> GameState:
        return state.copy()

    def is_terminal(self, state: GameState) -> bool:
        return False

    def get_result(self, state: GameState) -> GameResult:
        return GameResult(winner=None, score_black=0, score_white=0, reason="test", move_count=0)

    def get_winner(self, state: GameState) -> int | None:
        return None

    def to_tensor(self, state: GameState) -> Tensor:
        return torch.zeros(2, 5, 5)

    def get_symmetries(
        self, state: GameState, policy: np.ndarray | Tensor
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        return [(state, policy)]


class TestGameRegistry:
    """Tests for the GameRegistry singleton."""

    def test_singleton(self) -> None:
        r1 = GameRegistry()
        r2 = GameRegistry()
        assert r1 is r2

    def test_go_is_registered(self) -> None:
        """Go should be registered on import."""
        registry = GameRegistry()
        assert registry.is_registered("go")

    def test_register_new_game(self) -> None:
        registry = GameRegistry()
        registry.register("test_dummy_unique_1", DummyGame, override=True)
        assert registry.is_registered("test_dummy_unique_1")

    def test_register_duplicate_raises(self) -> None:
        registry = GameRegistry()
        registry.register("test_dup_game", DummyGame, override=True)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_dup_game", DummyGame, override=False)

    def test_register_override(self) -> None:
        registry = GameRegistry()
        registry.register("test_override_game", DummyGame, override=True)
        registry.register("test_override_game", DummyGame, override=True)
        assert registry.is_registered("test_override_game")

    def test_register_empty_name_raises(self) -> None:
        registry = GameRegistry()
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register("", DummyGame)

    def test_register_whitespace_name_raises(self) -> None:
        registry = GameRegistry()
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register("   ", DummyGame)

    def test_get_existing_game(self) -> None:
        registry = GameRegistry()
        registry.register("test_get_game", DummyGame, override=True)
        game = registry.get("test_get_game")
        assert game is not None
        assert isinstance(game, DummyGame)

    def test_get_nonexistent_game(self) -> None:
        registry = GameRegistry()
        game = registry.get("nonexistent_game_xyz")
        assert game is None

    def test_get_class(self) -> None:
        registry = GameRegistry()
        registry.register("test_get_class", DummyGame, override=True)
        cls = registry.get_class("test_get_class")
        assert cls is DummyGame

    def test_get_class_nonexistent(self) -> None:
        registry = GameRegistry()
        cls = registry.get_class("nonexistent_xyz_123")
        assert cls is None

    def test_list_games(self) -> None:
        registry = GameRegistry()
        games = registry.list_games()
        assert isinstance(games, list)
        assert "go" in games

    def test_get_all(self) -> None:
        registry = GameRegistry()
        all_games = registry.get_all()
        assert isinstance(all_games, dict)
        assert "go" in all_games

    def test_is_registered(self) -> None:
        registry = GameRegistry()
        assert registry.is_registered("go")
        assert not registry.is_registered("nonexistent_xyz_999")

    def test_get_info(self) -> None:
        registry = GameRegistry()
        info = registry.get_info("go")
        assert info is not None
        assert "name" in info
        assert "action_space_size" in info
        assert "state_channels" in info
        assert "n_players" in info
        assert info["n_players"] == 2

    def test_get_info_nonexistent(self) -> None:
        registry = GameRegistry()
        info = registry.get_info("nonexistent_xyz_888")
        assert info is None


class TestRegisterGameDecorator:
    """Tests for the @register_game decorator."""

    def test_decorator_registers(self) -> None:
        @register_game("test_decorator_game")
        class DecoratedGame(DummyGame):
            pass

        registry = GameRegistry()
        assert registry.is_registered("test_decorator_game")

    def test_decorator_sets_name(self) -> None:
        @register_game("test_name_set_game")
        class NamedGame(DummyGame):
            pass

        assert NamedGame.name == "test_name_set_game"


class TestGetGame:
    """Tests for get_game convenience function."""

    def test_get_existing(self) -> None:
        game = get_game("go")
        assert game is not None
        assert game.name == "go"

    def test_get_nonexistent_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_game("nonexistent_game_abc")


class TestListAvailableGames:
    """Tests for list_available_games function."""

    def test_returns_list(self) -> None:
        games = list_available_games()
        assert isinstance(games, list)
        assert len(games) > 0

    def test_game_info_structure(self) -> None:
        games = list_available_games()
        for info in games:
            assert "name" in info
            assert "description" in info
            assert "action_space_size" in info
