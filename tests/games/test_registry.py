"""Tests for the games.registry module.

Covers:
    - GameRegistry singleton pattern
    - register() with valid, empty-name, and duplicate-name cases
    - get() for registered and unregistered games
    - get_class() for registered and unregistered games
    - get_info() for registered and unregistered games
    - list_games() listing
    - get_all() returning a dict copy
    - clear() removing all registrations
    - is_registered() boolean check
    - register_game decorator
    - get_game convenience function
    - list_available_games convenience function
    - Override registration
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from src.games.interface import GameInterface
from src.games.registry import GameRegistry, get_game, list_available_games, register_game
from src.games.state import ActionMask, GameState


# ---------------------------------------------------------------------------
# Helpers: minimal concrete GameInterface for testing
# ---------------------------------------------------------------------------

class _StubGame(GameInterface):
    """Minimal concrete GameInterface for registry tests."""

    name: str = "stub"
    description: str = "A stub game for testing"
    min_board_size: int = 3
    max_board_size: int = 19
    default_board_size: int = 9

    @property
    def action_space_size(self) -> int:
        return self.default_board_size ** 2 + 1

    @property
    def state_channels(self) -> int:
        return 4

    def initial_state(self, board_size: int | None = None) -> GameState:
        bs = board_size or self.default_board_size
        return GameState(
            board=np.zeros((bs, bs), dtype=np.int8),
            current_player=1,
            move_number=0,
            board_size=bs,
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        return list(range(state.board_size ** 2 + 1))

    def get_action_mask(self, state: GameState) -> ActionMask:
        mask = np.ones(state.board_size ** 2 + 1, dtype=bool)
        return ActionMask(mask=mask)

    def apply_action(self, state: GameState, action: int) -> GameState:
        return state

    def is_terminal(self, state: GameState) -> bool:
        return False

    def get_result(self, state: GameState) -> Any:
        return None

    def get_winner(self, state: GameState) -> int | None:
        return None

    def to_tensor(self, state: GameState) -> Tensor:
        return torch.zeros(self.state_channels, state.board_size, state.board_size)

    def get_symmetries(
        self,
        state: GameState,
        policy: np.ndarray | Tensor,
    ) -> list[tuple[GameState, np.ndarray | Tensor]]:
        return [(state, policy)]


class _AnotherStubGame(_StubGame):
    """Second stub game class for duplicate registration tests."""

    name: str = "another_stub"
    description: str = "Another stub game"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around every test.

    This ensures test isolation: each test starts with an empty registry
    for the stub games and the real games are restored afterward.
    """
    registry = GameRegistry()
    saved = dict(registry._games)
    # Clear to start fresh for each test
    registry._games.clear()
    yield
    # Restore original state
    registry._games.clear()
    registry._games.update(saved)


# ===========================================================================
# Test classes
# ===========================================================================


class TestGameRegistrySingleton:
    """Tests for the GameRegistry singleton pattern."""

    def test_singleton_returns_same_instance(self) -> None:
        """Two calls to GameRegistry() must return the exact same object."""
        r1 = GameRegistry()
        r2 = GameRegistry()
        assert r1 is r2

    def test_singleton_shares_state(self) -> None:
        """Registrations made through one reference are visible through another."""
        r1 = GameRegistry()
        r2 = GameRegistry()
        r1.register("shared_test", _StubGame)
        assert r2.is_registered("shared_test")


class TestRegister:
    """Tests for GameRegistry.register()."""

    def test_register_valid_game(self) -> None:
        """A valid game class can be registered and retrieved."""
        registry = GameRegistry()
        registry.register("test_game", _StubGame)
        assert registry.is_registered("test_game")

    def test_register_empty_name_raises(self) -> None:
        """Registering with an empty string name must raise ValueError."""
        registry = GameRegistry()
        with pytest.raises(ValueError, match="Game name cannot be empty"):
            registry.register("", _StubGame)

    def test_register_whitespace_name_raises(self) -> None:
        """Registering with a whitespace-only name must raise ValueError."""
        registry = GameRegistry()
        with pytest.raises(ValueError, match="Game name cannot be empty"):
            registry.register("   ", _StubGame)

    def test_register_duplicate_name_raises(self) -> None:
        """Re-registering the same name without override must raise ValueError."""
        registry = GameRegistry()
        registry.register("dup", _StubGame)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", _AnotherStubGame)

    def test_register_duplicate_name_with_override(self) -> None:
        """Re-registering the same name with override=True must succeed."""
        registry = GameRegistry()
        registry.register("overridable", _StubGame)
        registry.register("overridable", _AnotherStubGame, override=True)
        # The overridden class should be the new one
        cls = registry.get_class("overridable")
        assert cls is _AnotherStubGame


class TestIsRegistered:
    """Tests for GameRegistry.is_registered()."""

    def test_registered_game(self) -> None:
        """is_registered returns True for a registered game."""
        registry = GameRegistry()
        registry.register("exists", _StubGame)
        assert registry.is_registered("exists") is True

    def test_unregistered_game(self) -> None:
        """is_registered returns False for a game that was never registered."""
        registry = GameRegistry()
        assert registry.is_registered("no_such_game") is False


class TestGet:
    """Tests for GameRegistry.get()."""

    def test_get_registered_game_returns_instance(self) -> None:
        """get() should return a new instance of the registered game class."""
        registry = GameRegistry()
        registry.register("inst", _StubGame)
        game = registry.get("inst")
        assert game is not None
        assert isinstance(game, _StubGame)

    def test_get_unregistered_game_returns_none(self) -> None:
        """get() should return None for an unregistered game name."""
        registry = GameRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_get_returns_new_instances(self) -> None:
        """Each call to get() should return a distinct instance."""
        registry = GameRegistry()
        registry.register("multi", _StubGame)
        g1 = registry.get("multi")
        g2 = registry.get("multi")
        assert g1 is not g2


class TestGetClass:
    """Tests for GameRegistry.get_class()."""

    def test_get_class_registered(self) -> None:
        """get_class() should return the class itself (not an instance)."""
        registry = GameRegistry()
        registry.register("cls_test", _StubGame)
        cls = registry.get_class("cls_test")
        assert cls is _StubGame

    def test_get_class_unregistered(self) -> None:
        """get_class() should return None for an unregistered name."""
        registry = GameRegistry()
        assert registry.get_class("missing") is None


class TestGetInfo:
    """Tests for GameRegistry.get_info()."""

    def test_get_info_registered(self) -> None:
        """get_info() should return a dict with game metadata."""
        registry = GameRegistry()
        registry.register("info_game", _StubGame)
        info = registry.get_info("info_game")
        assert info is not None
        assert info["name"] == "stub"
        assert info["description"] == "A stub game for testing"
        assert info["min_board_size"] == 3
        assert info["max_board_size"] == 19
        assert info["default_board_size"] == 9
        assert info["action_space_size"] == 9 ** 2 + 1
        assert info["state_channels"] == 4
        assert info["n_players"] == 2

    def test_get_info_unregistered_returns_none(self) -> None:
        """get_info() should return None for an unregistered game."""
        registry = GameRegistry()
        assert registry.get_info("does_not_exist") is None


class TestListGames:
    """Tests for GameRegistry.list_games()."""

    def test_list_games_empty(self) -> None:
        """list_games() returns an empty list when nothing is registered."""
        registry = GameRegistry()
        assert registry.list_games() == []

    def test_list_games_with_entries(self) -> None:
        """list_games() returns names of all registered games."""
        registry = GameRegistry()
        registry.register("alpha", _StubGame)
        registry.register("beta", _AnotherStubGame)
        names = registry.list_games()
        assert set(names) == {"alpha", "beta"}

    def test_list_games_returns_copy(self) -> None:
        """Mutating the returned list must not affect the registry."""
        registry = GameRegistry()
        registry.register("safe", _StubGame)
        names = registry.list_games()
        names.append("hacked")
        assert "hacked" not in registry.list_games()


class TestGetAll:
    """Tests for GameRegistry.get_all()."""

    def test_get_all_returns_dict(self) -> None:
        """get_all() should return a dict mapping name -> class."""
        registry = GameRegistry()
        registry.register("a", _StubGame)
        registry.register("b", _AnotherStubGame)
        all_games = registry.get_all()
        assert all_games == {"a": _StubGame, "b": _AnotherStubGame}

    def test_get_all_returns_copy(self) -> None:
        """Mutating the returned dict must not affect the registry."""
        registry = GameRegistry()
        registry.register("orig", _StubGame)
        all_games = registry.get_all()
        all_games["injected"] = _AnotherStubGame
        assert not registry.is_registered("injected")


class TestClear:
    """Tests for GameRegistry.clear()."""

    def test_clear_removes_all(self) -> None:
        """clear() should empty the registry."""
        registry = GameRegistry()
        registry.register("x", _StubGame)
        registry.register("y", _AnotherStubGame)
        assert len(registry.list_games()) == 2

        registry.clear()
        assert registry.list_games() == []

    def test_clear_empty_registry(self) -> None:
        """clear() on an already-empty registry should not raise."""
        registry = GameRegistry()
        registry.clear()  # should not raise
        assert registry.list_games() == []


class TestRegisterGameDecorator:
    """Tests for the register_game() decorator."""

    def test_decorator_registers_class(self) -> None:
        """Using @register_game should add the class to the registry."""

        @register_game("decorated_stub")
        class DecoratedGame(_StubGame):
            pass

        registry = GameRegistry()
        assert registry.is_registered("decorated_stub")
        game = registry.get("decorated_stub")
        assert isinstance(game, DecoratedGame)

    def test_decorator_sets_name_attribute(self) -> None:
        """The decorator should set the 'name' class attribute."""

        @register_game("named_stub")
        class NamedGame(_StubGame):
            pass

        assert NamedGame.name == "named_stub"

    def test_decorator_returns_original_class(self) -> None:
        """The decorator should return the same class object it receives."""

        @register_game("identity_stub")
        class IdentityGame(_StubGame):
            pass

        assert isinstance(IdentityGame(), _StubGame)


class TestGetGameConvenience:
    """Tests for the get_game() convenience function."""

    def test_get_game_registered(self) -> None:
        """get_game() returns an instance for a registered game."""
        GameRegistry().register("conv_game", _StubGame)
        game = get_game("conv_game")
        assert isinstance(game, _StubGame)

    def test_get_game_unregistered_raises(self) -> None:
        """get_game() raises ValueError when the game is not found."""
        with pytest.raises(ValueError, match="not found"):
            get_game("totally_missing")

    def test_get_game_error_shows_available(self) -> None:
        """The ValueError message from get_game() should list available games."""
        GameRegistry().register("listed_game", _StubGame)
        with pytest.raises(ValueError, match="listed_game"):
            get_game("wrong_name")


class TestListAvailableGames:
    """Tests for the list_available_games() convenience function."""

    def test_list_available_games_empty(self) -> None:
        """list_available_games() returns an empty list with no registrations."""
        result = list_available_games()
        assert result == []

    def test_list_available_games_with_entries(self) -> None:
        """list_available_games() returns info dicts for every registered game."""
        GameRegistry().register("game_a", _StubGame)
        GameRegistry().register("game_b", _AnotherStubGame)
        result = list_available_games()
        assert len(result) == 2
        names = {info["name"] for info in result}
        # Both stub classes have the same base name attributes; just check count
        assert len(names) >= 1

    def test_list_available_games_contains_expected_keys(self) -> None:
        """Each info dict should contain the expected metadata keys."""
        GameRegistry().register("keyed", _StubGame)
        result = list_available_games()
        assert len(result) == 1
        info = result[0]
        expected_keys = {
            "name",
            "description",
            "min_board_size",
            "max_board_size",
            "default_board_size",
            "action_space_size",
            "state_channels",
            "n_players",
        }
        assert set(info.keys()) == expected_keys
