"""Tests for GameRegistry and registration utilities.

Tests cover:
- Singleton pattern (same instance across calls)
- Register and get game
- list_games returns registered games
- get with unknown game returns None / raises via get_game
- Decorator @register_game
- Thread safety (concurrent registration)
- Registration validation (empty names, duplicates, override)
- Convenience functions (get_game, list_available_games)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from src.games.interface import GameInterface, GameResult
from src.games.registry import (
    GameRegistry,
    get_game,
    list_available_games,
    register_game,
)
from src.games.state import ActionMask, GameState

if TYPE_CHECKING:
    pass


# --- Helpers ---


class _StubGame(GameInterface):
    """Minimal concrete game implementation for testing."""

    name = "stub"
    description = "Stub game for tests"
    min_board_size = 3
    max_board_size = 19
    default_board_size = 9

    @property
    def action_space_size(self) -> int:
        return self.default_board_size ** 2 + 1

    @property
    def state_channels(self) -> int:
        return 3

    def initial_state(self, board_size: int | None = None) -> GameState:
        size = board_size or self.default_board_size
        return GameState(
            board=np.zeros((size, size), dtype=np.float32),
            current_player=1,
        )

    def get_legal_actions(self, state: GameState) -> list[int]:
        size = state.board_size
        return list(range(size * size + 1))

    def get_action_mask(self, state: GameState) -> ActionMask:
        size = state.board_size
        mask = np.ones(size * size + 1, dtype=bool)
        return ActionMask(mask=mask, action_space_size=size * size + 1)

    def apply_action(self, state: GameState, action: int) -> GameState:
        new_board = state.board.copy()
        size = state.board_size
        if action < size * size:
            row, col = divmod(action, size)
            new_board[row, col] = state.current_player
        return state.with_move(action=action, new_board=new_board)

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
        if self.is_terminal(state):
            return 1
        return None

    def to_tensor(self, state: GameState):
        import torch

        return torch.zeros(self.state_channels, state.board_size, state.board_size)

    def get_symmetries(self, state, policy):
        return [(state, policy)]


def _make_stub_class(name: str) -> type[_StubGame]:
    """Create a uniquely-named stub game class."""
    return type(name, (_StubGame,), {"name": name, "description": f"{name} game"})


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test.

    Avoids side-effects from registration leaking between tests while
    preserving the real 'go' and 'chess' registrations.
    """
    registry = GameRegistry()
    saved = dict(registry._games)
    yield
    registry._games.clear()
    registry._games.update(saved)


@pytest.fixture
def registry() -> GameRegistry:
    """Provide a reference to the singleton registry."""
    return GameRegistry()


# --- Singleton Tests ---


class TestSingletonPattern:
    """Tests for GameRegistry singleton behaviour."""

    def test_same_instance_returned(self) -> None:
        """Two calls to GameRegistry() return the identical object."""
        a = GameRegistry()
        b = GameRegistry()
        assert a is b

    def test_games_dict_shared(self) -> None:
        """The internal _games dict is shared across references."""
        a = GameRegistry()
        b = GameRegistry()
        assert a._games is b._games

    def test_class_attribute_preserved(self) -> None:
        """Singleton survives repeated construction."""
        _ = GameRegistry()
        _ = GameRegistry()
        instance = GameRegistry()
        assert hasattr(instance, "_games")


# --- Registration Tests ---


class TestRegistration:
    """Tests for registering and retrieving games."""

    def test_register_and_get(self, registry: GameRegistry) -> None:
        """Register a game and retrieve it."""
        stub_cls = _make_stub_class("test_reg")
        registry.register("test_reg", stub_cls)

        game = registry.get("test_reg")
        assert game is not None
        assert isinstance(game, _StubGame)

    def test_get_class_returns_class(self, registry: GameRegistry) -> None:
        """get_class returns the registered class itself, not an instance."""
        stub_cls = _make_stub_class("test_get_cls")
        registry.register("test_get_cls", stub_cls)

        cls = registry.get_class("test_get_cls")
        assert cls is stub_cls

    def test_get_unknown_returns_none(self, registry: GameRegistry) -> None:
        """get() returns None for an unregistered name."""
        assert registry.get("nonexistent_game_xyz") is None

    def test_get_class_unknown_returns_none(self, registry: GameRegistry) -> None:
        """get_class() returns None for an unregistered name."""
        assert registry.get_class("nonexistent_game_xyz") is None

    def test_is_registered_true(self, registry: GameRegistry) -> None:
        """is_registered returns True for known games."""
        stub_cls = _make_stub_class("test_is_reg")
        registry.register("test_is_reg", stub_cls)

        assert registry.is_registered("test_is_reg") is True

    def test_is_registered_false(self, registry: GameRegistry) -> None:
        """is_registered returns False for unknown games."""
        assert registry.is_registered("never_registered_xyz") is False

    def test_duplicate_registration_raises(self, registry: GameRegistry) -> None:
        """Registering the same name twice raises ValueError."""
        stub_cls = _make_stub_class("dup")
        registry.register("dup", stub_cls)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", stub_cls)

    def test_override_allows_duplicate(self, registry: GameRegistry) -> None:
        """override=True replaces an existing registration."""
        cls_a = _make_stub_class("over_a")
        cls_b = _make_stub_class("over_b")
        registry.register("override_test", cls_a)
        registry.register("override_test", cls_b, override=True)

        assert registry.get_class("override_test") is cls_b

    @pytest.mark.parametrize("bad_name", ["", "   ", "\t", "\n"])
    def test_empty_name_raises(self, registry: GameRegistry, bad_name: str) -> None:
        """Empty or whitespace-only names are rejected."""
        stub_cls = _make_stub_class("empty_name_test")
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register(bad_name, stub_cls)


# --- list_games Tests ---


class TestListGames:
    """Tests for listing registered games."""

    def test_list_includes_registered(self, registry: GameRegistry) -> None:
        """Newly registered game appears in list_games."""
        stub_cls = _make_stub_class("listed")
        registry.register("listed", stub_cls)

        names = registry.list_games()
        assert "listed" in names

    def test_list_returns_copy(self, registry: GameRegistry) -> None:
        """list_games returns a new list, not internal state."""
        names_a = registry.list_games()
        names_b = registry.list_games()
        assert names_a is not names_b

    def test_builtin_games_present(self, registry: GameRegistry) -> None:
        """Go and Chess should be registered after module import."""
        names = registry.list_games()
        assert "go" in names
        assert "chess" in names

    def test_get_all_returns_dict_copy(self, registry: GameRegistry) -> None:
        """get_all returns a copy of the internal mapping."""
        all_games = registry.get_all()
        assert isinstance(all_games, dict)
        assert all_games is not registry._games


# --- get_info Tests ---


class TestGetInfo:
    """Tests for GameRegistry.get_info."""

    def test_get_info_known_game(self, registry: GameRegistry) -> None:
        """get_info returns a dict with expected keys."""
        stub_cls = _make_stub_class("info_test")
        registry.register("info_test", stub_cls)

        info = registry.get_info("info_test")
        assert info is not None
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
        assert expected_keys <= set(info.keys())

    def test_get_info_unknown_returns_none(self, registry: GameRegistry) -> None:
        """get_info returns None for unregistered game."""
        assert registry.get_info("unknown_xyz") is None


# --- clear Tests ---


class TestClear:
    """Tests for clearing the registry."""

    def test_clear_removes_all(self, registry: GameRegistry) -> None:
        """clear() empties the registry."""
        stub_cls = _make_stub_class("clear_test")
        registry.register("clear_test", stub_cls)
        assert registry.is_registered("clear_test")

        registry.clear()

        assert len(registry.list_games()) == 0

    def test_clear_and_re_register(self, registry: GameRegistry) -> None:
        """After clear, new registrations work normally."""
        registry.clear()

        stub_cls = _make_stub_class("after_clear")
        registry.register("after_clear", stub_cls)
        assert registry.is_registered("after_clear")


# --- Decorator Tests ---


class TestRegisterGameDecorator:
    """Tests for the @register_game decorator."""

    def test_decorator_registers_class(self, registry: GameRegistry) -> None:
        """@register_game adds the class to the registry."""

        @register_game("decorated_game")
        class DecoratedGame(_StubGame):
            pass

        assert registry.is_registered("decorated_game")
        game = registry.get("decorated_game")
        assert isinstance(game, DecoratedGame)

    def test_decorator_sets_name_attribute(self) -> None:
        """@register_game sets .name on the class."""

        @register_game("named_by_decorator")
        class NamedGame(_StubGame):
            pass

        assert NamedGame.name == "named_by_decorator"

    def test_decorator_returns_original_class(self) -> None:
        """@register_game returns the same class object."""

        @register_game("identity_test")
        class IdentityGame(_StubGame):
            pass

        # The class should still be usable directly
        game = IdentityGame()
        assert isinstance(game, _StubGame)


# --- Convenience Function Tests ---


class TestConvenienceFunctions:
    """Tests for get_game and list_available_games module-level functions."""

    def test_get_game_existing(self) -> None:
        """get_game returns an instance for known games."""
        game = get_game("go")
        assert game is not None
        assert game.name == "go"

    def test_get_game_unknown_raises(self) -> None:
        """get_game raises ValueError for unknown game."""
        with pytest.raises(ValueError, match="not found"):
            get_game("nonexistent_game_xyz")

    def test_list_available_games_returns_list_of_dicts(self) -> None:
        """list_available_games returns info dicts for each game."""
        games = list_available_games()
        assert isinstance(games, list)
        assert len(games) >= 2  # at least Go and Chess

        for info in games:
            assert "name" in info
            assert "action_space_size" in info


# --- Thread Safety Tests ---


class TestThreadSafety:
    """Tests for concurrent access to GameRegistry."""

    @pytest.fixture
    def n_threads(self) -> int:
        """Number of threads for concurrency tests."""
        return 16

    def test_concurrent_registration(
        self, registry: GameRegistry, n_threads: int
    ) -> None:
        """Multiple threads registering different games should not corrupt state."""
        errors: list[Exception] = []
        barrier = threading.Barrier(n_threads)

        def register_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                name = f"thread_game_{idx}"
                cls = _make_stub_class(f"ThreadStub{idx}")
                registry.register(name, cls)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Errors during concurrent registration: {errors}"

        # All should be registered
        for i in range(n_threads):
            assert registry.is_registered(f"thread_game_{i}")

    def test_concurrent_get(self, registry: GameRegistry, n_threads: int) -> None:
        """Multiple threads reading the same game should not fail."""
        errors: list[Exception] = []
        results: list[GameInterface | None] = [None] * n_threads
        barrier = threading.Barrier(n_threads)

        def get_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                results[idx] = registry.get("go")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=get_worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Errors during concurrent get: {errors}"
        for r in results:
            assert r is not None
            assert r.name == "go"

    def test_concurrent_list_games(
        self, registry: GameRegistry, n_threads: int
    ) -> None:
        """Multiple threads calling list_games should not fail."""
        errors: list[Exception] = []
        results: list[list[str]] = [[] for _ in range(n_threads)]
        barrier = threading.Barrier(n_threads)

        def list_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                results[idx] = registry.list_games()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=list_worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Errors during concurrent list: {errors}"
        # All results should contain the same games
        for names in results:
            assert "go" in names
            assert "chess" in names

    def test_concurrent_register_and_read(
        self, registry: GameRegistry, n_threads: int
    ) -> None:
        """Mixing registration and reads should not corrupt state."""
        errors: list[Exception] = []
        barrier = threading.Barrier(n_threads)

        def mixed_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                if idx % 2 == 0:
                    name = f"mixed_game_{idx}"
                    cls = _make_stub_class(f"MixedStub{idx}")
                    registry.register(name, cls)
                else:
                    _ = registry.list_games()
                    _ = registry.get("go")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=mixed_worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Errors during mixed concurrent access: {errors}"
