"""Game registry for discovering and managing game implementations.

This module provides a registration system for game implementations,
enabling automatic discovery and factory-based instantiation.

Usage:
    from src.games import GameRegistry, register_game

    @register_game("my_game")
    class MyGame(GameInterface):
        ...

    # Get game by name
    game = GameRegistry.get("my_game")
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.games.interface import GameInterface

logger = structlog.get_logger(__name__)


class GameRegistry:
    """Central registry for all game implementations.

    Provides discovery, instantiation, and management of game types.
    Thread-safe singleton pattern for global access.

    Thread Safety:
        All operations are protected by a lock to ensure thread-safe
        registration and lookup.

    """

    _instance: GameRegistry | None = None
    _lock: threading.Lock = threading.Lock()
    _games: dict[str, type[GameInterface]]

    def __new__(cls) -> GameRegistry:
        """Ensure singleton instance with thread-safe initialization."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._games = {}
        return cls._instance

    def register(
        self,
        name: str,
        game_cls: type[GameInterface],
        override: bool = False,
    ) -> None:
        """Register a game implementation.

        Thread-safe registration with duplicate check.

        Args:
            name: Unique game identifier (must be non-empty).
            game_cls: Game class to register.
            override: Allow overriding existing registration.

        Raises:
            ValueError: If name is already registered (and not override).

        """
        if not name or not name.strip():
            raise ValueError("Game name cannot be empty")

        with self._lock:
            if name in self._games and not override:
                raise ValueError(f"Game '{name}' already registered by {self._games[name]}")

            self._games[name] = game_cls
            logger.debug("game_registered", name=name, cls=game_cls.__name__)

    def get(self, name: str) -> GameInterface | None:
        """Get an instantiated game by name.

        Thread-safe lookup and instantiation.

        Args:
            name: Game identifier.

        Returns:
            New instance of the game, or None if not found.

        """
        with self._lock:
            game_cls = self._games.get(name)

        if game_cls is None:
            return None

        return game_cls()

    def get_class(self, name: str) -> type[GameInterface] | None:
        """Get game class by name (without instantiation).

        Thread-safe lookup.

        Args:
            name: Game identifier.

        Returns:
            Game class, or None if not found.

        """
        with self._lock:
            return self._games.get(name)

    def list_games(self) -> list[str]:
        """List all registered game names.

        Thread-safe list copy.

        Returns:
            List of registered game names.

        """
        with self._lock:
            return list(self._games.keys())

    def get_all(self) -> dict[str, type[GameInterface]]:
        """Get all registered game classes.

        Thread-safe dict copy.

        Returns:
            Dictionary mapping names to game classes.

        """
        with self._lock:
            return dict(self._games)

    def clear(self) -> None:
        """Clear all registrations (primarily for testing).

        Thread-safe clear with warning.

        Warning:
            This should only be called in test contexts.

        """
        with self._lock:
            logger.warning(
                "registry_cleared",
                count=len(self._games),
                message="Game registry cleared - this should only happen in tests",
            )
            self._games.clear()

    def is_registered(self, name: str) -> bool:
        """Check if a game is registered.

        Args:
            name: Game identifier.

        Returns:
            True if game is registered.

        """
        with self._lock:
            return name in self._games

    def get_info(self, name: str) -> dict[str, Any] | None:
        """Get information about a registered game.

        Args:
            name: Game identifier.

        Returns:
            Dictionary with game information, or None if not found.

        """
        game = self.get(name)
        if game is None:
            return None

        return {
            "name": game.name,
            "description": game.description,
            "min_board_size": game.min_board_size,
            "max_board_size": game.max_board_size,
            "default_board_size": game.default_board_size,
            "action_space_size": game.action_space_size,
            "state_channels": game.state_channels,
            "n_players": game.n_players,
        }


def register_game(name: str) -> Callable[[type], type]:
    """Decorator to register a game class.

    Args:
        name: Unique game identifier.

    Returns:
        Class decorator.

    Example:
        @register_game("my_game")
        class MyGame(GameInterface):
            ...

    """

    def decorator(cls: type) -> type:
        GameRegistry().register(name, cls)
        cls.name = name  # type: ignore[attr-defined]  # Set name on class
        return cls

    return decorator


def get_game(name: str) -> GameInterface:
    """Get a game instance by name.

    Convenience function for accessing the registry.

    Args:
        name: Game identifier.

    Returns:
        Game instance.

    Raises:
        ValueError: If game not found.

    """
    game = GameRegistry().get(name)
    if game is None:
        available = GameRegistry().list_games()
        raise ValueError(f"Game '{name}' not found. Available: {available}")
    return game


def list_available_games() -> list[dict[str, Any]]:
    """List all available games with their info.

    Returns:
        List of game information dictionaries.

    """
    registry = GameRegistry()
    games = []

    for name in registry.list_games():
        info = registry.get_info(name)
        if info:
            games.append(info)

    return games
