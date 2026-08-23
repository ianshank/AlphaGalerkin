"""Generic, thread-safe registry pattern for AlphaGalerkin extensible modules."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

import structlog

from src.templates.registry import BaseRegistry, create_registry

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """Type-safe registry wrapper with thread-safe item registration and lookup.

    Provides a clean object-oriented interface over the template registry factory.

    Example:
        >>> registry = Registry[BaseSolver]("Solver")
        >>> @registry.register("my_solver", deprecated_names=["legacy_solver"])
        ... class MySolver(BaseSolver): ...
        >>> solver_cls = registry.get("my_solver")

    """

    def __init__(
        self,
        name: str,
        base_type: type[T] | None = None,
    ) -> None:
        self._name = name
        self._base_type = base_type
        self._lock = threading.Lock()
        self._items: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        *,
        deprecated_names: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register an implementation class."""

        def decorator(cls: type[T]) -> type[T]:
            with self._lock:
                self._items[name] = cls
                if deprecated_names:
                    for alias in deprecated_names:
                        self._aliases[alias] = name
            return cls

        return decorator

    def get(self, name: str) -> type[T] | None:
        """Get a registered class by name or alias (None if missing)."""
        with self._lock:
            resolved_name = self._aliases.get(name, name)
            return self._items.get(resolved_name)

    def get_or_raise(self, name: str) -> type[T]:
        """Get a registered class by name or raise KeyError."""
        item = self.get(name)
        if item is None:
            available = self.list_items()
            raise KeyError(f"Unknown {self._name}: {name!r}. Available: {available}")
        return item

    def list_items(self) -> list[str]:
        """List all primary registered item keys in sorted order."""
        with self._lock:
            return sorted(self._items.keys())

    def is_registered(self, name: str) -> bool:
        """Return True if name or an alias is registered."""
        with self._lock:
            return name in self._items or name in self._aliases

    def clear(self) -> None:
        """Clear all registered items (for test isolation)."""
        with self._lock:
            self._items.clear()
            self._aliases.clear()

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


__all__ = [
    "BaseRegistry",
    "Registry",
    "create_registry",
]
