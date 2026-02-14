"""Plugin registry for physics modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger("physics.registry")

T = TypeVar("T")


class PhysicsRegistry:
    """Registry for PDE physics modules.

    Usage:
        registry = PhysicsRegistry()

        @registry.register_decorator("poisson_2d")
        class PoissonModule:
            ...

        module = registry.get("poisson_2d")
    """

    _instance: PhysicsRegistry | None = None
    _plugins: dict[str, type] = {}
    _instances: dict[str, Any] = {}

    def __new__(cls) -> PhysicsRegistry:
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._plugins = {}
            cls._instances = {}
        return cls._instance

    def register(self, name: str, cls_to_register: type) -> None:
        """Register a physics module class."""
        if name in self._plugins:
            msg = f"'{name}' already registered in physics registry"
            raise ValueError(msg)
        self._plugins[name] = cls_to_register
        logger.info("physics.registry.registered", name=name)

    def register_decorator(
        self,
        name: str,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator for registering physics modules."""

        def decorator(cls: type[T]) -> type[T]:
            self.register(name, cls)
            return cls

        return decorator

    def get(self, name: str, **kwargs: Any) -> Any:
        """Get or create a physics module instance."""
        if name not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys()))
            msg = f"'{name}' not registered in physics registry. Available: {available}"
            raise KeyError(msg)
        if name not in self._instances:
            self._instances[name] = self._plugins[name](**kwargs)
        return self._instances[name]

    def list_modules(self) -> list[str]:
        """List all registered module names."""
        return sorted(self._plugins.keys())

    def clear_instances(self) -> None:
        """Clear cached instances (useful for testing)."""
        self._instances.clear()

    def clear_all(self) -> None:
        """Clear everything (useful for testing)."""
        self._plugins.clear()
        self._instances.clear()


# Global registry instance
_registry = PhysicsRegistry()


def register_physics(
    name: str,
) -> Callable[[type[T]], type[T]]:
    """Module-level decorator for registering physics modules.

    Usage:
        @register_physics("poisson_2d")
        class PoissonModule:
            ...
    """
    return _registry.register_decorator(name)
