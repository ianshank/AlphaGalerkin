"""Thread-safe registry pattern for AlphaGalerkin modules.

This module provides a reusable registry pattern with:
- Thread-safe singleton implementation
- Double-check locking for initialization
- Decorator-based registration
- Factory function for creating module-specific registries

Example:
    from src.templates.registry import create_registry

    # Define base class
    class BaseAnalyzer:
        def analyze(self, data): ...

    # Create registry and decorator
    AnalyzerRegistry, register_analyzer = create_registry("Analyzer", BaseAnalyzer)

    # Use decorator to register implementations
    @register_analyzer("statistical")
    class StatisticalAnalyzer(BaseAnalyzer):
        def analyze(self, data):
            return statistics.mean(data)

    # Retrieve registered class
    analyzer_cls = AnalyzerRegistry().get("statistical")

"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Thread-safe singleton registry base class.

    This class provides the core registry functionality with:
    - Thread-safe singleton pattern using double-check locking
    - Protected access to internal state
    - Clear API for registration and retrieval
    - Support for registry clearing (for testing)

    Subclasses are created automatically by create_registry().
    """

    _instance: BaseRegistry[T] | None = None
    _lock: threading.Lock
    _items: dict[str, type[T]]
    _registry_name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize class-level attributes for each subclass."""
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._lock = threading.Lock()

    def __new__(cls) -> BaseRegistry[T]:
        """Ensure singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._items = {}
                    cls._instance = instance
        return cls._instance

    def register(self, name: str, item_cls: type[T]) -> None:
        """Register an item class with the given name.

        Args:
            name: Unique identifier for the item.
            item_cls: Class to register.

        Raises:
            ValueError: If name is empty or already registered.

        """
        if not name or not name.strip():
            raise ValueError("Registration name cannot be empty")

        with self._lock:
            if name in self._items:
                existing = self._items[name]
                raise ValueError(
                    f"'{name}' already registered by {existing.__module__}.{existing.__name__}"
                )

            self._items[name] = item_cls
            logger.debug(
                "item_registered",
                registry=self._registry_name,
                name=name,
                cls=f"{item_cls.__module__}.{item_cls.__name__}",
            )

    def unregister(self, name: str) -> bool:
        """Unregister an item by name.

        Args:
            name: Name of the item to unregister.

        Returns:
            True if item was unregistered, False if not found.

        """
        with self._lock:
            if name in self._items:
                del self._items[name]
                logger.debug(
                    "item_unregistered",
                    registry=self._registry_name,
                    name=name,
                )
                return True
            return False

    def get(self, name: str) -> type[T] | None:
        """Get a registered item class by name.

        Args:
            name: Name of the item to retrieve.

        Returns:
            The registered class, or None if not found.

        """
        with self._lock:
            return self._items.get(name)

    def get_or_raise(self, name: str) -> type[T]:
        """Get a registered item class or raise an error.

        Args:
            name: Name of the item to retrieve.

        Returns:
            The registered class.

        Raises:
            KeyError: If the item is not registered.

        """
        item = self.get(name)
        if item is None:
            available = self.list_items()
            raise KeyError(
                f"'{name}' not registered in {self._registry_name}. Available: {available}"
            )
        return item

    def get_instance(self, name: str, *args: Any, **kwargs: Any) -> T | None:
        """Get an instantiated item by name.

        Args:
            name: Name of the item to retrieve.
            *args: Positional arguments for instantiation.
            **kwargs: Keyword arguments for instantiation.

        Returns:
            Instantiated item, or None if not found.

        """
        item_cls = self.get(name)
        if item_cls is None:
            return None
        return item_cls(*args, **kwargs)

    def list_items(self) -> list[str]:
        """List all registered item names.

        Returns:
            Sorted list of registered names.

        """
        with self._lock:
            return sorted(self._items.keys())

    def get_all(self) -> dict[str, type[T]]:
        """Get all registered items.

        Returns:
            Copy of the internal registry dictionary.

        """
        with self._lock:
            return dict(self._items)

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: Name to check.

        Returns:
            True if registered, False otherwise.

        """
        with self._lock:
            return name in self._items

    def clear(self) -> None:
        """Clear all registrations.

        Warning: This should only be used in tests.
        """
        with self._lock:
            count = len(self._items)
            self._items.clear()
            logger.warning(
                "registry_cleared",
                registry=self._registry_name,
                count=count,
                message="Registry cleared - this should only happen in tests",
            )

    def __len__(self) -> int:
        """Return number of registered items."""
        with self._lock:
            return len(self._items)

    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return self.is_registered(name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        return iter(self.list_items())


def create_registry(
    name: str,
    base_class: type[T],
) -> tuple[type[BaseRegistry[T]], Callable[[str], Callable[[type[T]], type[T]]]]:
    """Factory function to create a module-specific registry.

    Args:
        name: Name for the registry (used in logs and errors).
        base_class: Base class that all registered items must inherit from.

    Returns:
        Tuple of (RegistryClass, register_decorator).

    Example:
        from src.templates.registry import create_registry

        class BaseProcessor:
            def process(self, data): ...

        ProcessorRegistry, register_processor = create_registry(
            "Processor", BaseProcessor
        )

        @register_processor("fast")
        class FastProcessor(BaseProcessor):
            def process(self, data):
                return data  # Fast but simple

        @register_processor("accurate")
        class AccurateProcessor(BaseProcessor):
            def process(self, data):
                return complex_processing(data)

        # Usage
        processor_cls = ProcessorRegistry().get("fast")
        processor = processor_cls()

    """
    # Create the registry class dynamically
    class_name = f"{name}Registry"

    registry_cls = type(
        class_name,
        (BaseRegistry,),
        {"_registry_name": name},
    )

    # Create the decorator function
    def register_decorator(
        item_name: str,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register an item class.

        Args:
            item_name: Name to register the class under.

        Returns:
            Decorator function.

        """

        def decorator(cls: type[T]) -> type[T]:
            # Validate inheritance
            if not issubclass(cls, base_class):
                raise TypeError(f"Class {cls.__name__} must inherit from {base_class.__name__}")

            # Register the class
            registry_cls().register(item_name, cls)

            # Add registry name as class attribute via setattr to avoid attr-defined error
            # on arbitrary type[T] where T may not declare _registry_name.
            cls._registry_name = item_name

            return cls

        return decorator

    return registry_cls, register_decorator


def create_typed_registry(
    name: str,
) -> tuple[type[BaseRegistry[Any]], Callable[[str], Callable[[type], type]]]:
    """Create a registry without base class constraint.

    Useful when the base class is defined after the registry, or when
    registering heterogeneous types.

    Args:
        name: Name for the registry.

    Returns:
        Tuple of (RegistryClass, register_decorator).

    """
    class_name = f"{name}Registry"

    registry_cls = type(
        class_name,
        (BaseRegistry,),
        {"_registry_name": name},
    )

    def register_decorator(item_name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            registry_cls().register(item_name, cls)
            cls._registry_name = item_name
            return cls

        return decorator

    return registry_cls, register_decorator
