"""Tests for the templates.registry module."""

from __future__ import annotations

import threading

import pytest

from src.templates.registry import BaseRegistry, create_typed_registry


class TestBaseRegistry:
    """Tests for BaseRegistry base class."""

    def test_subclass_isolation(self) -> None:
        """Test that subclasses have isolated state."""

        class RegistryA(BaseRegistry):
            _registry_name = "A"

        class RegistryB(BaseRegistry):
            _registry_name = "B"

        reg_a = RegistryA()
        reg_b = RegistryB()

        class ItemA:
            pass

        class ItemB:
            pass

        reg_a.register("item", ItemA)
        reg_b.register("item", ItemB)

        assert reg_a.get("item") is ItemA
        assert reg_b.get("item") is ItemB

        # Cleanup
        reg_a.clear()
        reg_b.clear()


class TestCreateRegistry:
    """Tests for create_registry factory."""

    def test_creates_registry_class(self, sample_registry) -> None:
        """Test that factory creates a registry class."""
        SampleRegistry, _ = sample_registry
        assert SampleRegistry is not None
        assert issubclass(SampleRegistry, BaseRegistry)

    def test_creates_decorator(self, sample_registry) -> None:
        """Test that factory creates a registration decorator."""
        _, register_sample = sample_registry
        assert callable(register_sample)

    def test_singleton_behavior(self, sample_registry) -> None:
        """Test registry is a singleton."""
        SampleRegistry, _ = sample_registry

        reg1 = SampleRegistry()
        reg2 = SampleRegistry()

        assert reg1 is reg2

    def test_register_and_get(self, sample_registry) -> None:
        """Test registration and retrieval."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class TestImpl(SampleBase):
            def process(self, data: str) -> str:
                return data.upper()

        registry = SampleRegistry()
        registry.register("test_impl", TestImpl)

        assert registry.get("test_impl") is TestImpl
        assert "test_impl" in registry.list_items()

    def test_register_with_decorator(self, sample_registry) -> None:
        """Test registration via decorator."""
        SampleRegistry, register_sample = sample_registry

        from tests.templates.conftest import SampleBase

        @register_sample("decorated")
        class DecoratedImpl(SampleBase):
            def process(self, data: str) -> str:
                return data.lower()

        assert SampleRegistry().get("decorated") is DecoratedImpl
        assert DecoratedImpl._registry_name == "decorated"

    def test_decorator_validates_inheritance(self, sample_registry) -> None:
        """Test that decorator validates base class inheritance."""
        _, register_sample = sample_registry

        with pytest.raises(TypeError, match="must inherit from"):

            @register_sample("invalid")
            class NotInherited:
                pass

    def test_duplicate_registration_fails(self, sample_registry) -> None:
        """Test that duplicate names raise errors."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl1(SampleBase):
            def process(self, data: str) -> str:
                return data

        class Impl2(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("duplicate", Impl1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("duplicate", Impl2)

    def test_empty_name_fails(self, sample_registry) -> None:
        """Test that empty names are rejected."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()

        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register("", Impl)

        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register("   ", Impl)

    def test_get_nonexistent_returns_none(self, sample_registry) -> None:
        """Test that getting nonexistent item returns None."""
        SampleRegistry, _ = sample_registry

        registry = SampleRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_raise(self, sample_registry) -> None:
        """Test get_or_raise behavior."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("exists", Impl)

        assert registry.get_or_raise("exists") is Impl

        with pytest.raises(KeyError, match="not registered"):
            registry.get_or_raise("nonexistent")

    def test_get_instance(self, sample_registry) -> None:
        """Test instantiation via registry."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def __init__(self, value: int = 0):
                self.value = value

            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("impl", Impl)

        instance = registry.get_instance("impl", value=42)
        assert instance is not None
        assert instance.value == 42

        assert registry.get_instance("nonexistent") is None

    def test_unregister(self, sample_registry) -> None:
        """Test unregistration."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("to_remove", Impl)

        assert registry.is_registered("to_remove")
        assert registry.unregister("to_remove")
        assert not registry.is_registered("to_remove")

        # Unregistering nonexistent returns False
        assert not registry.unregister("nonexistent")

    def test_list_items_sorted(self, sample_registry) -> None:
        """Test that list_items returns sorted names."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("zebra", Impl)
        registry.register("alpha", Impl)
        registry.register("middle", Impl)

        items = registry.list_items()
        assert items == ["alpha", "middle", "zebra"]

    def test_get_all_returns_copy(self, sample_registry) -> None:
        """Test that get_all returns a copy."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("item", Impl)

        all_items = registry.get_all()
        all_items["new_item"] = Impl  # Modify the copy

        # Original should be unchanged
        assert "new_item" not in registry.list_items()

    def test_len_and_contains(self, sample_registry) -> None:
        """Test __len__ and __contains__."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        assert len(registry) == 0
        assert "item" not in registry

        registry.register("item", Impl)
        assert len(registry) == 1
        assert "item" in registry

    def test_iter(self, sample_registry) -> None:
        """Test iteration over registry."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("a", Impl)
        registry.register("b", Impl)

        items = list(registry)
        assert items == ["a", "b"]

    def test_clear(self, sample_registry) -> None:
        """Test clearing the registry."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        class Impl(SampleBase):
            def process(self, data: str) -> str:
                return data

        registry = SampleRegistry()
        registry.register("item1", Impl)
        registry.register("item2", Impl)

        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0

    def test_thread_safety(self, sample_registry) -> None:
        """Test thread-safe operations."""
        SampleRegistry, _ = sample_registry

        from tests.templates.conftest import SampleBase

        registry = SampleRegistry()
        errors: list[Exception] = []

        def register_item(name: str) -> None:
            try:

                class Impl(SampleBase):
                    def process(self, data: str) -> str:
                        return data

                registry.register(name, Impl)
            except ValueError:
                # Expected for duplicate names
                pass
            except Exception as e:
                errors.append(e)

        # Create threads that try to register items
        threads = []
        for i in range(10):
            for j in range(5):
                t = threading.Thread(target=register_item, args=(f"item_{j}",))
                threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have no unexpected errors
        assert len(errors) == 0

        # Should have exactly 5 items registered
        assert len(registry) == 5


class TestCreateTypedRegistry:
    """Tests for create_typed_registry factory."""

    def test_creates_registry_without_base_class(self) -> None:
        """Test creating registry without base class constraint."""
        TypedRegistry, register_typed = create_typed_registry("Typed")

        @register_typed("string")
        class StringProcessor:
            def process(self, data: str) -> str:
                return data

        @register_typed("int")
        class IntProcessor:
            def process(self, data: int) -> int:
                return data

        registry = TypedRegistry()
        assert registry.get("string") is StringProcessor
        assert registry.get("int") is IntProcessor

        # Cleanup
        registry.clear()

    def test_no_inheritance_validation(self) -> None:
        """Test that typed registry doesn't validate inheritance."""
        TypedRegistry, register_typed = create_typed_registry("NoValidation")

        # Should not raise TypeError
        @register_typed("anything")
        class Anything:
            pass

        assert TypedRegistry().get("anything") is Anything

        # Cleanup
        TypedRegistry().clear()
