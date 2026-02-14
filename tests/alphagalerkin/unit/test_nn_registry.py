"""Tests for architecture registry (nn/registry.py)."""

from __future__ import annotations

import pytest

from src.alphagalerkin.nn import registry as reg_module
from src.alphagalerkin.nn.registry import (
    get_architecture,
    list_architectures,
    register_architecture,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:  # type: ignore[misc]
    """Save and restore the global registry around each test."""
    original = dict(reg_module._ARCHITECTURE_REGISTRY)
    yield  # type: ignore[misc]
    reg_module._ARCHITECTURE_REGISTRY.clear()
    reg_module._ARCHITECTURE_REGISTRY.update(original)


class TestRegisterArchitecture:
    """register_architecture decorator."""

    def test_register_new_class(self) -> None:
        @register_architecture("test_arch")
        class TestArch:
            pass

        assert "test_arch" in list_architectures()

    def test_duplicate_registration_raises(self) -> None:
        @register_architecture("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_architecture("dup")
            class Second:
                pass

    def test_decorator_returns_class_unchanged(self) -> None:
        @register_architecture("orig")
        class MyArch:
            sentinel = 42

        assert MyArch.sentinel == 42


class TestGetArchitecture:
    """get_architecture instantiates registered classes."""

    def test_get_registered(self) -> None:
        @register_architecture("simple")
        class Simple:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        instance = get_architecture("simple", x=1)

        assert isinstance(instance, Simple)
        assert instance.kwargs == {"x": 1}

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_architecture("does_not_exist")

    def test_error_message_lists_available(self) -> None:
        @register_architecture("aaa")
        class A:
            pass

        @register_architecture("bbb")
        class B:
            pass

        with pytest.raises(KeyError, match="aaa") as exc_info:
            get_architecture("missing")

        assert "bbb" in str(exc_info.value)


class TestListArchitectures:
    """list_architectures returns sorted names."""

    def test_empty_after_clear(self) -> None:
        reg_module._ARCHITECTURE_REGISTRY.clear()

        assert list_architectures() == []

    def test_sorted_order(self) -> None:
        @register_architecture("z_arch")
        class Z:
            pass

        @register_architecture("a_arch")
        class A:
            pass

        names = list_architectures()

        # Must be sorted; a_arch before z_arch.
        assert names.index("a_arch") < names.index("z_arch")
