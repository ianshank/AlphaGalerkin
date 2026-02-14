"""Tests for physics registry."""
from __future__ import annotations

import pytest

import src.alphagalerkin.physics.poisson  # noqa: F401 trigger registration
from src.alphagalerkin.physics.registry import PhysicsRegistry


class TestPhysicsRegistry:
    """Tests for the singleton PhysicsRegistry."""

    def setup_method(self) -> None:
        """Clear cached instances between tests."""
        registry = PhysicsRegistry()
        registry.clear_instances()

    def test_builtin_poisson_registered(self) -> None:
        """The poisson_2d module should be auto-registered."""
        registry = PhysicsRegistry()
        registered = registry.list_modules()
        assert "poisson_2d" in registered

    def test_get_unregistered_raises(self) -> None:
        registry = PhysicsRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_physics")

    def test_registered_module_has_required_methods(
        self,
    ) -> None:
        registry = PhysicsRegistry()
        module = registry.get("poisson_2d")
        assert hasattr(module, "weak_form")
        assert hasattr(module, "boundary_conditions")
        assert hasattr(module, "manufactured_solution")
        assert hasattr(module, "reward_function")
        assert hasattr(module, "name")

    def test_get_returns_same_instance(self) -> None:
        """Registry caches instances (singleton per module)."""
        registry = PhysicsRegistry()
        m1 = registry.get("poisson_2d")
        m2 = registry.get("poisson_2d")
        assert m1 is m2

    def test_clear_instances_forces_recreation(self) -> None:
        registry = PhysicsRegistry()
        m1 = registry.get("poisson_2d")
        registry.clear_instances()
        m2 = registry.get("poisson_2d")
        assert m1 is not m2

    def test_list_modules_returns_sorted(self) -> None:
        registry = PhysicsRegistry()
        modules = registry.list_modules()
        assert modules == sorted(modules)

    def test_register_duplicate_name_raises(self) -> None:
        registry = PhysicsRegistry()
        if "poisson_2d" in registry.list_modules():
            with pytest.raises(
                ValueError, match="already registered"
            ):
                registry.register(
                    "poisson_2d", type("Dummy", (), {}),
                )
