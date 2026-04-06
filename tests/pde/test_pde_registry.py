"""Tests for PDE operator registry."""

from __future__ import annotations

import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import (
    AdvectionDiffusionOperator,
    BurgersOperator,
    HeatOperator,
    LShapedPoissonOperator,
    NavierStokesOperator,
    PoissonOperator,
)
from src.pde.registry import (
    PDEOperatorRegistry,
    get_pde_operator,
    list_pde_operators,
    register_pde_operator,
)


class TestPDEOperatorRegistry:
    """Tests for the PDE operator registry."""

    def test_singleton(self) -> None:
        r1 = PDEOperatorRegistry()
        r2 = PDEOperatorRegistry()
        assert r1 is r2

    def test_builtin_operators_registered(self) -> None:
        registry = PDEOperatorRegistry()
        assert registry.is_registered("poisson")
        assert registry.is_registered("burgers")
        assert registry.is_registered("advection_diffusion")
        assert registry.is_registered("heat")
        assert registry.is_registered("navier_stokes")
        assert registry.is_registered("poisson_lshaped")

    def test_get_poisson(self) -> None:
        cls = PDEOperatorRegistry().get("poisson")
        assert cls is PoissonOperator

    def test_get_burgers(self) -> None:
        cls = PDEOperatorRegistry().get("burgers")
        assert cls is BurgersOperator

    def test_get_advection_diffusion(self) -> None:
        cls = PDEOperatorRegistry().get("advection_diffusion")
        assert cls is AdvectionDiffusionOperator

    def test_get_heat(self) -> None:
        cls = PDEOperatorRegistry().get("heat")
        assert cls is HeatOperator

    def test_get_navier_stokes(self) -> None:
        cls = PDEOperatorRegistry().get("navier_stokes")
        assert cls is NavierStokesOperator

    def test_get_lshaped_poisson(self) -> None:
        cls = PDEOperatorRegistry().get("poisson_lshaped")
        assert cls is LShapedPoissonOperator

    def test_get_nonexistent_returns_none(self) -> None:
        cls = PDEOperatorRegistry().get("nonexistent_pde_xyz")
        assert cls is None

    def test_list_items(self) -> None:
        items = PDEOperatorRegistry().list_items()
        assert isinstance(items, list)
        assert "poisson" in items
        assert "burgers" in items
        assert "heat" in items
        assert "advection_diffusion" in items
        assert "navier_stokes" in items
        assert "poisson_lshaped" in items
        assert len(items) >= 6

    def test_list_items_sorted(self) -> None:
        items = PDEOperatorRegistry().list_items()
        assert items == sorted(items)

    def test_is_registered_true(self) -> None:
        registry = PDEOperatorRegistry()
        assert registry.is_registered("poisson") is True

    def test_is_registered_false(self) -> None:
        registry = PDEOperatorRegistry()
        assert registry.is_registered("nonexistent_xyz") is False

    def test_get_all(self) -> None:
        registry = PDEOperatorRegistry()
        all_ops = registry.get_all()
        assert isinstance(all_ops, dict)
        assert "poisson" in all_ops
        assert all_ops["poisson"] is PoissonOperator


class TestGetPDEOperator:
    """Tests for get_pde_operator convenience function."""

    def test_get_existing(self) -> None:
        cls = get_pde_operator("poisson")
        assert cls is PoissonOperator

    def test_get_burgers(self) -> None:
        cls = get_pde_operator("burgers")
        assert cls is BurgersOperator

    def test_get_nonexistent_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            get_pde_operator("nonexistent_pde_abc")

    def test_error_message_contains_available(self) -> None:
        with pytest.raises(KeyError, match="poisson"):
            get_pde_operator("does_not_exist")

    def test_instantiate_from_registry(self) -> None:
        cls = get_pde_operator("poisson")
        config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        operator = cls(config)
        assert operator is not None
        assert isinstance(operator, PoissonOperator)

    def test_instantiate_burgers_from_registry(self) -> None:
        cls = get_pde_operator("burgers")
        config = PDEConfig(
            name="test",
            pde_type=PDEType.BURGERS,
            is_time_dependent=True,
        )
        operator = cls(config)
        assert isinstance(operator, BurgersOperator)


class TestListPDEOperators:
    """Tests for list_pde_operators function."""

    def test_returns_list(self) -> None:
        ops = list_pde_operators()
        assert isinstance(ops, list)
        assert len(ops) >= 6

    def test_contains_builtins(self) -> None:
        ops = list_pde_operators()
        for name in [
            "poisson",
            "burgers",
            "heat",
            "advection_diffusion",
            "navier_stokes",
            "poisson_lshaped",
        ]:
            assert name in ops

    def test_returns_sorted(self) -> None:
        ops = list_pde_operators()
        assert ops == sorted(ops)


class TestRegisterCustomOperator:
    """Tests for registering custom operators."""

    def test_register_custom(self) -> None:
        @register_pde_operator("test_custom_pde_op")
        class CustomOp(PoissonOperator):
            pass

        registry = PDEOperatorRegistry()
        assert registry.is_registered("test_custom_pde_op")
        cls = registry.get("test_custom_pde_op")
        assert cls is CustomOp

    def test_register_duplicate_raises(self) -> None:
        """Registering the same name twice should raise ValueError."""
        # "poisson" is already registered
        with pytest.raises(ValueError, match="already registered"):
            @register_pde_operator("poisson")
            class DuplicateOp(PoissonOperator):
                pass

    def test_registered_custom_in_list(self) -> None:
        # test_custom_pde_op was registered in test_register_custom
        ops = list_pde_operators()
        assert "test_custom_pde_op" in ops
