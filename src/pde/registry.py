"""PDE Operator registry for AlphaGalerkin.

This module provides registration and discovery for PDE operators,
enabling dynamic loading and configuration of different PDE types.

Usage:
    from src.pde.registry import PDEOperatorRegistry, register_pde_operator

    @register_pde_operator("custom_pde")
    class CustomPDEOperator(PDEOperator):
        ...

    # Retrieve registered operator
    operator_cls = PDEOperatorRegistry().get("custom_pde")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.pde.operators import (
    AdvectionDiffusionOperator,
    BurgersOperator,
    HeatOperator,
    LShapedPoissonOperator,
    NavierStokesOperator,
    PDEOperator,
    PoissonOperator,
)
from src.templates.registry import create_registry

if TYPE_CHECKING:
    pass

# Create the registry and decorator
PDEOperatorRegistry, register_pde_operator = create_registry("PDEOperator", PDEOperator)


def _register_builtin_operators() -> None:
    """Register built-in PDE operators."""
    registry = PDEOperatorRegistry()

    # Only register if not already registered
    if not registry.is_registered("poisson"):
        register_pde_operator("poisson")(PoissonOperator)

    if not registry.is_registered("burgers"):
        register_pde_operator("burgers")(BurgersOperator)

    if not registry.is_registered("advection_diffusion"):
        register_pde_operator("advection_diffusion")(AdvectionDiffusionOperator)

    if not registry.is_registered("heat"):
        register_pde_operator("heat")(HeatOperator)

    if not registry.is_registered("navier_stokes"):
        register_pde_operator("navier_stokes")(NavierStokesOperator)

    if not registry.is_registered("poisson_lshaped"):
        register_pde_operator("poisson_lshaped")(LShapedPoissonOperator)


# Register built-in operators on import
_register_builtin_operators()


def get_pde_operator(name: str) -> type[PDEOperator]:
    """Get a PDE operator class by name.

    Args:
        name: Registered operator name.

    Returns:
        PDE operator class.

    Raises:
        KeyError: If operator not registered.

    """
    return PDEOperatorRegistry().get_or_raise(name)


def list_pde_operators() -> list[str]:
    """List all registered PDE operators.

    Returns:
        List of registered operator names.

    """
    return PDEOperatorRegistry().list_items()
