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
    BiharmonicOperator,
    BurgersOperator,
    HeatOperator,
    HelmholtzOperator,
    LShapedPoissonOperator,
    NavierStokesOperator,
    PDEOperator,
    PoissonOperator,
)
from src.pde.operators.multi_corner_poisson import (
    LShapeMultiCornerPoissonOperator,
    ZShapeMultiCornerPoissonOperator,
)
from src.pde.operators_picogk import (
    HelicalHeatOperator,
    HelicalMagnetostaticsOperator,
    HelicalStokesOperator,
)
from src.templates.registry import create_registry

if TYPE_CHECKING:
    pass

# Create the registry and decorator
PDEOperatorRegistry, register_pde_operator = create_registry("PDEOperator", PDEOperator)


#: Built-in operators by registry key, in registration order. Every caller
#: constructs a registered operator as ``cls(config)``, so every entry must build
#: from a config alone (``tests/pde/test_pde_registry_contract.py``).
_BUILTIN_OPERATORS: tuple[tuple[str, type[PDEOperator]], ...] = (
    ("poisson", PoissonOperator),
    ("burgers", BurgersOperator),
    ("advection_diffusion", AdvectionDiffusionOperator),
    ("heat", HeatOperator),
    ("navier_stokes", NavierStokesOperator),
    ("poisson_lshaped", LShapedPoissonOperator),
    # Polyomino presets of MultiCornerPoissonOperator (several reentrant corners).
    # The generic class also needs the ``domain`` and ``corners`` a PDEConfig
    # cannot carry, so it is not registered; each preset fixes both.
    (LShapeMultiCornerPoissonOperator.name, LShapeMultiCornerPoissonOperator),
    (ZShapeMultiCornerPoissonOperator.name, ZShapeMultiCornerPoissonOperator),
    # Out-of-distribution operators for held-out generalisation benchmarks.
    ("helmholtz", HelmholtzOperator),
    ("biharmonic", BiharmonicOperator),
    # Leap 71 / Noyron-targeted SDF-aware operators.
    ("helical_heat", HelicalHeatOperator),
    ("helical_stokes", HelicalStokesOperator),
    ("helical_magnetostatics", HelicalMagnetostaticsOperator),
)


def _register_builtin_operators() -> None:
    """Register each built-in operator under its key, unless the key is taken."""
    registry = PDEOperatorRegistry()
    for name, operator_cls in _BUILTIN_OPERATORS:
        if not registry.is_registered(name):
            register_pde_operator(name)(operator_cls)


# Register built-in operators on import
_register_builtin_operators()


def get_pde_operator(name: str) -> type[PDEOperator]:
    """Get a PDE operator class by name.

    Args:
    ----
        name: Registered operator name.

    Returns:
    -------
        PDE operator class.

    Raises:
    ------
        KeyError: If operator not registered.

    """
    return PDEOperatorRegistry().get_or_raise(name)


def list_pde_operators() -> list[str]:
    """List all registered PDE operators.

    Returns
    -------
        List of registered operator names.

    """
    return PDEOperatorRegistry().list_items()
