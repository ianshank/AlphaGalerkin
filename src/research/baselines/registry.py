"""Solver name → class registry. Extra solvers register into SOLVER_REGISTRY."""

from __future__ import annotations

from typing import Any

from src.research.baselines.amr import DorflerAMRSolver
from src.research.baselines.base import BaseSolver
from src.research.baselines.fdm import UniformFDMSolver
from src.research.baselines.navier_stokes import NavierStokesFDMSolver
from src.research.baselines.pinn import SimplePINNSolver

SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {
    "uniform_fdm": UniformFDMSolver,
    "dorfler_amr": DorflerAMRSolver,
    "pinn": SimplePINNSolver,
    "navier_stokes_fdm": NavierStokesFDMSolver,
}


def get_solver(name: str, **kwargs: Any) -> BaseSolver:
    """Get a solver instance by name.

    Args:
        name: Registered solver name.
        **kwargs: Solver-specific constructor arguments.

    Returns:
        Instantiated solver.

    Raises:
        KeyError: If solver name is not registered.

    """
    cls = SOLVER_REGISTRY.get(name)
    if cls is None:
        available = sorted(SOLVER_REGISTRY.keys())
        raise KeyError(f"Unknown solver '{name}'. Available: {available}")
    return cls(**kwargs)


def list_solvers() -> list[str]:
    """List all registered solver names."""
    return sorted(SOLVER_REGISTRY.keys())
