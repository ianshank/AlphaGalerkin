"""Base utilities shared across physics modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.alphagalerkin.core.types import PDEType


@dataclass
class ManufacturedSolution:
    """A manufactured solution for MMS testing.

    Provides exact solution, forcing term, and boundary data
    for verifying PDE solver convergence.
    """

    exact_solution: Callable[[np.ndarray], np.ndarray]
    forcing: Callable[[np.ndarray], np.ndarray]
    boundary_data: Callable[[np.ndarray], np.ndarray]
    expected_convergence_order: float = 2.0
    name: str = "unnamed"

    def compute_error(
        self,
        numerical: np.ndarray,
        points: np.ndarray,
    ) -> float:
        """Compute L2 error between numerical and exact."""
        exact = self.exact_solution(points)
        return float(np.sqrt(np.mean((numerical - exact) ** 2)))


@dataclass
class SolveResult:
    """Result from solving a PDE on a discretization."""

    solution: np.ndarray
    residual_norm: float
    condition_number: float = 1.0
    solve_time_ms: float = 0.0
    converged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""

    bc_type: str  # "dirichlet", "neumann", "robin"
    value: Callable[[np.ndarray], np.ndarray] | float
    region: str = "all"  # "all", "left", "right", "top", "bottom"


class PhysicsModuleBase:
    """Base class providing default placeholder implementations for physics modules.

    Subclasses must set ``name`` and ``pde_type`` class attributes,
    and should override ``manufactured_solution()`` and ``boundary_conditions()``
    for each specific PDE.
    """

    name: str = "unnamed"
    pde_type: PDEType = PDEType.ELLIPTIC

    def reward_function(
        self,
        _state: Any,
        _action: Any,
        _next_state: Any,
    ) -> float:
        """Reward based on residual reduction and DOF efficiency."""
        return 0.0  # Placeholder for RL integration

    def state_features(self, _discretization: Any) -> Any:
        """Per-element features for the GNN encoder."""
        return None  # Placeholder for GNN integration

    def action_validators(self) -> list[Any]:
        """PDE-specific action validators (override in subclasses)."""
        return []
