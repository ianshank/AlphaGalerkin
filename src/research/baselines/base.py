"""Abstract solver protocol for classical PDE baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.pde.operators import PDEOperator
from src.research.baselines.predicates import nodal_rms_l2_error
from src.research.baselines.schemas import SolverResult


class BaseSolver(ABC):
    """Protocol for classical PDE solvers.

    All baselines implement this interface so that the benchmark runner
    can treat them uniformly.
    """

    name: str = "abstract"
    description: str = "Abstract base solver"

    @abstractmethod
    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve the PDE problem.

        Args:
            operator: PDE operator defining the problem.
            n_dof: Target degrees of freedom.
            **kwargs: Solver-specific options.

        Returns:
            SolverResult with solution, timing, and error metrics.

        """
        ...

    def _compute_l2_error(
        self,
        solution: NDArray[np.float64],
        coords: NDArray[np.float64],
        operator: PDEOperator,
    ) -> float | None:
        """Compute L2 error against the exact solution, if available.

        Thin delegate to the module-level :func:`nodal_rms_l2_error` (same
        split as ``_dorfler_mark`` -> ``dorfler_mark``), so a caller outside
        this class hierarchy -- a ``RefinementSubstrate``, say -- reuses the
        formula instead of copying it. A copy was briefly made in
        ``SkfemTriSubstrate`` and immediately diverged: it dropped the
        ``exact is None`` guard below, which turns every operator without an
        analytic solution into a ``TypeError`` from inside ``np.asarray``.
        """
        return nodal_rms_l2_error(solution, coords, operator)
