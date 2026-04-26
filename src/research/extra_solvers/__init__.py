"""Additional baseline solvers used by SBIR proposal benchmarks.

This package complements :mod:`src.research.baselines` by adding the
solvers required by ``config/proposals/*.yaml`` that were not part of
the original baseline set.  Concretely:

* :mod:`.supg_fem` — :class:`SUPGFEMSolver` for advection-dominated
  problems (Streamline-Upwind / Petrov-Galerkin), required by
  ``doe_ascr_c59.yaml::advdiff_boundary_layer``.
* :mod:`.multigrid` — :class:`MultigridPoissonSolver` (PyAMG-backed)
  and :class:`DirectPoissonSolver` (scipy.sparse.linalg.spsolve) for
  weak-scaling studies (``doe_ascr_c59.yaml::poisson_scaling``).
* :mod:`.neural_op` — :class:`FNOBaselineSolver` and
  :class:`DeepONetBaselineSolver` for spectral-bias and operator
  learning comparisons (``nsf_sbir.yaml::spectral_bias_comparison``,
  ``doe_ascr_c59.yaml::poisson_scaling``).

Every solver is **opt-in via Python imports**: importing this package
triggers registration into :data:`src.research.baselines.SOLVER_REGISTRY`
so that ``get_solver(name)`` resolves them.  Solvers whose external
dependency (``pyamg``, ``neuraloperator``) is missing register a
*stub* solver that raises a clear :class:`ImportError` at construction
time.  This keeps the global 85% coverage gate green even on machines
without the extras installed.

Configuration values (Peclet thresholds, stabilization parameters,
FNO modes, DeepONet branch/trunk widths, etc.) are surfaced through
Pydantic config models — never hardcoded.
"""

from __future__ import annotations

# Import each solver module so its module-level registration runs.
# Failures (missing optional deps) surface a stub that raises a clear
# error at construction time; we never silently swallow them.
from src.research.extra_solvers import (  # noqa: F401
    multigrid,
    neural_op,
    supg_fem,
)

__all__ = [
    "multigrid",
    "neural_op",
    "supg_fem",
]
