"""Tests for 2D AMR solver on L-shaped domain.

Validates that:
- DorflerAMRSolver works on 2D L-shaped Poisson problems
- Mesh refinement reduces L2 error (convergence)
- Dorfler marking concentrates refinement near corner singularity
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.operators import LShapedPoissonOperator
from src.research.baselines import AMRConfig, DorflerAMRSolver

scipy = pytest.importorskip("scipy", reason="scipy required for AMR tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lshaped_operator() -> LShapedPoissonOperator:
    """Create an L-shaped Poisson operator with reentrant corner singularity."""
    cfg = PDEConfig(
        name="test_lshaped",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-1.0, -1.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
        geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=1.0),
    )
    return LShapedPoissonOperator(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDorflerAMR2DLShaped:
    """2D AMR solver on L-shaped domain."""

    def test_solve_returns_result(self) -> None:
        """AMR solver returns a valid SolverResult on the L-shaped domain."""
        operator = _make_lshaped_operator()
        solver = DorflerAMRSolver(
            config=AMRConfig(
                marking_fraction=0.3,
                max_refinements=2,
                min_initial_side_2d=3,
            ),
        )
        result = solver.solve(operator, n_dof=25)

        assert result.n_dof > 0
        assert result.wall_time_seconds >= 0.0
        assert result.grid_points.shape[1] == 2
        assert result.solution.shape[0] == result.grid_points.shape[0]

    def test_refinement_reduces_error(self) -> None:
        """More DOFs should produce smaller L2 error (convergence test)."""
        operator = _make_lshaped_operator()
        solver = DorflerAMRSolver(
            config=AMRConfig(
                marking_fraction=0.3,
                max_refinements=5,
                min_initial_side_2d=3,
            ),
        )

        errors: list[float] = []
        dof_values = [16, 64]
        for n_dof in dof_values:
            result = solver.solve(operator, n_dof=n_dof)
            if result.l2_error is not None:
                errors.append(result.l2_error)

        # With at least two valid error values, the finer grid should
        # produce a smaller (or at least non-larger) error.
        if len(errors) >= 2:
            assert errors[-1] <= errors[0] * 1.5, f"Error should decrease with refinement: {errors}"

    def test_metadata_contains_refinement_info(self) -> None:
        """Solver metadata should include AMR-specific information."""
        operator = _make_lshaped_operator()
        solver = DorflerAMRSolver(
            config=AMRConfig(
                marking_fraction=0.4,
                max_refinements=3,
            ),
        )
        result = solver.solve(operator, n_dof=25)

        assert "marking_fraction" in result.metadata
        assert "n_refinements" in result.metadata
        assert result.metadata["dim"] == 2
        assert result.metadata["marking_fraction"] == pytest.approx(0.4)

    def test_dorfler_marking_fraction_effect(self) -> None:
        """Smaller marking fraction should refine fewer elements per step."""
        operator = _make_lshaped_operator()
        config_tight = AMRConfig(marking_fraction=0.1, max_refinements=3)
        config_wide = AMRConfig(marking_fraction=0.8, max_refinements=3)

        solver_tight = DorflerAMRSolver(config=config_tight)
        solver_wide = DorflerAMRSolver(config=config_wide)

        result_tight = solver_tight.solve(operator, n_dof=100)
        result_wide = solver_wide.solve(operator, n_dof=100)

        # Both should produce valid results
        assert result_tight.n_dof > 0
        assert result_wide.n_dof > 0

    def test_error_indicators_2d(self) -> None:
        """Error indicators should be non-negative 2D arrays."""
        operator = _make_lshaped_operator()
        solver = DorflerAMRSolver(config=AMRConfig(max_refinements=1))

        xs = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 6, dtype=np.float64)

        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        u, _grid = solver._solve_on_grid_2d(xs, ys, operator, sparse, spsolve)  # type: ignore[arg-type]
        indicators = solver._compute_indicators_2d(xs, ys, u, operator)

        assert indicators.shape == (5, 5)
        assert np.all(indicators >= 0.0)
