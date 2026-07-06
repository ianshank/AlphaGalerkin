"""Backwards-compatibility + masking tests for the geometry-aware AMR solve.

The ``inside`` predicate added to ``DorflerAMRSolver._solve_on_grid_2d`` and
``_compute_indicators_2d`` is *optional*. This module is the load-bearing
regression guard that:

* ``inside=None`` is byte-for-byte identical to supplying an all-True predicate
  (the historical full-box behaviour is unchanged);
* out-of-domain interior nodes are pinned to their Dirichlet boundary value;
* out-of-domain elements receive a zero indicator (never marked).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.operators import LShapedPoissonOperator
from src.research.baselines import DorflerAMRSolver
from src.research.lshape_amr_compare import lshape_inside_predicate

scipy = pytest.importorskip("scipy", reason="scipy required for masked FD solve")

from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402


def _operator() -> LShapedPoissonOperator:
    cfg = PDEConfig(
        name="l",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-1.0, -1.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
        geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=1.0),
    )
    return LShapedPoissonOperator(cfg)


def _all_true(points: np.ndarray) -> np.ndarray:
    return np.ones(points.shape[0], dtype=bool)


class TestBackwardsCompat:
    """inside=None must equal an all-True predicate, byte-for-byte."""

    def test_solve_none_equals_all_true(self) -> None:
        op = _operator()
        xs = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 7, dtype=np.float64)

        u_none, grid_none = DorflerAMRSolver._solve_on_grid_2d(
            xs, ys, op, sparse, spsolve, inside=None
        )
        u_true, grid_true = DorflerAMRSolver._solve_on_grid_2d(
            xs, ys, op, sparse, spsolve, inside=_all_true
        )

        np.testing.assert_array_equal(u_none, u_true)
        np.testing.assert_array_equal(grid_none, grid_true)

    def test_indicators_none_equals_all_true(self) -> None:
        op = _operator()
        xs = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 7, dtype=np.float64)

        u_none, _ = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve, inside=None)
        ind_none = DorflerAMRSolver._compute_indicators_2d(xs, ys, u_none, op, inside=None)
        ind_true = DorflerAMRSolver._compute_indicators_2d(xs, ys, u_none, op, inside=_all_true)

        np.testing.assert_array_equal(ind_none, ind_true)


class TestMaskedNodes:
    """Out-of-domain interior nodes are pinned to the boundary/exact value."""

    def test_masked_nodes_equal_exact(self) -> None:
        op = _operator()
        inside = lshape_inside_predicate(1.0)
        xs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 9, dtype=np.float64)

        u_full, grid = DorflerAMRSolver._solve_on_grid_2d(
            xs, ys, op, sparse, spsolve, inside=inside
        )
        out_mask = ~np.asarray(inside(grid), dtype=bool)
        assert out_mask.any(), "the notch must remove at least one node"

        exact = np.asarray(op.exact_solution(grid.astype(np.float32)), dtype=np.float64).ravel()
        # Pinned identity rows / boundary fill => out-of-domain nodes carry the
        # exact Dirichlet value to machine precision.
        np.testing.assert_allclose(u_full[out_mask], exact[out_mask], atol=1e-9)

    def test_masked_solution_differs_from_full_box(self) -> None:
        """The mask actually changes the interior solve (not a silent no-op)."""
        op = _operator()
        inside = lshape_inside_predicate(1.0)
        xs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 9, dtype=np.float64)

        u_box, _ = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve, inside=None)
        u_masked, _ = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve, inside=inside)
        assert not np.allclose(u_box, u_masked)


class TestMaskedElements:
    """Out-of-domain elements receive a zero indicator (never marked)."""

    def test_out_of_domain_elements_zero(self) -> None:
        op = _operator()
        inside = lshape_inside_predicate(1.0)
        xs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 9, dtype=np.float64)

        u_full, _ = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve, inside=inside)
        ind = DorflerAMRSolver._compute_indicators_2d(xs, ys, u_full, op, inside=inside)

        nx, ny = len(xs) - 1, len(ys) - 1
        assert ind.shape == (nx, ny)
        n_zeroed = 0
        for i in range(nx):
            cx = 0.5 * (xs[i] + xs[i + 1])
            for j in range(ny):
                cy = 0.5 * (ys[j] + ys[j + 1])
                if cx > 0.0 and cy < 0.0:
                    assert ind[i, j] == 0.0
                    n_zeroed += 1
        assert n_zeroed > 0, "at least one element centre lies in the removed quadrant"

    def test_indicators_non_negative(self) -> None:
        op = _operator()
        inside = lshape_inside_predicate(1.0)
        xs = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        u_full, _ = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve, inside=inside)
        ind = DorflerAMRSolver._compute_indicators_2d(xs, ys, u_full, op, inside=inside)
        assert np.all(ind >= 0.0)
