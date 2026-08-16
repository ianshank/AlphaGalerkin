"""Convergence gate for the L-shaped Poisson AMR substrate.

Before *any* refinement-policy comparison is meaningful, the shared
discretisation must actually converge. These tests assert that directly, and
would have caught the reentrant-edge boundary-condition defect that made
``lshape_inside_predicate`` remove the *open* rather than the *closed* fourth
quadrant (see that function's docstring).

Under the defect the L2 error **grew** with DOF under uniform refinement --
5.0e-2 at 65 DOF rising to 1.15e-1 at 12545 DOF -- because the two reentrant
edges never received their ``u = 0`` Dirichlet condition. Every policy metric
built on that substrate compared two arms on a problem neither was solving.

The gate is deliberately cheap (grids up to 64 per side) so it can run on every
CI pass rather than as a manual sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.operators import LShapedPoissonOperator
from src.research.lshape_amr_compare import (
    ComparisonParams,
    lshape_inside_predicate,
    make_solve_fn,
    run_dorfler_arm,
)

pytest.importorskip("scipy", reason="scipy required for the masked FD solve")

SCALE = 1.0

# Uniform-refinement grid sizes (nodes per side). Even, so x=0 and y=0 are grid
# lines and the reentrant edges are resolved exactly.
UNIFORM_SIDES = (8, 16, 32, 64)

# Theoretical L2 rate for u = r^(2/3) sin(2*theta/3) on the L-shape under
# uniform refinement is O(h^(4/3)) ~ 1.333. The measured rate approaches it from
# below (1.19 -> 1.31 over the sweep), so the band is generous on the low side
# and tight enough to fail the divergent (negative-rate) substrate outright.
MIN_L2_RATE_VS_H = 1.0
MAX_L2_RATE_VS_H = 1.6


def _operator() -> LShapedPoissonOperator:
    return LShapedPoissonOperator(
        PDEConfig(
            name="lshape_convergence_gate",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-SCALE, -SCALE],
            domain_max=[SCALE, SCALE],
            advection_coeff=[0.0, 0.0],
            geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=SCALE),
        )
    )


def _uniform_sweep() -> tuple[list[int], list[float], list[float]]:
    """Solve on uniform grids; return (dofs, h values, L2 errors)."""
    solve = make_solve_fn(_operator(), lshape_inside_predicate(SCALE))
    dofs: list[int] = []
    hs: list[float] = []
    errors: list[float] = []
    for n in UNIFORM_SIDES:
        xs = np.linspace(-SCALE, SCALE, n + 1, dtype=np.float64)
        ys = np.linspace(-SCALE, SCALE, n + 1, dtype=np.float64)
        res = solve(xs, ys)
        dofs.append(res.n_dof)
        hs.append(2.0 * SCALE / n)
        errors.append(res.l2_error)
    return dofs, hs, errors


class TestUniformRefinementConverges:
    """The substrate must converge with no marking policy involved at all."""

    def test_error_decreases_monotonically_with_dof(self) -> None:
        dofs, _, errors = _uniform_sweep()
        assert dofs == sorted(dofs), "DOF must increase across the sweep"
        for (d_prev, e_prev), (d_next, e_next) in zip(
            zip(dofs, errors), zip(dofs[1:], errors[1:]), strict=False
        ):
            assert e_next < e_prev, (
                f"L2 error rose from {e_prev:.6e} at {d_prev} DOF to "
                f"{e_next:.6e} at {d_next} DOF -- the discretisation is not "
                "converging, so no refinement-policy comparison built on it is "
                "meaningful."
            )

    def test_observed_l2_rate_matches_theory(self) -> None:
        """Finest-pair rate must sit in the O(h^(4/3)) band, not merely be positive."""
        _, hs, errors = _uniform_sweep()
        rate = np.log(errors[-2] / errors[-1]) / np.log(hs[-2] / hs[-1])
        assert MIN_L2_RATE_VS_H <= rate <= MAX_L2_RATE_VS_H, (
            f"observed L2 convergence rate {rate:.3f} outside "
            f"[{MIN_L2_RATE_VS_H}, {MAX_L2_RATE_VS_H}]; theory for the "
            "r^(2/3) reentrant-corner singularity is O(h^(4/3))."
        )

    def test_finest_grid_error_is_small(self) -> None:
        """Absolute anchor: pins the fix's magnitude, not just its direction."""
        _, _, errors = _uniform_sweep()
        assert errors[-1] < 1e-3, (
            f"L2 error {errors[-1]:.6e} on the finest uniform grid; the "
            "converging substrate reaches ~6.4e-4 there."
        )


class TestReentrantEdgesArePinned:
    """The specific defect: slit-edge nodes must be Dirichlet, not unknowns."""

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (0.5, 0.0),  # {y=0, x>0} reentrant edge
            (0.0, -0.5),  # {x=0, y<0} reentrant edge
            (0.0, 0.0),  # reentrant corner itself
        ],
    )
    def test_reentrant_edge_excluded_from_unknowns(self, x: float, y: float) -> None:
        inside = lshape_inside_predicate(SCALE)
        pt = np.array([[x, y]], dtype=np.float64)
        assert not bool(inside(pt)[0]), (
            f"({x}, {y}) lies on the L-shape boundary where u=0; treating it as "
            "an interior unknown leaves its Dirichlet condition unimposed."
        )

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (0.5, 0.5),  # first quadrant
            (-0.5, 0.5),  # second quadrant
            (-0.5, -0.5),  # third quadrant
            (-0.5, 0.0),  # y=0 on the *non*-slit side, still interior
        ],
    )
    def test_genuine_interior_retained(self, x: float, y: float) -> None:
        inside = lshape_inside_predicate(SCALE)
        pt = np.array([[x, y]], dtype=np.float64)
        assert bool(inside(pt)[0]), f"({x}, {y}) is a genuine interior unknown"


class TestDorflerArmConverges:
    """The classical arm must reproduce the 40-year-old result on its own.

    This is the gate the comparison harness actually rides on: if adaptive
    Dorfler marking cannot drive the error down with DOF, an MCTS-vs-Dorfler
    ratio measures nothing regardless of which arm wins.
    """

    def test_dorfler_error_drops_with_dof(self) -> None:
        operator = _operator()
        solve = make_solve_fn(operator, lshape_inside_predicate(SCALE))
        params = ComparisonParams(seed=23799, scale=SCALE, max_dof=1200, max_steps=12)

        traj = run_dorfler_arm(operator, solve, params)
        errors = traj.errors()
        dofs = traj.dofs()

        assert len(errors) >= 4, "need several refinement levels to judge a trend"
        assert errors[-1] < errors[0], (
            f"Dorfler L2 error did not improve: {errors[0]:.6e} at {dofs[0]:.0f} "
            f"DOF -> {errors[-1]:.6e} at {dofs[-1]:.0f} DOF."
        )
        # A 10x DOF increase that buys under 2x error reduction indicates the
        # refinement is not reaching the singularity.
        assert errors[0] / errors[-1] > 2.0, (
            f"Dorfler error reduction {errors[0] / errors[-1]:.2f}x over "
            f"{dofs[-1] / dofs[0]:.1f}x DOF is too weak to support a policy comparison."
        )
