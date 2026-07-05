"""Tests for the L-shaped Poisson MCTS-vs-Dörfler comparison harness.

Covers the geometry predicate, the masked solve function, both arm runners,
the log-log interpolation helper, the ratio computation, a real end-to-end
micro-run, and the CSV/PNG artifact writers.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.operators import LShapedPoissonOperator
from src.research.lshape_amr_compare import (
    ArmTrajectory,
    ComparisonParams,
    TrajectoryPoint,
    _interp_log,
    compare_ratios,
    export_csv,
    export_plot,
    lshape_inside_predicate,
    make_solve_fn,
    run_comparison,
    run_dorfler_arm,
)

pytest.importorskip("scipy", reason="scipy required for the masked FD solve")


def _operator(scale: float = 1.0) -> LShapedPoissonOperator:
    cfg = PDEConfig(
        name="l",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-scale, -scale],
        domain_max=[scale, scale],
        advection_coeff=[0.0, 0.0],
        geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=scale),
    )
    return LShapedPoissonOperator(cfg)


def _game_config(scale: float = 1.0, **overrides: object) -> PDEGameConfig:
    pde_cfg = PDEConfig(
        name="l",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-scale, -scale],
        domain_max=[scale, scale],
        advection_coeff=[0.0, 0.0],
        geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=scale),
    )
    params: dict[str, object] = {
        "name": "g",
        "pde_config": pde_cfg,
        "game_mode": "mesh_refinement",
        "max_dof": 120,
        "max_steps": 6,
        "error_tolerance": 1e-6,
    }
    params.update(overrides)
    return PDEGameConfig(**params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Geometry predicate                                                           #
# --------------------------------------------------------------------------- #


class TestPredicate:
    def test_excludes_removed_quadrant(self) -> None:
        inside = lshape_inside_predicate(1.0)
        pts = np.array(
            [
                [0.5, -0.5],  # removed quadrant -> False
                [-0.5, -0.5],  # keep
                [0.5, 0.5],  # keep
                [-0.5, 0.5],  # keep
                [0.0, -0.5],  # on axis x=0, not strictly > 0 -> keep
            ],
            dtype=np.float64,
        )
        mask = inside(pts)
        np.testing.assert_array_equal(mask, [False, True, True, True, True])


# --------------------------------------------------------------------------- #
# Masked solve function                                                        #
# --------------------------------------------------------------------------- #


class TestMakeSolveFn:
    def test_finite_l2_and_reduced_dof(self) -> None:
        op = _operator()
        solve = make_solve_fn(op, lshape_inside_predicate(1.0))
        xs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
        result = solve(xs, ys)

        assert np.isfinite(result.l2_error)
        assert result.l2_error >= 0.0
        full_nodes = len(xs) * len(ys)
        # The notch removes interior nodes, so active DOF is strictly smaller.
        assert result.n_dof < full_nodes
        assert result.indicators.shape == (len(xs) - 1, len(ys) - 1)


# --------------------------------------------------------------------------- #
# Dörfler arm                                                                  #
# --------------------------------------------------------------------------- #


class TestDorflerArm:
    def test_trajectory_decreases(self) -> None:
        op = _operator()
        solve = make_solve_fn(op, lshape_inside_predicate(1.0))
        params = ComparisonParams(
            initial_side=4, max_dof=150, max_refinements=5, marking_fraction=0.5
        )
        traj = run_dorfler_arm(op, solve, params)
        assert traj.method == "dorfler"
        assert len(traj.points) >= 2
        # Refinement should not increase the error overall (allow small slack).
        assert traj.points[-1].l2_error <= traj.points[0].l2_error * 1.5
        assert traj.points[-1].n_dof >= traj.points[0].n_dof
        assert np.isfinite(traj.convergence_exponent())


# --------------------------------------------------------------------------- #
# _interp_log                                                                  #
# --------------------------------------------------------------------------- #


class TestInterpLog:
    def test_single_point(self) -> None:
        assert _interp_log(123.0, np.array([5.0]), np.array([3.0])) == pytest.approx(3.0)

    def test_geometric_midpoint(self) -> None:
        # log-log interpolation of a power law: y = x^2 at x = sqrt(10) -> 10.
        xs = np.array([1.0, 10.0])
        ys = np.array([1.0, 100.0])
        assert _interp_log(np.sqrt(10.0), xs, ys) == pytest.approx(10.0, rel=1e-6)

    def test_duplicate_x_keeps_last(self) -> None:
        xs = np.array([1.0, 2.0, 2.0, 4.0])
        ys = np.array([10.0, 20.0, 25.0, 40.0])
        # duplicate x=2 collapses to the last value (25); query at x=2 -> 25.
        assert _interp_log(2.0, xs, ys) == pytest.approx(25.0, rel=1e-6)

    def test_unsorted_input(self) -> None:
        xs = np.array([10.0, 1.0])
        ys = np.array([100.0, 1.0])
        assert _interp_log(np.sqrt(10.0), xs, ys) == pytest.approx(10.0, rel=1e-6)

    def test_non_positive_linear_fallback(self) -> None:
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([-1.0, 0.0, 1.0])
        assert _interp_log(2.0, xs, ys) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# compare_ratios                                                               #
# --------------------------------------------------------------------------- #


def _traj(method: str, dofs: list[int], errs: list[float], times: list[float]) -> ArmTrajectory:
    traj = ArmTrajectory(method=method)
    for level, (d, e, t) in enumerate(zip(dofs, errs, times, strict=True)):
        traj.points.append(TrajectoryPoint(level=level, n_dof=d, l2_error=e, wall_time_seconds=t))
    return traj


class TestCompareRatios:
    def test_finite_positive_ratios(self) -> None:
        dorfler = _traj("dorfler", [10, 40, 90], [1.0, 0.5, 0.25], [0.0, 0.1, 0.2])
        mcts = _traj("mcts", [10, 40, 90], [1.0, 0.4, 0.2], [0.0, 0.3, 0.6])
        l2_ratio, epd_ratio, matched_dof, matched_t = compare_ratios(
            dorfler, mcts, ComparisonParams()
        )
        assert np.isfinite(l2_ratio) and l2_ratio > 0.0
        assert np.isfinite(epd_ratio) and epd_ratio > 0.0
        assert matched_dof == 90.0
        assert matched_t == pytest.approx(0.2)
        # MCTS has lower error at matched DOF -> ratio < 1.
        assert l2_ratio < 1.0


# --------------------------------------------------------------------------- #
# Real run_comparison micro-run                                                #
# --------------------------------------------------------------------------- #


class TestRunComparison:
    def test_micro_run_finite_ratios(self) -> None:
        op = _operator()
        game_config = _game_config()
        params = ComparisonParams(
            initial_side=4,
            max_dof=110,
            max_steps=5,
            max_refinements=5,
            n_candidate_elements=4,
            n_simulations=4,
            add_noise=False,
        )
        result = run_comparison(op, game_config, params)
        metrics = result.metrics()
        assert np.isfinite(metrics["l2_error_ratio_at_matched_dof"])
        assert metrics["l2_error_ratio_at_matched_dof"] > 0.0
        assert np.isfinite(metrics["error_per_dof_ratio_mcts_over_dorfler"])
        assert metrics["error_per_dof_ratio_mcts_over_dorfler"] > 0.0
        assert result.dorfler.points and result.mcts.points


# --------------------------------------------------------------------------- #
# Artifact writers                                                             #
# --------------------------------------------------------------------------- #


class TestArtifacts:
    def _result(self):  # type: ignore[no-untyped-def]
        op = _operator()
        params = ComparisonParams(
            initial_side=4,
            max_dof=100,
            max_steps=4,
            max_refinements=4,
            n_candidate_elements=4,
            n_simulations=4,
            add_noise=False,
        )
        return run_comparison(op, _game_config(), params)

    def test_export_csv_columns_and_rows(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = self._result()
        path = export_csv(result, tmp_path / "out.csv")
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        assert header == [
            "problem",
            "method",
            "refinement_level",
            "n_dof",
            "l2_error",
            "wall_time_seconds",
            "error_per_dof",
            "seed",
        ]
        n_data = len(rows) - 1
        assert n_data == len(result.dorfler.points) + len(result.mcts.points)
        methods = {r[1] for r in rows[1:]}
        assert methods == {"dorfler", "mcts"}

    def test_export_plot_writes_png(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("matplotlib", reason="matplotlib required for the PNG")
        result = self._result()
        path = export_plot(result, tmp_path / "out.png")
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 0
