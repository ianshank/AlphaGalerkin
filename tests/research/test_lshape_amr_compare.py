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
    SEED_PRIME_STRIDE,
    ArmTrajectory,
    ComparisonParams,
    ComparisonResult,
    MultiSeedComparison,
    TrajectoryPoint,
    _area_weighted_l2,
    _interp_log,
    _trapezoidal_weights,
    compare_ratios,
    export_csv,
    export_plot,
    l2_ratio_at_matched_solves,
    lshape_inside_predicate,
    make_solve_fn,
    resolved_seeds,
    run_comparison,
    run_dorfler_arm,
    run_multiseed_comparison,
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


def _traj(
    method: str,
    dofs: list[int],
    errs: list[float],
    times: list[float],
    solves: list[int] | None = None,
) -> ArmTrajectory:
    traj = ArmTrajectory(method=method)
    solves = solves if solves is not None else [i + 1 for i in range(len(dofs))]
    for level, (d, e, t, s) in enumerate(zip(dofs, errs, times, solves, strict=True)):
        traj.points.append(
            TrajectoryPoint(level=level, n_dof=d, l2_error=e, wall_time_seconds=t, n_solves=s)
        )
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


class TestMatchedSolves:
    def test_ratio_at_matched_solves(self) -> None:
        # Dörfler reaches low error in few solves; MCTS spends more solves for
        # the same DOF/error curve -> at a matched solve budget MCTS trails.
        dorfler = _traj(
            "dorfler", [10, 40, 90], [1.0, 0.5, 0.25], [0.0, 0.1, 0.2], solves=[1, 2, 3]
        )
        mcts = _traj("mcts", [10, 40, 90], [1.0, 0.5, 0.25], [0.0, 0.3, 0.6], solves=[2, 8, 18])
        ratio, matched = l2_ratio_at_matched_solves(dorfler, mcts)
        # Largest common solve count is 3 (Dörfler's max).
        assert matched == 3.0
        assert np.isfinite(ratio) and ratio > 0.0
        # At 3 solves Dörfler is already at ~0.25 while MCTS is barely past its
        # first refinement -> MCTS is worse per solve (ratio > 1).
        assert ratio > 1.0


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
        # Matched-compute (solve-count) metric is present, finite and positive.
        assert np.isfinite(metrics["l2_error_ratio_at_matched_solves"])
        assert metrics["l2_error_ratio_at_matched_solves"] > 0.0
        assert metrics["matched_solves"] > 0.0
        assert result.dorfler.points and result.mcts.points
        # Solve counts are cumulative (non-decreasing) and the MCTS arm, which
        # replays solves inside every simulation, spends strictly more real
        # solves per accepted step than Dörfler's one-solve-per-level.
        for arm in (result.dorfler, result.mcts):
            counts = arm.solve_counts()
            assert np.all(np.diff(counts) >= 0)
            assert counts[-1] >= len(arm.points)
        assert result.mcts.points[-1].n_solves > result.dorfler.points[-1].n_solves


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
            "n_solves",
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


# --------------------------------------------------------------------------- #
# _trapezoidal_weights                                                         #
# --------------------------------------------------------------------------- #


class TestTrapezoidalWeights:
    def test_uniform_axis_interior_and_endpoints(self) -> None:
        axis = np.linspace(0.0, 1.0, 5, dtype=np.float64)  # h = 0.25
        w = _trapezoidal_weights(axis)
        assert w.shape == axis.shape
        # Endpoints get half-spacing, interior gets a full spacing.
        assert w[0] == pytest.approx(0.125)
        assert w[-1] == pytest.approx(0.125)
        np.testing.assert_allclose(w[1:-1], 0.25)
        # Sum equals the span.
        assert float(w.sum()) == pytest.approx(axis[-1] - axis[0])

    def test_non_uniform_axis_sum_equals_span(self) -> None:
        axis = np.array([-1.0, -0.5, 0.1, 0.4, 2.0], dtype=np.float64)
        w = _trapezoidal_weights(axis)
        assert float(w.sum()) == pytest.approx(axis[-1] - axis[0])
        assert np.all(w > 0.0)


# --------------------------------------------------------------------------- #
# _area_weighted_l2                                                            #
# --------------------------------------------------------------------------- #


class TestAreaWeightedL2:
    def test_empty_diff_is_nan(self) -> None:
        xs = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        in_mask = np.zeros(9, dtype=bool)
        assert np.isnan(_area_weighted_l2(np.array([], dtype=np.float64), xs, ys, in_mask))

    def test_constant_field_on_uniform_grid_recovers_constant(self) -> None:
        # On a uniform grid a constant error field returns exactly that constant,
        # independent of the weighting: sqrt(sum(w*c^2)/sum(w)) == |c|.
        xs = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        in_mask = np.ones(9, dtype=bool)
        c = 0.37
        diff = np.full(9, c, dtype=np.float64)
        val = _area_weighted_l2(diff, xs, ys, in_mask)
        assert val == pytest.approx(c)
        assert val > 0.0 and np.isfinite(val)

    def test_non_uniform_grid_differs_from_unweighted_rms(self) -> None:
        # A strongly non-uniform grid makes the area weighting active, so the
        # dual-cell L2 must diverge from the plain node-wise RMS.
        xs = np.array([0.0, 0.1, 1.0], dtype=np.float64)
        ys = np.array([0.0, 0.1, 1.0], dtype=np.float64)
        in_mask = np.ones(9, dtype=bool)
        diff = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        weighted = _area_weighted_l2(diff, xs, ys, in_mask)
        rms = float(np.sqrt(np.mean(diff**2)))
        assert np.isfinite(weighted) and weighted > 0.0
        assert weighted != pytest.approx(rms)


# --------------------------------------------------------------------------- #
# resolved_seeds                                                               #
# --------------------------------------------------------------------------- #


class TestResolvedSeeds:
    def test_deterministic_distinct_and_anchored(self) -> None:
        seeds = resolved_seeds(42, 5)
        assert len(seeds) == 5
        assert seeds[0] == 42  # first == base_seed
        assert len(set(seeds)) == 5  # distinct
        # Deterministic re-run.
        assert resolved_seeds(42, 5) == seeds
        # Stride is the named prime.
        assert seeds[1] - seeds[0] == SEED_PRIME_STRIDE


# --------------------------------------------------------------------------- #
# MultiSeedComparison                                                          #
# --------------------------------------------------------------------------- #


def _stub_result(
    l2_ratio: float, epd_ratio: float, seed: int, solve_ratio: float = 1.0
) -> ComparisonResult:
    """A minimal ComparisonResult carrying the headline ratios."""
    dorfler = _traj("dorfler", [10, 40], [1.0, 0.5], [0.0, 0.1])
    mcts = _traj("mcts", [10, 40], [1.0, 0.4], [0.0, 0.2])
    return ComparisonResult(
        dorfler=dorfler,
        mcts=mcts,
        l2_error_ratio_at_matched_dof=l2_ratio,
        error_per_dof_ratio_mcts_over_dorfler=epd_ratio,
        matched_dof=40.0,
        matched_wall_time_seconds=0.1,
        dorfler_convergence_exponent=-1.0,
        mcts_convergence_exponent=-1.0,
        seed=seed,
        l2_error_ratio_at_matched_solves=solve_ratio,
        matched_solves=10.0,
    )


class TestMultiSeedComparison:
    def test_metrics_median_and_win_fraction(self) -> None:
        l2 = [0.5, 2.0, 0.8]
        epd = [1.1, 3.0, 2.0]
        solve = [1.5, 3.0, 0.9]
        seeds = [1, 2, 3]
        per_seed = [
            _stub_result(a, b, s, solve_ratio=sr)
            for a, b, s, sr in zip(l2, epd, seeds, solve, strict=True)
        ]
        ms = MultiSeedComparison(per_seed=per_seed, seeds=seeds)

        assert ms.l2_ratios == l2
        assert ms.epd_ratios == epd
        assert ms.solve_ratios == solve

        metrics = ms.metrics()
        assert metrics["l2_error_ratio_at_matched_dof"] == pytest.approx(np.median(l2))
        assert metrics["error_per_dof_ratio_mcts_over_dorfler"] == pytest.approx(np.median(epd))
        assert metrics["l2_error_ratio_at_matched_solves"] == pytest.approx(np.median(solve))
        assert metrics["l2_ratio_seed_min"] == pytest.approx(min(l2))
        assert metrics["l2_ratio_seed_max"] == pytest.approx(max(l2))
        assert metrics["l2_ratio_seed_std"] == pytest.approx(float(np.std(l2)))
        # Two of three seeds have matched-DOF ratio < 1.
        assert metrics["mcts_win_fraction"] == pytest.approx(2.0 / 3.0)
        # One of three seeds has matched-solve ratio < 1.
        assert metrics["mcts_solve_win_fraction"] == pytest.approx(1.0 / 3.0)
        assert metrics["n_seeds"] == pytest.approx(3.0)

    def test_representative_is_median_ratio_seed(self) -> None:
        l2 = [0.5, 2.0, 0.8]
        seeds = [11, 22, 33]
        per_seed = [_stub_result(a, 1.0, s) for a, s in zip(l2, seeds, strict=True)]
        ms = MultiSeedComparison(per_seed=per_seed, seeds=seeds)
        # Sorted ratios [0.5, 0.8, 2.0] -> median is the 0.8 seed (seed 33).
        assert ms.representative.seed == 33
        assert ms.representative.l2_error_ratio_at_matched_dof == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# run_multiseed_comparison (tiny real run)                                     #
# --------------------------------------------------------------------------- #


class TestRunMultiSeedComparison:
    def test_two_seed_micro_run(self) -> None:
        op = _operator()
        game_config = _game_config()
        params = ComparisonParams(
            initial_side=2,
            max_dof=80,
            max_steps=4,
            max_refinements=4,
            n_candidate_elements=3,
            n_simulations=3,
            add_noise=False,
            n_seeds=2,
        )
        ms = run_multiseed_comparison(op, game_config, params)
        assert isinstance(ms, MultiSeedComparison)
        assert len(ms.per_seed) == 2
        assert ms.seeds == resolved_seeds(params.seed, 2)
        metrics = ms.metrics()
        assert np.isfinite(metrics["l2_error_ratio_at_matched_dof"])
        assert np.isfinite(metrics["error_per_dof_ratio_mcts_over_dorfler"])
        assert metrics["n_seeds"] == pytest.approx(2.0)
        assert 0.0 <= metrics["mcts_win_fraction"] <= 1.0
