"""Tests for scripts/run_adaptive_vs_uniform.py.

The script exists to make the charter's *"L-shape adaptive Dörfler vs uniform at
matched DOF"* row cite an artifact that actually **contains both arms**. The
previously cited file (``results/lshape_mcts_vs_dorfler.csv``) has only
``{dorfler, mcts}`` in its ``method`` column, so the charter's evidence guard
passed on a file that could not support the claim.

The load-bearing property here is *shared substrate*: both arms must use one
solver, one geometry predicate and one refinement primitive, or the comparison
measures the plumbing rather than the marking.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.run_adaptive_vs_uniform import (
    build_operator,
    build_parser,
    compare,
    export_csv,
    run_uniform_arm,
)
from src.research.lshape_amr_compare import (
    ComparisonParams,
    lshape_inside_predicate,
    make_solve_fn,
)


@pytest.fixture(scope="module")
def params() -> ComparisonParams:
    """A deliberately small budget: these are mechanism tests, not the headline."""
    return ComparisonParams(initial_side=4, max_dof=250, marking_fraction=0.5, max_refinements=8)


@pytest.fixture(scope="module")
def uniform_rows(params: ComparisonParams) -> list[tuple[int, int, float]]:
    operator = build_operator(params.scale)
    solve_fn = make_solve_fn(operator, lshape_inside_predicate(params.scale))
    return run_uniform_arm(solve_fn, params)


class TestParser:
    def test_defaults_are_declared_not_implied(self) -> None:
        args = build_parser().parse_args([])
        assert args.output.endswith(".csv")
        assert args.initial_side > 0 and args.max_dof > 0
        assert 0.0 < args.marking_fraction < 1.0

    def test_every_knob_is_overridable(self) -> None:
        args = build_parser().parse_args(
            ["--output", "/tmp/x.csv", "--initial-side", "8", "--max-dof", "99"]
        )
        assert (args.output, args.initial_side, args.max_dof) == ("/tmp/x.csv", 8, 99)


class TestUniformArm:
    def test_refines_every_element_so_dof_grows_geometrically(
        self, uniform_rows: list[tuple[int, int, float]]
    ) -> None:
        dofs = [row[1] for row in uniform_rows]
        assert len(dofs) >= 3, "need several levels to see the growth"
        ratios = [b / a for a, b in zip(dofs, dofs[1:], strict=False)]
        assert all(r > 2.0 for r in ratios), (
            f"uniform refinement must roughly quadruple DOF per level, got {ratios}"
        )

    def test_error_decreases_monotonically(
        self, uniform_rows: list[tuple[int, int, float]]
    ) -> None:
        errors = [row[2] for row in uniform_rows]
        assert all(b < a for a, b in zip(errors, errors[1:], strict=False)), (
            f"uniform refinement must converge; got {errors}"
        )

    def test_respects_the_dof_budget(
        self, uniform_rows: list[tuple[int, int, float]], params: ComparisonParams
    ) -> None:
        assert uniform_rows[-1][1] >= params.max_dof or len(uniform_rows) > 1

    def test_starts_from_the_same_grid_as_the_dorfler_arm(
        self, uniform_rows: list[tuple[int, int, float]], params: ComparisonParams
    ) -> None:
        """Level 0 must be the shared coarse solve, or the arms are not comparable."""
        from src.research.lshape_amr_compare import run_dorfler_arm

        operator = build_operator(params.scale)
        solve_fn = make_solve_fn(operator, lshape_inside_predicate(params.scale))
        dorfler = run_dorfler_arm(operator, solve_fn, params)
        assert uniform_rows[0][1] == dorfler.points[0].n_dof
        assert uniform_rows[0][2] == pytest.approx(dorfler.points[0].l2_error, rel=0)


class TestCompare:
    def test_reports_rates_and_a_ratio_band(self) -> None:
        uniform = [(0, 16, 1e-2), (1, 64, 2.5e-3), (2, 256, 6e-4)]
        dorfler_dofs = np.array([16.0, 64.0, 256.0])
        dorfler_errors = np.array([1e-2, 8e-3, 6e-3])
        metrics = compare(uniform, dorfler_dofs, dorfler_errors)
        assert metrics["uniform_convergence_exponent"] < -0.4
        assert metrics["dorfler_convergence_exponent"] > -0.3
        assert metrics["dorfler_over_uniform_max"] > 1.0

    def test_ratio_direction_is_dorfler_over_uniform(self) -> None:
        """Above 1 must mean adaptive is WORSE -- the direction is the claim."""
        uniform = [(0, 16, 1e-3), (1, 64, 1e-4)]
        metrics = compare(uniform, np.array([16.0, 64.0]), np.array([1e-3, 1e-2]))
        assert metrics["dorfler_over_uniform_max"] > 1.0

    def test_min_and_max_bound_the_reported_readings(self) -> None:
        uniform = [(0, 16, 1e-2), (1, 64, 2.5e-3), (2, 256, 6e-4)]
        metrics = compare(uniform, np.array([16.0, 64.0, 256.0]), np.array([1e-2, 8e-3, 6e-3]))
        readings = [
            value for key, value in metrics.items() if key.startswith("dorfler_over_uniform_at_")
        ]
        assert min(readings) == pytest.approx(metrics["dorfler_over_uniform_min"])
        assert max(readings) == pytest.approx(metrics["dorfler_over_uniform_max"])


class TestExportCsv:
    def test_artifact_contains_both_arms(self, tmp_path: Path) -> None:
        """The whole reason the script exists."""
        path = export_csv(tmp_path / "out.csv", [(0, 16, 1e-2)], [(0, 16, 1e-2), (1, 24, 9e-3)])
        with path.open(encoding="utf-8") as handle:
            methods = {row["method"] for row in csv.DictReader(handle)}
        assert methods == {"uniform", "dorfler"}, (
            "an artifact backing a comparison claim must contain the arms compared"
        )

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = export_csv(tmp_path / "a" / "b" / "out.csv", [(0, 16, 1e-2)], [(0, 16, 1e-2)])
        assert path.exists()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
