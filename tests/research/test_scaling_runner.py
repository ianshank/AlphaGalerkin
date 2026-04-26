"""Tests for the weak-scaling benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, BaseSolver, SolverResult
from src.research.scaling_runner import (
    ScalingConfig,
    WeakScalingRunner,
    estimate_scaling_exponent,
)


def _make_op() -> PoissonOperator:
    return PoissonOperator(
        PDEConfig(
            name="poisson_test",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            advection_coeff=[0.0, 0.0],
        )
    )


# ---------------------------------------------------------------------------
# Deterministic fake clock shared between mock solvers and the monkeypatch.
#
# Each _LinearMockSolver.solve() call advances _fake_now by (1e-4 * n_dof)
# seconds.  The runner measures wall time as:
#
#   t0    = time.perf_counter()   ← returns _fake_now before solve()
#   solve() runs                  ← advances _fake_now
#   elapsed = time.perf_counter() - t0  ← captures the delta
#
# This gives perfectly deterministic O(n) wall times with no CPU spin-wait.
# ---------------------------------------------------------------------------

_fake_now: list[float] = [0.0]


class _LinearMockSolver(BaseSolver):
    """Solver whose simulated wall-time scales as O(n).

    Advances the module-level *_fake_now* clock instead of spinning — see
    the *_mock_solvers_in_registry* fixture which patches
    ``time.perf_counter`` to read from that clock.
    """

    name = "_mock_linear"
    description = "Mock solver with linear scaling"

    def __init__(self) -> None:
        pass

    def solve(self, operator, n_dof, **kwargs):  # type: ignore[no-untyped-def]
        _fake_now[0] += 1e-4 * n_dof  # advance fake clock proportional to n
        return SolverResult(
            solution=np.zeros(n_dof, dtype=np.float64),
            grid_points=np.zeros((n_dof, 2), dtype=np.float64),
            n_dof=n_dof,
            wall_time_seconds=1e-4 * n_dof,
            l2_error=0.0,
        )


class _FailingSolver(BaseSolver):
    name = "_mock_fail"
    description = "Mock solver that always raises"

    def solve(self, operator, n_dof, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("intentional failure")


@pytest.fixture(autouse=True)
def _mock_solvers_in_registry(monkeypatch: pytest.MonkeyPatch) -> object:
    """Register mock solvers and patch time.perf_counter for determinism."""
    _fake_now[0] = 0.0

    monkeypatch.setattr("time.perf_counter", lambda: _fake_now[0])

    SOLVER_REGISTRY.setdefault("_mock_linear", _LinearMockSolver)
    SOLVER_REGISTRY.setdefault("_mock_fail", _FailingSolver)
    yield
    SOLVER_REGISTRY.pop("_mock_linear", None)
    SOLVER_REGISTRY.pop("_mock_fail", None)


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class TestScalingConfig:
    def test_defaults(self) -> None:
        cfg = ScalingConfig(solvers=["direct_solver"], n_dof_values=[16, 64])
        assert cfg.repeats == 1
        assert cfg.timeout_seconds is None
        assert cfg.drop_warmup is False

    def test_empty_solvers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScalingConfig(solvers=[], n_dof_values=[16])

    def test_empty_dofs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScalingConfig(solvers=["x"], n_dof_values=[])

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScalingConfig(solvers=["x"], n_dof_values=[16], unknown=1)  # type: ignore[call-arg]

    def test_zero_dof_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScalingConfig(solvers=["x"], n_dof_values=[0])

    def test_negative_dof_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScalingConfig(solvers=["x"], n_dof_values=[-1, 64])


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


class TestWeakScalingRunner:
    def test_run_records_each_cell(self) -> None:
        cfg = ScalingConfig(solvers=["_mock_linear"], n_dof_values=[16, 64, 256])
        report = WeakScalingRunner(cfg).run(_make_op())
        assert len(report.measurements) == 3
        assert all(m.success for m in report.measurements)
        # Wall times monotonically grow with DOF
        ts = [m.wall_time_seconds for m in report.measurements]
        assert ts == sorted(ts)

    def test_unknown_solver_recorded_as_failure(self) -> None:
        cfg = ScalingConfig(solvers=["nonexistent"], n_dof_values=[32])
        report = WeakScalingRunner(cfg).run(_make_op())
        assert len(report.measurements) == 1
        assert report.measurements[0].success is False
        assert "not registered" in (report.measurements[0].error_message or "")

    def test_solver_exception_isolated(self) -> None:
        cfg = ScalingConfig(
            solvers=["_mock_fail", "_mock_linear"],
            n_dof_values=[32],
        )
        report = WeakScalingRunner(cfg).run(_make_op())
        assert len(report.measurements) == 2
        # Failing solver records the failure, linear succeeds
        by_solver = {m.solver: m for m in report.measurements}
        assert by_solver["_mock_fail"].success is False
        assert "intentional failure" in (by_solver["_mock_fail"].error_message or "")
        assert by_solver["_mock_linear"].success is True

    def test_summary_fits_exponent(self) -> None:
        cfg = ScalingConfig(
            solvers=["_mock_linear"],
            n_dof_values=[64, 256, 1024, 4096],
        )
        report = WeakScalingRunner(cfg).run(_make_op())
        assert len(report.summaries) == 1
        summary = report.summaries[0]
        # Linear synthetic time -> exponent ≈ 1, R² close to 1
        assert summary.exponent == pytest.approx(1.0, abs=0.2)
        assert summary.r_squared > 0.9

    def test_save_emits_csv_and_json(self, tmp_path: Path) -> None:
        cfg = ScalingConfig(solvers=["_mock_linear"], n_dof_values=[32, 128])
        report = WeakScalingRunner(cfg).run(_make_op())
        artefacts = WeakScalingRunner(cfg).save(report, tmp_path)
        assert "csv" in artefacts and artefacts["csv"].exists()
        assert "json" in artefacts and artefacts["json"].exists()
        data = json.loads(artefacts["json"].read_text())
        assert data["config"]["solvers"] == ["_mock_linear"]
        assert len(data["measurements"]) == 2

    def test_repeats_average(self) -> None:
        cfg = ScalingConfig(
            solvers=["_mock_linear"],
            n_dof_values=[64],
            repeats=3,
            drop_warmup=True,
        )
        report = WeakScalingRunner(cfg).run(_make_op())
        assert len(report.measurements) == 1
        assert report.measurements[0].success
        assert np.isfinite(report.measurements[0].wall_time_seconds)

    def test_timeout_marks_failure(self) -> None:
        """A cell whose wall time exceeds the timeout budget must be failure."""
        # n_dof=256 → simulated time = 1e-4 * 256 = 0.0256 > 0.001 budget
        cfg = ScalingConfig(
            solvers=["_mock_linear"],
            n_dof_values=[256],
            timeout_seconds=0.001,
        )
        report = WeakScalingRunner(cfg).run(_make_op())
        assert report.measurements[0].success is False
        assert report.measurements[0].error_message is not None
        assert "TimeoutError" in report.measurements[0].error_message


# ---------------------------------------------------------------------------
# estimate_scaling_exponent
# ---------------------------------------------------------------------------


class TestEstimateScalingExponent:
    def test_linear_fits_exponent_one(self) -> None:
        n = [10, 100, 1000, 10000]
        t = [x * 1e-4 for x in n]
        assert estimate_scaling_exponent(n, t) == pytest.approx(1.0, abs=1e-6)

    def test_quadratic_fits_exponent_two(self) -> None:
        n = [10, 100, 1000]
        t = [x**2 * 1e-6 for x in n]
        assert estimate_scaling_exponent(n, t) == pytest.approx(2.0, abs=1e-6)

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_scaling_exponent([10], [1.0])
        with pytest.raises(ValueError):
            estimate_scaling_exponent([10, 100], [1.0])
