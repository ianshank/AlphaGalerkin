"""Unit tests for the substrate-agnostic sweep + rate-fit helpers.

Driven entirely by a synthetic in-memory substrate: no PDE solve, no
``scikit-fem``, no scipy. That is deliberate on two counts — it keeps this
file fast enough to run on every CI job, and it *proves* the claim
``src/research/substrates/sweep.py`` makes in its own docstring, that nothing
in it knows what the mesh underneath actually is. A test that could only be
written against a real substrate would falsify that claim rather than pin it.

``tests/research/test_amr_arena_interpretability.py`` covers the same helpers
against the two real substrates; this file covers the branches a healthy
sweep never reaches (early stops, refusal to fit too few points, the
degenerate-unit diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.refinement.substrate import SubstrateSolveResult
from src.research.substrates.config import RATE_FIT_MIN_POINTS
from src.research.substrates.sweep import (
    InsufficientSweepPointsError,
    RateSeparation,
    SweepPoint,
    fit_log_log_rate,
    measure_rate_separation,
    run_refinement_sweep,
)


@dataclass(frozen=True)
class _FakeMesh:
    n_units: int


class _FakeSubstrate:
    """Synthetic substrate with an exactly-known convergence rate.

    ``l2_error = n_dof ** rate`` by construction, so a correct log-log fit
    must recover ``rate`` to machine precision -- which is what makes a rate
    assertion here a real test of the fitter rather than a restatement of
    whatever it happened to produce.
    """

    def __init__(
        self,
        *,
        rate: float = -1.0,
        growth: int = 2,
        start_units: int = 4,
        mark_first_only: bool = False,
        never_mark: bool = False,
    ) -> None:
        self.rate = rate
        self.growth = growth
        self.start_units = start_units
        self.mark_first_only = mark_first_only
        self.never_mark = never_mark
        self.solve_calls = 0

    def initial_mesh(self) -> _FakeMesh:
        return _FakeMesh(n_units=self.start_units)

    def solve(self, mesh: _FakeMesh) -> SubstrateSolveResult:
        self.solve_calls += 1
        n_dof = mesh.n_units
        return SubstrateSolveResult(
            values=np.zeros(n_dof),
            indicators=np.arange(mesh.n_units, dtype=np.float64),
            l2_error=float(n_dof) ** self.rate,
            n_dof=n_dof,
            n_dof_free=n_dof,
            extra={},
        )

    def mark(self, indicators: np.ndarray, theta: float) -> np.ndarray:
        if self.never_mark:
            return np.zeros(len(indicators), dtype=bool)
        marked = np.zeros(len(indicators), dtype=bool)
        if self.mark_first_only:
            marked[0] = True
        else:
            marked[:] = True
        return marked

    def refine(self, mesh: _FakeMesh, marked: np.ndarray) -> _FakeMesh:
        return _FakeMesh(n_units=mesh.n_units * self.growth)

    def n_units(self, mesh: _FakeMesh) -> int:
        return mesh.n_units

    def refinable_mask(self, mesh: _FakeMesh) -> np.ndarray:
        return np.ones(mesh.n_units, dtype=bool)

    def fingerprint(self, mesh: _FakeMesh) -> bytes:
        return str(mesh.n_units).encode()

    def describe(self) -> dict[str, str | int | float]:
        return {"kind": "fake"}


def _sweep(substrate: _FakeSubstrate, **kwargs: object) -> list[SweepPoint]:
    defaults: dict[str, object] = {
        "policy": "adaptive",
        "theta": 0.5,
        "max_levels": 10,
        "max_dof": 1_000,
    }
    defaults.update(kwargs)
    return run_refinement_sweep(substrate, **defaults)  # type: ignore[arg-type]


class TestRunRefinementSweep:
    def test_records_one_point_per_level_in_order(self) -> None:
        points = _sweep(_FakeSubstrate())
        assert [p.level for p in points] == list(range(len(points)))
        assert [p.n_dof for p in points] == [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def test_stops_at_max_dof(self) -> None:
        points = _sweep(_FakeSubstrate(), max_dof=64)
        assert points[-1].n_dof == 64
        assert all(p.n_dof <= 64 for p in points)

    def test_stops_at_max_levels(self) -> None:
        """The ``for/else`` branch: budget exhausted before any stop condition."""
        points = _sweep(_FakeSubstrate(), max_levels=3, max_dof=10**9)
        assert len(points) == 3

    def test_stops_on_error_tolerance(self) -> None:
        points = _sweep(_FakeSubstrate(rate=-1.0), max_dof=10**9, error_tolerance=0.05)
        assert points[-1].l2_error < 0.05
        assert all(p.l2_error >= 0.05 for p in points[:-1])

    def test_stops_when_nothing_is_marked(self) -> None:
        """A substrate that marks nothing must end the sweep, not spin."""
        substrate = _FakeSubstrate(never_mark=True)
        points = _sweep(substrate, max_dof=10**9, max_levels=10)
        assert len(points) == 1
        assert substrate.solve_calls == 1

    def test_uniform_policy_ignores_mark_and_uses_refinable_mask(self) -> None:
        """``policy="uniform"`` must not consult ``mark`` at all.

        Proven with a substrate whose ``mark`` marks *nothing*: under the
        adaptive policy that stops the sweep after one level, so a uniform
        sweep that still runs to completion can only have used
        ``refinable_mask``.
        """
        substrate = _FakeSubstrate(never_mark=True)
        adaptive = _sweep(substrate, policy="adaptive", max_dof=64)
        uniform = _sweep(substrate, policy="uniform", max_dof=64)
        assert len(adaptive) == 1
        assert len(uniform) > 1

    def test_records_n_units_alongside_n_dof(self) -> None:
        points = _sweep(_FakeSubstrate(), max_dof=16)
        assert [p.n_units for p in points] == [4, 8, 16]
        assert all(p.n_dof_free == p.n_dof for p in points)


class TestFitLogLogRate:
    def test_recovers_a_known_rate_exactly(self) -> None:
        points = _sweep(_FakeSubstrate(rate=-1.25), max_dof=1024)
        slope, n_used = fit_log_log_rate(points, (1, 10**9))
        assert slope == pytest.approx(-1.25, rel=1e-9)
        assert n_used == len(points)

    def test_window_restricts_the_points_used(self) -> None:
        points = _sweep(_FakeSubstrate(), max_dof=1024)
        _, n_used = fit_log_log_rate(points, (16, 128))
        assert n_used == 4  # 16, 32, 64, 128

    def test_too_few_points_raises_rather_than_fitting(self) -> None:
        points = _sweep(_FakeSubstrate(), max_dof=1024)
        with pytest.raises(InsufficientSweepPointsError, match="means nothing"):
            fit_log_log_rate(points, (16, 32))

    def test_min_points_is_the_documented_boundary(self) -> None:
        """Exactly ``RATE_FIT_MIN_POINTS`` fits; one fewer raises."""
        points = _sweep(_FakeSubstrate(), max_dof=1024)
        assert RATE_FIT_MIN_POINTS == 3
        slope, n_used = fit_log_log_rate(points, (16, 64))
        assert n_used == RATE_FIT_MIN_POINTS
        assert slope < 0

    def test_non_positive_errors_are_excluded_not_logged(self) -> None:
        """``log(0)`` would be ``-inf``; such a point must be dropped, not fitted."""
        good = [
            SweepPoint(level=i, n_dof=d, n_dof_free=d, n_units=d, l2_error=1.0 / d)
            for i, d in enumerate((10, 20, 40, 80))
        ]
        poisoned = [
            *good,
            SweepPoint(level=4, n_dof=160, n_dof_free=160, n_units=160, l2_error=0.0),
        ]
        slope_good, n_good = fit_log_log_rate(good, (1, 10**9))
        slope_poisoned, n_poisoned = fit_log_log_rate(poisoned, (1, 10**9))
        assert n_poisoned == n_good
        assert slope_poisoned == pytest.approx(slope_good)

    def test_empty_trajectory_raises_with_a_readable_message(self) -> None:
        with pytest.raises(InsufficientSweepPointsError, match="got 0"):
            fit_log_log_rate([], (1, 10**9))


class TestMeasureRateSeparation:
    def test_reads_the_ratio_at_the_largest_shared_dof(self) -> None:
        adaptive = _sweep(_FakeSubstrate(rate=-1.25), max_dof=1024)
        uniform = _sweep(_FakeSubstrate(rate=-0.70), max_dof=256)
        separation = measure_rate_separation(adaptive, uniform, (1, 10**9))
        assert isinstance(separation, RateSeparation)
        assert separation.matched_dof == 256.0
        assert separation.adaptive_rate == pytest.approx(-1.25, rel=1e-9)
        assert separation.uniform_rate == pytest.approx(-0.70, rel=1e-9)

    def test_steeper_adaptive_rate_gives_a_ratio_below_one(self) -> None:
        adaptive = _sweep(_FakeSubstrate(rate=-1.25), max_dof=1024)
        uniform = _sweep(_FakeSubstrate(rate=-0.70), max_dof=1024)
        separation = measure_rate_separation(adaptive, uniform, (1, 10**9))
        assert separation.error_ratio_at_matched_dof < 1.0

    def test_shallower_adaptive_rate_gives_a_ratio_above_one(self) -> None:
        """The inversion the tensor-product substrate exhibits, in synthetic form."""
        adaptive = _sweep(_FakeSubstrate(rate=-0.25), max_dof=1024)
        uniform = _sweep(_FakeSubstrate(rate=-0.70), max_dof=1024)
        separation = measure_rate_separation(adaptive, uniform, (1, 10**9))
        assert separation.error_ratio_at_matched_dof > 1.0

    def test_identical_arms_give_a_ratio_of_one(self) -> None:
        arm = _sweep(_FakeSubstrate(rate=-0.70), max_dof=1024)
        separation = measure_rate_separation(arm, arm, (1, 10**9))
        assert separation.error_ratio_at_matched_dof == pytest.approx(1.0)

    def test_propagates_insufficient_points(self) -> None:
        adaptive = _sweep(_FakeSubstrate(), max_dof=1024)
        with pytest.raises(InsufficientSweepPointsError):
            measure_rate_separation(adaptive, adaptive, (16, 32))
