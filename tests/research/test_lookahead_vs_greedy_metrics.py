"""Gate 1 measurements: matched DOF against the best classical arm, span, break-even.

Pure functions on synthetic trajectories, so every expected value below is
computed independently of the code under test (power laws, hand-solved
interpolations), never by calling the function a second time. Defect classes:

* **M1 wrong reference** -- the ratio is read against one classical arm, or
  each arm at its own pairwise matched DOF, instead of the best arm at the one
  DOF every arm reached.
* **M2 silent shrink** -- a classical arm that stops short is not reported, so
  the matched DOF shrinks to its last level.
* **M3 wrong passage** -- equal accuracy is read after the first crossing (a
  non-monotone trajectory), extrapolated, or missed by one ulp.
* **M4 K* drift** -- ``K*`` is finite with no DOF saving, infinite with one, or
  not the pre-registered formula.

The mutation-kill record for the whole Gate 1 surface is in
``tests/research/test_lookahead_vs_greedy.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import src.research.lookahead_vs_greedy_metrics as metrics_module
from src.research.amr_arena_types import ArenaPoint, ArenaTrajectory
from src.research.lookahead_vs_greedy_metrics import (
    FIRST_PASSAGE_RTOL,
    GREEDY_LABEL,
    UNIFORM_LABEL,
    BreakEvenCalibrationError,
    ClassicalBudgetError,
    SearchStep,
    alpha_suffix,
    assert_classical_span,
    break_even,
    dorfler_label,
    first_passage,
    matched_reading,
    per_solve_cost_coefficient,
    reuses_to_break_even,
)


def traj(
    dofs: Sequence[float],
    errors: Sequence[float],
    walls: Sequence[float] | None = None,
    *,
    method: str = "mcts",
) -> ArenaTrajectory:
    """A trajectory with the given readings (wall-clock defaults to one second a step)."""
    out = ArenaTrajectory(method=method)  # type: ignore[arg-type]
    seconds = walls if walls is not None else [float(i) for i in range(len(dofs))]
    for level, (dof, err, wall) in enumerate(zip(dofs, errors, seconds, strict=True)):
        out.points.append(
            ArenaPoint(
                level=level,
                n_dof=int(dof),
                l2_error=float(err),
                wall_time_seconds=float(wall),
                n_cache_misses=level + 1,
                n_cache_hits=0,
                n_apply_actions=level,
            )
        )
    return out


class TestLabels:
    def test_dorfler_and_alpha_labels(self) -> None:
        assert [dorfler_label(theta) for theta in (0.1, 0.3, 0.5)] == [
            "dorfler_theta0.1",
            "dorfler_theta0.3",
            "dorfler_theta0.5",
        ]
        assert (alpha_suffix(1.0), alpha_suffix(1.5)) == ("alpha1", "alpha1.5")

    def test_search_step_diverged(self) -> None:
        kept = SearchStep(step=0, action=3, greedy_action=3, tree_depth=2, n_legal=4, dof=10)
        moved = SearchStep(step=1, action=5, greedy_action=3, tree_depth=2, n_legal=4, dof=12)
        assert (kept.diverged, moved.diverged) == (False, True)


class TestMatchedReading:
    def test_one_matched_dof_for_every_arm_and_the_best_classical(self) -> None:
        """M1: N* is the min over ALL arms' max DOF, read with log-log interpolation."""
        candidate = traj([100, 400], [1.0, 0.25])
        classical = {
            GREEDY_LABEL: traj([100, 300], [1.0, 0.4], method="greedy"),
            "dorfler_theta0.5": traj([100, 800], [1.0, 0.1], method="dorfler"),
            UNIFORM_LABEL: traj([100, 900], [1.0, 0.5], method="uniform"),
        }
        reading = matched_reading(candidate, classical)
        assert reading.matched_dof == pytest.approx(300.0)
        # Every arm is a power law between its points, so log-log reads are exact.
        cand = 1.0 * (300 / 100) ** (math.log(0.25) / math.log(4.0))
        dorf = 1.0 * (300 / 100) ** (math.log(0.1) / math.log(8.0))
        unif = 1.0 * (300 / 100) ** (math.log(0.5) / math.log(9.0))
        assert reading.candidate_l2 == pytest.approx(cand)
        assert reading.classical_l2[GREEDY_LABEL] == pytest.approx(0.4)
        assert reading.classical_l2["dorfler_theta0.5"] == pytest.approx(dorf)
        assert reading.classical_l2[UNIFORM_LABEL] == pytest.approx(unif)
        assert reading.best_classical == "dorfler_theta0.5"
        assert reading.ratio == pytest.approx(cand / dorf)
        assert reading.ratio_vs(GREEDY_LABEL) == pytest.approx(cand / 0.4)

    def test_a_candidate_that_stops_first_sets_the_matched_dof(self) -> None:
        """M1: the candidate is one of the arms N* is taken over, not only the classical ones."""
        candidate = traj([100, 250], [1.0, 0.3])
        classical = {
            GREEDY_LABEL: traj([100, 400], [1.0, 0.25], method="greedy"),
            UNIFORM_LABEL: traj([100, 900], [1.0, 0.5], method="uniform"),
        }
        reading = matched_reading(candidate, classical)
        assert reading.matched_dof == pytest.approx(250.0)
        assert reading.candidate_l2 == pytest.approx(0.3)
        greedy = 1.0 * (250 / 100) ** (math.log(0.25) / math.log(4.0))
        assert reading.classical_l2[GREEDY_LABEL] == pytest.approx(greedy)

    def test_a_tie_goes_to_the_arm_listed_first(self) -> None:
        same = [100, 200], [1.0, 0.5]
        reading = matched_reading(
            traj(*same),
            {GREEDY_LABEL: traj(*same), UNIFORM_LABEL: traj(*same)},
        )
        assert reading.best_classical == GREEDY_LABEL
        assert reading.ratio == pytest.approx(1.0)

    def test_empty_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one classical"):
            matched_reading(traj([1], [1.0]), {})
        with pytest.raises(ValueError, match="no points"):
            matched_reading(ArenaTrajectory(method="mcts"), {GREEDY_LABEL: traj([1], [1.0])})
        with pytest.raises(ValueError, match="greedy trajectory has no points"):
            matched_reading(traj([1], [1.0]), {GREEDY_LABEL: ArenaTrajectory(method="greedy")})


class TestClassicalSpan:
    def test_spanning_arms_pass(self) -> None:
        assert_classical_span(
            {"dorfler_theta0.1": traj([10, 500], [1, 0.1])},
            [traj([10, 300], [1, 0.2]), traj([10, 499], [1, 0.2])],
        )

    def test_an_arm_reaching_exactly_the_game_driven_dof_spans_it(self) -> None:
        """M2 boundary: reaching the largest game-driven DOF exactly is enough."""
        assert_classical_span(
            {UNIFORM_LABEL: traj([10, 300], [1, 0.1])},
            [traj([10, 300], [1, 0.2]), traj([10, 120], [1, 0.3])],
        )

    def test_a_short_arm_is_named(self) -> None:
        """M2: the arm that stops below the largest game-driven DOF is reported."""
        with pytest.raises(ClassicalBudgetError, match="dorfler_theta0.1"):
            assert_classical_span(
                {
                    "dorfler_theta0.1": traj([10, 200], [1, 0.1]),
                    UNIFORM_LABEL: traj([10, 900], [1, 0.1]),
                },
                [traj([10, 300], [1, 0.2])],
            )

    def test_an_empty_arm_is_short_and_no_game_points_is_refused(self) -> None:
        with pytest.raises(ClassicalBudgetError, match="uniform"):
            assert_classical_span(
                {UNIFORM_LABEL: ArenaTrajectory(method="uniform")}, [traj([10], [1.0])]
            )
        with pytest.raises(ValueError, match="nothing to span"):
            assert_classical_span({}, [ArenaTrajectory(method="mcts")])


class TestFirstPassage:
    def test_the_initial_point_already_meets_the_target(self) -> None:
        reached = first_passage(traj([50, 100], [0.2, 0.1], [0.0, 3.0]), 0.5)
        assert reached is not None
        assert (reached.dof, reached.wall_time_seconds) == (50.0, 0.0)

    def test_log_log_dof_and_linear_wall_inside_the_crossing_segment(self) -> None:
        """M3: t = ln(e*/e0)/ln(e1/e0); N log-interpolated, W linear at the same t."""
        reached = first_passage(traj([100, 400], [1.0, 0.25], [2.0, 6.0]), 0.5)
        assert reached is not None
        assert reached.dof == pytest.approx(200.0)  # t = 0.5 in log space
        assert reached.wall_time_seconds == pytest.approx(4.0)

    def test_the_first_crossing_wins_on_a_non_monotone_trajectory(self) -> None:
        """M3: a later re-crossing must not be read instead."""
        reached = first_passage(traj([100, 200, 300, 400], [1.0, 0.4, 0.8, 0.3], [0, 1, 2, 3]), 0.5)
        assert reached is not None
        assert 100.0 < reached.dof < 200.0

    def test_never_reached_is_none_not_an_extrapolation(self) -> None:
        assert first_passage(traj([100, 200], [1.0, 0.6]), 0.5) is None

    def test_an_arm_reaches_its_own_point_despite_a_one_ulp_round_trip(self) -> None:
        """M3: exp(log(e)) can land one ulp below e; the slack absorbs it."""
        target = 0.3 * (1.0 - FIRST_PASSAGE_RTOL / 10)
        reached = first_passage(traj([100, 200], [1.0, 0.3], [0, 5]), target)
        assert reached is not None
        assert reached.dof == pytest.approx(200.0)
        assert reached.wall_time_seconds == pytest.approx(5.0)

    @pytest.mark.parametrize("target", [0.0, -1.0, math.inf, math.nan])
    def test_a_target_that_cannot_be_read_in_log_space_is_refused(self, target: float) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            first_passage(traj([1, 2], [1.0, 0.5]), target)

    def test_a_non_positive_reading_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            first_passage(traj([1, 2], [1.0, 0.0]), 0.5)


class TestCostCoefficient:
    def test_measured_seconds_over_modelled_solve_cost(self) -> None:
        greedy = traj([100, 120, 150], [1.0, 0.9, 0.8], [0.0, 1.0, 3.0])
        expected = 3.0 / (120**1.5 + 150**1.5)
        assert per_solve_cost_coefficient(greedy, 1.5) == pytest.approx(expected)

    def test_no_step_or_no_time_cannot_be_calibrated(self) -> None:
        with pytest.raises(BreakEvenCalibrationError, match="committed step"):
            per_solve_cost_coefficient(traj([100], [1.0]), 1.0)
        with pytest.raises(BreakEvenCalibrationError, match="cannot calibrate"):
            per_solve_cost_coefficient(traj([100, 120], [1.0, 0.9], [2.0, 2.0]), 1.0)

    @pytest.mark.parametrize("alpha", [0.0, -1.0, math.inf])
    def test_alpha_must_be_positive(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            per_solve_cost_coefficient(traj([100, 120], [1.0, 0.9]), alpha)


class TestBreakEven:
    def test_identical_arms_save_nothing_and_never_pay_back(self) -> None:
        """M4: MCTS == greedy is a zero saving, so K* is infinite for every alpha."""
        dofs, errs, walls = [100, 140, 190], [1.0, 0.7, 0.5], [0.0, 1.0, 2.0]
        result = break_even(
            traj(dofs, errs, walls), traj(dofs, errs, walls), matched_dof=190, alphas=(1.0, 1.5)
        )
        assert result.dof_saving == 0.0
        assert result.reuses == {1.0: None, 1.5: None}
        assert result.finite is False

    def test_a_leaner_slower_candidate_pays_back_after_the_formula_count(self) -> None:
        """M4: K* = (W_m - W_g) / (kappa (N_g^a - N_m^a)), every input hand-computed."""
        greedy = traj([100, 400], [1.0, 0.25], [0.0, 2.0], method="greedy")
        mcts = traj([100, 400], [1.0, 0.0625], [0.0, 10.0])
        result = break_even(mcts, greedy, matched_dof=400, alphas=(1.0, 1.5))
        # Equal accuracy = the worse arm at N* = greedy's 0.25.
        assert result.equal_accuracy_l2 == pytest.approx(0.25)
        # MCTS: e = 0.0625 * (N/400)^-2 from 1.0 at 100 -> reaches 0.25 at N = 200 (t = 0.5).
        assert result.candidate.dof == pytest.approx(200.0)
        assert result.candidate.wall_time_seconds == pytest.approx(5.0)
        assert result.reference.dof == pytest.approx(400.0)
        assert result.reference.wall_time_seconds == pytest.approx(2.0)
        assert result.dof_saving == pytest.approx(200.0)
        for alpha in (1.0, 1.5):
            kappa = 2.0 / 400**alpha
            expected = (5.0 - 2.0) / (kappa * (400**alpha - 200**alpha))
            assert result.cost_coefficients[alpha] == pytest.approx(kappa)
            assert result.reuses[alpha] == pytest.approx(expected)
        assert result.finite is True
        assert result.extra_wall_seconds == pytest.approx(3.0)

    def test_a_faster_leaner_candidate_pays_back_at_once(self) -> None:
        greedy = traj([100, 400], [1.0, 0.25], [0.0, 8.0], method="greedy")
        mcts = traj([100, 400], [1.0, 0.0625], [0.0, 1.0])
        result = break_even(mcts, greedy, matched_dof=400, alphas=(1.0,))
        assert result.reuses[1.0] == 0.0
        assert result.finite is True

    def test_a_worse_candidate_has_a_negative_saving(self) -> None:
        greedy = traj([100, 400], [1.0, 0.0625], [0.0, 2.0], method="greedy")
        mcts = traj([100, 400], [1.0, 0.25], [0.0, 9.0])
        result = break_even(mcts, greedy, matched_dof=400, alphas=(1.0, 1.5))
        assert result.dof_saving == pytest.approx(-200.0)
        assert result.finite is False

    def test_no_alpha_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one cost exponent"):
            break_even(traj([1, 2], [1.0, 0.5]), traj([1, 2], [1.0, 0.5]), matched_dof=2, alphas=())

    def test_an_arm_that_cannot_reach_the_equal_accuracy_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive: the worse arm's reading bounds both arms, so a miss is a defect.

        Unreachable through the public inputs (each arm's interpolated L2 lies
        between two of its own points), so the passage is forced to miss.
        """
        monkeypatch.setattr(metrics_module, "first_passage", lambda trajectory, target: None)
        same = traj([100, 400], [1.0, 0.25])
        with pytest.raises(ValueError, match="never reaches"):
            break_even(same, same, matched_dof=400, alphas=(1.0,))

    @settings(max_examples=80, deadline=None)
    @given(
        n_cand=st.floats(min_value=10.0, max_value=1e5),
        saving=st.floats(min_value=-1e4, max_value=1e4),
        extra=st.floats(min_value=0.0, max_value=1e4),
        alpha=st.sampled_from((1.0, 1.5)),
    )
    def test_finite_iff_the_saving_is_positive(
        self, n_cand: float, saving: float, extra: float, alpha: float
    ) -> None:
        """M4: the pre-registered equivalence C4 rests on, over realistic magnitudes."""
        assume(abs(saving) >= 1e-6 * n_cand or saving == 0.0)
        reuses = reuses_to_break_even(extra, 1e-6, n_cand, saving, alpha)
        assert (reuses is not None) == (saving > 0.0)
        if reuses is not None:
            direct = extra / (1e-6 * ((n_cand + saving) ** alpha - n_cand**alpha))
            assert reuses == pytest.approx(direct, rel=1e-6)

    def test_a_tiny_positive_saving_stays_finite(self) -> None:
        """expm1/log1p keep the per-reuse saving positive where a difference rounds to 0."""
        reuses = reuses_to_break_even(1.0, 1e-3, 1e8, 1e-6, 1.5)
        assert reuses is not None
        assert np.isfinite(reuses)

    @pytest.mark.parametrize("coefficient", [0.0, 1e-320])
    def test_a_saving_worth_nothing_per_reuse_never_pays_back(self, coefficient: float) -> None:
        """M4: free downstream solves (or a subnormal cost) repay nothing -- K* is infinite.

        ``0.0`` makes the per-reuse saving zero; ``1e-320`` keeps it positive but
        so small that ``extra / saving`` overflows. Both must read as infinite,
        never as a finite count or a ZeroDivisionError.
        """
        assert reuses_to_break_even(5.0, coefficient, 100.0, 50.0, 1.5) is None
