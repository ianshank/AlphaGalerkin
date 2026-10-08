"""Gate 1 verdict: the metric set and GO / NO-GO against the pre-registered criteria.

Defect classes:

* **V1 lenient verdict** -- a criterion that fails alone still yields GO, or a
  missing / non-finite measurement passes (``-inf <= 0.98`` is True).
* **V2 off-by-boundary** -- ``<=`` / ``<`` / ``>`` drift at the pre-registered
  bars (0.98 passes C1, 1.0 fails C3, 0 fails C4).
* **V3 non-monotone verdict** -- improving a measurement turns GO into NO-GO.
* **V4 wrong aggregation** -- the robustness count uses ``<=``, the median is
  replaced by a mean, or ``K*`` is published while infinite.

The thresholds come from ``LookaheadVsGreedyConfig.get_default_thresholds()``
(the single source), never re-typed here. The mutation-kill record for the
Gate 1 surface is in ``tests/research/test_lookahead_vs_greedy.py``.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.poc.scenarios.lookahead_vs_greedy_config import LookaheadVsGreedyConfig
from src.research.amr_arena_types import ArenaPoint, ArenaTrajectory
from src.research.lookahead_vs_greedy_metrics import (
    BREAK_EVEN_FINITE_METRIC,
    GREEDY_LABEL,
    PRIMARY_DIVERGENCE_METRIC,
    PRIMARY_DOF_SAVING_METRIC,
    PRIMARY_RATIO_METRIC,
    ROBUST_MEDIAN_RATIO_METRIC,
    ROBUST_WINS_METRIC,
    UNIFORM_LABEL,
    SearchStep,
    break_even,
    matched_reading,
)
from src.research.lookahead_vs_greedy_verdict import (
    CRITERION_OF_METRIC,
    GO_LABEL,
    NO_GO_LABEL,
    VERDICT_GO_METRIC,
    VERDICT_PASSED_METRIC,
    VERDICT_TOTAL_METRIC,
    Gate1Verdict,
    NonFiniteMetricError,
    ScoredRun,
    collect_testbed_metrics,
    evaluate_verdict,
    require_finite_metrics,
)
from src.research.mcts_classical_amr_arena import compare_trajectories

CONFIG = LookaheadVsGreedyConfig()
THRESHOLDS = CONFIG.get_default_thresholds()

#: A measurement set that passes every pre-registered criterion with margin.
PASSING: dict[str, float] = {
    PRIMARY_RATIO_METRIC: 0.9,
    PRIMARY_DIVERGENCE_METRIC: 4.0,
    ROBUST_MEDIAN_RATIO_METRIC: 0.95,
    ROBUST_WINS_METRIC: 4.0,
    PRIMARY_DOF_SAVING_METRIC: 12.0,
}

#: One failing value per gated metric, each just past its bar.
FAILING_ALONE: dict[str, float] = {
    PRIMARY_RATIO_METRIC: 0.981,
    PRIMARY_DIVERGENCE_METRIC: 0.0,
    ROBUST_MEDIAN_RATIO_METRIC: 1.0,
    ROBUST_WINS_METRIC: 2.0,
    PRIMARY_DOF_SAVING_METRIC: 0.0,
}


def traj(dofs: list[int], errors: list[float], *, method: str = "mcts") -> ArenaTrajectory:
    out = ArenaTrajectory(method=method)  # type: ignore[arg-type]
    for level, (dof, err) in enumerate(zip(dofs, errors, strict=True)):
        out.points.append(
            ArenaPoint(
                level=level,
                n_dof=dof,
                l2_error=err,
                wall_time_seconds=float(level),
                n_cache_misses=level + 1,
                n_cache_hits=0,
                n_apply_actions=level,
            )
        )
    return out


CLASSICAL = {
    GREEDY_LABEL: traj([100, 400], [1.0, 0.25], method="greedy"),
    "dorfler_theta0.5": traj([100, 900], [1.0, 0.3], method="dorfler"),
    UNIFORM_LABEL: traj([100, 900], [1.0, 0.5], method="uniform"),
}

#: Greedy dips below its matched-DOF accuracy early and comes back up. That is
#: the only way a run can beat every classical arm at N* (C1) yet need more DOF
#: than greedy to reach their equal accuracy (C4 fails): with a monotone greedy,
#: a run better than greedy at N* always reaches greedy's accuracy first.
NON_MONOTONE_CLASSICAL = {
    GREEDY_LABEL: traj([100, 150, 400], [1.0, 0.2, 0.3], method="greedy"),
    "dorfler_theta0.5": traj([100, 900], [1.0, 0.3], method="dorfler"),
    UNIFORM_LABEL: traj([100, 900], [1.0, 0.5], method="uniform"),
}


def scored(
    final_error: float,
    *,
    seed: int = 42,
    divergences: int | None = 1,
    depths: tuple[int, ...] = (1, 3, 2),
    classical: dict[str, ArenaTrajectory] = CLASSICAL,
) -> ScoredRun:
    """A run from 100 to 400 DOF ending at ``final_error``, scored against ``classical``."""
    trajectory = traj([100, 400], [1.0, final_error])
    trajectory.decisions_diverging_from_greedy = divergences
    reading = matched_reading(trajectory, classical)
    return ScoredRun(
        label="mcts_primary",
        seed=seed,
        add_noise=False,
        trajectory=trajectory,
        steps=tuple(
            SearchStep(step=i, action=i, greedy_action=0, tree_depth=d, n_legal=4, dof=100)
            for i, d in enumerate(depths)
        ),
        reading=reading,
        versus_greedy=compare_trajectories(reference=classical[GREEDY_LABEL], candidate=trajectory),
        versus_best=compare_trajectories(
            reference=classical[reading.best_classical], candidate=trajectory
        ),
        break_even=break_even(
            trajectory,
            classical[GREEDY_LABEL],
            matched_dof=reading.matched_dof,
            alphas=CONFIG.break_even_alphas,
        ),
    )


def testbed_verdict(
    primary_error: float,
    robust_errors: tuple[float, ...],
    *,
    divergences: int = 1,
    classical: dict[str, ArenaTrajectory] = CLASSICAL,
) -> Gate1Verdict:
    """Score a whole synthetic testbed: trajectories -> readings -> metrics -> verdict."""
    primary = scored(primary_error, divergences=divergences, classical=classical)
    robust = [
        scored(error, seed=seed, classical=classical)
        for seed, error in zip(CONFIG.robust_seeds(), robust_errors, strict=True)
    ]
    metrics = collect_testbed_metrics(primary, robust, robust_win_ratio=CONFIG.robust_win_ratio)
    return evaluate_verdict("T", metrics, THRESHOLDS)


class TestEvaluateVerdict:
    def test_every_criterion_passing_is_go(self) -> None:
        verdict = evaluate_verdict("T1_lshape", PASSING, THRESHOLDS)
        assert verdict.go
        assert verdict.label == GO_LABEL
        assert verdict.metrics() == {
            VERDICT_GO_METRIC: 1.0,
            VERDICT_PASSED_METRIC: 5.0,
            VERDICT_TOTAL_METRIC: 5.0,
        }

    @pytest.mark.parametrize("metric", sorted(FAILING_ALONE))
    def test_each_criterion_failing_alone_is_no_go(self, metric: str) -> None:
        """V1: no criterion is optional."""
        verdict = evaluate_verdict(
            "T1_lshape", {**PASSING, metric: FAILING_ALONE[metric]}, THRESHOLDS
        )
        assert not verdict.go
        assert verdict.label == NO_GO_LABEL
        assert [item.metric for item in verdict.criteria if not item.passed] == [metric]

    @pytest.mark.parametrize(
        ("metric", "value", "passes"),
        [
            (PRIMARY_RATIO_METRIC, 0.98, True),
            (PRIMARY_DIVERGENCE_METRIC, 1.0, True),
            (ROBUST_MEDIAN_RATIO_METRIC, 1.0, False),
            (ROBUST_MEDIAN_RATIO_METRIC, 0.9999, True),
            (ROBUST_WINS_METRIC, 3.0, True),
            (PRIMARY_DOF_SAVING_METRIC, 0.0, False),
            (PRIMARY_DOF_SAVING_METRIC, 1e-9, True),
        ],
    )
    def test_the_pre_registered_boundaries(self, metric: str, value: float, passes: bool) -> None:
        """V2: <= 0.98, >= 1, < 1.0, >= 3, > 0 -- exactly."""
        verdict = evaluate_verdict("T", {**PASSING, metric: value}, THRESHOLDS)
        assert verdict.go is passes

    @pytest.mark.parametrize(
        ("metric", "value"),
        [
            (PRIMARY_RATIO_METRIC, -math.inf),
            (PRIMARY_RATIO_METRIC, math.nan),
            (PRIMARY_DOF_SAVING_METRIC, math.inf),
            (ROBUST_WINS_METRIC, math.inf),
        ],
    )
    def test_a_non_finite_measurement_fails_closed(self, metric: str, value: float) -> None:
        """V1: ``-inf <= 0.98`` is True in Python; the verdict must not believe it."""
        assert not evaluate_verdict("T", {**PASSING, metric: value}, THRESHOLDS).go

    def test_a_missing_measurement_fails(self) -> None:
        metrics = {k: v for k, v in PASSING.items() if k != PRIMARY_DOF_SAVING_METRIC}
        verdict = evaluate_verdict("T", metrics, THRESHOLDS)
        assert not verdict.go
        assert math.isnan(verdict.criteria[-1].value)

    def test_no_criterion_is_refused_not_a_vacuous_go(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            evaluate_verdict("T", PASSING, [])
        assert not Gate1Verdict(testbed="T", criteria=()).go

    def test_criterion_ids_follow_the_pre_registration(self) -> None:
        verdict = evaluate_verdict("T", PASSING, THRESHOLDS)
        assert [item.criterion for item in verdict.criteria] == ["C1", "C2", "C3", "C3", "C4"]
        assert set(CRITERION_OF_METRIC) == {item.metric for item in verdict.criteria}

    def test_summary_names_every_criterion_and_the_verdict(self) -> None:
        verdict = evaluate_verdict("T2_zshape", {**PASSING, PRIMARY_RATIO_METRIC: 1.0}, THRESHOLDS)
        summary = verdict.summary()
        assert summary.splitlines()[0] == "T2_zshape: NO-GO"
        assert "C1 primary_l2_ratio_vs_best_classical = 1 (needs <= 0.98): FAIL" in summary
        assert summary.count(": pass") == 4

    @settings(max_examples=150, deadline=None)
    @given(
        base=st.fixed_dictionaries(
            {
                PRIMARY_RATIO_METRIC: st.floats(0.5, 1.5),
                PRIMARY_DIVERGENCE_METRIC: st.integers(0, 30).map(float),
                ROBUST_MEDIAN_RATIO_METRIC: st.floats(0.5, 1.5),
                ROBUST_WINS_METRIC: st.integers(0, 5).map(float),
                PRIMARY_DOF_SAVING_METRIC: st.floats(-50.0, 50.0),
            }
        ),
        improvement=st.fixed_dictionaries(
            {
                PRIMARY_RATIO_METRIC: st.floats(0.0, 0.5),
                PRIMARY_DIVERGENCE_METRIC: st.integers(0, 5).map(float),
                ROBUST_MEDIAN_RATIO_METRIC: st.floats(0.0, 0.5),
                ROBUST_WINS_METRIC: st.integers(0, 5).map(float),
                PRIMARY_DOF_SAVING_METRIC: st.floats(0.0, 50.0),
            }
        ),
    )
    def test_the_verdict_is_monotone(
        self, base: dict[str, float], improvement: dict[str, float]
    ) -> None:
        """V3: lower ratios and higher counts / savings never turn GO into NO-GO."""
        better = {
            PRIMARY_RATIO_METRIC: base[PRIMARY_RATIO_METRIC] - improvement[PRIMARY_RATIO_METRIC],
            PRIMARY_DIVERGENCE_METRIC: base[PRIMARY_DIVERGENCE_METRIC]
            + improvement[PRIMARY_DIVERGENCE_METRIC],
            ROBUST_MEDIAN_RATIO_METRIC: base[ROBUST_MEDIAN_RATIO_METRIC]
            - improvement[ROBUST_MEDIAN_RATIO_METRIC],
            ROBUST_WINS_METRIC: base[ROBUST_WINS_METRIC] + improvement[ROBUST_WINS_METRIC],
            PRIMARY_DOF_SAVING_METRIC: base[PRIMARY_DOF_SAVING_METRIC]
            + improvement[PRIMARY_DOF_SAVING_METRIC],
        }
        if evaluate_verdict("T", base, THRESHOLDS).go:
            assert evaluate_verdict("T", better, THRESHOLDS).go
        passed_before = {
            i.metric for i in evaluate_verdict("T", base, THRESHOLDS).criteria if i.passed
        }
        passed_after = {
            i.metric for i in evaluate_verdict("T", better, THRESHOLDS).criteria if i.passed
        }
        assert passed_before <= passed_after


class TestCollectTestbedMetrics:
    def test_the_gated_values_come_from_the_runs(self) -> None:
        primary = scored(0.1, divergences=3)
        robust = [scored(0.2, seed=1), scored(0.3, seed=2), scored(0.1, seed=3)]
        metrics = collect_testbed_metrics(primary, robust, robust_win_ratio=1.0)
        assert metrics[PRIMARY_RATIO_METRIC] == pytest.approx(primary.reading.ratio)
        assert metrics[PRIMARY_DIVERGENCE_METRIC] == 3.0
        assert metrics[PRIMARY_DOF_SAVING_METRIC] == pytest.approx(primary.break_even.dof_saving)
        ratios = sorted(run.reading.ratio for run in robust)
        assert metrics[ROBUST_MEDIAN_RATIO_METRIC] == pytest.approx(ratios[1])
        assert metrics["robust_n_seeds"] == 3.0
        assert metrics["primary_tree_depth_max"] == 3.0
        assert metrics["primary_tree_depth_median"] == 2.0
        assert metrics["primary_tree_depth_min"] == 1.0
        assert metrics["primary_steps"] == 3.0
        assert metrics["classical_l2_at_matched_dof_dorfler_theta0.5"] == pytest.approx(
            primary.reading.classical_l2["dorfler_theta0.5"]
        )

    def test_a_seed_exactly_at_the_win_bar_is_not_a_win(self) -> None:
        """V4: 'ratio < 1.0' is strict, as pre-registered."""
        tie = scored(0.25)  # equal to greedy, the best classical arm here
        assert tie.reading.ratio == pytest.approx(1.0)
        better = scored(0.1)
        metrics = collect_testbed_metrics(better, [tie, better], robust_win_ratio=tie.reading.ratio)
        assert metrics[ROBUST_WINS_METRIC] == 1.0

    def test_k_star_is_published_only_when_finite(self) -> None:
        """V4: an infinite K* is a flag, never a number JSON would turn into null."""
        leaner = collect_testbed_metrics(scored(0.1), [scored(0.1)], robust_win_ratio=1.0)
        assert leaner[BREAK_EVEN_FINITE_METRIC] == 1.0
        assert {"primary_break_even_reuses_alpha1", "primary_break_even_reuses_alpha1.5"} <= set(
            leaner
        )
        same = collect_testbed_metrics(scored(0.25), [scored(0.25)], robust_win_ratio=1.0)
        assert same[BREAK_EVEN_FINITE_METRIC] == 0.0
        assert not any(key.startswith("primary_break_even_reuses_") for key in same)
        assert same[PRIMARY_DOF_SAVING_METRIC] == pytest.approx(0.0)

    def test_a_run_that_committed_nothing_has_zero_depth(self) -> None:
        metrics = collect_testbed_metrics(
            scored(0.1, depths=()), [scored(0.1, depths=())], robust_win_ratio=1.0
        )
        assert metrics["primary_tree_depth_max"] == 0.0
        assert metrics["robust_tree_depth_max"] == 0.0

    def test_no_robustness_seed_or_an_unmeasured_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one seed"):
            collect_testbed_metrics(scored(0.1), [], robust_win_ratio=1.0)
        with pytest.raises(ValueError, match="did not count"):
            collect_testbed_metrics(
                scored(0.1, divergences=None), [scored(0.1)], robust_win_ratio=1.0
            )


#: Five robustness runs against CLASSICAL (best arm greedy, L2 0.25 at N* = 400):
#: ratios 0.4, 0.8, 0.8, 1.2, 1.2 -- median 0.8, three seeds below 1.
ROBUST_PASSING = (0.1, 0.2, 0.2, 0.3, 0.3)
#: Every robustness run worse than greedy: median 1.2, no seed below 1.
ROBUST_FAILING = (0.3, 0.3, 0.3, 0.3, 0.3)


class TestVerdictOnSyntheticTrajectories:
    """V1 end to end: each criterion fails alone on real trajectories, not on metric dicts.

    The dict-level tests above prove the evaluator; these prove that the chain
    trajectories -> matched reading -> break-even -> metrics -> verdict lets
    every criterion fail on its own. Every expected value is hand-derived.
    """

    @pytest.mark.parametrize(
        ("case", "primary_error", "robust_errors", "divergences", "classical", "failing"),
        [
            # Primary 0.1 at N* = 400 against greedy's 0.25: ratio 0.4; it reaches
            # 0.25 at 100 * 4**(ln 0.25 / ln 0.1) = 230 DOF, greedy at 400.
            ("GO", 0.1, ROBUST_PASSING, 1, CLASSICAL, set()),
            # Ratio 0.248 / 0.25 = 0.992 > 0.98, but it still reaches greedy's
            # 0.25 first (396.9 < 400 DOF): C1 fails, C4 does not.
            ("C1", 0.248, ROBUST_PASSING, 1, CLASSICAL, {PRIMARY_RATIO_METRIC}),
            ("C2", 0.1, ROBUST_PASSING, 0, CLASSICAL, {PRIMARY_DIVERGENCE_METRIC}),
            (
                "C3",
                0.1,
                ROBUST_FAILING,
                1,
                CLASSICAL,
                {ROBUST_MEDIAN_RATIO_METRIC, ROBUST_WINS_METRIC},
            ),
            # Greedy (non-monotone) is best at N* with 0.3; primary 0.25 is ratio
            # 0.83, yet greedy reaches 0.3 at 135 DOF and primary only at 333.
            ("C4", 0.25, ROBUST_PASSING, 1, NON_MONOTONE_CLASSICAL, {PRIMARY_DOF_SAVING_METRIC}),
        ],
    )
    def test_each_criterion_fails_alone(
        self,
        case: str,
        primary_error: float,
        robust_errors: tuple[float, ...],
        divergences: int,
        classical: dict[str, ArenaTrajectory],
        failing: set[str],
    ) -> None:
        verdict = testbed_verdict(
            primary_error, robust_errors, divergences=divergences, classical=classical
        )
        assert {item.metric for item in verdict.criteria if not item.passed} == failing, case
        assert verdict.go is (not failing)
        assert {CRITERION_OF_METRIC[metric] for metric in failing} <= {case}

    def test_c4_reads_the_first_passage_of_a_non_monotone_greedy(self) -> None:
        """The C4 case's inputs, hand-solved: equal accuracy 0.3, passages 135.4 and 333.3."""
        run = scored(0.25, classical=NON_MONOTONE_CLASSICAL)
        assert run.reading.best_classical == GREEDY_LABEL
        assert run.reading.ratio == pytest.approx(0.25 / 0.3)
        result = run.break_even
        assert result.equal_accuracy_l2 == pytest.approx(0.3)
        greedy_dof = 100.0 * 1.5 ** (math.log(0.3) / math.log(0.2))
        assert result.reference.dof == pytest.approx(greedy_dof)
        assert result.candidate.dof == pytest.approx(1000.0 / 3.0)
        assert result.dof_saving == pytest.approx(greedy_dof - 1000.0 / 3.0)
        assert result.dof_saving < 0.0
        assert result.finite is False


class TestRequireFiniteMetrics:
    def test_finite_metrics_pass_through_as_floats(self) -> None:
        assert require_finite_metrics({"a": 1, "b": 0.5}) == {"a": 1.0, "b": 0.5}

    def test_every_non_finite_key_is_named(self) -> None:
        with pytest.raises(NonFiniteMetricError, match=r"\['a', 'c'\]"):
            require_finite_metrics({"a": math.inf, "b": 1.0, "c": math.nan})
