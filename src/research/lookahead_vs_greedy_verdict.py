"""Gate 1 scoring: one testbed's metric set and its GO / NO-GO verdict.

The verdict is the pre-registered ``MetricThreshold`` list
(``LookaheadVsGreedyConfig.get_default_thresholds()``), evaluated exactly once
here. The PoC scenario's pass/fail evaluates the same objects, so the recorded
verdict and the scenario status cannot disagree; neither holds a second copy of
a threshold. A missing or non-finite measurement fails its criterion (fail
closed), and a threshold list that is empty is refused rather than read as a
vacuous GO.

Recorded metrics must be finite: the run sidecar is JSON, where an infinite
``K*`` or a NaN ratio would serialise as ``null`` and fail to load back.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from src.research.lookahead_vs_greedy_metrics import (
    BREAK_EVEN_FINITE_METRIC,
    GREEDY_LABEL,
    PRIMARY_DIVERGENCE_METRIC,
    PRIMARY_DOF_SAVING_METRIC,
    PRIMARY_RATIO_METRIC,
    ROBUST_DIVERGENCE_MAX_METRIC,
    ROBUST_DIVERGENCE_MEDIAN_METRIC,
    ROBUST_MEDIAN_RATIO_METRIC,
    ROBUST_WINS_METRIC,
    alpha_suffix,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.poc.config import MetricThreshold
    from src.research.amr_arena_types import ArenaTrajectory
    from src.research.lookahead_vs_greedy_metrics import (
        BreakEven,
        MatchedReading,
        SearchStep,
    )
    from src.research.mcts_classical_amr_arena import MatchedComparison

#: ``1.0`` on GO, ``0.0`` on NO-GO.
VERDICT_GO_METRIC: Final[str] = "verdict_go"
VERDICT_PASSED_METRIC: Final[str] = "verdict_criteria_passed"
VERDICT_TOTAL_METRIC: Final[str] = "verdict_criteria_total"
GO_LABEL: Final[str] = "GO"
NO_GO_LABEL: Final[str] = "NO-GO"

#: Pre-registered criterion id of each gated metric (C3 has two parts).
CRITERION_OF_METRIC: Final[dict[str, str]] = {
    PRIMARY_RATIO_METRIC: "C1",
    PRIMARY_DIVERGENCE_METRIC: "C2",
    ROBUST_MEDIAN_RATIO_METRIC: "C3",
    ROBUST_WINS_METRIC: "C3",
    PRIMARY_DOF_SAVING_METRIC: "C4",
}

#: Summary statistics of the realized tree depth.
_DEPTH_STATS: Final[tuple[str, ...]] = ("max", "median", "min")


class NonFiniteMetricError(ValueError):
    """A recorded metric is NaN or infinite, which the JSON sidecar cannot hold."""


@dataclass(frozen=True)
class ScoredRun:
    """One MCTS run with every reading the pre-registration asks for."""

    label: str
    seed: int
    add_noise: bool
    trajectory: ArenaTrajectory
    steps: tuple[SearchStep, ...]
    reading: MatchedReading
    versus_greedy: MatchedComparison
    versus_best: MatchedComparison
    break_even: BreakEven

    @property
    def divergences(self) -> int:
        """``decisions_diverging_from_greedy`` as the run counted it.

        Raises:
            ValueError: The run did not measure it (``None`` means unmeasured,
                never zero).

        """
        counted = self.trajectory.decisions_diverging_from_greedy
        if counted is None:
            raise ValueError(f"{self.label} seed {self.seed} did not count its divergences")
        return counted

    def tree_depths(self) -> NDArray[np.float64]:
        """Realized tree depth before each committed action."""
        return np.array([step.tree_depth for step in self.steps], dtype=np.float64)


def _depth_metrics(prefix: str, depths: NDArray[np.float64]) -> dict[str, float]:
    """Max / median / min realized depth; all ``0`` for a run that committed nothing."""
    if depths.size == 0:
        return {f"{prefix}_tree_depth_{stat}": 0.0 for stat in _DEPTH_STATS}
    return {
        f"{prefix}_tree_depth_max": float(np.max(depths)),
        f"{prefix}_tree_depth_median": float(np.median(depths)),
        f"{prefix}_tree_depth_min": float(np.min(depths)),
    }


def _break_even_metrics(result: BreakEven) -> dict[str, float]:
    """The C4 inputs and ``K*`` per exponent (``K*`` only when it is finite)."""
    metrics = {
        PRIMARY_DOF_SAVING_METRIC: result.dof_saving,
        BREAK_EVEN_FINITE_METRIC: 1.0 if result.finite else 0.0,
        "primary_equal_accuracy_l2": result.equal_accuracy_l2,
        "primary_dof_at_equal_accuracy": result.candidate.dof,
        "greedy_dof_at_equal_accuracy": result.reference.dof,
        "primary_wall_seconds_at_equal_accuracy": result.candidate.wall_time_seconds,
        "greedy_wall_seconds_at_equal_accuracy": result.reference.wall_time_seconds,
    }
    for alpha, coefficient in result.cost_coefficients.items():
        metrics[f"break_even_cost_coefficient_{alpha_suffix(alpha)}"] = coefficient
    for alpha, reuses in result.reuses.items():
        if reuses is not None:
            metrics[f"primary_break_even_reuses_{alpha_suffix(alpha)}"] = reuses
    return metrics


def _primary_metrics(primary: ScoredRun) -> dict[str, float]:
    """C1, C2, C4 and the primary run's ungated readings."""
    reading = primary.reading
    metrics = {
        PRIMARY_RATIO_METRIC: reading.ratio,
        PRIMARY_DIVERGENCE_METRIC: float(primary.divergences),
        "primary_matched_dof": reading.matched_dof,
        "primary_l2_at_matched_dof": reading.candidate_l2,
        "primary_best_classical_l2_at_matched_dof": reading.best_classical_l2,
        "primary_l2_ratio_vs_greedy": reading.ratio_vs(GREEDY_LABEL),
        "primary_steps": float(len(primary.steps)),
        "primary_l2_ratio_at_matched_solves_vs_greedy": (
            primary.versus_greedy.l2_error_ratio_at_matched_solves
        ),
        "primary_error_per_dof_ratio_at_matched_wall_clock_vs_greedy": (
            primary.versus_greedy.error_per_dof_ratio_at_matched_wall_clock
        ),
        "primary_l2_ratio_at_matched_solves_vs_best_classical": (
            primary.versus_best.l2_error_ratio_at_matched_solves
        ),
        "primary_error_per_dof_ratio_at_matched_wall_clock_vs_best_classical": (
            primary.versus_best.error_per_dof_ratio_at_matched_wall_clock
        ),
        **{
            f"classical_l2_at_matched_dof_{label}": value
            for label, value in reading.classical_l2.items()
        },
        **_depth_metrics("primary", primary.tree_depths()),
    }
    metrics.update(_break_even_metrics(primary.break_even))
    return metrics


def _robust_metrics(robust: Sequence[ScoredRun], *, robust_win_ratio: float) -> dict[str, float]:
    """C3 (median ratio and win count) and the robustness arm's spread.

    Raises:
        ValueError: If ``robust`` is empty.

    """
    if not robust:
        raise ValueError("the robustness arm needs at least one seed")
    ratios = np.array([run.reading.ratio for run in robust], dtype=np.float64)
    divergences = np.array([run.divergences for run in robust], dtype=np.float64)
    savings = np.array([run.break_even.dof_saving for run in robust], dtype=np.float64)
    depths = np.concatenate([run.tree_depths() for run in robust])
    return {
        ROBUST_MEDIAN_RATIO_METRIC: float(np.median(ratios)),
        ROBUST_WINS_METRIC: float(np.count_nonzero(ratios < robust_win_ratio)),
        "robust_n_seeds": float(len(robust)),
        "robust_l2_ratio_min": float(np.min(ratios)),
        "robust_l2_ratio_max": float(np.max(ratios)),
        ROBUST_DIVERGENCE_MEDIAN_METRIC: float(np.median(divergences)),
        ROBUST_DIVERGENCE_MAX_METRIC: float(np.max(divergences)),
        "robust_tree_depth_max": float(np.max(depths)) if depths.size else 0.0,
        "robust_dof_saving_vs_greedy_median": float(np.median(savings)),
    }


def collect_testbed_metrics(
    primary: ScoredRun,
    robust: Sequence[ScoredRun],
    *,
    robust_win_ratio: float,
) -> dict[str, float]:
    """Every pre-registered reading of one testbed, gated and ungated."""
    return {
        **_primary_metrics(primary),
        **_robust_metrics(robust, robust_win_ratio=robust_win_ratio),
    }


def require_finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return ``metrics`` as a dict, refusing any NaN or infinite value.

    Raises:
        NonFiniteMetricError: Naming every offending key.

    """
    bad = sorted(name for name, value in metrics.items() if not math.isfinite(float(value)))
    if bad:
        raise NonFiniteMetricError(
            f"non-finite metrics {bad}: the JSON sidecar would record them as null and "
            "fail to load back; a non-finite reading is an invalid run, not a result"
        )
    return {name: float(value) for name, value in metrics.items()}


@dataclass(frozen=True)
class CriterionResult:
    """One pre-registered criterion: what it reads, the bar, the value, the outcome."""

    criterion: str
    metric: str
    operator: str
    threshold: float
    value: float
    passed: bool


@dataclass(frozen=True)
class Gate1Verdict:
    """GO iff every criterion passed (and there is at least one)."""

    testbed: str
    criteria: tuple[CriterionResult, ...]

    @property
    def go(self) -> bool:
        """The go/no-go decision."""
        return bool(self.criteria) and all(item.passed for item in self.criteria)

    @property
    def label(self) -> str:
        """``GO`` or ``NO-GO``."""
        return GO_LABEL if self.go else NO_GO_LABEL

    def metrics(self) -> dict[str, float]:
        """The verdict as recordable numbers."""
        return {
            VERDICT_GO_METRIC: 1.0 if self.go else 0.0,
            VERDICT_PASSED_METRIC: float(sum(item.passed for item in self.criteria)),
            VERDICT_TOTAL_METRIC: float(len(self.criteria)),
        }

    def summary(self) -> str:
        """One line per criterion, prefixed by the verdict."""
        lines = [f"{self.testbed}: {self.label}"]
        lines.extend(
            f"  {item.criterion} {item.metric} = {item.value:.6g} "
            f"(needs {item.operator} {item.threshold:g}): {'pass' if item.passed else 'FAIL'}"
            for item in self.criteria
        )
        return "\n".join(lines)


def evaluate_verdict(
    testbed: str,
    metrics: Mapping[str, float],
    thresholds: Sequence[MetricThreshold],
) -> Gate1Verdict:
    """Evaluate each pre-registered threshold once; fail closed on a missing or non-finite value.

    Raises:
        ValueError: If ``thresholds`` is empty -- a verdict with no criterion
            would be a vacuous GO.

    """
    if not thresholds:
        raise ValueError("a Gate 1 verdict needs at least one pre-registered criterion")
    results = []
    for threshold in thresholds:
        value = float(metrics.get(threshold.name, math.nan))
        results.append(
            CriterionResult(
                criterion=CRITERION_OF_METRIC.get(threshold.name, threshold.name),
                metric=threshold.name,
                operator=threshold.operator,
                threshold=float(threshold.value),
                value=value,
                passed=math.isfinite(value) and threshold.evaluate(value),
            )
        )
    return Gate1Verdict(testbed=testbed, criteria=tuple(results))


__all__ = [
    "CRITERION_OF_METRIC",
    "GO_LABEL",
    "NO_GO_LABEL",
    "VERDICT_GO_METRIC",
    "VERDICT_PASSED_METRIC",
    "VERDICT_TOTAL_METRIC",
    "CriterionResult",
    "Gate1Verdict",
    "NonFiniteMetricError",
    "ScoredRun",
    "collect_testbed_metrics",
    "evaluate_verdict",
    "require_finite_metrics",
]
