"""Measurements for the Gate 1 look-ahead-vs-greedy experiment.

Pure functions over finished trajectories: nothing here solves, refines or
searches. Every formula is the one pre-registered in
``specs/lookahead_vs_greedy.spec.md``, and the C1-C4 names below are the metric
keys the scenario config gates on (the verdict itself is
``src.research.lookahead_vs_greedy_verdict``):

* **Matched DOF** -- the largest DOF *every* arm reached. One MCTS run is read
  against all five classical arms (greedy, Dörfler at each θ, uniform) at that
  one DOF, so "beats the best classical policy" means the same mesh size for
  every arm. Readings use ``lshape_amr_compare._interp_log``, the log-log reader
  behind ``compare_trajectories``; it is not re-implemented here.
* **Break-even reuse count** ``K*`` -- how many downstream solves on the
  equal-accuracy mesh repay MCTS's extra construction wall-clock, with solve
  cost modelled as ``kappa_alpha * N**alpha`` and ``kappa_alpha`` calibrated
  from the greedy arm's measured per-step time. Finite iff MCTS reaches the
  equal accuracy with fewer DOF than greedy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from src.constants import DEFAULT_RATIO_FLOOR
from src.research.greedy_control import GREEDY_METHOD
from src.research.lshape_amr_compare import _interp_log
from src.research.substrates.config import RATIO_FLOOR

if TYPE_CHECKING:
    from src.research.amr_arena_types import ArenaTrajectory

# ---------------------------------------------------------------------------
# Arm labels (the CSV ``method`` column) and the gated metric names
# ---------------------------------------------------------------------------

#: Classical control arms, in the order they are read and reported.
GREEDY_LABEL: Final[str] = GREEDY_METHOD
UNIFORM_LABEL: Final[str] = "uniform"
DORFLER_LABEL_PREFIX: Final[str] = "dorfler_theta"
#: The two MCTS arms.
PRIMARY_LABEL: Final[str] = "mcts_primary"
ROBUST_LABEL: Final[str] = "mcts_robust"

#: C1 -- MCTS-primary L2 over the best classical arm at matched DOF.
PRIMARY_RATIO_METRIC: Final[str] = "primary_l2_ratio_vs_best_classical"
#: C2 -- MCTS-primary decisions that differ from greedy on the same state.
PRIMARY_DIVERGENCE_METRIC: Final[str] = "primary_decisions_diverging_from_greedy"
#: Recorded, ungated -- the same count over the robustness seeds. Root noise lives
#: in that arm's decision rule, so its departures from greedy are not search.
ROBUST_DIVERGENCE_MEDIAN_METRIC: Final[str] = "robust_decisions_diverging_from_greedy_median"
ROBUST_DIVERGENCE_MAX_METRIC: Final[str] = "robust_decisions_diverging_from_greedy_max"
#: C3 -- median over the robustness seeds of the same ratio.
ROBUST_MEDIAN_RATIO_METRIC: Final[str] = "robust_median_l2_ratio_vs_best_classical"
#: C3 -- robustness seeds whose ratio is below the win bar.
ROBUST_WINS_METRIC: Final[str] = "robust_seeds_below_best_classical"
#: C4 -- greedy DOF minus MCTS DOF at equal accuracy (> 0 iff ``K*`` is finite).
PRIMARY_DOF_SAVING_METRIC: Final[str] = "primary_dof_saving_vs_greedy"
#: ``1.0`` when every ``K*_alpha`` is finite, else ``0.0``: JSON keeps no infinity.
BREAK_EVEN_FINITE_METRIC: Final[str] = "primary_break_even_finite"

#: Relative slack when testing ``L2 <= target`` in a first passage. The target
#: is often an arm's own point read back through ``exp(log(e))``, which can land
#: one ulp below ``e``; without slack that arm would never "reach" itself.
FIRST_PASSAGE_RTOL: Final[float] = 1e-9
#: A per-solve time needs a measured step: the initial point plus one more.
MIN_CALIBRATION_POINTS: Final[int] = 2
#: Ratio denominators are floored exactly as ``compare_trajectories`` floors them.
RATIO_DENOMINATOR_FLOOR: Final[float] = max(RATIO_FLOOR, DEFAULT_RATIO_FLOOR)


def dorfler_label(theta: float) -> str:
    """CSV ``method`` of the Dörfler arm at ``theta`` (``dorfler_theta0.1``)."""
    return f"{DORFLER_LABEL_PREFIX}{theta:g}"


def alpha_suffix(alpha: float) -> str:
    """Metric-name suffix for a cost exponent (``alpha1``, ``alpha1.5``)."""
    return f"alpha{alpha:g}"


class ClassicalBudgetError(ValueError):
    """A classical arm stopped below the largest DOF a game-driven arm reached.

    The pre-registered classical budget exists so that every classical
    trajectory spans the matched DOF. If one does not, the matched DOF would
    silently shrink to that arm's last level -- a different experiment.
    """


class BreakEvenCalibrationError(ValueError):
    """The greedy arm's per-solve time cannot be calibrated (no step, or no time)."""


@dataclass(frozen=True)
class SearchStep:
    """One committed MCTS action and what the search looked like when it was chosen."""

    step: int
    action: int
    greedy_action: int
    #: ``src.mcts.node.subtree_depth`` of the root, after the search, before the action.
    tree_depth: int
    n_legal: int
    dof: int

    @property
    def diverged(self) -> bool:
        """Whether the committed action differs from greedy's on the same state."""
        return self.action != self.greedy_action


# ---------------------------------------------------------------------------
# Matched DOF against the best classical arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedReading:
    """One candidate run read against every classical arm at their shared matched DOF."""

    matched_dof: float
    candidate_l2: float
    classical_l2: Mapping[str, float]
    best_classical: str

    @property
    def best_classical_l2(self) -> float:
        """The lowest classical L2 at the matched DOF."""
        return self.classical_l2[self.best_classical]

    @property
    def ratio(self) -> float:
        """Candidate L2 over the best classical L2 (``< 1``: the candidate is better)."""
        return self.ratio_vs(self.best_classical)

    def ratio_vs(self, label: str) -> float:
        """Candidate L2 over classical arm ``label``'s L2, denominator floored."""
        return self.candidate_l2 / max(self.classical_l2[label], RATIO_DENOMINATOR_FLOOR)


def matched_reading(
    candidate: ArenaTrajectory,
    classical: Mapping[str, ArenaTrajectory],
) -> MatchedReading:
    """Read ``candidate`` and every classical arm at the largest DOF all of them reached.

    The best classical arm is the one with the lowest interpolated L2 there; a
    tie goes to the arm listed first.

    Raises:
        ValueError: If ``classical`` is empty or any trajectory has no points.

    """
    if not classical:
        raise ValueError("the best classical arm needs at least one classical trajectory")
    for label, trajectory in (("candidate", candidate), *classical.items()):
        if not trajectory.points:
            raise ValueError(f"{label} trajectory has no points; nothing to read at matched DOF")
    matched = float(min([candidate.dofs().max(), *(t.dofs().max() for t in classical.values())]))
    l2 = {
        label: _interp_log(matched, trajectory.dofs(), trajectory.errors())
        for label, trajectory in classical.items()
    }
    best = min(l2, key=l2.__getitem__)
    return MatchedReading(
        matched_dof=matched,
        candidate_l2=_interp_log(matched, candidate.dofs(), candidate.errors()),
        classical_l2=l2,
        best_classical=best,
    )


def assert_classical_span(
    sweep_arms: Mapping[str, ArenaTrajectory],
    game_driven: Sequence[ArenaTrajectory],
) -> None:
    """Every sweep-driven classical arm must reach the largest game-driven DOF.

    Raises:
        ValueError: If ``game_driven`` holds no point.
        ClassicalBudgetError: Naming each arm that stopped short.

    """
    reached = [float(t.dofs().max()) for t in game_driven if t.points]
    if not reached:
        raise ValueError("no game-driven trajectory has a point; nothing to span")
    reach = max(reached)
    short = {
        label: float(t.dofs().max()) if t.points else 0.0
        for label, t in sweep_arms.items()
        if not t.points or float(t.dofs().max()) < reach
    }
    if short:
        raise ClassicalBudgetError(
            f"classical arms stop below the largest game-driven DOF {reach:g}: {short}; "
            "the matched DOF would shrink to their last level -- raise classical_max_dof"
        )


# ---------------------------------------------------------------------------
# Break-even reuse count against greedy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """Where a trajectory first reaches an accuracy: its DOF and wall-clock there."""

    dof: float
    wall_time_seconds: float


def first_passage(trajectory: ArenaTrajectory, target_l2: float) -> Passage | None:
    """The first point at which ``trajectory`` reaches ``target_l2``, or ``None``.

    Within the first segment ``(k-1 -> k)`` with ``e_k <= target``, the crossing
    fraction ``t`` is taken in log-error space; DOF is interpolated log-log and
    wall-clock linearly at that same ``t``. Walking the segments in order (not
    interpolating error -> DOF globally) is what makes this correct for an L2
    trajectory that is not monotone.

    Raises:
        ValueError: If ``target_l2`` is not finite and positive, or any L2 or DOF
            on the trajectory is not positive (both are read in log space).

    """
    if not (math.isfinite(target_l2) and target_l2 > 0.0):
        raise ValueError(f"target_l2 must be finite and positive, got {target_l2!r}")
    dofs, errors, walls = trajectory.dofs(), trajectory.errors(), trajectory.wall_times()
    if bool(np.any(errors <= 0.0)) or bool(np.any(dofs <= 0.0)):
        raise ValueError("first passage reads log(L2) and log(DOF); both must be positive")
    reached = errors <= target_l2 * (1.0 + FIRST_PASSAGE_RTOL)
    if not bool(np.any(reached)):
        return None
    k = int(np.argmax(reached))
    if k == 0:
        return Passage(dof=float(dofs[0]), wall_time_seconds=float(walls[0]))
    log_before, log_after = math.log(errors[k - 1]), math.log(errors[k])
    fraction = (math.log(target_l2) - log_before) / (log_after - log_before)
    fraction = min(1.0, max(0.0, fraction))
    log_dof = math.log(dofs[k - 1]) + fraction * (math.log(dofs[k]) - math.log(dofs[k - 1]))
    wall = float(walls[k - 1]) + fraction * float(walls[k] - walls[k - 1])
    return Passage(dof=math.exp(log_dof), wall_time_seconds=wall)


def per_solve_cost_coefficient(reference: ArenaTrajectory, alpha: float) -> float:
    """``kappa_alpha``: the reference arm's measured seconds per modelled solve cost.

    ``(W_K - W_0) / sum_{k=1..K} N_k**alpha`` over the reference arm's committed
    steps, each of which is one refine and one solve.

    Raises:
        ValueError: If ``alpha`` is not finite and positive.
        BreakEvenCalibrationError: Fewer than one committed step, or no measured
            elapsed time.

    """
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise ValueError(f"alpha must be finite and positive, got {alpha!r}")
    points = reference.points
    if len(points) < MIN_CALIBRATION_POINTS:
        raise BreakEvenCalibrationError(
            f"calibrating the per-solve time needs a committed step; the reference arm "
            f"has {len(points)} point(s)"
        )
    elapsed = points[-1].wall_time_seconds - points[0].wall_time_seconds
    modelled = math.fsum(float(point.n_dof) ** alpha for point in points[1:])
    if not (math.isfinite(elapsed) and elapsed > 0.0 and modelled > 0.0):
        raise BreakEvenCalibrationError(
            f"cannot calibrate the per-solve time: elapsed={elapsed!r}s over modelled "
            f"cost {modelled!r}"
        )
    return elapsed / modelled


@dataclass(frozen=True)
class BreakEven:
    """``K*`` against a reference arm at their equal accuracy, per cost exponent."""

    equal_accuracy_l2: float
    candidate: Passage
    reference: Passage
    #: ``alpha -> kappa_alpha`` (seconds per ``DOF**alpha``).
    cost_coefficients: Mapping[float, float]
    #: ``alpha -> K*_alpha``; ``None`` is an infinite break-even (never pays back).
    reuses: Mapping[float, float | None]

    @property
    def dof_saving(self) -> float:
        """Reference DOF minus candidate DOF at equal accuracy (positive: candidate leaner)."""
        return self.reference.dof - self.candidate.dof

    @property
    def extra_wall_seconds(self) -> float:
        """Extra construction wall-clock the candidate spent reaching equal accuracy."""
        return max(0.0, self.candidate.wall_time_seconds - self.reference.wall_time_seconds)

    @property
    def finite(self) -> bool:
        """Whether every ``K*_alpha`` is finite."""
        return bool(self.reuses) and all(value is not None for value in self.reuses.values())


def reuses_to_break_even(
    extra_wall_seconds: float,
    coefficient: float,
    candidate_dof: float,
    dof_saving: float,
    alpha: float,
) -> float | None:
    """``K*_alpha``, or ``None`` when the candidate saves no DOF (it never pays back).

    The per-reuse saving ``kappa * (N_ref**alpha - N_cand**alpha)`` is computed as
    ``kappa * N_cand**alpha * expm1(alpha * log1p(saving / N_cand))``, which stays
    positive for any positive saving where the direct difference could round to 0.
    """
    if dof_saving <= 0.0:
        return None
    per_reuse = (
        coefficient
        * candidate_dof**alpha
        * math.expm1(alpha * math.log1p(dof_saving / candidate_dof))
    )
    if not (math.isfinite(per_reuse) and per_reuse > 0.0):
        return None
    reuses = extra_wall_seconds / per_reuse
    return reuses if math.isfinite(reuses) else None


def break_even(
    candidate: ArenaTrajectory,
    reference: ArenaTrajectory,
    *,
    matched_dof: float,
    alphas: Sequence[float],
) -> BreakEven:
    """``K*_alpha`` of ``candidate`` against ``reference`` at their equal accuracy.

    The equal accuracy is the *worse* of the two L2 values at ``matched_dof``,
    so both arms provably reach it inside their recorded trajectories and the
    DOF saving is never extrapolated.

    Raises:
        ValueError: If ``alphas`` is empty, or an arm does not reach the equal
            accuracy (impossible for trajectories that both span ``matched_dof``).
        BreakEvenCalibrationError: See :func:`per_solve_cost_coefficient`.

    """
    if not alphas:
        raise ValueError("break_even needs at least one cost exponent alpha")
    target = max(
        _interp_log(matched_dof, candidate.dofs(), candidate.errors()),
        _interp_log(matched_dof, reference.dofs(), reference.errors()),
    )
    reached_candidate = first_passage(candidate, target)
    reached_reference = first_passage(reference, target)
    if reached_candidate is None or reached_reference is None:
        raise ValueError(
            f"an arm never reaches the equal accuracy {target!r} at or before matched DOF "
            f"{matched_dof!r}; both trajectories must span the matched DOF"
        )
    coefficients = {alpha: per_solve_cost_coefficient(reference, alpha) for alpha in alphas}
    saving = reached_reference.dof - reached_candidate.dof
    extra = max(0.0, reached_candidate.wall_time_seconds - reached_reference.wall_time_seconds)
    return BreakEven(
        equal_accuracy_l2=target,
        candidate=reached_candidate,
        reference=reached_reference,
        cost_coefficients=coefficients,
        reuses={
            alpha: reuses_to_break_even(
                extra, coefficients[alpha], reached_candidate.dof, saving, alpha
            )
            for alpha in alphas
        },
    )


__all__ = [
    "BREAK_EVEN_FINITE_METRIC",
    "DORFLER_LABEL_PREFIX",
    "FIRST_PASSAGE_RTOL",
    "GREEDY_LABEL",
    "MIN_CALIBRATION_POINTS",
    "PRIMARY_DIVERGENCE_METRIC",
    "PRIMARY_DOF_SAVING_METRIC",
    "PRIMARY_LABEL",
    "PRIMARY_RATIO_METRIC",
    "RATIO_DENOMINATOR_FLOOR",
    "ROBUST_DIVERGENCE_MAX_METRIC",
    "ROBUST_DIVERGENCE_MEDIAN_METRIC",
    "ROBUST_LABEL",
    "ROBUST_MEDIAN_RATIO_METRIC",
    "ROBUST_WINS_METRIC",
    "UNIFORM_LABEL",
    "BreakEven",
    "BreakEvenCalibrationError",
    "ClassicalBudgetError",
    "MatchedReading",
    "Passage",
    "SearchStep",
    "alpha_suffix",
    "assert_classical_span",
    "break_even",
    "dorfler_label",
    "first_passage",
    "matched_reading",
    "per_solve_cost_coefficient",
    "reuses_to_break_even",
]
