"""L-shaped Poisson AMR head-to-head: MCTS refinement vs Dörfler marking.

This module runs the thesis-critical controlled experiment: on the standard
L-shaped Poisson benchmark (reentrant-corner singularity), refine adaptively
with (a) classical Dörfler bulk marking and (b) an MCTS refinement policy,
using the **same** masked finite-difference solver, the **same** residual
error estimator, the **same** geometry mask and the **same** active-DOF
accounting. The only difference between the arms is the marking policy.

Two honest comparisons are reported (see ``specs/lshape_amr_compare.spec.md``
and the docstring on :func:`compare_ratios`):

* **Matched DOF** — isolates *policy quality* (MCTS search cost excluded):
  ``l2_error_ratio_at_matched_dof`` (lower is better for MCTS).
* **Matched wall-clock** — end-to-end efficiency (search cost included):
  ``error_per_dof_ratio_mcts_over_dorfler`` (the headline; lower is better).

MCTS runs ``n_simulations`` real solves per accepted refinement, so it is
*expected* to trail on wall-clock; the matched-DOF number is where a genuine
policy-quality win, if any, shows up. Both numbers are always emitted so a
matched-DOF win with a wall-clock loss is legible rather than hidden.

The refinement primitives (``_solve_on_grid_2d``, ``_compute_indicators_2d``,
``_dorfler_mark_2d``, ``_refine_grid``) are reused verbatim from
:class:`~src.research.baselines.DorflerAMRSolver` so the Dörfler arm is
provably the same algorithm callers already trust.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

from src.pde.games.lshape_amr import (
    EncodedValueEvaluator,
    GridSolveResult,
    LShapeAMRGame,
)
from src.research.baselines import AMRConfig, DorflerAMRSolver

# Re-exported for backwards-compatible imports (callers import these from here).
from src.research.seed_sweep import SEED_PRIME_STRIDE, resolved_seeds  # noqa: F401

if TYPE_CHECKING:
    from src.pde.config import PDEGameConfig
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Named numerical constants (no magic numbers)                                 #
# --------------------------------------------------------------------------- #

# Floor applied to any denominator (DOF, error, wall-clock, ratio) so ratios
# stay finite even for a degenerate (single-point / zero-time) trajectory.
RATIO_FLOOR: float = 1e-15

# ``SEED_PRIME_STRIDE`` and ``resolved_seeds`` are imported from src.research.seed_sweep
# (shared with transfer_baseline_compare) and re-exported above.


# --------------------------------------------------------------------------- #
# Geometry predicate                                                          #
# --------------------------------------------------------------------------- #


def lshape_inside_predicate(
    scale: float = 1.0,
) -> Callable[[NDArray[np.float64]], NDArray[np.bool_]]:
    r"""Return the L-shape membership predicate for ``[-s,s]^2 \ (0,s]x[-s,0)``.

    Mirrors :meth:`src.pde.geometry.LShapedDomain.contains_point`: a point is
    inside unless it lies in the removed bottom-right quadrant ``x>0 and y<0``.
    The bounding box is already enforced by the grid, so only the notch is
    excluded here.

    Args:
        scale: Domain half-width ``s`` (accepted for API symmetry; the notch
            rule is scale-independent).

    Returns:
        A vectorised predicate mapping ``(N, 2)`` coords to an ``(N,)`` mask.

    """

    def _inside(points: NDArray[np.float64]) -> NDArray[np.bool_]:
        x = points[:, 0]
        y = points[:, 1]
        removed = (x > 0.0) & (y < 0.0)
        return ~removed

    return _inside


# --------------------------------------------------------------------------- #
# Data contracts                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComparisonParams:
    """All tunables for one L-shape MCTS-vs-Dörfler comparison run.

    Every field is explicit and defaulted so the harness is reusable and
    testable without the PoC scenario layer. The PoC config
    (:class:`~src.poc.scenarios.lshape_amr_compare_config.LShapeAMRCompareConfig`)
    supplies these from validated Pydantic fields.
    """

    seed: int = 42
    scale: float = 1.0
    initial_side: int = 4
    max_dof: int = 400
    max_steps: int = 30
    marking_fraction: float = 0.5
    max_refinements: int = 30
    error_tolerance: float = 1e-6
    # MCTS arm
    n_candidate_elements: int = 6
    n_simulations: int = 12
    value_scale: float = 4.0
    c_puct: float = 1.4
    add_noise: bool = True
    # Backup semantics for the MCTS arm. L-shape AMR is a single-agent game,
    # so ``"single_agent"`` is correct. ``"legacy_adversarial"`` reproduces the
    # pre-fix two-player backup used to generate the originally committed
    # results/lshape_mcts_vs_dorfler.csv.
    search_mode: str = "single_agent"
    # Comparison
    n_seeds: int = 5

    def __post_init__(self) -> None:
        """Validate invariants that direct harness callers must also honour.

        Mirrors ``LShapeAMRCompareConfig``: ``initial_side`` must be even so the
        reentrant corner at the origin is a grid node, and ``n_seeds >= 1``.
        """
        if self.initial_side < 2 or self.initial_side % 2 != 0:
            raise ValueError(f"initial_side must be an even integer >= 2, got {self.initial_side}")
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        valid_modes = {"single_agent", "zero_sum", "legacy_adversarial"}
        if self.search_mode not in valid_modes:
            raise ValueError(
                f"search_mode must be one of {sorted(valid_modes)}, got {self.search_mode!r}"
            )


@dataclass
class TrajectoryPoint:
    """One recorded (level, DOF, error, wall-clock, solve-count) sample.

    ``n_solves`` is the *cumulative* number of real PDE solves the arm has
    performed up to and including this point. For Dörfler this is ~1 per level;
    for MCTS it also counts every ``apply_action`` solve replayed inside the
    ``n_simulations`` tree descents (``src/mcts/search.py`` clones the game per
    simulation and re-solves each edge of the selected path), so it is the
    honest *matched-compute* axis — cleaner than wall-clock, which additionally
    conflates growing linear-system size and Python/scipy overhead.
    """

    level: int
    n_dof: int
    l2_error: float
    wall_time_seconds: float
    n_solves: int = 0

    @property
    def error_per_dof(self) -> float:
        """Error divided by active DOF (efficiency at this point)."""
        return self.l2_error / max(self.n_dof, 1)


@dataclass
class ArmTrajectory:
    """A method's full refinement trajectory."""

    method: str
    points: list[TrajectoryPoint] = field(default_factory=list)

    def dofs(self) -> NDArray[np.float64]:
        """Active-DOF values along the trajectory."""
        return np.array([p.n_dof for p in self.points], dtype=np.float64)

    def errors(self) -> NDArray[np.float64]:
        """L2 errors along the trajectory."""
        return np.array([p.l2_error for p in self.points], dtype=np.float64)

    def wall_times(self) -> NDArray[np.float64]:
        """Cumulative wall-clock seconds along the trajectory."""
        return np.array([p.wall_time_seconds for p in self.points], dtype=np.float64)

    def solve_counts(self) -> NDArray[np.float64]:
        """Cumulative real-solve counts along the trajectory."""
        return np.array([p.n_solves for p in self.points], dtype=np.float64)

    def convergence_exponent(self) -> float:
        """Least-squares slope of ``log(l2)`` vs ``log(dof)`` (negative)."""
        d = self.dofs()
        e = self.errors()
        good = (d > 0) & (e > 0)
        if good.sum() < 2:
            return float("nan")
        slope = np.polyfit(np.log(d[good]), np.log(e[good]), 1)[0]
        return float(slope)


@dataclass
class ComparisonResult:
    """Outcome of a full comparison: both trajectories + the headline ratios."""

    dorfler: ArmTrajectory
    mcts: ArmTrajectory
    l2_error_ratio_at_matched_dof: float
    error_per_dof_ratio_mcts_over_dorfler: float
    matched_dof: float
    matched_wall_time_seconds: float
    dorfler_convergence_exponent: float
    mcts_convergence_exponent: float
    seed: int
    # Matched-compute (solve-count) comparison — the honest primary axis. A
    # value < 1 means MCTS reaches lower L2 for the same number of real solves.
    # Defaulted so existing positional/keyword constructions stay valid.
    l2_error_ratio_at_matched_solves: float = float("nan")
    matched_solves: float = 0.0

    def metrics(self) -> dict[str, float]:
        """Flat metric dict for the PoC scenario / baseline harness."""
        return {
            "l2_error_ratio_at_matched_dof": self.l2_error_ratio_at_matched_dof,
            "error_per_dof_ratio_mcts_over_dorfler": (self.error_per_dof_ratio_mcts_over_dorfler),
            "l2_error_ratio_at_matched_solves": self.l2_error_ratio_at_matched_solves,
            "matched_dof": self.matched_dof,
            "matched_solves": self.matched_solves,
            "matched_wall_time_seconds": self.matched_wall_time_seconds,
            "dorfler_convergence_exponent": self.dorfler_convergence_exponent,
            "mcts_convergence_exponent": self.mcts_convergence_exponent,
            "dorfler_final_dof": float(self.dorfler.points[-1].n_dof),
            "mcts_final_dof": float(self.mcts.points[-1].n_dof),
            "dorfler_final_l2": float(self.dorfler.points[-1].l2_error),
            "mcts_final_l2": float(self.mcts.points[-1].l2_error),
            "dorfler_final_solves": float(self.dorfler.points[-1].n_solves),
            "mcts_final_solves": float(self.mcts.points[-1].n_solves),
        }


# --------------------------------------------------------------------------- #
# Solve function                                                              #
# --------------------------------------------------------------------------- #


def _trapezoidal_weights(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dual-cell (trapezoidal) integration weights for a 1-D grid axis.

    Interior nodes get half of each adjacent spacing; the two endpoints get a
    single half-spacing. ``sum(weights) == axis[-1] - axis[0]``.
    """
    h = np.diff(axis)
    return np.concatenate([[h[0] / 2.0], 0.5 * (h[:-1] + h[1:]), [h[-1] / 2.0]])


def _area_weighted_l2(
    diff: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    in_mask: NDArray[np.bool_],
) -> float:
    """Area-weighted discrete L2 norm of ``diff`` over in-domain nodes.

    ``diff`` is the already-masked error vector (length ``in_mask.sum()``);
    ``in_mask`` selects the in-domain nodes from the full ``(len(xs), len(ys))``
    tensor grid (i-major). Returns ``sqrt(sum(w * diff^2) / sum(w))`` where
    ``w`` is each node's dual-cell area — the mesh-independent continuous norm.
    """
    if diff.size == 0:
        return float("nan")
    weights = np.outer(_trapezoidal_weights(xs), _trapezoidal_weights(ys)).ravel()
    w_in = weights[in_mask]
    total = float(np.sum(w_in))
    if total <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(w_in * diff**2) / total))


def make_solve_fn(
    operator: PDEOperator,
    inside: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], GridSolveResult]:
    """Build a masked tensor-product solve function for the L-shape.

    The returned callable reuses :class:`DorflerAMRSolver`'s static solve and
    residual-indicator primitives with the notch mask applied, and reports the
    L2 error over *in-domain* nodes and the active-DOF count — the matched
    accounting shared by both arms.

    Args:
        operator: The L-shaped Poisson operator.
        inside: Domain membership predicate.

    Returns:
        ``solve(xs, ys) -> GridSolveResult``.

    Raises:
        ImportError: If scipy is unavailable.

    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "L-shape AMR comparison requires scipy. Install with: pip install scipy"
        ) from exc

    def solve(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> GridSolveResult:
        u_full, grid = DorflerAMRSolver._solve_on_grid_2d(
            xs, ys, operator, sparse, spsolve, inside=inside
        )
        indicators = DorflerAMRSolver._compute_indicators_2d(
            xs, ys, u_full, operator, inside=inside
        )
        in_mask = np.asarray(inside(grid), dtype=bool)
        exact = np.asarray(
            operator.exact_solution(grid.astype(np.float32)), dtype=np.float64
        ).ravel()
        diff = (u_full.ravel() - exact)[in_mask]
        # Area-weighted (dual-cell / lumped-mass) discrete L2 norm. A plain
        # node-wise RMS over-weights the densely-refined singular region on the
        # non-uniform AMR grid; since MCTS and Dörfler cluster nodes
        # differently, that bias would distort the *ratio* itself. Weighting
        # each node by its trapezoidal dual-cell area (wx_i * wy_j) recovers the
        # mesh-independent continuous norm ||u_h - u||_L2 / sqrt(|Omega|).
        l2 = _area_weighted_l2(diff, xs, ys, in_mask)
        n_dof = int(in_mask.sum())
        return GridSolveResult(
            solution=u_full,
            grid=grid,
            l2_error=l2,
            n_dof=n_dof,
            indicators=indicators,
        )

    return solve


# --------------------------------------------------------------------------- #
# Arms                                                                        #
# --------------------------------------------------------------------------- #


class _SolveCounter:
    """Wraps a ``solve_fn``, counting every real solve it performs.

    For the MCTS arm the *same* counter instance is handed to
    :class:`LShapeAMRGame`, so it also tallies the solves replayed inside MCTS
    tree descents (``src/mcts/search.py`` clones the game per simulation and
    re-solves each edge on the selected path) — the true matched-compute cost.
    Delegation is transparent: the wrapped callable's return value is passed
    straight through.
    """

    def __init__(
        self,
        solve_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], GridSolveResult],
    ) -> None:
        self._solve_fn = solve_fn
        self.count = 0

    def __call__(self, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> GridSolveResult:
        self.count += 1
        return self._solve_fn(xs, ys)


def run_dorfler_arm(
    operator: PDEOperator,
    solve_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], GridSolveResult],
    params: ComparisonParams,
) -> ArmTrajectory:
    """Run the classical Dörfler-marking arm, recording its trajectory.

    Reuses :meth:`DorflerAMRSolver._dorfler_mark_2d` / ``_refine_grid`` verbatim
    so the marking is provably identical to the trusted baseline solver. The
    solve function is wrapped in a :class:`_SolveCounter` so each trajectory
    point records the cumulative real-solve count (~1 per level) — the shared
    matched-compute axis compared against the MCTS arm.
    """
    lo = float(np.asarray(operator.domain_min, dtype=np.float64)[0])
    hi = float(np.asarray(operator.domain_max, dtype=np.float64)[0])
    lo_y = float(np.asarray(operator.domain_min, dtype=np.float64)[1])
    hi_y = float(np.asarray(operator.domain_max, dtype=np.float64)[1])

    marker = DorflerAMRSolver(
        config=AMRConfig(
            marking_fraction=params.marking_fraction,
            max_refinements=params.max_refinements,
        )
    )

    xs = np.linspace(lo, hi, params.initial_side + 1, dtype=np.float64)
    ys = np.linspace(lo_y, hi_y, params.initial_side + 1, dtype=np.float64)

    counter = _SolveCounter(solve_fn)
    traj = ArmTrajectory(method="dorfler")
    t0 = time.perf_counter()
    for level in range(params.max_refinements + 1):
        solve = counter(xs, ys)
        traj.points.append(
            TrajectoryPoint(
                level=level,
                n_dof=solve.n_dof,
                l2_error=solve.l2_error,
                wall_time_seconds=time.perf_counter() - t0,
                n_solves=counter.count,
            )
        )
        if solve.n_dof >= params.max_dof or solve.l2_error < params.error_tolerance:
            break
        marked_x, marked_y = marker._dorfler_mark_2d(solve.indicators, xs, ys)
        xs = marker._refine_grid(xs, marked_x)
        ys = marker._refine_grid(ys, marked_y)
    logger.debug(
        "dorfler_arm_done",
        levels=len(traj.points),
        final_dof=traj.points[-1].n_dof,
        final_l2=traj.points[-1].l2_error,
    )
    return traj


def run_mcts_arm(
    operator: PDEOperator,
    solve_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], GridSolveResult],
    game_config: PDEGameConfig,
    params: ComparisonParams,
) -> ArmTrajectory:
    """Run the MCTS refinement arm through the real MCTS engine.

    Wall-clock accumulates the MCTS search cost (``n_simulations`` real solves
    per accepted refinement), so this trajectory is honestly end-to-end.
    """
    # Imported here so the module imports cleanly without the MCTS engine
    # available (e.g. for pure geometry / marker unit tests).
    from src.mcts.search import MCTS, SearchMode
    from src.pde.mcts_adapter import PDEGameAdapter

    np.random.seed(params.seed)

    # Count every real solve, including those replayed inside MCTS tree descents
    # (the game receives the counter, not the raw solve_fn).
    counter = _SolveCounter(solve_fn)
    game = LShapeAMRGame(
        operator,
        game_config,
        solve_fn=counter,
        initial_side=params.initial_side,
        n_candidate_elements=params.n_candidate_elements,
        value_scale=params.value_scale,
    )
    evaluator = EncodedValueEvaluator(n_actions=game.action_space_size)
    # L-shape AMR is single-agent: SINGLE_AGENT is the correct backup. The
    # legacy_adversarial escape hatch reproduces the pre-fix committed CSV.
    mcts = MCTS(
        evaluator=evaluator,
        n_simulations=params.n_simulations,
        c_puct=params.c_puct,
        search_mode=SearchMode(params.search_mode),
    )

    traj = ArmTrajectory(method="mcts")
    # Start the clock *before* PDEGameAdapter triggers the initial coarse solve,
    # so the level-0 point counts that solve for both its wall-clock and its
    # solve count — symmetric with the Dörfler arm (whose t0 precedes its first
    # solve). Previously the adapter (and its solve) ran before t0, undercounting
    # the MCTS arm's initial cost.
    t0 = time.perf_counter()
    adapter = PDEGameAdapter(game)
    traj.points.append(
        TrajectoryPoint(
            level=0,
            n_dof=adapter.state.dof,
            l2_error=adapter.state.error_estimate,
            wall_time_seconds=time.perf_counter() - t0,
            n_solves=counter.count,
        )
    )
    for level in range(1, params.max_steps + 1):
        if adapter.is_terminal() or not adapter.get_legal_actions():
            break
        action = mcts.get_action(adapter, temperature=0.0, add_noise=params.add_noise)
        adapter.apply_action(action)
        mcts.advance(action)
        traj.points.append(
            TrajectoryPoint(
                level=level,
                n_dof=adapter.state.dof,
                l2_error=adapter.state.error_estimate,
                wall_time_seconds=time.perf_counter() - t0,
                n_solves=counter.count,
            )
        )
        if adapter.state.dof >= params.max_dof:
            break
    logger.debug(
        "mcts_arm_done",
        levels=len(traj.points),
        final_dof=traj.points[-1].n_dof,
        final_l2=traj.points[-1].l2_error,
    )
    return traj


# --------------------------------------------------------------------------- #
# Comparison                                                                  #
# --------------------------------------------------------------------------- #


def _interp_log(
    x_query: float,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
) -> float:
    """Interpolate ``ys`` at ``x_query`` in log-log space (monotone ``xs``).

    Falls back to linear interpolation for non-positive values. ``xs`` need not
    be strictly increasing; duplicate x are de-duplicated keeping the last.
    """
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    # np.interp needs strictly increasing x. Collapse duplicate x, keeping the
    # last occurrence (finest-grid value), which requires no-op when unique.
    if len(np.unique(xs_s)) != len(xs_s):
        seen: dict[float, float] = {}
        for xv, yv in zip(xs_s, ys_s, strict=False):
            seen[float(xv)] = float(yv)
        xs_s = np.array(sorted(seen.keys()), dtype=np.float64)
        ys_s = np.array([seen[k] for k in xs_s], dtype=np.float64)
    if xs_s.size == 1:
        return float(ys_s[0])
    if np.all(xs_s > 0) and np.all(ys_s > 0):
        lx = np.log(xs_s)
        ly = np.log(ys_s)
        return float(np.exp(np.interp(np.log(max(x_query, RATIO_FLOOR)), lx, ly)))
    return float(np.interp(x_query, xs_s, ys_s))


def compare_ratios(
    dorfler: ArmTrajectory,
    mcts: ArmTrajectory,
    params: ComparisonParams,
) -> tuple[float, float, float, float]:
    """Compute the matched-DOF and matched-wall-clock comparison ratios.

    Returns:
        ``(l2_ratio_at_matched_dof, epd_ratio_at_matched_wall_clock,
        matched_dof, matched_wall_time)``. Ratios are ``mcts / dorfler`` so a
        value ``< 1`` means MCTS is better on that axis. Both are evaluated at
        the *largest common* checkpoint (no cherry-picking).

    """
    # Matched DOF: largest DOF both arms reached.
    matched_dof = float(min(dorfler.dofs().max(), mcts.dofs().max()))
    l2_dorfler = _interp_log(matched_dof, dorfler.dofs(), dorfler.errors())
    l2_mcts = _interp_log(matched_dof, mcts.dofs(), mcts.errors())
    l2_ratio = l2_mcts / max(l2_dorfler, RATIO_FLOOR)

    # Matched wall-clock: largest cumulative time both arms reached.
    matched_t = float(min(dorfler.wall_times().max(), mcts.wall_times().max()))
    epd_dorfler_curve = dorfler.errors() / np.maximum(dorfler.dofs(), 1.0)
    epd_mcts_curve = mcts.errors() / np.maximum(mcts.dofs(), 1.0)
    epd_dorfler = _interp_log(matched_t, dorfler.wall_times(), epd_dorfler_curve)
    epd_mcts = _interp_log(matched_t, mcts.wall_times(), epd_mcts_curve)
    epd_ratio = epd_mcts / max(epd_dorfler, RATIO_FLOOR)

    return l2_ratio, epd_ratio, matched_dof, matched_t


def _step_read(
    x_query: float,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
) -> float:
    """Piecewise-constant 'last observed value at or before ``x_query``' read.

    Unlike :func:`_interp_log`, this does **not** interpolate: it returns the
    ``ys`` value of the last point whose ``xs`` is ``<= x_query`` (a step/floor
    read). This is the faithful semantics for the **solve-count** compute axis,
    where an arm's committed L2 error is piecewise-constant between recorded
    points — it does not improve until a refinement step is *applied*, yet the
    solve count jumps by large amounts between points (MCTS burns
    ``n_simulations`` solves on clones per accepted step without improving the
    committed error). Interpolating would credit the arm with an L2 it never
    achieved at that budget. If ``x_query`` precedes the first point, the first
    value is returned (the arm's initial, pre-refinement error).
    """
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    idx = int(np.searchsorted(xs_s, x_query, side="right")) - 1
    idx = max(0, min(idx, len(ys_s) - 1))
    return float(ys_s[idx])


def l2_ratio_at_matched_solves(
    dorfler: ArmTrajectory,
    mcts: ArmTrajectory,
) -> tuple[float, float]:
    """L2-error ratio (mcts/dorfler) at the largest common real-solve count.

    This is the honest *matched-compute* comparison: both arms are read at the
    same total number of real solves, so it is not confounded by wall-clock
    implementation cost. A value ``< 1`` means MCTS reaches a lower L2 error for
    the same solve budget. Returns ``(ratio, matched_solves)``.

    The read is **stepwise** (:func:`_step_read`), not interpolated: the L2 error
    is piecewise-constant on the solve-count axis (it only drops when a
    refinement step is applied, while solve counts jump between recorded points),
    so each arm is read at the last point it had actually reached within the
    matched budget.
    """
    d_solves = dorfler.solve_counts()
    m_solves = mcts.solve_counts()
    matched = float(min(d_solves.max(), m_solves.max()))
    l2_dorfler = _step_read(matched, d_solves, dorfler.errors())
    l2_mcts = _step_read(matched, m_solves, mcts.errors())
    return l2_mcts / max(l2_dorfler, RATIO_FLOOR), matched


def run_comparison(
    operator: PDEOperator,
    game_config: PDEGameConfig,
    params: ComparisonParams,
    dorfler: ArmTrajectory | None = None,
) -> ComparisonResult:
    """Run both arms and assemble the :class:`ComparisonResult`.

    Args:
        operator: L-shaped Poisson operator.
        game_config: Config for :class:`LShapeAMRGame` (budget / winner
            thresholds / tolerance).
        params: All comparison tunables (fixed seed for reproducibility).
        dorfler: Optional pre-computed Dörfler arm. The Dörfler arm is fully
            deterministic and has **no** ``params.seed`` dependence, so a
            multi-seed sweep can solve it once and reuse it across seeds
            (:func:`run_multiseed_comparison`). When ``None`` it is computed
            here — the historical behaviour, preserved for direct callers.

    Returns:
        The full comparison with both trajectories and the headline ratios.

    """
    inside = lshape_inside_predicate(params.scale)
    solve_fn = make_solve_fn(operator, inside)

    logger.info(
        "lshape_comparison_start",
        seed=params.seed,
        max_dof=params.max_dof,
        n_simulations=params.n_simulations,
    )
    if dorfler is None:
        dorfler = run_dorfler_arm(operator, solve_fn, params)
    mcts = run_mcts_arm(operator, solve_fn, game_config, params)

    l2_ratio, epd_ratio, matched_dof, matched_t = compare_ratios(dorfler, mcts, params)
    l2_solve_ratio, matched_solves = l2_ratio_at_matched_solves(dorfler, mcts)

    result = ComparisonResult(
        dorfler=dorfler,
        mcts=mcts,
        l2_error_ratio_at_matched_dof=l2_ratio,
        error_per_dof_ratio_mcts_over_dorfler=epd_ratio,
        matched_dof=matched_dof,
        matched_wall_time_seconds=matched_t,
        dorfler_convergence_exponent=dorfler.convergence_exponent(),
        mcts_convergence_exponent=mcts.convergence_exponent(),
        seed=params.seed,
        l2_error_ratio_at_matched_solves=l2_solve_ratio,
        matched_solves=matched_solves,
    )
    logger.info(
        "lshape_comparison_done",
        l2_error_ratio_at_matched_dof=l2_ratio,
        error_per_dof_ratio_mcts_over_dorfler=epd_ratio,
        l2_error_ratio_at_matched_solves=l2_solve_ratio,
        matched_dof=matched_dof,
        matched_solves=matched_solves,
    )
    return result


# --------------------------------------------------------------------------- #
# Multi-seed aggregation                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class MultiSeedComparison:
    """Aggregate of ``n_seeds`` single-seed comparisons.

    A single MCTS run is high-variance, so the headline ratios are the
    **median** across seeds (robust to an unlucky seed), with the full per-seed
    spread recorded for honesty. Mirrors the ``noyron_basis`` / ``scaling_law``
    median-over-seeds convention.
    """

    per_seed: list[ComparisonResult]
    seeds: list[int]

    @property
    def l2_ratios(self) -> list[float]:
        """Per-seed matched-DOF L2 ratios."""
        return [r.l2_error_ratio_at_matched_dof for r in self.per_seed]

    @property
    def epd_ratios(self) -> list[float]:
        """Per-seed matched-wall-clock error-per-DOF ratios."""
        return [r.error_per_dof_ratio_mcts_over_dorfler for r in self.per_seed]

    @property
    def solve_ratios(self) -> list[float]:
        """Per-seed matched-solve-count L2 ratios (the matched-compute axis)."""
        return [r.l2_error_ratio_at_matched_solves for r in self.per_seed]

    @property
    def representative(self) -> ComparisonResult:
        """The per-seed result whose L2 ratio is the median — for the artifact."""
        ratios = self.l2_ratios
        order = sorted(range(len(ratios)), key=lambda i: ratios[i])
        return self.per_seed[order[len(order) // 2]]

    def metrics(self) -> dict[str, float]:
        """Headline (median) metrics plus per-seed spread and win fraction.

        The gated key ``l2_error_ratio_at_matched_dof`` is the **median** across
        seeds. Per-arm final DOF/L2 come from the representative (median) seed.
        """
        l2 = np.array(self.l2_ratios, dtype=np.float64)
        epd = np.array(self.epd_ratios, dtype=np.float64)
        solve = np.array(self.solve_ratios, dtype=np.float64)
        out = dict(self.representative.metrics())
        out.update(
            {
                "l2_error_ratio_at_matched_dof": float(np.median(l2)),
                "error_per_dof_ratio_mcts_over_dorfler": float(np.median(epd)),
                "l2_error_ratio_at_matched_solves": float(np.median(solve)),
                "l2_ratio_seed_min": float(np.min(l2)),
                "l2_ratio_seed_max": float(np.max(l2)),
                "l2_ratio_seed_std": float(np.std(l2)),
                "mcts_win_fraction": float(np.mean(l2 < 1.0)),
                "mcts_solve_win_fraction": float(np.mean(solve < 1.0)),
                "n_seeds": float(len(self.per_seed)),
            }
        )
        return out


def run_multiseed_comparison(
    operator: PDEOperator,
    game_config: PDEGameConfig,
    params: ComparisonParams,
) -> MultiSeedComparison:
    """Run ``params.n_seeds`` comparisons and aggregate the median headline.

    Seeds are derived from ``params.seed`` via :func:`resolved_seeds` so the run
    is fully reproducible. Each seed runs an independent :func:`run_comparison`.

    The Dörfler arm is deterministic (no ``params.seed`` dependence), so it is
    solved **once** here and reused across every seed — only the stochastic MCTS
    arm re-runs per seed. This leaves the gated matched-DOF L2 ratio (a function
    of ``n_dof``/``l2_error`` only) byte-identical while cutting the sweep cost of
    the redundant sparse solves.
    """
    seeds = resolved_seeds(params.seed, params.n_seeds)
    # Compute the seed-invariant Dörfler arm once and share it across seeds.
    inside = lshape_inside_predicate(params.scale)
    solve_fn = make_solve_fn(operator, inside)
    dorfler = run_dorfler_arm(operator, solve_fn, params)
    per_seed = [
        run_comparison(operator, game_config, replace(params, seed=s), dorfler=dorfler)
        for s in seeds
    ]
    result = MultiSeedComparison(per_seed=per_seed, seeds=seeds)
    logger.info(
        "lshape_multiseed_done",
        n_seeds=len(seeds),
        median_l2_ratio=float(np.median(result.l2_ratios)),
        win_fraction=float(np.mean(np.array(result.l2_ratios) < 1.0)),
    )
    return result


# --------------------------------------------------------------------------- #
# Artifact writers                                                            #
# --------------------------------------------------------------------------- #


def export_csv(result: ComparisonResult, output_path: str | Path) -> Path:
    """Write both arms' trajectories to CSV.

    Columns: ``problem, method, refinement_level, n_dof, l2_error,
    wall_time_seconds, n_solves, error_per_dof, seed``. ``n_solves`` is the
    cumulative real-solve count (the matched-compute axis). The matched
    comparisons are recomputable from these raw rows, so the CSV is the
    reproducible record.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
        for arm in (result.dorfler, result.mcts):
            for p in arm.points:
                writer.writerow(
                    [
                        "poisson_lshaped",
                        arm.method,
                        p.level,
                        p.n_dof,
                        f"{p.l2_error:.8e}",
                        f"{p.wall_time_seconds:.6f}",
                        p.n_solves,
                        f"{p.error_per_dof:.8e}",
                        result.seed,
                    ]
                )
    logger.info("lshape_csv_export", path=str(path))
    return path


def export_plot(result: ComparisonResult, output_path: str | Path) -> Path | None:
    """Render the error-vs-{DOF, solves, wall-clock} Pareto PNG (MCTS vs Dörfler).

    The middle panel is the honest matched-*compute* axis (real-solve count),
    unconfounded by wall-clock implementation cost. Matplotlib is imported lazily
    (``Agg`` backend) so importing this module never requires it. Returns
    ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - environment guard
        logger.warning("lshape_plot_skipped", reason="matplotlib unavailable")
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_dof, ax_solve, ax_time) = plt.subplots(1, 3, figsize=(16, 4.5))
    for arm, colour in ((result.dorfler, "#1f77b4"), (result.mcts, "#d62728")):
        ax_dof.plot(arm.dofs(), arm.errors(), "o-", color=colour, label=arm.method)
        ax_solve.plot(arm.solve_counts(), arm.errors(), "o-", color=colour, label=arm.method)
        ax_time.plot(arm.wall_times(), arm.errors(), "o-", color=colour, label=arm.method)

    ax_dof.set_xscale("log")
    ax_dof.set_yscale("log")
    ax_dof.set_xlabel("active DOF")
    ax_dof.set_ylabel("L2 error")
    ax_dof.set_title(
        f"Matched DOF\nL2 ratio (mcts/dorfler) @ {result.matched_dof:.0f} DOF = "
        f"{result.l2_error_ratio_at_matched_dof:.3f}"
    )
    ax_dof.legend()
    ax_dof.grid(True, which="both", alpha=0.3)

    ax_solve.set_xscale("log")
    ax_solve.set_yscale("log")
    ax_solve.set_xlabel("real solves (matched compute)")
    ax_solve.set_ylabel("L2 error")
    ax_solve.set_title(
        f"Matched compute\nL2 ratio (mcts/dorfler) @ {result.matched_solves:.0f} solves = "
        f"{result.l2_error_ratio_at_matched_solves:.3f}"
    )
    ax_solve.legend()
    ax_solve.grid(True, which="both", alpha=0.3)

    ax_time.set_yscale("log")
    ax_time.set_xlabel("wall-clock (s)")
    ax_time.set_ylabel("L2 error")
    ax_time.set_title(
        "Matched wall-clock\nerror/DOF ratio (mcts/dorfler) = "
        f"{result.error_per_dof_ratio_mcts_over_dorfler:.3f}"
    )
    ax_time.legend()
    ax_time.grid(True, which="both", alpha=0.3)

    fig.suptitle("L-shaped Poisson AMR: MCTS refinement vs Dörfler marking")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("lshape_plot_export", path=str(path))
    return path
