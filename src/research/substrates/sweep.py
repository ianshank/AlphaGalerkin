"""Substrate-agnostic refinement sweep + log-log convergence-rate fitting.

The adequacy gate (``tests/research/test_amr_arena_interpretability.py``, AC7)
and any future MCTS-vs-classical arena both need the same three things: drive a
``RefinementSubstrate`` through repeated solve/mark/refine steps, fit a
convergence rate over a pinned DOF window, and compare two arms' rates. Putting
them here rather than in the gate's test file is what stops the arena change
from growing a second, subtly-different copy -- the failure mode that made
``dorfler_mark`` necessary in the first place.

Everything is generic over the ``RefinementSubstrate`` Protocol: nothing here
knows whether the mesh underneath is a tensor-product grid or a ``scikit-fem``
triangulation, which is exactly what lets the gate assert that the *same*
measurement passes on one substrate and fails on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog

from src.research.substrates.config import AREA_FLOOR, RATE_FIT_MIN_POINTS, RATIO_FLOOR

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.refinement.substrate import RefinementSubstrate

logger = structlog.get_logger(__name__)

#: How a sweep chooses which units to refine at each level.
#: ``"adaptive"`` delegates to the substrate's own ``mark`` (Dörfler bulk
#: chasing); ``"uniform"`` marks every refinable unit, which is the control
#: arm the adequacy gate measures against.
MarkingPolicy = Literal["adaptive", "uniform"]


@dataclass(frozen=True)
class SweepPoint:
    """One (level, cost, error) reading along a refinement trajectory."""

    level: int
    n_dof: int
    n_dof_free: int
    n_units: int
    l2_error: float


@dataclass(frozen=True)
class RateSeparation:
    """The adequacy gate's actual measurement: two arms' rates and their ratio.

    Attributes:
        adaptive_rate: Log-log slope of the adaptive arm (negative; steeper is
            better).
        uniform_rate: Log-log slope of the uniform control arm.
        error_ratio_at_matched_dof: ``adaptive_error / uniform_error``
            interpolated at the largest DOF both arms reach. Below 1.0 means
            adaptive wins.
        matched_dof: The DOF count that ratio was read at.
        n_adaptive_points: Points the adaptive fit used (>= RATE_FIT_MIN_POINTS).
        n_uniform_points: Points the uniform fit used (>= RATE_FIT_MIN_POINTS).

    """

    adaptive_rate: float
    uniform_rate: float
    error_ratio_at_matched_dof: float
    matched_dof: float
    n_adaptive_points: int
    n_uniform_points: int


class InsufficientSweepPointsError(ValueError):
    """Raised when a rate fit has fewer than ``RATE_FIT_MIN_POINTS`` points.

    Fitting a slope through one or two points always "succeeds" and always
    means nothing; refusing loudly is what stops a too-short sweep from being
    read as a convergence result.
    """


def run_refinement_sweep(
    substrate: RefinementSubstrate[Any],
    *,
    policy: MarkingPolicy,
    theta: float,
    max_levels: int,
    max_dof: int,
    error_tolerance: float = 0.0,
) -> list[SweepPoint]:
    """Drive ``substrate`` through solve -> mark -> refine, recording each level.

    Args:
        substrate: Any ``RefinementSubstrate`` implementation.
        policy: ``"adaptive"`` (the substrate's own Dörfler marking) or
            ``"uniform"`` (mark every refinable unit) -- the control arm.
        theta: Dörfler bulk fraction, passed to ``substrate.mark``. Ignored
            under ``policy="uniform"``.
        max_levels: Hard ceiling on refinement levels (a runaway guard;
            uniform refinement multiplies DOF each level).
        max_dof: Stop once a level's ``n_dof`` reaches this.
        error_tolerance: Stop early if ``l2_error`` drops below this. Defaults
            to 0.0, i.e. never.

    Returns:
        One ``SweepPoint`` per level actually solved, in order.

    """
    log = logger.bind(
        policy=policy,
        theta=theta,
        max_dof=max_dof,
        substrate=substrate.describe().get("kind", "unknown"),
    )
    log.info("refinement_sweep_start", max_levels=max_levels)

    mesh = substrate.initial_mesh()
    points: list[SweepPoint] = []

    for level in range(max_levels):
        result = substrate.solve(mesh)
        n_units = substrate.n_units(mesh)
        points.append(
            SweepPoint(
                level=level,
                n_dof=result.n_dof,
                n_dof_free=result.n_dof_free,
                n_units=n_units,
                l2_error=result.l2_error,
            )
        )
        log.debug(
            "refinement_sweep_level",
            level=level,
            n_dof=result.n_dof,
            n_units=n_units,
            l2_error=result.l2_error,
        )

        if result.n_dof >= max_dof:
            log.info("refinement_sweep_stop", reason="max_dof", level=level, n_dof=result.n_dof)
            break
        if result.l2_error < error_tolerance:
            log.info(
                "refinement_sweep_stop",
                reason="error_tolerance",
                level=level,
                l2_error=result.l2_error,
            )
            break

        if policy == "uniform":
            marked = substrate.refinable_mask(mesh)
        else:
            marked = substrate.mark(result.indicators, theta)

        if not bool(np.any(marked)):
            log.warning("refinement_sweep_stop", reason="nothing_marked", level=level)
            break

        mesh = substrate.refine(mesh, marked)
    else:
        log.info("refinement_sweep_stop", reason="max_levels", level=max_levels - 1)

    log.info(
        "refinement_sweep_done",
        n_levels=len(points),
        final_dof=points[-1].n_dof if points else 0,
        final_l2=points[-1].l2_error if points else float("nan"),
    )
    return points


def fit_log_log_rate(
    points: list[SweepPoint],
    dof_range: tuple[float, float],
) -> tuple[float, int]:
    """Least-squares slope of ``log(l2_error)`` vs ``log(n_dof)`` over a window.

    Args:
        points: A sweep trajectory.
        dof_range: ``(low, high)`` inclusive DOF window to fit over. Restricting
            the window is what makes the rate an *asymptotic* statement rather
            than an average over the pre-asymptotic levels where neither arm has
            separated yet.

    Returns:
        ``(slope, n_points_used)``. Slope is negative for a converging method.

    Raises:
        InsufficientSweepPointsError: If fewer than ``RATE_FIT_MIN_POINTS``
            usable points fall inside ``dof_range``.

    """
    dofs = np.array([p.n_dof for p in points], dtype=np.float64)
    errors = np.array([p.l2_error for p in points], dtype=np.float64)

    low, high = dof_range
    usable = (dofs > 0) & (errors > 0) & (dofs >= low) & (dofs <= high)
    n_used = int(usable.sum())

    if n_used < RATE_FIT_MIN_POINTS:
        raise InsufficientSweepPointsError(
            f"rate fit needs >= {RATE_FIT_MIN_POINTS} points inside DOF window "
            f"{dof_range}, got {n_used} (sweep DOF range was "
            f"[{dofs.min() if dofs.size else float('nan')}, "
            f"{dofs.max() if dofs.size else float('nan')}]) -- a slope through "
            "fewer points always 'succeeds' and always means nothing"
        )

    slope = float(np.polyfit(np.log(dofs[usable]), np.log(errors[usable]), 1)[0])
    logger.debug("rate_fit", slope=slope, n_points=n_used, dof_range=dof_range)
    return slope, n_used


def _interp_error_at_dof(points: list[SweepPoint], target_dof: float) -> float:
    """Log-log interpolate a trajectory's error at ``target_dof``."""
    dofs = np.array([p.n_dof for p in points], dtype=np.float64)
    errors = np.array([p.l2_error for p in points], dtype=np.float64)
    good = (dofs > 0) & (errors > 0)
    return float(
        np.exp(
            np.interp(
                np.log(max(target_dof, RATIO_FLOOR)),
                np.log(dofs[good]),
                np.log(errors[good]),
            )
        )
    )


def measure_rate_separation(
    adaptive: list[SweepPoint],
    uniform: list[SweepPoint],
    dof_range: tuple[float, float],
) -> RateSeparation:
    """Fit both arms and compare them at matched DOF.

    The matched-DOF reading is taken at the largest DOF *both* arms reach, so
    neither arm is credited for a level the other never got to.
    """
    adaptive_rate, n_adaptive = fit_log_log_rate(adaptive, dof_range)
    uniform_rate, n_uniform = fit_log_log_rate(uniform, dof_range)

    matched_dof = float(
        min(
            max(p.n_dof for p in adaptive),
            max(p.n_dof for p in uniform),
        )
    )
    adaptive_error = _interp_error_at_dof(adaptive, matched_dof)
    uniform_error = _interp_error_at_dof(uniform, matched_dof)
    ratio = adaptive_error / max(uniform_error, RATIO_FLOOR)

    separation = RateSeparation(
        adaptive_rate=adaptive_rate,
        uniform_rate=uniform_rate,
        error_ratio_at_matched_dof=ratio,
        matched_dof=matched_dof,
        n_adaptive_points=n_adaptive,
        n_uniform_points=n_uniform,
    )
    logger.info(
        "rate_separation",
        adaptive_rate=adaptive_rate,
        uniform_rate=uniform_rate,
        ratio_at_matched_dof=ratio,
        matched_dof=matched_dof,
    )
    return separation


def warn_on_degenerate_units(
    substrate: RefinementSubstrate[Any],
    mesh: Any,
    areas: NDArray[np.float64],
) -> int:
    """Log a warning for any unit whose area is below ``AREA_FLOOR``.

    A degenerate (near-zero-area) element silently zeroes its own error
    indicator -- it can never be marked, so refinement quietly stops improving
    exactly where the mesh is worst. That is a *diagnostic* concern rather than
    a numerical one (the estimator's arithmetic is unchanged), so this reports
    rather than clamps.

    Returns:
        The number of degenerate units found.

    """
    degenerate = int(np.count_nonzero(areas < AREA_FLOOR))
    if degenerate:
        logger.warning(
            "degenerate_units",
            n_degenerate=degenerate,
            n_units=substrate.n_units(mesh),
            area_floor=AREA_FLOOR,
            min_area=float(areas.min()) if areas.size else float("nan"),
        )
    return degenerate
