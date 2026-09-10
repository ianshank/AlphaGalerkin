"""Shared geometry / error predicates used by both solvers and substrates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from src.pde.operators import PDEOperator

# Optional geometry predicate for masked (non-rectangular) domains.  Given
# node coordinates of shape ``(N, 2)`` it returns a boolean mask ``(N,)`` that
# is True for nodes *inside* the physical domain.  ``None`` (the default on
# every parameter below) means "full bounding box" — the historical behaviour,
# so every existing caller and test is byte-for-byte unchanged.
InsidePredicate = Callable[[NDArray[np.float64]], NDArray[np.bool_]]


def element_inside_mask(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    inside: InsidePredicate,
) -> NDArray[np.bool_]:
    """Which tensor-grid elements lie inside the physical domain, by centre test.

    Shape ``(len(xs) - 1, len(ys) - 1)``. An element counts as inside iff its
    centre satisfies ``inside`` -- the same rule
    ``DorflerAMRSolver._compute_indicators_2d`` uses to decide which elements
    get a zero indicator, and therefore the rule any *refinability* answer has
    to agree with: an element whose indicator is forced to zero can never be
    marked, so calling it refinable is a claim the estimator contradicts.

    Module-level so the indicator path and ``TensorGridSubstrate.refinable_mask``
    share **one** definition. The centres are built through the identical
    ``float32 -> float64`` round-trip the per-element midpoint uses, so the
    predicate sees byte-for-byte the same input; a second hand-rolled copy of
    that round-trip would be free to drift by a ULP and silently disagree at
    the boundary.
    """
    cx_all = 0.5 * (xs[:-1] + xs[1:])
    cy_all = 0.5 * (ys[:-1] + ys[1:])
    cx_grid, cy_grid = np.meshgrid(cx_all, cy_all, indexing="ij")
    centres = (
        np.stack([cx_grid.ravel(), cy_grid.ravel()], axis=-1).astype(np.float32).astype(np.float64)
    )
    return np.asarray(inside(centres), dtype=bool).reshape(len(xs) - 1, len(ys) - 1)


def require_exact_solution(operator: PDEOperator, owner: str) -> None:
    """Fail at construction when ``operator`` has no analytic exact solution.

    Every ``RefinementSubstrate`` measures error against an exact solution, and
    ``SubstrateSolveResult.l2_error`` is a bare ``float`` with nowhere to put a
    ``None``. Without this, such an operator surfaces as a ``TypeError`` from
    ``np.asarray(None, dtype=np.float64)`` several frames inside a solve or a
    quadrature form -- true, but unhelpful.

    Probed at ``operator.dim`` rather than a hardcoded 2: both substrates
    previously wrote ``np.zeros((1, 2))``, which silently assumed 2-D and was
    the third copy-paste between that pair (after ``_dirichlet_dof_indices``
    and the nodal-RMS formula). Shared here, next to the other primitives both
    substrates already import, so a fourth cannot drift.

    Args:
        operator: The operator to probe.
        owner: Caller name, used verbatim in the error so the message names the
            class the user actually constructed.

    Raises:
        ValueError: If ``exact_solution`` returns ``None``.

    """
    probe = np.zeros((1, operator.dim), dtype=np.float32)
    if operator.exact_solution(probe) is None:
        raise ValueError(
            f"{owner} measures error against an analytic exact solution; "
            f"{type(operator).__name__}.exact_solution returned None. Failing at "
            f"construction rather than as a TypeError from inside solve()."
        )


def require_measurable_l2(nodal_rms: float | None, owner: str) -> float:
    """Return ``nodal_rms``, or raise -- never substitute ``0.0``.

    :func:`nodal_rms_l2_error` documents that it returns ``None`` "not a crash,
    and **not** ``0.0``", because ``0.0`` is the strongest possible correctness
    claim and an unmeasurable error is the weakest possible evidence for it.
    Both substrates nevertheless wrote ``float(nodal_rms or 0.0)`` at six sites,
    converting the sentinel straight back into the value the helper refuses to
    return, so an unmeasurable solve published a *perfect* L2 error.

    Two triggers reach that ``None``. The first -- an operator with no analytic
    exact solution -- is already guarded at construction by
    :func:`require_exact_solution`. The second is an **empty diff array**
    (``n == 0``): an empty in-domain node set, reachable from an over-restrictive
    geometry predicate, and guarded by nothing. That is what makes this a live
    path rather than defensive dead code.

    Args:
        nodal_rms: The value from :func:`nodal_rms_l2_error`.
        owner: Caller name, used verbatim so the error names the substrate the
            user actually constructed.

    Returns:
        ``nodal_rms``, when it is not ``None``.

    Raises:
        ValueError: If ``nodal_rms`` is ``None``.

    """
    if nodal_rms is None:
        raise ValueError(
            f"{owner}: the nodal-RMS L2 error is unmeasurable (no exact solution, or an "
            f"empty in-domain node set). Refusing to report 0.0, which would publish a "
            f"perfect score for a solve that was never measured."
        )
    return float(nodal_rms)


def nodal_rms_l2_error(
    solution: NDArray[np.float64],
    coords: NDArray[np.float64],
    operator: PDEOperator,
) -> float | None:
    """Root-mean-square nodal error against the exact solution, if available.

    The historical ``l2_error`` every ``SolverResult`` in this module reports.
    Deliberately **not** a true L2 norm: it is unweighted, so on a graded mesh
    it over-counts the densely-refined region. ``fem_baseline.quadrature_l2_error``
    is the mesh-independent metric to compare two refinement policies with; this
    one exists so those historical numbers stay reproducible byte-for-byte.

    Returns ``None`` -- not a crash, and not ``0.0`` -- when ``operator`` has no
    analytic exact solution, which is the majority of real operators.
    """
    exact = operator.exact_solution(coords.astype(np.float32))
    if exact is None:
        return None
    if isinstance(exact, torch.Tensor):
        exact = exact.detach().cpu().numpy()
    exact = np.asarray(exact, dtype=np.float64)
    diff = solution.flatten() - exact.flatten()
    n = len(diff)
    return float(np.sqrt(np.sum(diff**2) / n)) if n > 0 else None
