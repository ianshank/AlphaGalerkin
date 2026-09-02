"""Shared Dörfler bulk-marking, unifying two historically divergent implementations.

``DorflerAMRSolver._dorfler_mark`` (``src.research.baselines``) and
``ScikitFEMPoissonSolver._dorfler_mark`` (``src.research.fem_baseline``) compute the
same idea — mark the smallest subset of elements whose indicators make up a
``theta`` fraction of the total — but diverge on two points that matter once a
substrate needs to reproduce either one exactly: whether the bulk quantity is
squared or linear, and what an all-zero indicator array marks. ``dorfler_mark``
reproduces both behaviours byte-for-byte behind one function, keyed by
``variant``, so ``RefinementSubstrate`` implementations (and the legacy solvers
themselves, via a thin delegate) share one marking primitive instead of two
copies that can silently drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MarkingVariant = Literal["squared", "linear"]


def dorfler_mark(
    indicators: NDArray[np.float64],
    theta: float,
    variant: MarkingVariant,
) -> NDArray[np.bool_]:
    """Mark elements via Dörfler bulk-chasing.

    Args:
        indicators: Non-negative per-element error indicators.
        theta: Bulk fraction (``marking_fraction`` in both legacy callers).
        variant: ``"squared"`` reproduces ``DorflerAMRSolver._dorfler_mark`` — the
            bulk quantity is ``indicators ** 2`` and an all-zero array still marks
            exactly one element (an all-zero cumulative sum's ``searchsorted``
            against a zero threshold returns index 0). ``"linear"`` reproduces
            ``ScikitFEMPoissonSolver._dorfler_mark`` — the bulk quantity is
            ``indicators`` itself and a non-positive total marks nothing.

    Returns:
        Boolean mask, same shape as ``indicators``, True where marked.

    """
    if variant == "squared":
        weighted = indicators**2
        total = float(np.sum(weighted))
        threshold = theta * total
        order = np.argsort(indicators)[::-1]
        cumulative = np.cumsum(weighted[order])
        n_mark = int(np.searchsorted(cumulative, threshold)) + 1
        marked = np.zeros(len(indicators), dtype=bool)
        marked[order[:n_mark]] = True
        return marked

    if variant == "linear":
        total = float(np.sum(indicators))
        if total <= 0.0:
            return np.zeros_like(indicators, dtype=bool)
        threshold = theta * total
        order = np.argsort(indicators)[::-1]
        cumulative = np.cumsum(indicators[order])
        n_mark = int(np.searchsorted(cumulative, threshold) + 1)
        marked = np.zeros_like(indicators, dtype=bool)
        marked[order[:n_mark]] = True
        return marked

    raise ValueError(f"Unknown marking variant: {variant!r}")
