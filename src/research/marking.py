"""Shared Dörfler bulk-marking, unifying two historically divergent implementations.

``DorflerAMRSolver._dorfler_mark`` (``src.research.baselines``) and
``ScikitFEMPoissonSolver._dorfler_mark`` (``src.research.fem_baseline``) compute the
same idea — mark the smallest subset of elements whose indicators make up a
``theta`` fraction of the total — but diverge on two points that matter once a
substrate needs to reproduce either one exactly: whether the bulk quantity is
squared or linear, and what an all-zero indicator array marks. ``dorfler_mark``
reproduces both behaviours byte-for-byte behind one function, keyed by
``variant``, so ``RefinementSubstrate`` implementations under
``src.research.substrates`` (and the legacy solvers themselves, via a thin
delegate) share one marking primitive instead of two copies that can silently
drift apart.

Lives under ``src.research`` rather than ``src.refinement`` (its originally
planned home in ``openspec/changes/element-local-substrate/``): both legacy
solvers this module de-duplicates are scoped by
``tests/regression/test_import_contracts.py``'s
``reference-baselines-do-not-import-the-candidate`` contract, which forbids
``src/research/baselines.py``/``fem_baseline.py`` from importing anything
under ``src.refinement`` at all -- unlike the contract's one existing
exemption (``src/mcts/gumbel.py`` importing an inert protocol/type), this
module is active marking *behaviour*, exactly what the contract exists to
keep out of a reference baseline. See the correction note in
``specs/refinement_substrate.spec.md``.
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
        indicators: Non-negative per-element error indicators, 1D.
        theta: Bulk fraction (``marking_fraction`` in both legacy callers).
        variant: ``"squared"`` reproduces ``DorflerAMRSolver._dorfler_mark`` — the
            bulk quantity is ``indicators ** 2`` and an all-zero array still marks
            exactly one element (an all-zero cumulative sum's ``searchsorted``
            against a zero threshold returns index 0). ``"linear"`` reproduces
            ``ScikitFEMPoissonSolver._dorfler_mark`` — the bulk quantity is
            ``indicators`` itself and a non-positive total marks nothing.

    Returns:
        1D boolean mask, ``len(result) == len(indicators)``, True where marked.

    Raises:
        ValueError: If ``theta`` is outside ``(0, 1]``, or if ``indicators``
            contains NaN/Inf. Both otherwise "work" and silently mark the wrong
            set: ``theta <= 0`` chases zero bulk, ``theta > 1`` chases bulk that
            does not exist, and a single NaN makes ``np.argsort`` sort it *last*
            -- so the ``[::-1]`` below puts it *first*, poisons the cumulative
            sum, and hands ``searchsorted`` a meaningless threshold. Silently
            marking the wrong elements is the worst available failure mode here,
            because every downstream number stays finite and plausible.

    """
    if not 0.0 < theta <= 1.0:
        raise ValueError(f"theta (Dörfler bulk fraction) must be in (0, 1]; got {theta!r}")
    if indicators.size and not np.all(np.isfinite(indicators)):
        n_bad = int(np.count_nonzero(~np.isfinite(indicators)))
        raise ValueError(
            f"indicators must be finite; got {n_bad} non-finite value(s) of "
            f"{indicators.size}. A NaN sorts last, so the descending order below "
            f"would rank it first and mark an arbitrary set."
        )

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
