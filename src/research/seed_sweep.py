"""Shared deterministic seed-sweep helpers for ``src/research/`` comparison harnesses.

Both the L-shape AMR (`lshape_amr_compare`) and transfer-baseline
(`transfer_baseline_compare`) comparisons run a median-over-seeds sweep, and both
derive the per-seed RNG seeds identically. This single definition is imported by both
(and re-exported from each for backwards-compatible imports) so the stride and
derivation never drift apart — mirroring the ``_centaur_common.median_of`` extraction.
"""

from __future__ import annotations

# Prime stride decorrelating per-seed RNG streams in a multi-seed sweep (mirrors the
# noyron_basis / scaling_law scenarios). A prime keeps successive seed streams from
# sharing low-order factors.
SEED_PRIME_STRIDE: int = 7919


def resolved_seeds(base_seed: int, n_seeds: int) -> list[int]:
    """Deterministic, decorrelated per-seed RNG seeds for a sweep.

    Args:
        base_seed: The first seed; subsequent seeds are strided by
            :data:`SEED_PRIME_STRIDE`.
        n_seeds: Number of seeds to derive (``>= 0``).

    Returns:
        ``[base_seed + i * SEED_PRIME_STRIDE for i in range(n_seeds)]``.

    """
    return [base_seed + i * SEED_PRIME_STRIDE for i in range(n_seeds)]
