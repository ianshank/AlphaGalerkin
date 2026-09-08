"""Fingerprint-keyed LRU solve cache for ``RefinementSubstrate``.

The only production reader of ``RefinementSubstrate.fingerprint``: memoises
``solve(mesh)`` results so MCTS simulations that revisit a mesh identity do
not re-pay the estimator-dominated cost measured in the element-local spike.
Bounded by ``SubstrateConfig.solve_cache_max_entries``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from src.refinement.substrate import SubstrateSolveResult

logger = structlog.get_logger(__name__)


class _FingerprintedSubstrate(Protocol):
    """Minimal surface the cache needs (solve + fingerprint)."""

    def solve(self, mesh: object) -> SubstrateSolveResult: ...

    def fingerprint(self, mesh: object) -> bytes: ...


class FingerprintSolveCache:
    """LRU cache keyed by ``substrate.fingerprint(mesh)``."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, SubstrateSolveResult] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._log = logger.bind(max_entries=max_entries)

    @property
    def size(self) -> int:
        """Current number of cached solves."""
        return len(self._entries)

    def clear(self) -> None:
        """Drop every cached entry (does not reset hit/miss counters)."""
        self._entries.clear()
        self._log.debug("solve_cache_cleared")

    def get_or_solve(
        self,
        substrate: _FingerprintedSubstrate,
        mesh: object,
    ) -> SubstrateSolveResult:
        """Return a cached solve for ``mesh``, or compute and store one.

        Always calls ``substrate.fingerprint(mesh)`` so the Protocol member has
        a real ``src/`` reader (retires the staged abstraction-audit exemption).
        """
        key = substrate.fingerprint(mesh)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            self._log.debug(
                "solve_cache_hit",
                fingerprint_nbytes=len(key),
                hits=self.hits,
                size=len(self._entries),
            )
            return cached

        result = substrate.solve(mesh)
        self._entries[key] = result
        self.misses += 1
        while len(self._entries) > self._max_entries:
            evicted, _ = self._entries.popitem(last=False)
            self._log.debug(
                "solve_cache_evict",
                fingerprint_nbytes=len(evicted),
                size=len(self._entries),
            )
        self._log.debug(
            "solve_cache_miss",
            fingerprint_nbytes=len(key),
            misses=self.misses,
            n_dof=result.n_dof,
            l2_error=result.l2_error,
            size=len(self._entries),
        )
        return result


__all__ = ["FingerprintSolveCache"]
