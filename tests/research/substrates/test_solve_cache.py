"""Tests for FingerprintSolveCache — the production fingerprint reader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.refinement.substrate import SubstrateSolveResult
from src.research.substrates.solve_cache import FingerprintSolveCache


@dataclass
class _FakeMesh:
    tag: bytes


class _FakeSubstrate:
    def __init__(self) -> None:
        self.solve_calls = 0

    def fingerprint(self, mesh: _FakeMesh) -> bytes:
        return mesh.tag

    def solve(self, mesh: _FakeMesh) -> SubstrateSolveResult:
        self.solve_calls += 1
        return SubstrateSolveResult(
            values=np.array([1.0], dtype=np.float64),
            indicators=np.array([0.5], dtype=np.float64),
            l2_error=0.25,
            n_dof=1,
            n_dof_free=1,
            extra={},
        )


def test_cache_hit_skips_second_solve() -> None:
    substrate = _FakeSubstrate()
    cache = FingerprintSolveCache(max_entries=8)
    mesh = _FakeMesh(tag=b"a")
    first = cache.get_or_solve(substrate, mesh)
    second = cache.get_or_solve(substrate, mesh)
    assert first.l2_error == second.l2_error
    assert substrate.solve_calls == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_miss_on_distinct_fingerprint() -> None:
    substrate = _FakeSubstrate()
    cache = FingerprintSolveCache(max_entries=8)
    cache.get_or_solve(substrate, _FakeMesh(tag=b"a"))
    cache.get_or_solve(substrate, _FakeMesh(tag=b"b"))
    assert substrate.solve_calls == 2
    assert cache.misses == 2


def test_lru_eviction() -> None:
    substrate = _FakeSubstrate()
    cache = FingerprintSolveCache(max_entries=2)
    cache.get_or_solve(substrate, _FakeMesh(tag=b"a"))
    cache.get_or_solve(substrate, _FakeMesh(tag=b"b"))
    cache.get_or_solve(substrate, _FakeMesh(tag=b"c"))  # evicts a
    assert cache.size == 2
    cache.get_or_solve(substrate, _FakeMesh(tag=b"a"))  # miss again
    assert substrate.solve_calls == 4


def test_rejects_non_positive_max_entries() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        FingerprintSolveCache(max_entries=0)
