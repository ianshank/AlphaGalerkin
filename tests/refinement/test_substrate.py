"""Contract tests for the domain-free ``RefinementSubstrate`` interface.

Covers ``SubstrateSolveResult`` (a frozen dataclass), the generic
``RefinementSubstrate`` Protocol's structural (``runtime_checkable``)
compliance checking, and the ``RefinementSubstrateRegistry`` round-trip —
mirroring the conventions in ``tests/refinement/test_refinement.py``
(``RefinementGame``'s own registry test).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.refinement.substrate import RefinementSubstrate, SubstrateSolveResult
from src.refinement.substrate_registry import (
    RefinementSubstrateRegistry,
    register_refinement_substrate,
)


class _ToyMesh:
    """Stand-in ``TMesh`` for a substrate over an opaque domain object."""

    def __init__(self, n: int) -> None:
        self.n = n


class _ToySubstrate:
    """A minimal, fully-conforming ``RefinementSubstrate`` implementation.

    Deliberately does NOT inherit from ``RefinementSubstrate`` — Protocols are
    structural, and a real substrate (``TensorGridSubstrate``,
    ``SkfemTriSubstrate``) should not either.
    """

    def initial_mesh(self) -> _ToyMesh:
        return _ToyMesh(n=2)

    def solve(self, mesh: _ToyMesh) -> SubstrateSolveResult:
        return SubstrateSolveResult(
            values=np.zeros(mesh.n),
            indicators=np.zeros(mesh.n),
            l2_error=0.1,
            n_dof=mesh.n,
            n_dof_free=mesh.n,
            extra={},
        )

    def mark(self, indicators: np.ndarray, theta: float) -> np.ndarray:
        return indicators > theta

    def refine(self, mesh: _ToyMesh, marked: np.ndarray) -> _ToyMesh:
        return _ToyMesh(n=mesh.n + int(marked.sum()))

    def n_units(self, mesh: _ToyMesh) -> int:
        return mesh.n

    def refinable_mask(self, mesh: _ToyMesh) -> np.ndarray:
        return np.ones(mesh.n, dtype=bool)

    def fingerprint(self, mesh: _ToyMesh) -> bytes:
        return str(mesh.n).encode()

    def describe(self) -> dict[str, str | int | float]:
        return {"kind": "toy"}


class _IncompleteSubstrate:
    """Missing every member except ``initial_mesh`` — must fail structural checks."""

    def initial_mesh(self) -> _ToyMesh:
        return _ToyMesh(n=1)


class TestSubstrateSolveResult:
    def test_is_frozen(self) -> None:
        result = SubstrateSolveResult(
            values=np.zeros(2),
            indicators=np.zeros(2),
            l2_error=0.5,
            n_dof=2,
            n_dof_free=2,
            extra={"l2_error_nodal_rms": 0.6},
        )
        assert dataclasses.is_dataclass(result)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.l2_error = 0.0  # type: ignore[misc]

    def test_extra_carries_auxiliary_metrics_without_replacing_l2_error(self) -> None:
        result = SubstrateSolveResult(
            values=np.zeros(1),
            indicators=np.zeros(1),
            l2_error=0.1,
            n_dof=1,
            n_dof_free=1,
            extra={"l2_error_nodal_rms": 0.2},
        )
        assert result.l2_error == 0.1
        assert result.extra["l2_error_nodal_rms"] == 0.2

    def test_n_dof_free_exceeding_n_dof_raises(self) -> None:
        """AC5: n_dof_free <= n_dof must hold, enforced at construction."""
        with pytest.raises(ValueError, match="n_dof_free"):
            SubstrateSolveResult(
                values=np.zeros(1),
                indicators=np.zeros(1),
                l2_error=0.1,
                n_dof=1,
                n_dof_free=2,
                extra={},
            )

    def test_negative_n_dof_raises(self) -> None:
        with pytest.raises(ValueError, match="n_dof"):
            SubstrateSolveResult(
                values=np.zeros(1),
                indicators=np.zeros(1),
                l2_error=0.1,
                n_dof=-1,
                n_dof_free=0,
                extra={},
            )

    def test_negative_n_dof_free_raises(self) -> None:
        with pytest.raises(ValueError, match="n_dof_free"):
            SubstrateSolveResult(
                values=np.zeros(1),
                indicators=np.zeros(1),
                l2_error=0.1,
                n_dof=1,
                n_dof_free=-1,
                extra={},
            )

    def test_n_dof_free_equal_to_n_dof_is_allowed(self) -> None:
        result = SubstrateSolveResult(
            values=np.zeros(1),
            indicators=np.zeros(1),
            l2_error=0.1,
            n_dof=1,
            n_dof_free=1,
            extra={},
        )
        assert result.n_dof_free == result.n_dof


class TestRefinementSubstrateProtocol:
    def test_conforming_class_satisfies_isinstance(self) -> None:
        assert isinstance(_ToySubstrate(), RefinementSubstrate)

    def test_non_conforming_class_does_not_satisfy_isinstance(self) -> None:
        assert not isinstance(_IncompleteSubstrate(), RefinementSubstrate)

    def test_full_solve_mark_refine_loop(self) -> None:
        substrate: RefinementSubstrate[_ToyMesh] = _ToySubstrate()
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        marked = substrate.mark(result.indicators, theta=-1.0)
        refined = substrate.refine(mesh, marked)
        assert substrate.n_units(refined) >= substrate.n_units(mesh)
        assert substrate.fingerprint(mesh) != substrate.fingerprint(refined)
        assert substrate.describe()["kind"] == "toy"


class TestRefinementSubstrateRegistry:
    def setup_method(self) -> None:
        RefinementSubstrateRegistry().clear()

    def teardown_method(self) -> None:
        RefinementSubstrateRegistry().clear()

    def test_register_and_retrieve(self) -> None:
        register_refinement_substrate("toy")(_ToySubstrate)
        registry = RefinementSubstrateRegistry()
        assert "toy" in registry.list_items()
        cls = registry.get_or_raise("toy")
        assert isinstance(cls(), RefinementSubstrate)

    def test_rejects_structurally_incomplete_class(self) -> None:
        with pytest.raises(TypeError, match="must inherit from RefinementSubstrate"):
            register_refinement_substrate("bad")(_IncompleteSubstrate)
