"""Tests for invariant checkers (env/invariants.py)."""

from __future__ import annotations

import pytest

from src.alphagalerkin.core.exceptions import InvariantViolationError
from src.alphagalerkin.core.types import BasisSpec, ElementID
from src.alphagalerkin.env.invariants import (
    check_all_invariants,
    check_basis_coverage,
    check_dof_consistency,
    check_no_orphan_basis,
)
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState


def _default_state() -> DiscretizationState:
    """Create a valid 2x2 quad mesh state."""
    mesh = MeshGraph.create_uniform_quad(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        num_elements=(2, 2),
    )
    return DiscretizationState.from_mesh(mesh)


class TestCheckDofConsistency:
    """check_dof_consistency requires positive DOF count."""

    def test_valid_state(self) -> None:
        state = _default_state()

        assert check_dof_consistency(state) is True

    def test_valid_higher_order(self) -> None:
        state = _default_state()
        # Increase polynomial order on all elements.
        for eid in state.basis_assignments:
            state.basis_assignments[eid] = BasisSpec(
                polynomial_order=3,
            )

        assert check_dof_consistency(state) is True


class TestCheckBasisCoverage:
    """check_basis_coverage: every element needs a basis."""

    def test_valid_state(self) -> None:
        state = _default_state()

        assert check_basis_coverage(state) is True

    def test_missing_assignment(self) -> None:
        state = _default_state()
        # Remove one assignment.
        first_eid = state.mesh.element_ids[0]
        del state.basis_assignments[first_eid]

        assert check_basis_coverage(state) is False


class TestCheckNoOrphanBasis:
    """check_no_orphan_basis: no assignment for non-existent elements."""

    def test_valid_state(self) -> None:
        state = _default_state()

        assert check_no_orphan_basis(state) is True

    def test_orphan_basis(self) -> None:
        state = _default_state()
        # Add an assignment for a nonexistent element.
        fake_eid = ElementID("phantom_element")
        state.basis_assignments[fake_eid] = BasisSpec(
            polynomial_order=1,
        )

        assert check_no_orphan_basis(state) is False


class TestCheckAllInvariants:
    """check_all_invariants aggregates and raises on failure."""

    def test_valid_state_passes(self) -> None:
        state = _default_state()

        # Should not raise.
        check_all_invariants(state)

    def test_raises_on_missing_basis(self) -> None:
        state = _default_state()
        first_eid = state.mesh.element_ids[0]
        del state.basis_assignments[first_eid]

        with pytest.raises(InvariantViolationError, match="basis"):
            check_all_invariants(state)

    def test_detects_orphan_basis(self) -> None:
        """check_no_orphan_basis detects extra keys not in mesh."""
        state = _default_state()
        fake_eid = ElementID("orphan_elem")
        state.basis_assignments[fake_eid] = BasisSpec(
            polynomial_order=1,
        )
        assert not check_no_orphan_basis(state)

    def test_first_failure_wins(self) -> None:
        """check_all_invariants raises on the first failing check."""
        state = _default_state()
        # Remove a basis to trigger basis coverage failure.
        first_eid = state.mesh.element_ids[0]
        del state.basis_assignments[first_eid]

        # Basis coverage fails before orphan check runs.
        with pytest.raises(InvariantViolationError, match="basis"):
            check_all_invariants(state)
