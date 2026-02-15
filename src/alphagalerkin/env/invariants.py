"""Invariant checkers for discretization states.

Each public function checks one invariant and returns ``True`` on
success.  :func:`check_all_invariants` aggregates them and raises
:class:`~alphagalerkin.core.exceptions.InvariantViolationError` on the
first failure, providing a clear error message for debugging.
"""
from __future__ import annotations

import structlog

from src.alphagalerkin.core.exceptions import (
    InvariantViolationError,
)
from src.alphagalerkin.env.state import DiscretizationState

logger = structlog.get_logger("env.invariants")


def check_dof_consistency(
    state: DiscretizationState,
) -> bool:
    """``dof_count`` must be strictly positive."""
    return state.dof_count > 0


def check_basis_coverage(
    state: DiscretizationState,
) -> bool:
    """Every active element must have a basis assignment."""
    return all(
        eid in state.basis_assignments
        for eid in state.mesh.element_ids
    )


def check_no_orphan_basis(
    state: DiscretizationState,
) -> bool:
    """No basis assignment may reference a non-existent element."""
    return all(
        eid in state.mesh.element_ids
        for eid in state.basis_assignments
    )


def check_all_invariants(
    state: DiscretizationState,
) -> None:
    """Run every invariant check.

    Raises
    ------
    InvariantViolationError
        On the first check that fails.

    """
    if not check_dof_consistency(state):
        logger.warning(
            "invariants.violation_detected",
            check="dof_consistency",
            details="DOF count inconsistent",
        )
        raise InvariantViolationError("DOF count inconsistent")
    if not check_basis_coverage(state):
        logger.warning(
            "invariants.violation_detected",
            check="basis_coverage",
            details="Not all elements have basis assignments",
        )
        raise InvariantViolationError(
            "Not all elements have basis assignments"
        )
    if not check_no_orphan_basis(state):
        logger.warning(
            "invariants.violation_detected",
            check="no_orphan_basis",
            details="Orphan basis assignments found",
        )
        raise InvariantViolationError(
            "Orphan basis assignments found"
        )
    logger.debug("invariants.all_passed")
