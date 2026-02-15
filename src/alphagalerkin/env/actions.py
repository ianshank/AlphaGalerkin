"""Action definitions and application logic for discretization games.

An :class:`Action` is an immutable description of a single
discretization modification (h/p-refinement, coarsening, basis swap,
or no-op) targeting one mesh element.  Validation is performed
against a :class:`~alphagalerkin.env.state.DiscretizationState`
to ensure the action is legal before it is applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.alphagalerkin.core.types import ActionType, ElementID

if TYPE_CHECKING:
    from src.alphagalerkin.env.state import DiscretizationState

logger = structlog.get_logger("env.actions")

# Action types that operate on the entire mesh rather than
# a single element.  They do not require a valid element_id.
GLOBAL_ACTION_TYPES: frozenset[ActionType] = frozenset({
    ActionType.REFINE_ALL_BOUNDARY,
    ActionType.COARSEN_ALL_INTERIOR,
    ActionType.UNIFORM_P_REFINE,
})


@dataclass(frozen=True)
class Action:
    """A discretization action on a specific element.

    Instances are frozen (immutable) so they can be used as dict keys
    and stored in sets.

    Attributes
    ----------
    element_id:
        Target element.  Ignored for ``NO_OP``.
    action_type:
        Kind of modification.
    params:
        Extra parameters consumed by certain action types (e.g.
        ``{"basis_family": "legendre"}`` for ``SWAP_BASIS``).

    """

    element_id: ElementID
    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)

    # -- validation --------------------------------------------------

    def validate(self, state: DiscretizationState) -> bool:
        """Return ``True`` if this action is legal in *state*.

        Checks
        ------
        * ``NO_OP`` is always valid.
        * Global actions (``REFINE_ALL_BOUNDARY``,
          ``COARSEN_ALL_INTERIOR``, ``UNIFORM_P_REFINE``) are valid
          when the mesh has at least one element.
        * The target element must exist in the active mesh.
        * ``H_COARSEN`` requires ``level > 0``.
        * ``P_COARSEN`` requires ``polynomial_order > 1``.
        """
        if self.action_type == ActionType.NO_OP:
            return True

        # Global actions do not need a specific element_id
        if self.action_type in GLOBAL_ACTION_TYPES:
            valid = state.mesh.num_elements > 0
            if not valid:
                logger.debug(
                    "action.invalid.global_empty_mesh",
                    action_type=self.action_type.value,
                )
            return valid

        if self.element_id not in state.mesh.element_ids:
            logger.debug(
                "action.invalid.element_not_found",
                element_id=str(self.element_id),
                action_type=self.action_type.value,
            )
            return False

        if self.action_type == ActionType.H_COARSEN:
            element = state.mesh.get_element(self.element_id)
            valid = element.level > 0
            if not valid:
                logger.debug(
                    "action.invalid.coarsen_at_root",
                    element_id=str(self.element_id),
                )
            return valid

        if self.action_type == ActionType.P_COARSEN:
            basis = state.basis_assignments.get(self.element_id)
            valid = (
                basis is not None
                and basis.polynomial_order > 1
            )
            if not valid:
                logger.debug(
                    "action.invalid.p_coarsen",
                    element_id=str(self.element_id),
                    has_basis=basis is not None,
                )
            return valid

        return True

    # -- hashing / equality ------------------------------------------

    def __hash__(self) -> int:
        return hash((self.element_id, self.action_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.element_id == other.element_id
            and self.action_type == other.action_type
        )
