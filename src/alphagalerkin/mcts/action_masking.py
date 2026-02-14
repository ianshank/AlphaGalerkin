"""Action masking for invalid discretization actions."""
from __future__ import annotations

import structlog

from src.alphagalerkin.core.config import EnvironmentConfig
from src.alphagalerkin.core.types import ActionType, ElementID
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.state import DiscretizationState

logger = structlog.get_logger("mcts.action_masking")


class ActionMasker:
    """Filters invalid actions based on environment constraints.

    The masker inspects the current mesh, basis assignments, and
    DOF budget to determine which refinement / coarsening moves
    are legal.  It also provides a helper to zero-out and
    re-normalise network priors so that only legal actions have
    non-zero probability.

    Args:
        config: Environment configuration providing DOF budget,
            element-size limits, and polynomial-order bounds.

    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config

    # ---------------------------------------------------------------
    # Valid action enumeration
    # ---------------------------------------------------------------

    def valid_actions(
        self,
        state: DiscretizationState,
    ) -> list[Action]:
        """Generate all valid actions for *state*.

        The list always contains at least a ``NO_OP`` action.

        Args:
            state: Current discretization state.

        Returns:
            Sorted list of valid :class:`Action` instances.

        """
        actions: list[Action] = []

        # NO_OP is always valid
        noop_eid = (
            state.mesh.element_ids[0]
            if state.mesh.element_ids
            else ElementID("e0")
        )
        actions.append(
            Action(
                element_id=noop_eid,
                action_type=ActionType.NO_OP,
            ),
        )

        for eid in state.mesh.element_ids:
            elem = state.mesh.get_element(eid)
            basis = state.basis_assignments.get(eid)

            # H-refine: check DOF budget and min element size
            child_size = elem.size / 2.0
            if (
                state.dof_count < self._config.max_dof
                and child_size
                > self._config.min_element_size
            ):
                actions.append(
                    Action(
                        element_id=eid,
                        action_type=ActionType.H_REFINE,
                    ),
                )

            # P-refine: check DOF budget and max poly order
            if (
                basis is not None
                and basis.polynomial_order
                < self._config.max_polynomial_order
                and state.dof_count
                < self._config.max_dof
            ):
                actions.append(
                    Action(
                        element_id=eid,
                        action_type=ActionType.P_REFINE,
                    ),
                )

            # H-coarsen: only if element was refined before
            if elem.level > 0:
                actions.append(
                    Action(
                        element_id=eid,
                        action_type=ActionType.H_COARSEN,
                    ),
                )

            # P-coarsen: only if polynomial order > 1
            if (
                basis is not None
                and basis.polynomial_order > 1
            ):
                actions.append(
                    Action(
                        element_id=eid,
                        action_type=ActionType.P_COARSEN,
                    ),
                )

        logger.debug(
            "mcts.masking.valid_actions",
            n_valid=len(actions),
            dof_count=state.dof_count,
        )
        return actions

    # ---------------------------------------------------------------
    # Prior masking
    # ---------------------------------------------------------------

    def mask_priors(
        self,
        priors: dict[Action, float],
        state: DiscretizationState,
    ) -> dict[Action, float]:
        """Zero out priors for invalid actions and renormalise.

        If no prior corresponds to a valid action, returns a
        uniform distribution over the valid action set.

        Args:
            priors: Network output mapping actions to raw priors.
            state: Current discretization state.

        Returns:
            Filtered and normalised prior distribution.

        """
        valid = set(self.valid_actions(state))

        masked: dict[Action, float] = {}
        for action, prior in priors.items():
            if action in valid:
                masked[action] = prior

        # Renormalise
        total = sum(masked.values())
        if total > 0:
            return {a: p / total for a, p in masked.items()}

        # Fallback: uniform over valid actions
        if valid:
            uniform = 1.0 / len(valid)
            return dict.fromkeys(valid, uniform)

        return {}
