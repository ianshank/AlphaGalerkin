"""Adapter bridging DiscretizationEnvironment to GameInterface protocol.

:class:`DiscretizationGame` wraps the existing
:class:`~alphagalerkin.env.environment.DiscretizationEnvironment` and
:class:`~alphagalerkin.mcts.action_masking.ActionMasker` so that the
PDE discretization environment conforms to the shared
:class:`~alphagalerkin.mcts.protocol.GameInterface` protocol.

This allows the PDE discretization to be played by **any** MCTS
implementation that accepts :class:`GameInterface`, bridging the
gap between the ``alphagalerkin.mcts`` (discretization-specific)
and ``src.mcts`` (general-purpose) backends.
"""
from __future__ import annotations

import structlog

from src.alphagalerkin.core.config import AlphaGalerkinConfig
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.environment import DiscretizationEnvironment
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker

logger = structlog.get_logger("mcts.adapter")


class DiscretizationGame:
    """Adapter making DiscretizationEnvironment protocol-compatible.

    Makes the :class:`~alphagalerkin.env.environment.DiscretizationEnvironment`
    conform to the :class:`~alphagalerkin.mcts.protocol.GameInterface` protocol,
    allowing the PDE discretization to be played by any MCTS
    implementation that accepts ``GameInterface``.

    Parameters
    ----------
    config:
        Root ``AlphaGalerkinConfig`` controlling environment and
        MCTS behaviour.

    """

    def __init__(self, config: AlphaGalerkinConfig) -> None:
        self._config = config
        self._env = DiscretizationEnvironment(config.environment)
        self._masker = ActionMasker(config.environment)

    def get_initial_state(self) -> DiscretizationState:
        """Reset the environment and return the initial state.

        Returns
        -------
        DiscretizationState
            A fresh discretization state from the environment.

        """
        state = self._env.reset()
        logger.debug(
            "adapter.initial_state",
            dof_count=state.dof_count,
            num_elements=state.mesh.num_elements,
        )
        return state

    def get_valid_actions(
        self,
        state: DiscretizationState,
    ) -> list[Action]:
        """Return all valid actions for *state*.

        Delegates to :class:`ActionMasker` which checks DOF budget,
        element size constraints, and polynomial order bounds.

        Parameters
        ----------
        state:
            Current discretization state.

        Returns
        -------
        list[Action]
            Sorted list of valid actions (always includes NO_OP).

        """
        return self._masker.valid_actions(state)

    def apply_action(
        self,
        state: DiscretizationState,
        action: Action,
    ) -> DiscretizationState:
        """Apply *action* to *state* and return the new state.

        The original *state* is not mutated.

        Parameters
        ----------
        state:
            Current discretization state.
        action:
            The discretization action to apply.

        Returns
        -------
        DiscretizationState
            The resulting state after applying the action.

        """
        return state.apply_action(action)

    def is_terminal(
        self,
        state: DiscretizationState,
    ) -> bool:
        """Check whether *state* is terminal.

        A state is terminal if the DOF budget is exceeded or
        the step limit has been reached.

        Parameters
        ----------
        state:
            The discretization state to check.

        Returns
        -------
        bool
            ``True`` if the episode should terminate.

        """
        return (
            state.dof_count > self._config.environment.max_dof
            or state.step >= self._config.environment.max_steps
        )

    def get_reward(
        self,
        state: DiscretizationState,
    ) -> float:
        """Compute reward for terminal state.

        Uses a simple inverse-DOF heuristic: fewer DOFs for the
        same accuracy is better.  The reward is bounded in
        ``(0, 1]``.

        Parameters
        ----------
        state:
            A terminal discretization state.

        Returns
        -------
        float
            Scalar reward value.

        """
        return 1.0 / max(1, state.dof_count)
