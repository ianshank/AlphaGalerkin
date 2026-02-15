"""Gym-like discretization environment.

:class:`DiscretizationEnvironment` wraps the state-transition logic
in a familiar ``reset`` / ``step`` API.  At each step it:

1. Applies the :class:`~alphagalerkin.env.actions.Action` to produce
   a new :class:`~alphagalerkin.env.state.DiscretizationState`.
2. Checks DOF budget and step-limit termination conditions.
3. Computes a scalar reward via :class:`RewardComposer`.
4. Returns a :class:`StepResult` bundle.

All tuneable parameters (DOF budget, step limit, reward weights) come
from :class:`~alphagalerkin.core.config.EnvironmentConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from src.alphagalerkin.core.config import EnvironmentConfig
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.rewards import RewardComposer
from src.alphagalerkin.env.state import DiscretizationState

logger = structlog.get_logger("env.environment")


# -------------------------------------------------------------------
# Step result
# -------------------------------------------------------------------


@dataclass
class StepResult:
    """Value object returned by :meth:`DiscretizationEnvironment.step`.

    Attributes
    ----------
    state:
        The post-action discretization state.
    reward:
        Scalar reward for this transition.
    done:
        ``True`` if the episode has terminated.
    info:
        Diagnostic dict (DOF count, element count, step index,
        budget flag).

    """

    state: DiscretizationState
    reward: float
    done: bool
    info: dict[str, Any]


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------


class DiscretizationEnvironment:
    """Gym-like environment for PDE discretization games.

    Parameters
    ----------
    config:
        Environment configuration (budget, step limit, weights).
    initial_mesh:
        Starting mesh.  When ``None`` a default 4x4 unit-square
        quad mesh is created.

    """

    def __init__(
        self,
        config: EnvironmentConfig,
        initial_mesh: MeshGraph | None = None,
        physics_module: Any = None,
    ) -> None:
        self._config = config
        self._reward_composer = RewardComposer(config.reward_weights)
        self._initial_mesh = initial_mesh or MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(4, 4),
        )
        self._state: DiscretizationState | None = None
        self._prev_residual: float | None = None
        self._physics = physics_module

    # -- episode lifecycle -------------------------------------------

    def reset(self) -> DiscretizationState:
        """Reset the environment to its initial state.

        Returns the fresh :class:`DiscretizationState`.
        """
        self._state = DiscretizationState.from_mesh(
            mesh=self._initial_mesh.clone(),
            initial_polynomial_order=(self._config.initial_polynomial_order),
            basis_family=self._config.default_basis_family,
        )
        self._prev_residual = None
        logger.info(
            "env.reset",
            num_elements=self._state.mesh.num_elements,
            dof_count=self._state.dof_count,
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        """Apply *action* and return transition data.

        Raises
        ------
        RuntimeError
            If :meth:`reset` has not been called.

        """
        if self._state is None:
            msg = "Environment not reset. Call reset() first."
            raise RuntimeError(msg)

        new_state = self._state.apply_action(action)

        # -- termination conditions ----------------------------------
        budget_exceeded = new_state.dof_count > self._config.max_dof
        if budget_exceeded:
            logger.warning(
                "env.dof_budget_exceeded",
                dof=new_state.dof_count,
                max=self._config.max_dof,
            )

        step_limit_reached = new_state.step >= self._config.max_steps

        # -- reward computation --
        # State-based heuristic: residual decreases as DOFs increase,
        # rewarding refinement even without a physics module.
        residual_norm = 1.0 / max(1, new_state.dof_count)
        if self._physics is not None:
            try:
                solve_result = self._physics.solve(new_state)
                residual_norm = float(solve_result.residual_norm)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                logger.warning(
                    "env.physics_solve_failed",
                    element=str(action.element_id),
                    error=type(exc).__name__,
                    exc_info=True,
                )
                residual_norm = 1.0 / max(
                    1,
                    new_state.dof_count,
                )

        accuracy = self._reward_composer.accuracy_reward(residual_norm, self._prev_residual)
        self._prev_residual = residual_norm
        efficiency = self._reward_composer.efficiency_reward(
            new_state.dof_count, self._config.max_dof
        )
        reward = self._reward_composer.compute(accuracy=accuracy, efficiency=efficiency)

        done = budget_exceeded or step_limit_reached

        logger.debug(
            "env.step",
            action_type=action.action_type.value,
            reward=round(reward, 6),
            done=done,
            dof_count=new_state.dof_count,
            step=new_state.step,
        )

        self._state = new_state

        return StepResult(
            state=new_state,
            reward=reward,
            done=done,
            info={
                "dof_count": new_state.dof_count,
                "num_elements": new_state.mesh.num_elements,
                "step": new_state.step,
                "budget_exceeded": budget_exceeded,
            },
        )

    # -- accessors ---------------------------------------------------

    @property
    def state(self) -> DiscretizationState | None:
        """Current state, or ``None`` before :meth:`reset`."""
        return self._state
