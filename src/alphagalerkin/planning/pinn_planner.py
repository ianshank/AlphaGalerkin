"""MCTS-based planning for physics-informed neural network training.

This module treats PINN training as a sequential decision problem where
an MCTS-style planner decides when and how to modify collocation points,
loss weights, and optimizer choice to accelerate convergence.

State: current collocation points, loss weights, optimizer choice.
Actions: modify collocation, adjust weights, switch optimizer.
Reward: reduction in PDE residual normalized by computational cost.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("planning.pinn")


class PINNActionType(str, Enum):
    """Actions available during PINN training planning."""

    ADD_COLLOCATION = "add_collocation"
    """Add collocation points in high-residual region."""

    REMOVE_COLLOCATION = "remove_collocation"
    """Remove low-error collocation points."""

    INCREASE_PHYSICS_WEIGHT = "increase_physics_weight"
    """Increase PDE residual loss weight."""

    DECREASE_PHYSICS_WEIGHT = "decrease_physics_weight"
    """Decrease PDE residual loss weight."""

    SWITCH_OPTIMIZER = "switch_optimizer"
    """Switch between Adam and L-BFGS."""

    INCREASE_BOUNDARY_WEIGHT = "increase_boundary_weight"
    """Increase boundary condition loss weight."""

    DECREASE_BOUNDARY_WEIGHT = "decrease_boundary_weight"
    """Decrease boundary condition loss weight."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


@dataclass
class PINNTrainingState:
    """State of a PINN training session.

    Captures everything needed to describe the current training
    configuration: collocation geometry, loss weighting, optimizer
    choice, and training history.
    """

    collocation_points: np.ndarray
    """Shape (N, dim) array of interior collocation coordinates."""

    boundary_points: np.ndarray
    """Shape (M, dim) array of boundary collocation coordinates."""

    physics_weight: float = 1.0
    """Weight applied to the PDE residual loss term."""

    boundary_weight: float = 1.0
    """Weight applied to the boundary condition loss term."""

    optimizer_type: str = "adam"
    """Current optimizer name ('adam' or 'lbfgs')."""

    current_loss: float = float("inf")
    """Most recent total training loss."""

    current_residual: float = float("inf")
    """Most recent PDE residual (L2 norm)."""

    step: int = 0
    """Number of planning steps taken so far."""

    training_history: list[dict[str, float]] = field(default_factory=list)
    """List of per-step metric dictionaries."""

    @property
    def num_collocation(self) -> int:
        """Return the number of interior collocation points."""
        return len(self.collocation_points)

    def clone(self) -> PINNTrainingState:
        """Return a deep, independent copy of this state."""
        return PINNTrainingState(
            collocation_points=self.collocation_points.copy(),
            boundary_points=self.boundary_points.copy(),
            physics_weight=self.physics_weight,
            boundary_weight=self.boundary_weight,
            optimizer_type=self.optimizer_type,
            current_loss=self.current_loss,
            current_residual=self.current_residual,
            step=self.step,
            training_history=[dict(h) for h in self.training_history],
        )


@dataclass
class PINNAction:
    """A single PINN training action.

    Attributes:
        action_type: The type of action to take.
        params: Optional parameters for the action (e.g. number of
            points to add, region bounds).

    """

    action_type: PINNActionType
    params: dict[str, Any] = field(default_factory=dict)


class PINNPlanner:
    """Plans PINN training using MCTS-style look-ahead.

    The planner treats PINN training as a sequential decision problem:
    - State: current collocation points, loss weights, optimizer choice
    - Actions: modify collocation, adjust weights, switch optimizer
    - Reward: reduction in PDE residual normalized by computational cost

    All tunable parameters are exposed through the constructor -- no
    magic numbers are hard-coded.

    Parameters
    ----------
    domain_bounds:
        Bounding box as a list of (min, max) tuples per dimension.
    max_collocation:
        Upper limit on interior collocation point count.
    min_collocation:
        Lower limit on interior collocation point count.
    weight_step:
        Multiplicative step when adjusting loss weights (added/subtracted).
    num_simulations:
        Number of look-ahead simulations per planning step.
    min_weight:
        Minimum allowed loss weight for physics/boundary terms.
    max_weight:
        Maximum allowed loss weight for physics/boundary terms.
    collocation_batch:
        Number of points to add/remove per action.

    """

    def __init__(
        self,
        domain_bounds: list[tuple[float, float]],
        max_collocation: int = 1000,
        min_collocation: int = 50,
        weight_step: float = 0.1,
        num_simulations: int = 50,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        collocation_batch: int = 20,
    ) -> None:
        self._bounds = domain_bounds
        self._max_collocation = max_collocation
        self._min_collocation = min_collocation
        self._weight_step = weight_step
        self._num_simulations = num_simulations
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._collocation_batch = collocation_batch
        self._dim = len(domain_bounds)
        self._rng = np.random.default_rng(42)

        logger.info(
            "pinn_planner.init",
            dim=self._dim,
            max_collocation=max_collocation,
            num_simulations=num_simulations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_next_action(
        self,
        state: PINNTrainingState,
        residual_fn: Callable[[np.ndarray], np.ndarray],
    ) -> PINNAction:
        """Use look-ahead to decide the next training modification.

        Evaluates every valid action by simulating its effect and
        scoring with the reward function.  Returns the action with
        the highest expected reward.

        Parameters
        ----------
        state:
            Current PINN training state.
        residual_fn:
            Callable ``(points: ndarray) -> ndarray`` that returns
            the absolute PDE residual at each query point.

        Returns
        -------
        PINNAction
            The best action found.

        """
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return PINNAction(action_type=PINNActionType.NO_OP)

        best_action = valid_actions[0]
        best_reward = -float("inf")

        for action in valid_actions:
            total_reward = 0.0
            for _ in range(self._num_simulations):
                new_state = self._simulate_action(state, action)
                # Evaluate residual at new collocation points
                residuals = residual_fn(new_state.collocation_points)
                new_state.current_residual = float(np.mean(np.abs(residuals)))
                reward = self._compute_reward(state, new_state)
                total_reward += reward

            avg_reward = total_reward / self._num_simulations
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action

        logger.info(
            "pinn_planner.plan",
            chosen_action=best_action.action_type.value,
            expected_reward=best_reward,
            step=state.step,
        )
        return best_action

    def get_valid_actions(self, state: PINNTrainingState) -> list[PINNAction]:
        """Return valid actions for the current state.

        Enforces collocation count bounds and weight bounds to
        prevent degenerate configurations.
        """
        actions: list[PINNAction] = []

        # Collocation management
        if state.num_collocation < self._max_collocation:
            actions.append(
                PINNAction(
                    action_type=PINNActionType.ADD_COLLOCATION,
                    params={"count": self._collocation_batch},
                )
            )
        if state.num_collocation > self._min_collocation:
            actions.append(
                PINNAction(
                    action_type=PINNActionType.REMOVE_COLLOCATION,
                    params={"count": min(
                        self._collocation_batch,
                        state.num_collocation - self._min_collocation,
                    )},
                )
            )

        # Physics weight
        if state.physics_weight + self._weight_step <= self._max_weight:
            actions.append(
                PINNAction(
                    action_type=PINNActionType.INCREASE_PHYSICS_WEIGHT,
                )
            )
        if state.physics_weight - self._weight_step >= self._min_weight:
            actions.append(
                PINNAction(
                    action_type=PINNActionType.DECREASE_PHYSICS_WEIGHT,
                )
            )

        # Boundary weight
        if state.boundary_weight + self._weight_step <= self._max_weight:
            actions.append(
                PINNAction(
                    action_type=PINNActionType.INCREASE_BOUNDARY_WEIGHT,
                )
            )
        if state.boundary_weight - self._weight_step >= self._min_weight:
            actions.append(
                PINNAction(
                    action_type=PINNActionType.DECREASE_BOUNDARY_WEIGHT,
                )
            )

        # Optimizer switch
        other_opt = "lbfgs" if state.optimizer_type == "adam" else "adam"
        actions.append(
            PINNAction(
                action_type=PINNActionType.SWITCH_OPTIMIZER,
                params={"target": other_opt},
            )
        )

        # No-op is always valid
        actions.append(PINNAction(action_type=PINNActionType.NO_OP))

        return actions

    def _simulate_action(
        self,
        state: PINNTrainingState,
        action: PINNAction,
    ) -> PINNTrainingState:
        """Simulate applying an action to get the next state.

        Returns a new state with the action applied.  Does not
        evaluate the neural network -- residual must be computed
        externally.
        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == PINNActionType.ADD_COLLOCATION:
            count = action.params.get("count", self._collocation_batch)
            new_points = self._sample_domain_points(count)
            new_state.collocation_points = np.concatenate(
                [new_state.collocation_points, new_points], axis=0,
            )

        elif action.action_type == PINNActionType.REMOVE_COLLOCATION:
            count = action.params.get("count", self._collocation_batch)
            count = min(
                count,
                len(new_state.collocation_points) - self._min_collocation,
            )
            if count > 0:
                # Remove random points (in real usage, remove low-residual)
                indices = self._rng.choice(
                    len(new_state.collocation_points),
                    size=len(new_state.collocation_points) - count,
                    replace=False,
                )
                new_state.collocation_points = (
                    new_state.collocation_points[indices]
                )

        elif action.action_type == PINNActionType.INCREASE_PHYSICS_WEIGHT:
            new_state.physics_weight = min(
                state.physics_weight + self._weight_step,
                self._max_weight,
            )

        elif action.action_type == PINNActionType.DECREASE_PHYSICS_WEIGHT:
            new_state.physics_weight = max(
                state.physics_weight - self._weight_step,
                self._min_weight,
            )

        elif action.action_type == PINNActionType.INCREASE_BOUNDARY_WEIGHT:
            new_state.boundary_weight = min(
                state.boundary_weight + self._weight_step,
                self._max_weight,
            )

        elif action.action_type == PINNActionType.DECREASE_BOUNDARY_WEIGHT:
            new_state.boundary_weight = max(
                state.boundary_weight - self._weight_step,
                self._min_weight,
            )

        elif action.action_type == PINNActionType.SWITCH_OPTIMIZER:
            target = action.params.get("target")
            if target is not None:
                new_state.optimizer_type = target
            else:
                new_state.optimizer_type = (
                    "lbfgs" if state.optimizer_type == "adam" else "adam"
                )

        # NO_OP: do nothing

        return new_state

    def _compute_reward(
        self,
        old_state: PINNTrainingState,
        new_state: PINNTrainingState,
    ) -> float:
        """Compute reward as residual reduction normalised by cost.

        Reward = (old_residual - new_residual) / cost_factor

        The cost factor accounts for the computational expense of
        having more collocation points and the optimizer choice.
        """
        # Residual improvement (positive = good)
        residual_delta = old_state.current_residual - new_state.current_residual

        # Cost factor: more points cost more, L-BFGS is more expensive
        point_cost = new_state.num_collocation / self._max_collocation
        optimizer_cost = 2.0 if new_state.optimizer_type == "lbfgs" else 1.0
        cost_factor = max(point_cost * optimizer_cost, 1e-8)

        return residual_delta / cost_factor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_domain_points(self, count: int) -> np.ndarray:
        """Sample uniform random points within domain bounds."""
        points = np.empty((count, self._dim))
        for d in range(self._dim):
            lo, hi = self._bounds[d]
            points[:, d] = self._rng.uniform(lo, hi, size=count)
        return points
