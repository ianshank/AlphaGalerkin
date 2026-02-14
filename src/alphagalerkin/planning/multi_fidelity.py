"""Multi-fidelity simulation management as a sequential decision game.

This module provides a planning layer that decides optimally between
running expensive high-fidelity simulations, cheaper low-fidelity
approximations, or training/querying a neural surrogate model.

The goal is to maximise information gain per unit computational cost
within a fixed budget, treating fidelity selection as a sequential
decision problem amenable to look-ahead planning.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("planning.multi_fidelity")


class FidelityLevel(str, Enum):
    """Available simulation fidelity levels."""

    HIGH = "high"
    """Full resolution solve -- most accurate, most expensive."""

    MEDIUM = "medium"
    """Reduced resolution -- moderate accuracy and cost."""

    LOW = "low"
    """Coarse approximation -- cheap but noisy."""

    SURROGATE = "surrogate"
    """Neural network surrogate -- near-zero cost after training."""


@dataclass
class SimulationPoint:
    """A point in parameter space with associated results.

    Attributes:
        parameters: Location in parameter space.
        fidelity: The fidelity level used for evaluation.
        result: The scalar simulation output (None if not yet evaluated).
        uncertainty: Estimated uncertainty of the result.
        cost: Computational cost incurred for this evaluation.

    """

    parameters: np.ndarray
    fidelity: FidelityLevel
    result: float | None = None
    uncertainty: float = float("inf")
    cost: float = 0.0


class FidelityActionType(str, Enum):
    """Actions for multi-fidelity management."""

    RUN_HIGH_FIDELITY = "run_high_fidelity"
    """Run a high-fidelity simulation at the target parameters."""

    RUN_MEDIUM_FIDELITY = "run_medium_fidelity"
    """Run a medium-fidelity simulation at the target parameters."""

    RUN_LOW_FIDELITY = "run_low_fidelity"
    """Run a low-fidelity simulation at the target parameters."""

    UPDATE_SURROGATE = "update_surrogate"
    """Retrain the surrogate model on all available data."""

    QUERY_SURROGATE = "query_surrogate"
    """Query the neural surrogate at the target parameters."""

    NO_OP = "no_op"
    """Do nothing -- useful when budget is exhausted."""


@dataclass
class MultiFidelityState:
    """State of a multi-fidelity simulation campaign.

    Attributes:
        parameter_space_bounds: Per-dimension (min, max) bounds.
        evaluated_points: All evaluated simulation points.
        surrogate_trained: Whether a surrogate is currently trained.
        total_cost: Cumulative computational cost so far.
        budget: Maximum allowed total cost.
        step: Number of planning steps taken.

    """

    parameter_space_bounds: list[tuple[float, float]]
    evaluated_points: list[SimulationPoint] = field(default_factory=list)
    surrogate_trained: bool = False
    total_cost: float = 0.0
    budget: float = 100.0
    step: int = 0

    @property
    def remaining_budget(self) -> float:
        """Return the budget that has not yet been spent."""
        return self.budget - self.total_cost

    def clone(self) -> MultiFidelityState:
        """Return a deep, independent copy of this state."""
        return MultiFidelityState(
            parameter_space_bounds=list(self.parameter_space_bounds),
            evaluated_points=[
                SimulationPoint(
                    parameters=pt.parameters.copy(),
                    fidelity=pt.fidelity,
                    result=pt.result,
                    uncertainty=pt.uncertainty,
                    cost=pt.cost,
                )
                for pt in self.evaluated_points
            ],
            surrogate_trained=self.surrogate_trained,
            total_cost=self.total_cost,
            budget=self.budget,
            step=self.step,
        )


@dataclass
class FidelityAction:
    """A multi-fidelity simulation action.

    Attributes:
        action_type: What kind of evaluation to perform.
        target_parameters: Where in parameter space (None for
            UPDATE_SURROGATE and NO_OP).

    """

    action_type: FidelityActionType
    target_parameters: np.ndarray | None = None


class MultiFidelityManager:
    """Manages multi-fidelity simulations using look-ahead planning.

    Decides optimally between running expensive high-fidelity
    simulations, running cheaper low-fidelity approximations,
    and training/querying a neural surrogate model.

    The goal is to maximise information gain per unit computational
    cost within a fixed evaluation budget.

    Parameters
    ----------
    parameter_bounds:
        Per-dimension (min, max) bounds for the parameter space.
    cost_ratios:
        Mapping of fidelity/operation names to cost values.
        Defaults to ``{"high": 10.0, "medium": 3.0, "low": 1.0,
        "surrogate": 0.01, "update_surrogate": 2.0}``.
    budget:
        Total computational budget.

    """

    def __init__(
        self,
        parameter_bounds: list[tuple[float, float]],
        cost_ratios: dict[str, float] | None = None,
        budget: float = 100.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._bounds = parameter_bounds
        self._costs = cost_ratios or {
            "high": 10.0,
            "medium": 3.0,
            "low": 1.0,
            "surrogate": 0.01,
            "update_surrogate": 2.0,
        }
        self._budget = budget
        self._dim = len(parameter_bounds)
        self._rng = np.random.default_rng(seed)

        logger.info(
            "multi_fidelity.init",
            dim=self._dim,
            budget=budget,
            costs=self._costs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_next_evaluation(
        self,
        state: MultiFidelityState,
        objective_fn: Callable[..., Any] | None = None,
    ) -> FidelityAction:
        """Plan the next evaluation using uncertainty-guided selection.

        Selects the action that maximises estimated information gain
        per unit cost, subject to the remaining budget constraint.

        Parameters
        ----------
        state:
            Current multi-fidelity campaign state.
        objective_fn:
            Optional callable for evaluating candidate points.
            If provided, may be used for look-ahead simulation.

        Returns
        -------
        FidelityAction
            The recommended next action.

        """
        valid = self.get_valid_actions(state)
        if not valid:
            return FidelityAction(action_type=FidelityActionType.NO_OP)

        best_action = valid[0]
        best_score = -float("inf")

        for action in valid:
            score = self._compute_information_gain(state, action)
            if score > best_score:
                best_score = score
                best_action = action

        logger.info(
            "multi_fidelity.plan",
            chosen_action=best_action.action_type.value,
            score=best_score,
            remaining_budget=state.remaining_budget,
        )
        return best_action

    def get_valid_actions(
        self,
        state: MultiFidelityState,
    ) -> list[FidelityAction]:
        """Return actions that fit within the remaining budget.

        Only actions whose cost does not exceed ``remaining_budget``
        are included.
        """
        actions: list[FidelityAction] = []
        remaining = state.remaining_budget
        target = self._sample_candidate_point(state)

        # Simulation actions at different fidelities
        if remaining >= self._costs["high"]:
            actions.append(
                FidelityAction(
                    action_type=FidelityActionType.RUN_HIGH_FIDELITY,
                    target_parameters=target,
                )
            )
        if remaining >= self._costs["medium"]:
            actions.append(
                FidelityAction(
                    action_type=FidelityActionType.RUN_MEDIUM_FIDELITY,
                    target_parameters=target,
                )
            )
        if remaining >= self._costs["low"]:
            actions.append(
                FidelityAction(
                    action_type=FidelityActionType.RUN_LOW_FIDELITY,
                    target_parameters=target,
                )
            )

        # Surrogate operations
        if (
            remaining >= self._costs["update_surrogate"]
            and len(state.evaluated_points) >= 3
        ):
            actions.append(
                FidelityAction(
                    action_type=FidelityActionType.UPDATE_SURROGATE,
                )
            )
        if state.surrogate_trained and remaining >= self._costs["surrogate"]:
            actions.append(
                FidelityAction(
                    action_type=FidelityActionType.QUERY_SURROGATE,
                    target_parameters=target,
                )
            )

        # No-op is always valid
        actions.append(FidelityAction(action_type=FidelityActionType.NO_OP))

        return actions

    def apply_action(
        self,
        state: MultiFidelityState,
        action: FidelityAction,
        result: float | None = None,
    ) -> MultiFidelityState:
        """Apply an action to produce a new state.

        Parameters
        ----------
        state:
            Current campaign state.
        action:
            The action to apply.
        result:
            The simulation result (required for simulation actions).

        Returns
        -------
        MultiFidelityState
            The updated state.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        cost = self._action_cost(action)
        new_state.total_cost += cost

        if action.action_type == FidelityActionType.RUN_HIGH_FIDELITY:
            if action.target_parameters is not None:
                new_state.evaluated_points.append(
                    SimulationPoint(
                        parameters=action.target_parameters.copy(),
                        fidelity=FidelityLevel.HIGH,
                        result=result,
                        uncertainty=0.01,
                        cost=cost,
                    )
                )

        elif action.action_type == FidelityActionType.RUN_MEDIUM_FIDELITY:
            if action.target_parameters is not None:
                new_state.evaluated_points.append(
                    SimulationPoint(
                        parameters=action.target_parameters.copy(),
                        fidelity=FidelityLevel.MEDIUM,
                        result=result,
                        uncertainty=0.1,
                        cost=cost,
                    )
                )

        elif action.action_type == FidelityActionType.RUN_LOW_FIDELITY:
            if action.target_parameters is not None:
                new_state.evaluated_points.append(
                    SimulationPoint(
                        parameters=action.target_parameters.copy(),
                        fidelity=FidelityLevel.LOW,
                        result=result,
                        uncertainty=0.5,
                        cost=cost,
                    )
                )

        elif action.action_type == FidelityActionType.UPDATE_SURROGATE:
            new_state.surrogate_trained = True

        elif action.action_type == FidelityActionType.QUERY_SURROGATE:
            if action.target_parameters is not None:
                new_state.evaluated_points.append(
                    SimulationPoint(
                        parameters=action.target_parameters.copy(),
                        fidelity=FidelityLevel.SURROGATE,
                        result=result,
                        uncertainty=0.3,
                        cost=cost,
                    )
                )

        # NO_OP: do nothing

        return new_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_information_gain(
        self,
        state: MultiFidelityState,
        action: FidelityAction,
    ) -> float:
        """Estimate information gain from an evaluation.

        Uses a simple heuristic: higher-fidelity evaluations yield
        more information (lower uncertainty), and the gain is
        normalised by cost.  Points far from existing evaluations
        also score higher (exploration bonus).
        """
        if action.action_type == FidelityActionType.NO_OP:
            return 0.0

        cost = self._action_cost(action)
        if cost <= 0:
            return 0.0

        # Information quality based on fidelity
        fidelity_info = {
            FidelityActionType.RUN_HIGH_FIDELITY: 1.0,
            FidelityActionType.RUN_MEDIUM_FIDELITY: 0.6,
            FidelityActionType.RUN_LOW_FIDELITY: 0.3,
            FidelityActionType.QUERY_SURROGATE: 0.2,
            FidelityActionType.UPDATE_SURROGATE: 0.4,
        }
        info = fidelity_info.get(action.action_type, 0.0)

        # Exploration bonus: distance to nearest evaluated point
        exploration_bonus = 1.0
        if (
            action.target_parameters is not None
            and state.evaluated_points
        ):
            min_dist = float("inf")
            for pt in state.evaluated_points:
                dist = float(np.linalg.norm(
                    action.target_parameters - pt.parameters,
                ))
                min_dist = min(min_dist, dist)
            # Normalise by domain diameter
            diameter = np.sqrt(sum(
                (hi - lo) ** 2 for lo, hi in self._bounds
            ))
            if diameter > 0:
                exploration_bonus = 1.0 + min_dist / diameter

        return (info * exploration_bonus) / cost

    def _action_cost(self, action: FidelityAction) -> float:
        """Return the computational cost of an action."""
        cost_map = {
            FidelityActionType.RUN_HIGH_FIDELITY: self._costs["high"],
            FidelityActionType.RUN_MEDIUM_FIDELITY: self._costs["medium"],
            FidelityActionType.RUN_LOW_FIDELITY: self._costs["low"],
            FidelityActionType.UPDATE_SURROGATE: self._costs["update_surrogate"],
            FidelityActionType.QUERY_SURROGATE: self._costs["surrogate"],
            FidelityActionType.NO_OP: 0.0,
        }
        return cost_map.get(action.action_type, 0.0)

    def _sample_candidate_point(
        self,
        state: MultiFidelityState,
    ) -> np.ndarray:
        """Sample a candidate evaluation point.

        Uses a simple space-filling heuristic: generate several
        random candidates and pick the one farthest from all
        existing evaluations.
        """
        n_candidates = 10
        candidates = np.empty((n_candidates, self._dim))
        for d in range(self._dim):
            lo, hi = self._bounds[d]
            candidates[:, d] = self._rng.uniform(lo, hi, size=n_candidates)

        if not state.evaluated_points:
            return candidates[0]  # type: ignore[no-any-return]

        existing = np.array([pt.parameters for pt in state.evaluated_points])
        best_idx = 0
        best_min_dist = -1.0
        for i in range(n_candidates):
            dists = np.linalg.norm(existing - candidates[i], axis=1)
            min_dist = float(np.min(dists))
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        return candidates[best_idx]  # type: ignore[no-any-return]
