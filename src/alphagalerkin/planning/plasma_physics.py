"""MCTS-based planning for plasma physics and stellarator optimization.

This module applies look-ahead planning to:

1. **Stellarator coil design**: Non-convex optimization over coil
   geometry with competing physics objectives (MHD stability,
   neoclassical transport, fast particle confinement).

2. **Plasma model selection**: Deciding when to use kinetic vs fluid
   descriptions in different plasma regions, treated as a sequential
   game.

3. **Anticipatory plasma control**: Planning ahead through possible
   evolution trajectories for real-time ELM suppression.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("planning.plasma")


# ======================================================================
# Stellarator Coil Design
# ======================================================================


class CoilActionType(str, Enum):
    """Actions for stellarator coil optimization."""

    ADJUST_CURRENT = "adjust_current"
    """Modify the current flowing through a coil."""

    MOVE_COIL_POINT = "move_coil_point"
    """Shift a Fourier control point on a coil."""

    ADD_COIL = "add_coil"
    """Insert a new coil into the configuration."""

    REMOVE_COIL = "remove_coil"
    """Remove an existing coil from the configuration."""

    ADJUST_WINDING = "adjust_winding"
    """Change the winding number of a coil."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


@dataclass
class CoilGeometry:
    """Represents a single stellarator coil.

    Attributes:
        control_points: Shape (num_points, 3) Fourier control points.
        current: Coil current in MA.
        winding_number: Number of toroidal windings.

    """

    control_points: np.ndarray
    current: float = 1.0
    winding_number: int = 1

    def clone(self) -> CoilGeometry:
        """Return a deep, independent copy of this coil."""
        return CoilGeometry(
            control_points=self.control_points.copy(),
            current=self.current,
            winding_number=self.winding_number,
        )


@dataclass
class StellaratorState:
    """State of a stellarator coil optimization.

    Attributes:
        coils: Current coil configurations.
        num_field_periods: Number of toroidal field periods.
        target_aspect_ratio: Desired aspect ratio of the plasma.
        mhd_stability_metric: MHD stability metric (lower is better).
        neoclassical_transport: Neoclassical transport level (lower is better).
        fast_particle_loss: Fast particle loss fraction (lower is better).
        coil_complexity: Geometric complexity penalty (lower is better).
        step: Number of planning steps taken so far.
        max_coils: Upper limit on the number of coils.

    """

    coils: list[CoilGeometry]
    num_field_periods: int = 5
    target_aspect_ratio: float = 10.0

    # Physics objectives (lower is better)
    mhd_stability_metric: float = float("inf")
    neoclassical_transport: float = float("inf")
    fast_particle_loss: float = float("inf")
    coil_complexity: float = 0.0

    step: int = 0
    max_coils: int = 20

    def clone(self) -> StellaratorState:
        """Return a deep, independent copy of this state."""
        return StellaratorState(
            coils=[c.clone() for c in self.coils],
            num_field_periods=self.num_field_periods,
            target_aspect_ratio=self.target_aspect_ratio,
            mhd_stability_metric=self.mhd_stability_metric,
            neoclassical_transport=self.neoclassical_transport,
            fast_particle_loss=self.fast_particle_loss,
            coil_complexity=self.coil_complexity,
            step=self.step,
            max_coils=self.max_coils,
        )

    @property
    def total_objective(self) -> float:
        """Multi-objective scalar (weighted sum of metrics).

        Returns the sum of all finite objective components plus the
        coil complexity.  Infinite values are excluded so the sum
        remains meaningful even before all metrics are populated.
        """
        total = 0.0
        for metric in (
            self.mhd_stability_metric,
            self.neoclassical_transport,
            self.fast_particle_loss,
        ):
            if np.isfinite(metric):
                total += metric
        total += self.coil_complexity
        return total


@dataclass
class CoilAction:
    """A coil optimization action.

    Attributes:
        action_type: The type of coil modification.
        coil_index: Index of the coil to modify.
        params: Optional parameters for the action (e.g. delta values).

    """

    action_type: CoilActionType
    coil_index: int = 0
    params: dict[str, Any] = field(default_factory=dict)


class StellaratorOptimizer:
    """Optimizes stellarator coil design using look-ahead planning.

    Evaluates candidate actions by simulating their effect on a
    multi-objective cost function combining MHD stability,
    neoclassical transport, fast particle confinement, and coil
    geometric complexity.

    All tunable parameters are exposed through the constructor.

    Parameters
    ----------
    max_coils:
        Upper limit on the number of coils.
    num_simulations:
        Number of look-ahead simulations per planning step.
    stability_weight:
        Weight for the MHD stability objective.
    transport_weight:
        Weight for the neoclassical transport objective.
    complexity_weight:
        Weight for the coil complexity penalty.
    current_step:
        Magnitude of current adjustments (MA).
    position_step:
        Magnitude of control point perturbations.

    """

    def __init__(
        self,
        max_coils: int = 20,
        num_simulations: int = 50,
        stability_weight: float = 1.0,
        transport_weight: float = 1.0,
        complexity_weight: float = 0.3,
        current_step: float = 0.01,
        position_step: float = 0.01,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._max_coils = max_coils
        self._num_simulations = num_simulations
        self._stability_weight = stability_weight
        self._transport_weight = transport_weight
        self._complexity_weight = complexity_weight
        self._current_step = current_step
        self._position_step = position_step
        self._rng = np.random.default_rng(seed)

        logger.info(
            "stellarator_optimizer.init",
            max_coils=max_coils,
            num_simulations=num_simulations,
            stability_weight=stability_weight,
            transport_weight=transport_weight,
            complexity_weight=complexity_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_next_action(
        self,
        state: StellaratorState,
        physics_fn: Callable[[StellaratorState], dict[str, float]] | None = None,
    ) -> CoilAction:
        """Use look-ahead to decide the next coil modification.

        Evaluates every valid action by simulating its effect and
        scoring with the reward function.  Returns the action with
        the highest expected reward.

        Parameters
        ----------
        state:
            Current stellarator optimization state.
        physics_fn:
            Optional callable ``(state) -> dict`` returning physics
            metrics (keys: ``mhd_stability``, ``neoclassical_transport``,
            ``fast_particle_loss``).  When ``None``, a simple
            heuristic proxy is used.

        Returns
        -------
        CoilAction
            The best action found.

        """
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return CoilAction(action_type=CoilActionType.NO_OP)

        best_action = valid_actions[0]
        best_reward = -float("inf")

        for action in valid_actions:
            total_reward = 0.0
            for _ in range(self._num_simulations):
                new_state = self.apply_action(state, action)
                if physics_fn is not None:
                    metrics = physics_fn(new_state)
                    new_state.mhd_stability_metric = metrics.get(
                        "mhd_stability", new_state.mhd_stability_metric,
                    )
                    new_state.neoclassical_transport = metrics.get(
                        "neoclassical_transport",
                        new_state.neoclassical_transport,
                    )
                    new_state.fast_particle_loss = metrics.get(
                        "fast_particle_loss", new_state.fast_particle_loss,
                    )
                reward = self._compute_reward(state, new_state)
                total_reward += reward

            avg_reward = total_reward / self._num_simulations
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action

        logger.info(
            "stellarator_optimizer.plan",
            chosen_action=best_action.action_type.value,
            expected_reward=best_reward,
            step=state.step,
        )
        return best_action

    def get_valid_actions(
        self,
        state: StellaratorState,
    ) -> list[CoilAction]:
        """Return valid actions for the current state.

        Enforces coil count bounds and ensures indices are within
        range to prevent degenerate configurations.
        """
        actions: list[CoilAction] = []

        # Per-coil actions
        for i in range(len(state.coils)):
            actions.append(
                CoilAction(
                    action_type=CoilActionType.ADJUST_CURRENT,
                    coil_index=i,
                    params={"delta": self._current_step},
                )
            )
            actions.append(
                CoilAction(
                    action_type=CoilActionType.ADJUST_CURRENT,
                    coil_index=i,
                    params={"delta": -self._current_step},
                )
            )
            actions.append(
                CoilAction(
                    action_type=CoilActionType.MOVE_COIL_POINT,
                    coil_index=i,
                    params={"step_size": self._position_step},
                )
            )
            actions.append(
                CoilAction(
                    action_type=CoilActionType.ADJUST_WINDING,
                    coil_index=i,
                    params={"delta": 1},
                )
            )

        # Add coil (if below max)
        if len(state.coils) < state.max_coils:
            actions.append(
                CoilAction(
                    action_type=CoilActionType.ADD_COIL,
                    params={"num_control_points": 10},
                )
            )

        # Remove coil (if more than 1)
        if len(state.coils) > 1:
            for i in range(len(state.coils)):
                actions.append(
                    CoilAction(
                        action_type=CoilActionType.REMOVE_COIL,
                        coil_index=i,
                    )
                )

        # No-op is always valid
        actions.append(CoilAction(action_type=CoilActionType.NO_OP))

        return actions

    def apply_action(
        self,
        state: StellaratorState,
        action: CoilAction,
    ) -> StellaratorState:
        """Apply an action to produce a new stellarator state.

        Returns a new state with the action applied.  Does not
        evaluate physics -- metrics must be computed externally
        or via a physics function passed to ``plan_next_action``.

        Parameters
        ----------
        state:
            Current stellarator state.
        action:
            The coil modification action to apply.

        Returns
        -------
        StellaratorState
            The updated state.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == CoilActionType.ADJUST_CURRENT:
            idx = action.coil_index
            if 0 <= idx < len(new_state.coils):
                delta = action.params.get("delta", self._current_step)
                new_state.coils[idx].current += delta

        elif action.action_type == CoilActionType.MOVE_COIL_POINT:
            idx = action.coil_index
            if 0 <= idx < len(new_state.coils):
                step_size = action.params.get(
                    "step_size", self._position_step,
                )
                # Perturb a random control point
                coil = new_state.coils[idx]
                pt_idx = int(
                    self._rng.integers(0, len(coil.control_points))
                )
                perturbation = self._rng.normal(
                    0, step_size, size=coil.control_points.shape[1],
                )
                coil.control_points[pt_idx] += perturbation

        elif action.action_type == CoilActionType.ADD_COIL:
            num_pts = action.params.get("num_control_points", 10)
            if len(new_state.coils) < new_state.max_coils:
                new_coil = CoilGeometry(
                    control_points=self._rng.normal(0, 1, size=(num_pts, 3)),
                    current=1.0,
                    winding_number=1,
                )
                new_state.coils.append(new_coil)
                # Complexity increases with coil count
                new_state.coil_complexity += 1.0

        elif action.action_type == CoilActionType.REMOVE_COIL:
            idx = action.coil_index
            if 0 <= idx < len(new_state.coils) and len(new_state.coils) > 1:
                new_state.coils.pop(idx)
                # Complexity decreases with coil count
                new_state.coil_complexity = max(
                    0.0, new_state.coil_complexity - 1.0,
                )

        elif action.action_type == CoilActionType.ADJUST_WINDING:
            idx = action.coil_index
            if 0 <= idx < len(new_state.coils):
                delta = action.params.get("delta", 1)
                new_state.coils[idx].winding_number = max(
                    1, new_state.coils[idx].winding_number + delta,
                )

        # NO_OP: do nothing

        return new_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        old_state: StellaratorState,
        new_state: StellaratorState,
    ) -> float:
        """Compute reward as weighted objective improvement.

        Reward is positive when the total objective decreases (i.e.
        physics metrics improve).  The reward is normalized by a
        complexity cost factor.

        Parameters
        ----------
        old_state:
            State before the action.
        new_state:
            State after the action.

        Returns
        -------
        float
            Signed reward (positive = improvement).

        """
        old_obj = self._weighted_objective(old_state)
        new_obj = self._weighted_objective(new_state)

        # Objective improvement (positive = good, since lower obj is better)
        improvement = old_obj - new_obj

        # Normalize by complexity cost
        complexity_cost = max(
            1.0 + self._complexity_weight * new_state.coil_complexity,
            1e-8,
        )

        return improvement / complexity_cost

    def _weighted_objective(self, state: StellaratorState) -> float:
        """Compute the weighted multi-objective scalar."""
        total = 0.0

        if np.isfinite(state.mhd_stability_metric):
            total += self._stability_weight * state.mhd_stability_metric

        if np.isfinite(state.neoclassical_transport):
            total += self._transport_weight * state.neoclassical_transport

        if np.isfinite(state.fast_particle_loss):
            total += state.fast_particle_loss

        total += self._complexity_weight * state.coil_complexity

        return total


# ======================================================================
# Plasma Model Selection
# ======================================================================


class PlasmaModelType(str, Enum):
    """Available plasma physics models."""

    MHD = "mhd"
    """Magnetohydrodynamics."""

    KINETIC = "kinetic"
    """Full kinetic (Vlasov)."""

    GYROKINETIC = "gyrokinetic"
    """Gyrokinetic approximation."""

    HYBRID = "hybrid"
    """Kinetic ions + fluid electrons."""

    FLUID = "fluid"
    """Multi-fluid description."""


class ModelSelectionActionType(str, Enum):
    """Actions for plasma model selection."""

    SET_REGION_MODEL = "set_region_model"
    """Assign a physics model to a spatial region."""

    SPLIT_REGION = "split_region"
    """Split a region into two sub-regions."""

    MERGE_REGIONS = "merge_regions"
    """Merge two adjacent regions."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


@dataclass
class PlasmaRegion:
    """A spatial region with an assigned physics model.

    Attributes:
        bounds: 1D interval (min, max) for this region.
        model: The physics model assigned to this region.
        accuracy: Solution quality estimate (lower is better).
        cost_per_step: Relative computational cost per time step.

    """

    bounds: tuple[float, float]
    model: PlasmaModelType = PlasmaModelType.MHD
    accuracy: float = 0.0
    cost_per_step: float = 1.0


@dataclass
class PlasmaModelState:
    """State of plasma model selection across regions.

    Attributes:
        regions: Spatial regions with assigned models.
        total_cost: Cumulative computational cost so far.
        budget: Maximum allowed total cost.
        accuracy_target: Target accuracy threshold.
        step: Number of planning steps taken.

    """

    regions: list[PlasmaRegion]
    total_cost: float = 0.0
    budget: float = 100.0
    accuracy_target: float = 0.01
    step: int = 0

    def clone(self) -> PlasmaModelState:
        """Return a deep, independent copy of this state."""
        return PlasmaModelState(
            regions=[
                PlasmaRegion(
                    bounds=r.bounds,
                    model=r.model,
                    accuracy=r.accuracy,
                    cost_per_step=r.cost_per_step,
                )
                for r in self.regions
            ],
            total_cost=self.total_cost,
            budget=self.budget,
            accuracy_target=self.accuracy_target,
            step=self.step,
        )


@dataclass
class ModelSelectionAction:
    """A model selection action.

    Attributes:
        action_type: The type of model selection modification.
        region_index: Index of the region to modify.
        target_model: The target physics model (for SET_REGION_MODEL).
        params: Optional parameters for the action.

    """

    action_type: ModelSelectionActionType
    region_index: int = 0
    target_model: PlasmaModelType = PlasmaModelType.MHD
    params: dict[str, Any] = field(default_factory=dict)


class PlasmaModelSelector:
    """Selects optimal physics models for plasma regions.

    Uses look-ahead planning to balance accuracy and computational
    cost when assigning physics models (MHD, kinetic, gyrokinetic,
    hybrid, fluid) to spatial regions of a plasma simulation.

    Parameters
    ----------
    num_simulations:
        Number of look-ahead simulations per planning step.
    budget:
        Total computational budget for the simulation campaign.
    model_costs:
        Mapping of model names to relative computational costs.
        Defaults to reasonable physics-based ratios.

    """

    DEFAULT_MODEL_COSTS: dict[str, float] = {
        "mhd": 1.0,
        "fluid": 2.0,
        "hybrid": 5.0,
        "gyrokinetic": 10.0,
        "kinetic": 50.0,
    }

    def __init__(
        self,
        num_simulations: int = 50,
        budget: float = 100.0,
        model_costs: dict[str, float] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._num_simulations = num_simulations
        self._budget = budget
        self._model_costs = model_costs or dict(self.DEFAULT_MODEL_COSTS)
        self._rng = np.random.default_rng(seed)

        logger.info(
            "plasma_model_selector.init",
            num_simulations=num_simulations,
            budget=budget,
            model_costs=self._model_costs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_next_action(
        self,
        state: PlasmaModelState,
    ) -> ModelSelectionAction:
        """Use look-ahead to decide the next model selection action.

        Evaluates every valid action by simulating its effect and
        scoring with the reward function.  Returns the action with
        the highest expected reward.

        Parameters
        ----------
        state:
            Current plasma model selection state.

        Returns
        -------
        ModelSelectionAction
            The best action found.

        """
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return ModelSelectionAction(
                action_type=ModelSelectionActionType.NO_OP,
            )

        best_action = valid_actions[0]
        best_reward = -float("inf")

        for action in valid_actions:
            total_reward = 0.0
            for _ in range(self._num_simulations):
                new_state = self.apply_action(state, action)
                reward = self._compute_reward(state, new_state)
                total_reward += reward

            avg_reward = total_reward / self._num_simulations
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action

        logger.info(
            "plasma_model_selector.plan",
            chosen_action=best_action.action_type.value,
            expected_reward=best_reward,
            step=state.step,
        )
        return best_action

    def get_valid_actions(
        self,
        state: PlasmaModelState,
    ) -> list[ModelSelectionAction]:
        """Return valid actions for the current state.

        Enforces budget constraints and ensures region indices are
        within range.
        """
        actions: list[ModelSelectionAction] = []
        remaining = state.budget - state.total_cost

        # Set model for each region (to every model type that fits budget)
        for i, region in enumerate(state.regions):
            for model_type in PlasmaModelType:
                cost = self._model_costs.get(model_type.value, 1.0)
                if cost <= remaining and model_type != region.model:
                    actions.append(
                        ModelSelectionAction(
                            action_type=ModelSelectionActionType.SET_REGION_MODEL,
                            region_index=i,
                            target_model=model_type,
                        )
                    )

        # Split region (if region is wide enough)
        for i, region in enumerate(state.regions):
            width = region.bounds[1] - region.bounds[0]
            if width > 0.1:  # Minimum region width threshold
                actions.append(
                    ModelSelectionAction(
                        action_type=ModelSelectionActionType.SPLIT_REGION,
                        region_index=i,
                    )
                )

        # Merge adjacent regions
        for i in range(len(state.regions) - 1):
            if state.regions[i].model == state.regions[i + 1].model:
                actions.append(
                    ModelSelectionAction(
                        action_type=ModelSelectionActionType.MERGE_REGIONS,
                        region_index=i,
                    )
                )

        # No-op is always valid
        actions.append(
            ModelSelectionAction(
                action_type=ModelSelectionActionType.NO_OP,
            )
        )

        return actions

    def apply_action(
        self,
        state: PlasmaModelState,
        action: ModelSelectionAction,
    ) -> PlasmaModelState:
        """Apply an action to produce a new state.

        Parameters
        ----------
        state:
            Current model selection state.
        action:
            The model selection action to apply.

        Returns
        -------
        PlasmaModelState
            The updated state.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == ModelSelectionActionType.SET_REGION_MODEL:
            idx = action.region_index
            if 0 <= idx < len(new_state.regions):
                old_cost = self._model_costs.get(
                    new_state.regions[idx].model.value, 1.0,
                )
                new_cost = self._model_costs.get(
                    action.target_model.value, 1.0,
                )
                new_state.regions[idx].model = action.target_model
                new_state.regions[idx].cost_per_step = new_cost
                # Update total cost with the cost difference
                new_state.total_cost += new_cost - old_cost

        elif action.action_type == ModelSelectionActionType.SPLIT_REGION:
            idx = action.region_index
            if 0 <= idx < len(new_state.regions):
                region = new_state.regions[idx]
                mid = (region.bounds[0] + region.bounds[1]) / 2.0
                left = PlasmaRegion(
                    bounds=(region.bounds[0], mid),
                    model=region.model,
                    accuracy=region.accuracy,
                    cost_per_step=region.cost_per_step,
                )
                right = PlasmaRegion(
                    bounds=(mid, region.bounds[1]),
                    model=region.model,
                    accuracy=region.accuracy,
                    cost_per_step=region.cost_per_step,
                )
                new_state.regions[idx:idx + 1] = [left, right]

        elif action.action_type == ModelSelectionActionType.MERGE_REGIONS:
            idx = action.region_index
            if 0 <= idx < len(new_state.regions) - 1:
                r1 = new_state.regions[idx]
                r2 = new_state.regions[idx + 1]
                merged = PlasmaRegion(
                    bounds=(r1.bounds[0], r2.bounds[1]),
                    model=r1.model,
                    accuracy=max(r1.accuracy, r2.accuracy),
                    cost_per_step=r1.cost_per_step,
                )
                new_state.regions[idx:idx + 2] = [merged]

        # NO_OP: do nothing

        return new_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        old_state: PlasmaModelState,
        new_state: PlasmaModelState,
    ) -> float:
        """Compute reward based on accuracy improvement per unit cost.

        Reward is positive when accuracy improves (total accuracy
        error decreases) relative to the computational cost incurred.
        """
        old_accuracy_error = sum(r.accuracy for r in old_state.regions)
        new_accuracy_error = sum(r.accuracy for r in new_state.regions)

        # Accuracy improvement (positive = good)
        improvement = old_accuracy_error - new_accuracy_error

        # Cost factor: higher cost reduces reward
        cost_factor = max(
            sum(r.cost_per_step for r in new_state.regions)
            / max(self._budget, 1e-8),
            1e-8,
        )

        return improvement / cost_factor
