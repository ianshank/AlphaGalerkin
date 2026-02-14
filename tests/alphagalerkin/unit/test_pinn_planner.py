"""Tests for the PINN-as-Planning framework."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.pinn_planner import (
    PINNAction,
    PINNActionType,
    PINNPlanner,
    PINNTrainingState,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def sample_state() -> PINNTrainingState:
    """A minimal PINN training state for testing."""
    rng = np.random.default_rng(0)
    return PINNTrainingState(
        collocation_points=rng.uniform(0, 1, size=(100, 2)),
        boundary_points=rng.uniform(0, 1, size=(40, 2)),
        physics_weight=1.0,
        boundary_weight=1.0,
        optimizer_type="adam",
        current_loss=1.5,
        current_residual=0.8,
        step=0,
        training_history=[],
    )


@pytest.fixture()
def planner() -> PINNPlanner:
    """A planner configured for the [0,1]^2 domain."""
    return PINNPlanner(
        domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
        max_collocation=200,
        min_collocation=50,
        weight_step=0.1,
        num_simulations=5,
        collocation_batch=10,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestPINNStateClone:
    """PINNTrainingState.clone produces an independent copy."""

    def test_pinn_state_clone(self, sample_state: PINNTrainingState) -> None:
        cloned = sample_state.clone()

        # Must be a different object
        assert cloned is not sample_state
        assert cloned.collocation_points is not sample_state.collocation_points
        assert cloned.boundary_points is not sample_state.boundary_points
        assert cloned.training_history is not sample_state.training_history

        # Values must match
        np.testing.assert_array_equal(
            cloned.collocation_points, sample_state.collocation_points,
        )
        assert cloned.physics_weight == sample_state.physics_weight
        assert cloned.optimizer_type == sample_state.optimizer_type
        assert cloned.step == sample_state.step

    def test_clone_mutation_independence(
        self, sample_state: PINNTrainingState,
    ) -> None:
        cloned = sample_state.clone()
        cloned.physics_weight = 999.0
        cloned.collocation_points[0, 0] = -1.0

        assert sample_state.physics_weight == 1.0
        assert sample_state.collocation_points[0, 0] != -1.0


class TestPINNValidActions:
    """PINNPlanner.get_valid_actions returns correct actions."""

    def test_pinn_valid_actions(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        actions = planner.get_valid_actions(sample_state)

        # Must always contain at least NO_OP
        action_types = {a.action_type for a in actions}
        assert PINNActionType.NO_OP in action_types

        # With 100 points (< 200 max), ADD_COLLOCATION should be valid
        assert PINNActionType.ADD_COLLOCATION in action_types

        # With 100 points (> 50 min), REMOVE_COLLOCATION should be valid
        assert PINNActionType.REMOVE_COLLOCATION in action_types

        # Optimizer switch always valid
        assert PINNActionType.SWITCH_OPTIMIZER in action_types

    def test_no_add_at_max_collocation(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        """ADD_COLLOCATION is excluded at the max point limit."""
        rng = np.random.default_rng(1)
        sample_state.collocation_points = rng.uniform(0, 1, size=(200, 2))

        actions = planner.get_valid_actions(sample_state)
        action_types = {a.action_type for a in actions}
        assert PINNActionType.ADD_COLLOCATION not in action_types

    def test_no_remove_at_min_collocation(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        """REMOVE_COLLOCATION is excluded at the min point limit."""
        rng = np.random.default_rng(2)
        sample_state.collocation_points = rng.uniform(0, 1, size=(50, 2))

        actions = planner.get_valid_actions(sample_state)
        action_types = {a.action_type for a in actions}
        assert PINNActionType.REMOVE_COLLOCATION not in action_types


class TestPINNSimulateAddCollocation:
    """Simulating ADD_COLLOCATION increases point count."""

    def test_pinn_simulate_add_collocation(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        action = PINNAction(
            action_type=PINNActionType.ADD_COLLOCATION,
            params={"count": 10},
        )
        new_state = planner._simulate_action(sample_state, action)

        assert new_state.num_collocation == sample_state.num_collocation + 10
        assert new_state.step == sample_state.step + 1
        # Original state is unmodified
        assert sample_state.num_collocation == 100


class TestPINNSimulateWeightChange:
    """Simulating weight changes modifies the correct field."""

    def test_pinn_simulate_weight_change(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        # Increase physics weight
        action_up = PINNAction(
            action_type=PINNActionType.INCREASE_PHYSICS_WEIGHT,
        )
        state_up = planner._simulate_action(sample_state, action_up)
        assert state_up.physics_weight == pytest.approx(1.1)

        # Decrease physics weight
        action_down = PINNAction(
            action_type=PINNActionType.DECREASE_PHYSICS_WEIGHT,
        )
        state_down = planner._simulate_action(sample_state, action_down)
        assert state_down.physics_weight == pytest.approx(0.9)

        # Boundary weights
        action_bup = PINNAction(
            action_type=PINNActionType.INCREASE_BOUNDARY_WEIGHT,
        )
        state_bup = planner._simulate_action(sample_state, action_bup)
        assert state_bup.boundary_weight == pytest.approx(1.1)


class TestPINNSimulateOptimizerSwitch:
    """Simulating SWITCH_OPTIMIZER toggles the optimizer."""

    def test_pinn_simulate_optimizer_switch(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        assert sample_state.optimizer_type == "adam"

        action = PINNAction(
            action_type=PINNActionType.SWITCH_OPTIMIZER,
            params={"target": "lbfgs"},
        )
        new_state = planner._simulate_action(sample_state, action)
        assert new_state.optimizer_type == "lbfgs"

        # Switch back
        action_back = PINNAction(
            action_type=PINNActionType.SWITCH_OPTIMIZER,
            params={"target": "adam"},
        )
        state_back = planner._simulate_action(new_state, action_back)
        assert state_back.optimizer_type == "adam"


class TestPINNPlannerPlan:
    """PINNPlanner.plan_next_action returns a valid action."""

    def test_pinn_planner_plan(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        # A residual function that returns constant residual
        def residual_fn(points: np.ndarray) -> np.ndarray:
            return np.full(len(points), 0.5)

        action = planner.plan_next_action(sample_state, residual_fn)
        assert isinstance(action, PINNAction)
        assert isinstance(action.action_type, PINNActionType)

        # The returned action must be among the valid set
        valid_types = {
            a.action_type for a in planner.get_valid_actions(sample_state)
        }
        assert action.action_type in valid_types


class TestPINNRewardPositiveForImprovement:
    """Reward is positive when residual decreases."""

    def test_pinn_reward_positive_for_improvement(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        # Improved state has lower residual
        improved = sample_state.clone()
        improved.current_residual = 0.3  # was 0.8

        reward = planner._compute_reward(sample_state, improved)
        assert reward > 0.0

    def test_pinn_reward_negative_for_degradation(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        worse = sample_state.clone()
        worse.current_residual = 1.5  # was 0.8

        reward = planner._compute_reward(sample_state, worse)
        assert reward < 0.0

    def test_pinn_reward_zero_for_no_change(
        self,
        planner: PINNPlanner,
        sample_state: PINNTrainingState,
    ) -> None:
        same = sample_state.clone()
        same.current_residual = sample_state.current_residual

        reward = planner._compute_reward(sample_state, same)
        assert reward == pytest.approx(0.0)
