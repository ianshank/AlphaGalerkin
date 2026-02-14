"""Tests for the Multi-Fidelity Simulation Manager."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.multi_fidelity import (
    FidelityAction,
    FidelityActionType,
    FidelityLevel,
    MultiFidelityManager,
    MultiFidelityState,
    SimulationPoint,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def parameter_bounds() -> list[tuple[float, float]]:
    """2D parameter space on [0, 1]^2."""
    return [(0.0, 1.0), (0.0, 1.0)]


@pytest.fixture()
def manager(parameter_bounds: list[tuple[float, float]]) -> MultiFidelityManager:
    """A manager with default cost ratios and 100-unit budget."""
    return MultiFidelityManager(
        parameter_bounds=parameter_bounds,
        budget=100.0,
    )


@pytest.fixture()
def initial_state(
    parameter_bounds: list[tuple[float, float]],
) -> MultiFidelityState:
    """A fresh campaign state."""
    return MultiFidelityState(
        parameter_space_bounds=parameter_bounds,
        budget=100.0,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestMultiFidelityStateBudget:
    """MultiFidelityState budget tracking."""

    def test_multi_fidelity_state_budget(
        self,
        initial_state: MultiFidelityState,
    ) -> None:
        assert initial_state.remaining_budget == 100.0
        assert initial_state.total_cost == 0.0

    def test_remaining_budget_decreases(
        self,
        initial_state: MultiFidelityState,
    ) -> None:
        initial_state.total_cost = 30.0
        assert initial_state.remaining_budget == pytest.approx(70.0)

    def test_clone_preserves_budget(
        self,
        initial_state: MultiFidelityState,
    ) -> None:
        initial_state.total_cost = 25.0
        cloned = initial_state.clone()
        assert cloned.remaining_budget == pytest.approx(75.0)
        assert cloned is not initial_state


class TestMultiFidelityValidActionsBudgetConstraint:
    """Valid actions respect the budget constraint."""

    def test_multi_fidelity_valid_actions_budget_constraint(
        self,
        manager: MultiFidelityManager,
    ) -> None:
        # With only 2 units of budget: HIGH (10) and MEDIUM (3)
        # should be excluded, but LOW (1) still fits.
        state = MultiFidelityState(
            parameter_space_bounds=[(0.0, 1.0), (0.0, 1.0)],
            total_cost=98.0,
            budget=100.0,
        )
        actions = manager.get_valid_actions(state)
        action_types = {a.action_type for a in actions}

        assert FidelityActionType.RUN_HIGH_FIDELITY not in action_types
        assert FidelityActionType.RUN_MEDIUM_FIDELITY not in action_types
        assert FidelityActionType.RUN_LOW_FIDELITY in action_types
        assert FidelityActionType.NO_OP in action_types

    def test_all_fidelities_with_full_budget(
        self,
        manager: MultiFidelityManager,
        initial_state: MultiFidelityState,
    ) -> None:
        actions = manager.get_valid_actions(initial_state)
        action_types = {a.action_type for a in actions}

        assert FidelityActionType.RUN_HIGH_FIDELITY in action_types
        assert FidelityActionType.RUN_MEDIUM_FIDELITY in action_types
        assert FidelityActionType.RUN_LOW_FIDELITY in action_types
        assert FidelityActionType.NO_OP in action_types

    def test_no_actions_with_zero_budget(
        self,
        manager: MultiFidelityManager,
    ) -> None:
        state = MultiFidelityState(
            parameter_space_bounds=[(0.0, 1.0), (0.0, 1.0)],
            total_cost=100.0,
            budget=100.0,
        )
        actions = manager.get_valid_actions(state)
        action_types = {a.action_type for a in actions}

        # Only NO_OP should remain
        assert action_types == {FidelityActionType.NO_OP}


class TestMultiFidelityApplyActionCost:
    """apply_action correctly tracks costs."""

    def test_multi_fidelity_apply_action_cost(
        self,
        manager: MultiFidelityManager,
        initial_state: MultiFidelityState,
    ) -> None:
        target = np.array([0.5, 0.5])
        action = FidelityAction(
            action_type=FidelityActionType.RUN_HIGH_FIDELITY,
            target_parameters=target,
        )
        new_state = manager.apply_action(initial_state, action, result=1.23)

        assert new_state.total_cost == pytest.approx(10.0)
        assert new_state.remaining_budget == pytest.approx(90.0)
        assert len(new_state.evaluated_points) == 1
        assert new_state.evaluated_points[0].result == pytest.approx(1.23)
        assert new_state.evaluated_points[0].fidelity == FidelityLevel.HIGH

    def test_low_fidelity_cost(
        self,
        manager: MultiFidelityManager,
        initial_state: MultiFidelityState,
    ) -> None:
        target = np.array([0.2, 0.8])
        action = FidelityAction(
            action_type=FidelityActionType.RUN_LOW_FIDELITY,
            target_parameters=target,
        )
        new_state = manager.apply_action(initial_state, action, result=0.5)
        assert new_state.total_cost == pytest.approx(1.0)

    def test_update_surrogate_marks_trained(
        self,
        manager: MultiFidelityManager,
        initial_state: MultiFidelityState,
    ) -> None:
        action = FidelityAction(
            action_type=FidelityActionType.UPDATE_SURROGATE,
        )
        new_state = manager.apply_action(initial_state, action)
        assert new_state.surrogate_trained is True
        assert new_state.total_cost == pytest.approx(2.0)

    def test_no_op_costs_nothing(
        self,
        manager: MultiFidelityManager,
        initial_state: MultiFidelityState,
    ) -> None:
        action = FidelityAction(action_type=FidelityActionType.NO_OP)
        new_state = manager.apply_action(initial_state, action)
        assert new_state.total_cost == pytest.approx(0.0)
        assert new_state.step == 1


class TestMultiFidelityPlanEvaluation:
    """plan_next_evaluation returns a valid action."""

    def test_multi_fidelity_plan_evaluation(
        self,
        manager: MultiFidelityManager,
        initial_state: MultiFidelityState,
    ) -> None:
        action = manager.plan_next_evaluation(initial_state)
        assert isinstance(action, FidelityAction)
        assert isinstance(action.action_type, FidelityActionType)

        # Must be within the valid set
        valid_types = {a.action_type for a in manager.get_valid_actions(initial_state)}
        assert action.action_type in valid_types

    def test_plan_with_exhausted_budget(
        self,
        manager: MultiFidelityManager,
    ) -> None:
        state = MultiFidelityState(
            parameter_space_bounds=[(0.0, 1.0), (0.0, 1.0)],
            total_cost=100.0,
            budget=100.0,
        )
        action = manager.plan_next_evaluation(state)
        assert action.action_type == FidelityActionType.NO_OP


class TestSimulationPointDataclass:
    """SimulationPoint behaves correctly as a dataclass."""

    def test_simulation_point_dataclass(self) -> None:
        params = np.array([0.3, 0.7])
        pt = SimulationPoint(
            parameters=params,
            fidelity=FidelityLevel.HIGH,
            result=42.0,
            uncertainty=0.01,
            cost=10.0,
        )
        assert pt.fidelity == FidelityLevel.HIGH
        assert pt.result == pytest.approx(42.0)
        assert pt.uncertainty == pytest.approx(0.01)
        assert pt.cost == pytest.approx(10.0)
        np.testing.assert_array_equal(pt.parameters, params)

    def test_simulation_point_defaults(self) -> None:
        pt = SimulationPoint(
            parameters=np.array([0.0]),
            fidelity=FidelityLevel.LOW,
        )
        assert pt.result is None
        assert pt.uncertainty == float("inf")
        assert pt.cost == 0.0
