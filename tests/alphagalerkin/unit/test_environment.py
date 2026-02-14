"""Tests for discretization environment."""
from __future__ import annotations

import pytest

from src.alphagalerkin.core.config import EnvironmentConfig
from src.alphagalerkin.core.types import ActionType, ElementID
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.environment import (
    DiscretizationEnvironment,
    StepResult,
)


class TestDiscretizationEnvironment:
    """Tests for the Gym-like discretization environment."""

    def test_reset_returns_valid_state(self) -> None:
        config = EnvironmentConfig(
            max_steps=10, max_dof=500,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        assert state is not None
        assert state.validate()

    def test_reset_creates_default_mesh(self) -> None:
        config = EnvironmentConfig()
        env = DiscretizationEnvironment(config)
        state = env.reset()
        # Default 4x4 mesh = 16 elements
        assert state.mesh.num_elements == 16

    def test_step_returns_step_result(self) -> None:
        config = EnvironmentConfig(
            max_steps=10, max_dof=50000,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        eid = state.mesh.element_ids[0]
        action = Action(eid, ActionType.NO_OP, {})
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert result.state is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_limit_terminates(self) -> None:
        config = EnvironmentConfig(
            max_steps=2, max_dof=50000,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        eid = state.mesh.element_ids[0]
        action = Action(eid, ActionType.NO_OP, {})
        result1 = env.step(action)
        result2 = env.step(action)
        assert result2.done

    def test_step_without_reset_raises(self) -> None:
        config = EnvironmentConfig()
        env = DiscretizationEnvironment(config)
        eid = ElementID("e0")
        action = Action(eid, ActionType.NO_OP, {})
        with pytest.raises(RuntimeError, match="not reset"):
            env.step(action)

    def test_info_contains_diagnostics(self) -> None:
        config = EnvironmentConfig(
            max_steps=10, max_dof=50000,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        eid = state.mesh.element_ids[0]
        action = Action(eid, ActionType.NO_OP, {})
        result = env.step(action)
        assert "dof_count" in result.info
        assert "num_elements" in result.info
        assert "step" in result.info
        assert "budget_exceeded" in result.info

    def test_state_property_before_reset(self) -> None:
        config = EnvironmentConfig()
        env = DiscretizationEnvironment(config)
        assert env.state is None

    def test_state_property_after_reset(self) -> None:
        config = EnvironmentConfig()
        env = DiscretizationEnvironment(config)
        state = env.reset()
        assert env.state is state

    def test_h_refine_action_in_environment(self) -> None:
        config = EnvironmentConfig(
            max_steps=10, max_dof=50000,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        initial_count = state.mesh.num_elements
        eid = state.mesh.element_ids[0]
        action = Action(eid, ActionType.H_REFINE, {})
        result = env.step(action)
        assert (
            result.state.mesh.num_elements > initial_count
        )

    def test_dof_budget_exceeded_terminates(self) -> None:
        config = EnvironmentConfig(
            max_steps=100, max_dof=10,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        eid = state.mesh.element_ids[0]
        action = Action(eid, ActionType.H_REFINE, {})
        result = env.step(action)
        # After refinement on default 4x4 mesh (16 elem),
        # DOF count should exceed budget of 10
        assert result.info["budget_exceeded"]
        assert result.done

    def test_custom_initial_mesh(self) -> None:
        from src.alphagalerkin.env.mesh_graph import MeshGraph

        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        config = EnvironmentConfig(max_steps=10)
        env = DiscretizationEnvironment(
            config, initial_mesh=mesh,
        )
        state = env.reset()
        assert state.mesh.num_elements == 4
