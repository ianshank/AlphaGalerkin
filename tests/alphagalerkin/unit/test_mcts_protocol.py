"""Tests for MCTS protocol and game adapter."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.alphagalerkin.core.config import (
    AlphaGalerkinConfig,
    EnvironmentConfig,
    MCTSConfig,
)
from src.alphagalerkin.core.types import ActionType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker
from src.alphagalerkin.mcts.game_adapter import DiscretizationGame
from src.alphagalerkin.mcts.protocol import (
    GameInterface,
    MCTSEvaluable,
    MCTSSearchable,
)
from src.alphagalerkin.mcts.tree import TreeManager

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_config(**overrides: Any) -> AlphaGalerkinConfig:
    """Create a minimal AlphaGalerkinConfig for testing."""
    env_kwargs: dict[str, Any] = {
        "max_steps": 5,
        "max_dof": 500,
    }
    env_kwargs.update(overrides.get("environment", {}))
    mcts_kwargs: dict[str, Any] = {
        "num_simulations": 3,
        "max_tree_depth": 3,
        "action_topk": 3,
    }
    mcts_kwargs.update(overrides.get("mcts", {}))
    return AlphaGalerkinConfig(
        environment=EnvironmentConfig(**env_kwargs),
        mcts=MCTSConfig(**mcts_kwargs),
        device="cpu",
    )


def _make_eval_fn() -> Callable:
    """Create a dummy evaluation function for TreeManager."""
    def eval_fn(
        state: DiscretizationState,
    ) -> tuple[dict[Action, float], float]:
        masker = ActionMasker(EnvironmentConfig(max_steps=5, max_dof=500))
        valid = masker.valid_actions(state)
        n = len(valid)
        priors = dict.fromkeys(valid, 1.0 / n)
        return priors, 0.5
    return eval_fn


def _make_valid_actions_fn() -> Callable:
    """Create a valid_actions function for TreeManager."""
    masker = ActionMasker(EnvironmentConfig(max_steps=5, max_dof=500))
    return masker.valid_actions


# -------------------------------------------------------------------
# Protocol satisfaction tests
# -------------------------------------------------------------------

class TestTreeManagerSatisfiesProtocol:
    """Verify TreeManager satisfies the MCTSSearchable protocol."""

    def test_tree_manager_satisfies_protocol(self) -> None:
        """TreeManager has the correct search() signature."""
        config = MCTSConfig(
            num_simulations=3,
            max_tree_depth=3,
            action_topk=3,
        )
        eval_fn = _make_eval_fn()
        valid_fn = _make_valid_actions_fn()
        manager = TreeManager(config, eval_fn, valid_fn)

        # runtime_checkable Protocol check
        assert isinstance(manager, MCTSSearchable)

    def test_tree_manager_search_returns_correct_types(self) -> None:
        """search() returns (Action, dict[Action, float])."""
        config = MCTSConfig(
            num_simulations=3,
            max_tree_depth=3,
            action_topk=3,
        )
        eval_fn = _make_eval_fn()
        valid_fn = _make_valid_actions_fn()
        manager = TreeManager(config, eval_fn, valid_fn)

        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(mesh)

        action, policy = manager.search(state, step=0)

        assert isinstance(action, Action)
        assert isinstance(policy, dict)
        assert len(policy) > 0
        for k, v in policy.items():
            assert isinstance(k, Action)
            assert isinstance(v, float)


class TestDiscretizationGameProtocol:
    """Verify DiscretizationGame satisfies the GameInterface protocol."""

    def test_discretization_game_protocol(self) -> None:
        """DiscretizationGame satisfies GameInterface."""
        config = _make_config()
        game = DiscretizationGame(config)
        assert isinstance(game, GameInterface)

    def test_game_adapter_initial_state(self) -> None:
        """get_initial_state() returns a valid DiscretizationState."""
        config = _make_config()
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        assert isinstance(state, DiscretizationState)
        assert state.dof_count > 0
        assert state.step == 0
        assert state.mesh.num_elements > 0

    def test_game_adapter_valid_actions(self) -> None:
        """get_valid_actions() returns a non-empty list of Actions."""
        config = _make_config()
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        actions = game.get_valid_actions(state)

        assert isinstance(actions, list)
        assert len(actions) > 0
        for a in actions:
            assert isinstance(a, Action)

        # Should always include at least NO_OP
        action_types = {a.action_type for a in actions}
        assert ActionType.NO_OP in action_types

    def test_game_adapter_apply_action(self) -> None:
        """apply_action() returns a new state without mutating original."""
        config = _make_config()
        game = DiscretizationGame(config)
        state = game.get_initial_state()
        original_step = state.step

        actions = game.get_valid_actions(state)
        # Pick a non-NOOP action if available
        action = actions[0]
        for a in actions:
            if a.action_type != ActionType.NO_OP:
                action = a
                break

        new_state = game.apply_action(state, action)

        # New state should have incremented step
        assert new_state.step == original_step + 1
        # Original state should be unchanged
        assert state.step == original_step

    def test_game_adapter_apply_action_noop(self) -> None:
        """NO_OP action increments step but preserves mesh."""
        config = _make_config()
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        noop = Action(
            element_id=state.mesh.element_ids[0],
            action_type=ActionType.NO_OP,
        )
        new_state = game.apply_action(state, noop)

        assert new_state.step == state.step + 1
        assert new_state.dof_count == state.dof_count

    def test_game_adapter_terminal(self) -> None:
        """is_terminal() returns True when step limit is reached."""
        config = _make_config(environment={"max_steps": 3, "max_dof": 500})
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        # Initially should not be terminal
        assert not game.is_terminal(state)

        # Apply actions until step limit
        current = state
        for _ in range(3):
            noop = Action(
                element_id=current.mesh.element_ids[0],
                action_type=ActionType.NO_OP,
            )
            current = game.apply_action(current, noop)

        assert game.is_terminal(current)

    def test_game_adapter_terminal_dof_exceeded(self) -> None:
        """is_terminal() returns True when DOF budget is exceeded."""
        # Very low DOF budget
        config = _make_config(environment={"max_steps": 100, "max_dof": 10})
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        # A 4x4 quad mesh at p=1 has 4*4=16 elements with ~11 DOFs
        # So even the initial state may exceed the budget
        # Let's check: the default mesh is 4x4 with p=1
        if state.dof_count > 10:
            assert game.is_terminal(state)

    def test_game_adapter_reward(self) -> None:
        """get_reward() returns a positive value."""
        config = _make_config()
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        reward = game.get_reward(state)
        assert reward > 0.0
        assert reward <= 1.0

    def test_game_adapter_reward_inverse_dof(self) -> None:
        """Reward is inversely proportional to DOF count."""
        config = _make_config()
        game = DiscretizationGame(config)
        state = game.get_initial_state()

        reward = game.get_reward(state)
        expected = 1.0 / max(1, state.dof_count)
        assert abs(reward - expected) < 1e-10


class TestMCTSEvaluableProtocol:
    """Verify that callable evaluation functions satisfy MCTSEvaluable."""

    def test_eval_fn_satisfies_protocol(self) -> None:
        """A properly typed callable satisfies MCTSEvaluable."""
        eval_fn = _make_eval_fn()

        # MCTSEvaluable only checks for __call__, which any
        # function has
        assert isinstance(eval_fn, MCTSEvaluable)

    def test_eval_fn_returns_correct_types(self) -> None:
        """Eval function returns (dict[Action, float], float)."""
        eval_fn = _make_eval_fn()
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(mesh)

        priors, value = eval_fn(state)
        assert isinstance(priors, dict)
        assert isinstance(value, float)
        for k, v in priors.items():
            assert isinstance(k, Action)
            assert isinstance(v, float)


class TestGameInterfaceFullCycle:
    """Integration-style tests for full game cycles via the protocol."""

    def test_full_game_cycle(self) -> None:
        """Play a short game through the adapter interface."""
        config = _make_config(environment={"max_steps": 3, "max_dof": 500})
        game = DiscretizationGame(config)

        state = game.get_initial_state()
        steps_taken = 0

        while not game.is_terminal(state):
            actions = game.get_valid_actions(state)
            assert len(actions) > 0
            # Always pick the first action (NO_OP)
            state = game.apply_action(state, actions[0])
            steps_taken += 1
            # Safety cap
            if steps_taken > 10:
                break

        assert game.is_terminal(state)
        reward = game.get_reward(state)
        assert reward > 0.0
