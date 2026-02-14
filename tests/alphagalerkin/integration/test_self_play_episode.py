"""Integration: full self-play episode via MCTS tree search.

This test builds a complete search pipeline -- action masker,
dummy evaluation function, tree manager -- and verifies that
the MCTS tree search produces a valid action and policy for
the discretization environment.
"""
from __future__ import annotations

import pytest

from src.alphagalerkin.core.config import (
    EnvironmentConfig,
    MCTSConfig,
)
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.environment import (
    DiscretizationEnvironment,
)
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker
from src.alphagalerkin.mcts.tree import TreeManager


def _dummy_eval_fn(
    state: DiscretizationState,
) -> tuple[dict[Action, float], float]:
    """Uniform prior over first few actions, constant value."""
    masker = ActionMasker(EnvironmentConfig())
    actions = masker.valid_actions(state)
    n = len(actions)
    priors = {a: 1.0 / max(1, n) for a in actions}
    return priors, 0.5


@pytest.mark.integration
class TestSelfPlayEpisode:
    """Integration test for a complete MCTS episode."""

    def test_mcts_search_returns_action_and_policy(
        self,
    ) -> None:
        """TreeManager.search should return a valid action and policy."""
        config = MCTSConfig(
            num_simulations=5,
            max_tree_depth=3,
            action_topk=3,
        )
        env_config = EnvironmentConfig(
            max_steps=5, max_dof=50000,
        )
        masker = ActionMasker(env_config)
        tree = TreeManager(
            config=config,
            eval_fn=_dummy_eval_fn,
            valid_actions_fn=masker.valid_actions,
        )

        env = DiscretizationEnvironment(env_config)
        state = env.reset()

        action, policy = tree.search(state, step=0)
        assert action is not None
        assert len(policy) > 0
        assert sum(policy.values()) > 0

    def test_full_episode_loop(self) -> None:
        """Run a complete episode: reset, search, step, repeat until done."""
        mcts_config = MCTSConfig(
            num_simulations=3,
            max_tree_depth=3,
            action_topk=2,
        )
        env_config = EnvironmentConfig(
            max_steps=3, max_dof=50000,
        )
        masker = ActionMasker(env_config)
        tree = TreeManager(
            config=mcts_config,
            eval_fn=_dummy_eval_fn,
            valid_actions_fn=masker.valid_actions,
        )

        env = DiscretizationEnvironment(env_config)
        state = env.reset()

        states: list[DiscretizationState] = []
        policies: list[dict[Action, float]] = []
        rewards: list[float] = []
        done = False
        step = 0

        while not done:
            action, policy = tree.search(state, step=step)
            result = env.step(action)

            states.append(state)
            policies.append(policy)
            rewards.append(result.reward)

            state = result.state
            done = result.done
            step += 1

        assert step > 0
        assert len(states) == step
        assert len(policies) == step
        assert len(rewards) == step

    def test_episode_respects_step_limit(self) -> None:
        """Episode should terminate at max_steps."""
        mcts_config = MCTSConfig(
            num_simulations=3,
            max_tree_depth=2,
            action_topk=2,
        )
        env_config = EnvironmentConfig(
            max_steps=2, max_dof=50000,
        )
        masker = ActionMasker(env_config)
        tree = TreeManager(
            config=mcts_config,
            eval_fn=_dummy_eval_fn,
            valid_actions_fn=masker.valid_actions,
        )

        env = DiscretizationEnvironment(env_config)
        state = env.reset()

        step = 0
        done = False
        while not done:
            action, _ = tree.search(state, step=step)
            result = env.step(action)
            state = result.state
            done = result.done
            step += 1

        assert step <= env_config.max_steps

    def test_episode_collects_experiences(self) -> None:
        """Collected (state, policy, reward) tuples should be usable as training data."""
        mcts_config = MCTSConfig(
            num_simulations=3,
            max_tree_depth=2,
            action_topk=2,
        )
        env_config = EnvironmentConfig(
            max_steps=3, max_dof=50000,
        )
        masker = ActionMasker(env_config)
        tree = TreeManager(
            config=mcts_config,
            eval_fn=_dummy_eval_fn,
            valid_actions_fn=masker.valid_actions,
        )

        env = DiscretizationEnvironment(env_config)
        state = env.reset()

        experiences: list[
            tuple[DiscretizationState, dict, float]
        ] = []
        done = False
        step = 0

        while not done:
            action, policy = tree.search(state, step=step)
            result = env.step(action)
            experiences.append(
                (state, policy, result.reward)
            )
            state = result.state
            done = result.done
            step += 1

        for s, p, r in experiences:
            assert s.validate()
            assert isinstance(p, dict)
            assert isinstance(r, float)
