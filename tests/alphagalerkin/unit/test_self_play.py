"""Tests for self-play episode generation and SelfPlayEngine.

Covers:
- Episode dataclass: length, total_reward, to_experiences conversion
- Episode edge cases (empty lists)
- SelfPlayEngine creation from AlphaGalerkinConfig
- set_eval_fn replaces the default evaluation function
- _default_eval_fn returns uniform policy with zero value
- play_episode produces a valid Episode (mocked env and _search)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.alphagalerkin.core.config import AlphaGalerkinConfig
from src.alphagalerkin.core.types import ActionType, ElementID
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.training.replay_buffer import Experience
from src.alphagalerkin.training.self_play import (
    Episode,
    SelfPlayEngine,
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_action(
    eid: str = "e0",
    action_type: ActionType = ActionType.NO_OP,
) -> Action:
    """Create a simple Action for testing."""
    return Action(
        element_id=ElementID(eid),
        action_type=action_type,
    )


def _make_mock_state() -> MagicMock:
    """Create a mock DiscretizationState with a to_feature_tensor method."""
    state = MagicMock()
    feature_tensor = MagicMock()
    feature_tensor.numpy.return_value = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        dtype=np.float32,
    )
    state.to_feature_tensor.return_value = feature_tensor
    state.dof_count = 16
    return state


def _make_policy(
    actions: list[Action] | None = None,
) -> dict[Action, float]:
    """Create a simple uniform policy dict."""
    if actions is None:
        actions = [_make_action("e0"), _make_action("e1")]
    uniform = 1.0 / max(1, len(actions))
    return dict.fromkeys(actions, uniform)


# -------------------------------------------------------------------
# Episode dataclass tests
# -------------------------------------------------------------------


class TestEpisode:
    """Tests for the Episode dataclass."""

    def test_empty_episode_length_is_zero(self) -> None:
        ep = Episode()
        assert ep.length == 0

    def test_empty_episode_total_reward_is_zero(self) -> None:
        ep = Episode()
        assert ep.total_reward == 0.0

    def test_length_matches_states_count(self) -> None:
        states = [
            _make_mock_state(),
            _make_mock_state(),
            _make_mock_state(),
        ]
        ep = Episode(states=states)
        assert ep.length == 3

    def test_total_reward_sums_rewards(self) -> None:
        ep = Episode(rewards=[1.0, 2.5, -0.5, 0.0])
        assert ep.total_reward == pytest.approx(3.0)

    def test_total_reward_with_negative_rewards(self) -> None:
        ep = Episode(rewards=[-1.0, -2.0, -3.0])
        assert ep.total_reward == pytest.approx(-6.0)

    def test_to_experiences_empty_episode(self) -> None:
        ep = Episode()
        experiences = ep.to_experiences(iteration=0)
        assert experiences == []

    def test_to_experiences_creates_correct_count(self) -> None:
        actions = [_make_action("e0"), _make_action("e1")]
        policies = [_make_policy(actions), _make_policy(actions)]
        states = [_make_mock_state(), _make_mock_state()]
        rewards = [1.0, 2.0]

        ep = Episode(
            states=states,
            policies=policies,
            rewards=rewards,
            actions=actions,
        )
        experiences = ep.to_experiences(iteration=5)
        assert len(experiences) == 2

    def test_to_experiences_returns_experience_objects(self) -> None:
        actions = [_make_action("e0")]
        policies = [_make_policy(actions)]
        states = [_make_mock_state()]
        rewards = [1.0]

        ep = Episode(
            states=states,
            policies=policies,
            rewards=rewards,
            actions=actions,
        )
        experiences = ep.to_experiences(iteration=3)
        assert len(experiences) == 1
        assert isinstance(experiences[0], Experience)

    def test_to_experiences_value_target_is_average_reward(self) -> None:
        actions = [_make_action("e0"), _make_action("e1")]
        policies = [_make_policy(actions), _make_policy(actions)]
        states = [_make_mock_state(), _make_mock_state()]
        rewards = [2.0, 4.0]

        ep = Episode(
            states=states,
            policies=policies,
            rewards=rewards,
            actions=actions,
        )
        experiences = ep.to_experiences(iteration=0)
        # outcome = total_reward / length = 6.0 / 2 = 3.0
        for exp in experiences:
            assert exp.value_target == pytest.approx(3.0)

    def test_to_experiences_iteration_propagated(self) -> None:
        actions = [_make_action("e0")]
        policies = [_make_policy(actions)]
        states = [_make_mock_state()]
        rewards = [1.0]

        ep = Episode(
            states=states,
            policies=policies,
            rewards=rewards,
            actions=actions,
        )
        experiences = ep.to_experiences(iteration=42)
        assert experiences[0].iteration == 42

    def test_to_experiences_default_iteration_is_zero(self) -> None:
        actions = [_make_action("e0")]
        policies = [_make_policy(actions)]
        states = [_make_mock_state()]
        rewards = [1.0]

        ep = Episode(
            states=states,
            policies=policies,
            rewards=rewards,
            actions=actions,
        )
        experiences = ep.to_experiences()
        assert experiences[0].iteration == 0

    def test_to_experiences_state_features_from_feature_tensor(self) -> None:
        actions = [_make_action("e0")]
        policies = [_make_policy(actions)]
        state = _make_mock_state()
        states = [state]
        rewards = [1.0]

        ep = Episode(
            states=states,
            policies=policies,
            rewards=rewards,
            actions=actions,
        )
        experiences = ep.to_experiences(iteration=0)
        state.to_feature_tensor.assert_called_once()
        np.testing.assert_array_equal(
            experiences[0].state_features,
            np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=np.float32,
            ),
        )

    def test_to_experiences_policy_target_is_numpy_array(self) -> None:
        action = _make_action("e0")
        policy = {action: 0.7, _make_action("e1"): 0.3}
        states = [_make_mock_state()]
        rewards = [1.0]

        ep = Episode(
            states=states,
            policies=[policy],
            rewards=rewards,
            actions=[action],
        )
        experiences = ep.to_experiences(iteration=0)
        assert isinstance(experiences[0].policy_target, np.ndarray)
        assert experiences[0].policy_target.dtype == np.float32

    def test_to_experiences_single_step_episode(self) -> None:
        """An episode with one step should produce one experience."""
        action = _make_action("e0")
        policy = {action: 1.0}
        states = [_make_mock_state()]
        rewards = [5.0]

        ep = Episode(
            states=states,
            policies=[policy],
            rewards=rewards,
            actions=[action],
        )
        experiences = ep.to_experiences(iteration=1)
        assert len(experiences) == 1
        assert experiences[0].value_target == pytest.approx(5.0)


# -------------------------------------------------------------------
# SelfPlayEngine tests
# -------------------------------------------------------------------

_SP_MODULE = "src.alphagalerkin.training.self_play"


class TestSelfPlayEngineCreation:
    """Tests for SelfPlayEngine construction and configuration."""

    def _make_config(self) -> AlphaGalerkinConfig:
        return AlphaGalerkinConfig()

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_creation_succeeds(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = self._make_config()
        engine = SelfPlayEngine(config)
        assert engine is not None

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_creates_env_with_environment_config(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = self._make_config()
        SelfPlayEngine(config)
        mock_env_cls.assert_called_once_with(config.environment)

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_creates_masker_with_environment_config(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = self._make_config()
        SelfPlayEngine(config)
        mock_masker_cls.assert_called_once_with(config.environment)

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_set_eval_fn_replaces_default(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = self._make_config()
        engine = SelfPlayEngine(config)
        original_fn = engine._eval_fn

        custom_fn = MagicMock(return_value=({}, 0.5))
        engine.set_eval_fn(custom_fn)

        assert engine._eval_fn is custom_fn
        assert engine._eval_fn is not original_fn

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_default_eval_fn_returns_uniform_policy(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = self._make_config()
        engine = SelfPlayEngine(config)

        state = _make_mock_state()
        action1 = _make_action("e0")
        action2 = _make_action("e1")
        mock_masker_cls.return_value.valid_actions.return_value = [
            action1,
            action2,
        ]

        priors, value = engine._default_eval_fn(state)

        assert len(priors) == 2
        for p in priors.values():
            assert p == pytest.approx(0.5)
        assert value == 0.0

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_default_eval_fn_no_valid_actions(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = self._make_config()
        engine = SelfPlayEngine(config)

        state = _make_mock_state()
        mock_masker_cls.return_value.valid_actions.return_value = []

        priors, value = engine._default_eval_fn(state)
        assert priors == {}
        assert value == 0.0


class TestSelfPlayEnginePlayEpisode:
    """Tests for play_episode.

    We mock ``_search`` to avoid exercising the full MCTS tree
    (which needs real states for node expansion). This isolates
    the episode-loop logic.
    """

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_returns_episode(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        step_result = MagicMock()
        step_result.reward = 1.0
        step_result.done = True
        step_result.state = _make_mock_state()
        mock_env.step.return_value = step_result

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}
        engine._search = MagicMock(return_value=(action, policy))

        episode = engine.play_episode()

        assert isinstance(episode, Episode)
        assert episode.length >= 1
        assert len(episode.rewards) >= 1
        assert len(episode.actions) >= 1
        assert len(episode.policies) >= 1

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_records_rewards(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        r1 = MagicMock()
        r1.reward = 1.5
        r1.done = False
        r1.state = _make_mock_state()

        r2 = MagicMock()
        r2.reward = 2.5
        r2.done = True
        r2.state = _make_mock_state()

        mock_env.step.side_effect = [r1, r2]

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}
        engine._search = MagicMock(return_value=(action, policy))

        episode = engine.play_episode()

        assert episode.length == 2
        assert episode.rewards == [1.5, 2.5]
        assert episode.total_reward == pytest.approx(4.0)

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_respects_max_steps(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig(
            environment={"max_steps": 3},
        )
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        step_result = MagicMock()
        step_result.reward = 0.1
        step_result.done = False
        step_result.state = _make_mock_state()
        mock_env.step.return_value = step_result

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}
        engine._search = MagicMock(return_value=(action, policy))

        episode = engine.play_episode()

        assert episode.length <= 3

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_calls_env_reset(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        step_result = MagicMock()
        step_result.reward = 0.0
        step_result.done = True
        step_result.state = _make_mock_state()
        mock_env.step.return_value = step_result

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}
        engine._search = MagicMock(return_value=(action, policy))

        engine.play_episode()

        mock_env.reset.assert_called_once()

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_with_custom_eval_fn(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        step_result = MagicMock()
        step_result.reward = 1.0
        step_result.done = True
        step_result.state = _make_mock_state()
        mock_env.step.return_value = step_result

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}

        custom_eval = MagicMock(return_value=(policy, 0.8))
        engine.set_eval_fn(custom_eval)

        # Mock _search so that play_episode exercises the loop
        engine._search = MagicMock(return_value=(action, policy))

        episode = engine.play_episode()

        assert isinstance(episode, Episode)
        assert episode.length >= 1

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_states_policies_actions_rewards_same_length(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        results = []
        for i in range(3):
            r = MagicMock()
            r.reward = float(i)
            r.done = i == 2
            r.state = _make_mock_state()
            results.append(r)
        mock_env.step.side_effect = results

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}
        engine._search = MagicMock(return_value=(action, policy))

        episode = engine.play_episode()

        assert len(episode.states) == len(episode.policies)
        assert len(episode.states) == len(episode.actions)
        assert len(episode.states) == len(episode.rewards)

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_calls_search_each_step(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        r1 = MagicMock()
        r1.reward = 0.5
        r1.done = False
        r1.state = _make_mock_state()

        r2 = MagicMock()
        r2.reward = 1.0
        r2.done = True
        r2.state = _make_mock_state()

        mock_env.step.side_effect = [r1, r2]

        action = _make_action("e0", ActionType.NO_OP)
        policy = {action: 1.0}
        mock_search = MagicMock(return_value=(action, policy))
        engine._search = mock_search

        engine.play_episode()

        # _search should be called once per step
        assert mock_search.call_count == 2

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_play_episode_calls_env_step_with_selected_action(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        config = AlphaGalerkinConfig()
        engine = SelfPlayEngine(config)

        mock_env = mock_env_cls.return_value
        mock_env.reset.return_value = _make_mock_state()

        step_result = MagicMock()
        step_result.reward = 1.0
        step_result.done = True
        step_result.state = _make_mock_state()
        mock_env.step.return_value = step_result

        action = _make_action("e0", ActionType.P_REFINE)
        policy = {action: 1.0}
        engine._search = MagicMock(return_value=(action, policy))

        engine.play_episode()

        mock_env.step.assert_called_once_with(action)


class TestSelfPlayEngineSearch:
    """Tests for the _search MCTS method.

    These exercise the actual MCTS loop by providing a mock state
    that supports apply_action and a mock eval_fn.
    """

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_search_returns_action_and_policy(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        """The _search method returns (Action, dict) tuple."""
        config = AlphaGalerkinConfig(
            mcts={"num_simulations": 2, "max_tree_depth": 3},
        )
        engine = SelfPlayEngine(config)

        action_a = _make_action("e0", ActionType.NO_OP)
        action_b = _make_action("e1", ActionType.H_REFINE)

        # Build a mock state that supports apply_action for node expansion
        def make_child_state() -> MagicMock:
            child = MagicMock()
            child.apply_action.return_value = child
            return child

        state = MagicMock()
        state.apply_action.return_value = make_child_state()

        # eval_fn returns priors for two actions
        priors = {action_a: 0.6, action_b: 0.4}
        eval_fn = MagicMock(return_value=(priors, 0.5))
        engine.set_eval_fn(eval_fn)

        action, policy = engine._search(state, step=0)

        assert isinstance(action, Action)
        assert isinstance(policy, dict)
        assert len(policy) > 0
        # All probabilities should sum to ~1
        assert sum(policy.values()) == pytest.approx(1.0, abs=0.01)

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_search_with_empty_priors_falls_back_to_uniform(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        """When eval_fn returns empty priors, _search uses uniform policy."""
        config = AlphaGalerkinConfig(
            mcts={"num_simulations": 1, "max_tree_depth": 2},
        )
        engine = SelfPlayEngine(config)

        action_a = _make_action("e0", ActionType.NO_OP)
        action_b = _make_action("e1", ActionType.H_REFINE)

        # Masker returns valid actions
        mock_masker_cls.return_value.valid_actions.return_value = [
            action_a,
            action_b,
        ]

        state = MagicMock()
        state.apply_action.return_value = state

        # First call returns empty priors; second returns uniform
        eval_fn = MagicMock(
            side_effect=[
                ({}, 0.0),  # root expansion (empty priors)
                ({action_a: 0.5, action_b: 0.5}, 0.3),  # child eval
            ],
        )
        engine.set_eval_fn(eval_fn)

        action, policy = engine._search(state, step=0)

        assert isinstance(action, Action)
        assert isinstance(policy, dict)

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_search_with_no_valid_actions_returns_noop(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        """When no valid actions exist, _search returns NO_OP."""
        config = AlphaGalerkinConfig(
            mcts={"num_simulations": 1},
        )
        engine = SelfPlayEngine(config)

        mock_masker_cls.return_value.valid_actions.return_value = []

        state = MagicMock()

        # eval_fn returns empty priors
        eval_fn = MagicMock(return_value=({}, 0.0))
        engine.set_eval_fn(eval_fn)

        action, policy = engine._search(state, step=0)

        assert action.action_type == ActionType.NO_OP
        assert policy == {}

    @patch(f"{_SP_MODULE}.DiscretizationEnvironment")
    @patch(f"{_SP_MODULE}.ActionMasker")
    def test_search_calls_eval_fn_at_least_once(
        self,
        mock_masker_cls: MagicMock,
        mock_env_cls: MagicMock,
    ) -> None:
        """The _search calls eval_fn at least once for root expansion."""
        config = AlphaGalerkinConfig(
            mcts={"num_simulations": 2, "max_tree_depth": 2},
        )
        engine = SelfPlayEngine(config)

        action_a = _make_action("e0", ActionType.NO_OP)
        action_b = _make_action("e1", ActionType.H_REFINE)
        priors = {action_a: 0.6, action_b: 0.4}

        # Each state.apply_action returns a *new* mock so nodes differ
        def new_child(action: object) -> MagicMock:
            child = MagicMock()
            child.apply_action.side_effect = new_child
            return child

        state = MagicMock()
        state.apply_action.side_effect = new_child

        eval_fn = MagicMock(return_value=(priors, 0.5))
        engine.set_eval_fn(eval_fn)

        engine._search(state, step=0)

        # eval_fn called at least once for root expansion
        assert eval_fn.call_count >= 1
