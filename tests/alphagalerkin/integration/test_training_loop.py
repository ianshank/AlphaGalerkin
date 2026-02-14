"""Integration: training loop components.

This test verifies that the replay buffer, curriculum manager,
and environment can work together in a simulated training loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.core.config import (
    CurriculumConfig,
    EnvironmentConfig,
    MCTSConfig,
    ReplayConfig,
)
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.environment import (
    DiscretizationEnvironment,
)
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker
from src.alphagalerkin.mcts.tree import TreeManager
from src.alphagalerkin.training.curriculum import (
    CurriculumManager,
)
from src.alphagalerkin.training.replay_buffer import (
    Experience,
    ReplayBuffer,
)


def _dummy_eval_fn(
    state: DiscretizationState,
) -> tuple[dict[Action, float], float]:
    """Uniform prior over valid actions, constant value."""
    masker = ActionMasker(EnvironmentConfig())
    actions = masker.valid_actions(state)
    n = len(actions)
    priors = {a: 1.0 / max(1, n) for a in actions}
    return priors, 0.5


@pytest.mark.integration
class TestTrainingLoop:
    """Integration test for the training pipeline components."""

    def test_episode_to_replay_buffer(self) -> None:
        """Experiences from a self-play episode should be storable in the replay buffer."""
        env_config = EnvironmentConfig(
            max_steps=3, max_dof=50000,
        )
        mcts_config = MCTSConfig(
            num_simulations=3,
            max_tree_depth=2,
            action_topk=2,
        )
        replay_config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )

        masker = ActionMasker(env_config)
        tree = TreeManager(
            config=mcts_config,
            eval_fn=_dummy_eval_fn,
            valid_actions_fn=masker.valid_actions,
        )
        env = DiscretizationEnvironment(env_config)
        buf = ReplayBuffer(replay_config)

        # Play one episode
        state = env.reset()
        done = False
        step = 0
        while not done:
            action, policy = tree.search(
                state, step=step,
            )
            result = env.step(action)

            # Store experience
            features = state.to_feature_tensor().numpy()
            policy_dict = {
                str(a.action_type.value): p
                for a, p in policy.items()
            }
            exp = Experience(
                state_features=features.mean(axis=0),
                policy_target=policy_dict,
                value_target=result.reward,
            )
            buf.add(exp)

            state = result.state
            done = result.done
            step += 1

        assert buf.size > 0
        assert buf.is_ready

    def test_multiple_episodes_fill_buffer(self) -> None:
        """Multiple episodes should accumulate experiences."""
        env_config = EnvironmentConfig(
            max_steps=2, max_dof=50000,
        )
        mcts_config = MCTSConfig(
            num_simulations=2,
            max_tree_depth=2,
            action_topk=2,
        )
        replay_config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )

        masker = ActionMasker(env_config)
        tree = TreeManager(
            config=mcts_config,
            eval_fn=_dummy_eval_fn,
            valid_actions_fn=masker.valid_actions,
        )
        env = DiscretizationEnvironment(env_config)
        buf = ReplayBuffer(replay_config)

        for episode_idx in range(3):
            state = env.reset()
            done = False
            step = 0
            while not done:
                action, policy = tree.search(
                    state, step=step,
                )
                result = env.step(action)
                features = (
                    state.to_feature_tensor()
                    .numpy()
                    .mean(axis=0)
                )
                exp = Experience(
                    state_features=features,
                    policy_target={},
                    value_target=result.reward,
                    iteration=episode_idx,
                )
                buf.add(exp)
                state = result.state
                done = result.done
                step += 1

        assert buf.size >= 3

    def test_curriculum_with_environment(self) -> None:
        """Curriculum manager should advance stages based on simulated win rates."""
        curriculum_config = CurriculumConfig(
            enabled=True,
            stages=[
                {"max_dof": 100},
                {"max_dof": 500},
                {"max_dof": 1000},
            ],
            advance_threshold=0.7,
            evaluation_window=10,
        )
        cm = CurriculumManager(curriculum_config)

        # Simulate several rounds of "training"
        assert cm.current_stage_index == 0

        # Low performance: stay at stage 0
        for _ in range(5):
            cm.update(0.5)
        assert cm.current_stage_index == 0

        # High performance: advance
        for _ in range(10):
            cm.update(0.9)
        assert cm.current_stage_index >= 1

    def test_replay_buffer_checkpoint_roundtrip(
        self,
    ) -> None:
        """Buffer state should survive serialization."""
        config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        for i in range(10):
            buf.add(
                Experience(
                    state_features=np.random.randn(8),
                    policy_target={},
                    value_target=float(i) / 10.0,
                )
            )

        state = buf.get_state()
        buf2 = ReplayBuffer(config)
        buf2.load_state(state)

        assert buf2.size == buf.size
        assert buf2.is_ready

    def test_sample_from_filled_buffer(self) -> None:
        """Sampling from a filled buffer should return valid experiences."""
        config = ReplayConfig(
            capacity=1000, min_size_to_train=5,
        )
        buf = ReplayBuffer(config)
        for i in range(20):
            buf.add(
                Experience(
                    state_features=np.random.randn(8),
                    policy_target={"h_refine": 0.5},
                    value_target=np.random.rand(),
                )
            )

        batch = buf.sample(4)
        assert len(batch) == 4
        for exp in batch:
            assert isinstance(exp, Experience)
            assert exp.state_features.shape == (8,)
