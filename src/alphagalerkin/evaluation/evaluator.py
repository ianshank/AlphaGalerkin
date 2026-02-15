"""Policy evaluation on held-out problems."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog

from src.alphagalerkin.core.config import AlphaGalerkinConfig
from src.alphagalerkin.env.environment import DiscretizationEnvironment
from src.alphagalerkin.mcts.action_masking import ActionMasker
from src.alphagalerkin.mcts.tree import EvalFn, TreeManager

logger = structlog.get_logger("evaluation.evaluator")


class PolicyEvaluator:
    """Evaluates trained policy on test problems.

    Runs MCTS-guided episodes using a trained neural network
    and collects performance metrics (reward, episode length,
    final DOF count).

    Parameters
    ----------
    config:
        Full AlphaGalerkin configuration.

    """

    def __init__(self, config: AlphaGalerkinConfig) -> None:
        self._config = config
        self._env = DiscretizationEnvironment(
            config.environment,
        )
        self._masker = ActionMasker(config.environment)

    def evaluate(
        self,
        eval_fn: EvalFn,
        num_episodes: int = 10,
    ) -> dict[str, float]:
        """Run evaluation episodes and compute metrics.

        Parameters
        ----------
        eval_fn:
            Neural network evaluation callback that maps a
            ``DiscretizationState`` to
            ``(action_priors, value_estimate)``.
        num_episodes:
            Number of episodes to run.

        Returns
        -------
        dict[str, float]
            Evaluation metrics keyed by name.

        """
        rewards: list[float] = []
        lengths: list[int] = []
        final_dofs: list[int] = []

        tree = TreeManager(
            config=self._config.mcts,
            eval_fn=eval_fn,
            valid_actions_fn=self._masker.valid_actions,
        )

        for ep in range(num_episodes):
            state = self._env.reset()
            episode_reward = 0.0
            step = 0

            for step in range(
                self._config.environment.max_steps,
            ):
                action, _ = tree.search(state, step=step)
                result = self._env.step(action)
                episode_reward += result.reward

                if result.done:
                    break
                state = result.state

            rewards.append(episode_reward)
            lengths.append(step + 1)
            final_dofs.append(state.dof_count)

            logger.debug(
                "evaluation.episode",
                episode=ep,
                reward=round(episode_reward, 6),
                length=step + 1,
                final_dof=state.dof_count,
            )

        metrics = {
            "eval/avg_reward": float(np.mean(rewards)),
            "eval/avg_length": float(np.mean(lengths)),
            "eval/avg_final_dof": float(
                np.mean(final_dofs),
            ),
            "eval/std_reward": float(np.std(rewards)),
            "eval/min_reward": float(np.min(rewards)),
            "eval/max_reward": float(np.max(rewards)),
        }

        logger.info("evaluation.complete", **metrics)
        return metrics

    # ---------------------------------------------------------------
    # Head-to-head comparison
    # ---------------------------------------------------------------

    def evaluate_head_to_head(
        self,
        current_fn: EvalFn,
        baseline_fn: EvalFn,
        num_episodes: int = 10,
    ) -> dict[str, float]:
        """Compare two policies on the same problems.

        Both policies play the same episodes (same initial states)
        and we compare their total rewards.

        Parameters
        ----------
        current_fn:
            Evaluation callback for the candidate policy.
        baseline_fn:
            Evaluation callback for the baseline policy.
        num_episodes:
            Number of episodes to evaluate.

        Returns
        -------
        dict[str, float]
            win_rate: fraction of episodes where current > baseline.
            avg_reward_diff: mean(current_reward - baseline_reward).
            policy_agreement: fraction of matching action selections.

        """
        current_rewards: list[float] = []
        baseline_rewards: list[float] = []
        agreement_counts: list[int] = []
        total_steps: list[int] = []

        current_tree = TreeManager(
            config=self._config.mcts,
            eval_fn=current_fn,
            valid_actions_fn=self._masker.valid_actions,
        )
        baseline_tree = TreeManager(
            config=self._config.mcts,
            eval_fn=baseline_fn,
            valid_actions_fn=self._masker.valid_actions,
        )

        for ep in range(num_episodes):
            # Reset once to get the same initial state for both.
            state = self._env.reset()
            initial_state = state.clone()

            # --- Run current policy ---
            cur_reward = 0.0
            cur_actions: list[Any] = []
            cur_env = DiscretizationEnvironment(
                self._config.environment,
            )
            cur_env._state = initial_state.clone()
            cur_state_live = initial_state.clone()

            for step_idx in range(
                self._config.environment.max_steps,
            ):
                action_c, _ = current_tree.search(
                    cur_state_live,
                    step=step_idx,
                )
                cur_actions.append(action_c)
                result_c = cur_env.step(action_c)
                cur_reward += result_c.reward
                if result_c.done:
                    break
                cur_state_live = result_c.state

            # --- Run baseline policy ---
            base_reward = 0.0
            base_actions: list[Any] = []
            base_env = DiscretizationEnvironment(
                self._config.environment,
            )
            base_env._state = initial_state.clone()
            base_state_live = initial_state.clone()

            for step_idx in range(
                self._config.environment.max_steps,
            ):
                action_b, _ = baseline_tree.search(
                    base_state_live,
                    step=step_idx,
                )
                base_actions.append(action_b)
                result_b = base_env.step(action_b)
                base_reward += result_b.reward
                if result_b.done:
                    break
                base_state_live = result_b.state

            current_rewards.append(cur_reward)
            baseline_rewards.append(base_reward)

            # Count action agreement over shared steps.
            shared_len = min(len(cur_actions), len(base_actions))
            agree = sum(
                1
                for a_c, a_b in zip(
                    cur_actions[:shared_len],
                    base_actions[:shared_len],
                    strict=True,
                )
                if a_c == a_b
            )
            agreement_counts.append(agree)
            total_steps.append(max(shared_len, 1))

            logger.debug(
                "evaluation.head_to_head.episode",
                episode=ep,
                current_reward=round(cur_reward, 6),
                baseline_reward=round(base_reward, 6),
                agreement=agree,
                steps=shared_len,
            )

        wins = sum(1 for c, b in zip(current_rewards, baseline_rewards, strict=True) if c > b)
        reward_diffs = [c - b for c, b in zip(current_rewards, baseline_rewards, strict=True)]
        total_agree = sum(agreement_counts)
        total_step_sum = sum(total_steps)

        metrics = {
            "h2h/win_rate": wins / max(1, num_episodes),
            "h2h/avg_reward_diff": float(np.mean(reward_diffs)),
            "h2h/policy_agreement": (total_agree / max(1, total_step_sum)),
        }

        logger.info("evaluation.head_to_head.complete", **metrics)
        return metrics

    def compute_policy_agreement(
        self,
        fn_a: EvalFn,
        fn_b: EvalFn,
        num_states: int = 100,
    ) -> float:
        """Measure how often two policies agree on action selection.

        Generates *num_states* states from the environment and
        compares the top-1 action each policy would select via MCTS.

        Parameters
        ----------
        fn_a:
            First evaluation callback.
        fn_b:
            Second evaluation callback.
        num_states:
            Number of states to sample.

        Returns
        -------
        float
            Fraction of states where both policies select the
            same action (in [0, 1]).

        """
        tree_a = TreeManager(
            config=self._config.mcts,
            eval_fn=fn_a,
            valid_actions_fn=self._masker.valid_actions,
        )
        tree_b = TreeManager(
            config=self._config.mcts,
            eval_fn=fn_b,
            valid_actions_fn=self._masker.valid_actions,
        )

        agreements = 0
        evaluated = 0

        for _ in range(num_states):
            state = self._env.reset()
            action_a, _ = tree_a.search(state, step=0)
            action_b, _ = tree_b.search(state, step=0)
            evaluated += 1
            if action_a == action_b:
                agreements += 1

        agreement_rate = agreements / max(1, evaluated)
        logger.info(
            "evaluation.policy_agreement",
            agreements=agreements,
            evaluated=evaluated,
            rate=round(agreement_rate, 4),
        )
        return agreement_rate

    def evaluate_from_checkpoint(
        self,
        checkpoint_path: Path,
        num_episodes: int = 10,
    ) -> dict[str, float]:
        """Load a checkpoint and evaluate the policy.

        Parameters
        ----------
        checkpoint_path:
            Path to a saved model checkpoint.
        num_episodes:
            Number of evaluation episodes.

        Returns
        -------
        dict[str, float]
            Evaluation metrics keyed by name.

        """
        import torch

        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location=self._config.device,
            weights_only=False,
        )

        logger.info(
            "evaluation.checkpoint_loaded",
            path=str(checkpoint_path),
            step=checkpoint.get("step", "unknown"),
        )

        # Build network and load state dict
        from src.alphagalerkin.nn.model import AlphaGalerkinNetwork

        network = AlphaGalerkinNetwork(self._config.network)
        network.load_state_dict(
            checkpoint["model_state_dict"],
        )
        network.eval()

        device = torch.device(self._config.device)
        network = network.to(device)

        def eval_fn(
            state: Any,
        ) -> tuple[dict[Any, float], float]:
            """Wrap network inference as EvalFn."""
            with torch.no_grad():
                policy, value = network.predict(state)
            return cast(dict[Any, float], policy), float(value)

        return self.evaluate(
            eval_fn=eval_fn,
            num_episodes=num_episodes,
        )
