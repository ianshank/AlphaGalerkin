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

logger = structlog.get_logger("evaluation")


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
