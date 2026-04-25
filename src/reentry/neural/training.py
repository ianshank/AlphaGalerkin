"""Self-play training loop for compressible flow MCTS evaluator.

Generates mesh refinement episodes via MCTS self-play, then
trains the neural evaluator (value + policy heads) on the
collected experience.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
from torch import Tensor

from src.reentry.neural.encoder import FlowFieldEncoder

logger = structlog.get_logger(__name__)


@dataclass
class TrainingEpisode:
    """A single self-play episode for training.

    Attributes:
        states: Flow field tensors at each step (n_steps, C, H, W).
        actions: Actions taken at each step (n_steps,).
        rewards: Rewards received (n_steps,).
        value_target: Final value (error reduction achieved).
        policy_target: MCTS visit distribution at each step (n_steps, n_actions).

    """

    states: list[Tensor]
    actions: list[int]
    rewards: list[float]
    value_target: float
    policy_target: list[Tensor]


class ReentryEvaluatorTrainer:
    """Trains the neural evaluator for compressible flow mesh refinement.

    Uses a simple value + policy architecture on top of the
    FlowFieldEncoder for MCTS evaluation.
    """

    def __init__(
        self,
        n_actions: int = 17,
        encode_size: int = 32,
        latent_dim: int = 128,
        learning_rate: float = 1e-3,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
    ) -> None:
        self.n_actions = n_actions
        self.encoder = FlowFieldEncoder(encode_size=encode_size, latent_dim=latent_dim)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions),
        )
        params = (
            list(self.encoder.parameters())
            + list(self.value_head.parameters())
            + list(self.policy_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        self.value_weight = value_weight
        self.policy_weight = policy_weight

    def predict(self, state_tensor: Tensor) -> tuple[float, Tensor]:
        """Predict value and policy for a state.

        Args:
            state_tensor: (1, C, H, W) flow field tensor.

        Returns:
            Tuple of (value, policy_logits).

        """
        with torch.no_grad():
            latent = self.encoder(state_tensor)
            value = self.value_head(latent)
            policy = self.policy_head(latent)
        return float(value.item()), policy.squeeze(0)

    def train_step(self, episodes: list[TrainingEpisode]) -> dict[str, float]:
        """Train on a batch of episodes.

        Args:
            episodes: List of self-play episodes.

        Returns:
            Dictionary of loss components.

        """
        total_loss = torch.tensor(0.0)
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        n_samples = 0

        for ep in episodes:
            for state, policy_target in zip(ep.states, ep.policy_target, strict=False):
                if state.dim() == 3:
                    state = state.unsqueeze(0)

                latent = self.encoder(state)
                value_pred = self.value_head(latent).squeeze()
                policy_pred = self.policy_head(latent).squeeze()

                # Value loss: MSE
                v_target = torch.tensor(ep.value_target, dtype=torch.float32)
                v_loss = torch.nn.functional.mse_loss(value_pred, v_target)

                # Policy loss: cross-entropy with MCTS visit distribution
                p_loss = -torch.sum(policy_target * torch.log_softmax(policy_pred, dim=-1))

                total_loss = total_loss + self.value_weight * v_loss + self.policy_weight * p_loss
                value_loss_sum += v_loss.item()
                policy_loss_sum += p_loss.item()
                n_samples += 1

        if n_samples > 0:
            total_loss = total_loss / n_samples
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return {
            "total_loss": total_loss.item() if n_samples > 0 else 0.0,
            "value_loss": value_loss_sum / max(n_samples, 1),
            "policy_loss": policy_loss_sum / max(n_samples, 1),
            "n_samples": n_samples,
        }
