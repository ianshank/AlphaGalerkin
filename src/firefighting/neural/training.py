"""Self-play training for fire spread MCTS evaluator.

Generates fire mesh refinement episodes and trains a neural
evaluator for the MCTS fire prediction system.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
from torch import Tensor

from src.firefighting.neural.encoder import FireStateEncoder

logger = structlog.get_logger(__name__)


@dataclass
class FireTrainingEpisode:
    """A single fire prediction self-play episode."""

    states: list[Tensor]
    actions: list[int]
    rewards: list[float]
    value_target: float
    policy_target: list[Tensor]


class FireEvaluatorTrainer:
    """Trains neural evaluator for fire spread mesh refinement."""

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
        self.encoder = FireStateEncoder(encode_size=encode_size, latent_dim=latent_dim)
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
        """Predict value and policy for a fire state."""
        with torch.no_grad():
            latent = self.encoder(state_tensor)
            value = self.value_head(latent)
            policy = self.policy_head(latent)
        return float(value.item()), policy.squeeze(0)

    def train_step(self, episodes: list[FireTrainingEpisode]) -> dict[str, float]:
        """Train on a batch of fire prediction episodes."""
        total_loss = torch.tensor(0.0)
        v_sum = 0.0
        p_sum = 0.0
        n = 0

        for ep in episodes:
            for state, policy_target in zip(ep.states, ep.policy_target, strict=False):
                if state.dim() == 3:
                    state = state.unsqueeze(0)

                latent = self.encoder(state)
                v_pred = self.value_head(latent).squeeze()
                p_pred = self.policy_head(latent).squeeze()

                v_target = torch.tensor(ep.value_target, dtype=torch.float32)
                v_loss = torch.nn.functional.mse_loss(v_pred, v_target)
                p_loss = -torch.sum(policy_target * torch.log_softmax(p_pred, dim=-1))

                total_loss = total_loss + self.value_weight * v_loss + self.policy_weight * p_loss
                v_sum += v_loss.item()
                p_sum += p_loss.item()
                n += 1

        if n > 0:
            total_loss = total_loss / n
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return {
            "total_loss": total_loss.item() if n > 0 else 0.0,
            "value_loss": v_sum / max(n, 1),
            "policy_loss": p_sum / max(n, 1),
            "n_samples": n,
        }
