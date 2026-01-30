"""Neural networks for MCTS rate control.

Implements MuZero-style networks for learned rate control:
- RepresentationNetwork: Encodes frame features to state
- DynamicsNetwork: Predicts next state given action
- PredictionNetwork: Outputs policy and value
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F


class PolicyOutput(NamedTuple):
    """Output from policy network."""

    logits: Tensor  # Action logits
    probs: Tensor  # Action probabilities


class PredictionOutput(NamedTuple):
    """Output from prediction network."""

    policy: PolicyOutput
    value: Tensor


class RepresentationNetwork(nn.Module):
    """Encodes frame features to hidden state for MCTS.

    Takes encoded frame latent and produces a compact state
    representation for the dynamics model.
    """

    def __init__(
        self,
        latent_channels: int,
        state_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        """Initialize representation network.

        Args:
            latent_channels: Channels in encoded frame latent.
            state_dim: Hidden state dimension.
            n_layers: Number of processing layers.
        """
        super().__init__()
        self.state_dim = state_dim

        # Spatial pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(4)

        layers = []
        ch = latent_channels * 16  # After 4x4 pooling flattened

        for i in range(n_layers):
            out_ch = state_dim if i == n_layers - 1 else ch
            layers.extend([
                nn.Linear(ch, out_ch),
                nn.LayerNorm(out_ch),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
            ])
            ch = out_ch

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        y: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch state_dim"]:
        """Encode latent to state.

        Args:
            y: Frame latent.

        Returns:
            Hidden state vector.
        """
        # Pool to fixed size and flatten
        x = self.pool(y)
        x = x.flatten(1)

        return self.mlp(x)


class DynamicsNetwork(nn.Module):
    """Predicts next state given current state and action.

    The action is the QP value (0-51) or a mode selection.
    Uses a residual MLP to predict state transition.
    """

    def __init__(
        self,
        state_dim: int = 256,
        num_actions: int = 52,  # QP values 0-51
        n_layers: int = 2,
    ) -> None:
        """Initialize dynamics network.

        Args:
            state_dim: Hidden state dimension.
            num_actions: Number of possible actions (QP values).
            n_layers: Number of residual layers.
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, state_dim)

        # Residual MLP for state transition
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(state_dim, state_dim * 2),
                    nn.GELU(),
                    nn.Linear(state_dim * 2, state_dim),
                    nn.LayerNorm(state_dim),
                )
            )

        # Reward prediction (R-D cost)
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(
        self,
        state: Float[Tensor, "batch state_dim"],
        action: Tensor,
    ) -> tuple[Float[Tensor, "batch state_dim"], Float[Tensor, "batch 1"]]:
        """Predict next state and reward.

        Args:
            state: Current hidden state.
            action: Selected action (QP value).

        Returns:
            Tuple of (next_state, reward).
        """
        # Embed action and combine with state
        action_emb = self.action_embed(action)
        x = state + action_emb

        # Residual layers
        for layer in self.layers:
            x = x + layer(x)

        # Predict reward (negative R-D cost, to maximize)
        reward = self.reward_head(x)

        return x, reward


class PolicyNetwork(nn.Module):
    """Predicts action distribution (QP selection).

    Outputs probability distribution over QP values based on
    current state and target bitrate constraints.
    """

    def __init__(
        self,
        state_dim: int = 256,
        num_actions: int = 52,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize policy network.

        Args:
            state_dim: Hidden state dimension.
            num_actions: Number of possible actions.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        state: Float[Tensor, "batch state_dim"],
        temperature: float = 1.0,
    ) -> PolicyOutput:
        """Predict action distribution.

        Args:
            state: Hidden state.
            temperature: Softmax temperature.

        Returns:
            PolicyOutput with logits and probabilities.
        """
        logits = self.net(state)
        probs = F.softmax(logits / temperature, dim=-1)

        return PolicyOutput(logits=logits, probs=probs)


class ValueNetwork(nn.Module):
    """Predicts expected R-D value from current state.

    Uses categorical value distribution (MuZero-style) for
    more stable value estimation.
    """

    def __init__(
        self,
        state_dim: int = 256,
        support_size: int = 51,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize value network.

        Args:
            state_dim: Hidden state dimension.
            support_size: Size of categorical support.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.support_size = support_size

        # Support for categorical distribution
        support = torch.linspace(-support_size // 2, support_size // 2, support_size)
        self.register_buffer("support", support)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, support_size),
        )

    def forward(
        self,
        state: Float[Tensor, "batch state_dim"],
    ) -> Float[Tensor, "batch"]:
        """Predict value.

        Args:
            state: Hidden state.

        Returns:
            Expected value.
        """
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)

        # Expected value from categorical distribution
        value = (probs * self.support).sum(dim=-1)

        return value

    def get_distribution(
        self,
        state: Float[Tensor, "batch state_dim"],
    ) -> Float[Tensor, "batch support_size"]:
        """Get full value distribution.

        Args:
            state: Hidden state.

        Returns:
            Value distribution probabilities.
        """
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


class PredictionNetwork(nn.Module):
    """Combined prediction network for policy and value.

    Shares early layers between policy and value heads
    for computational efficiency.
    """

    def __init__(
        self,
        state_dim: int = 256,
        num_actions: int = 52,
        support_size: int = 51,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize prediction network.

        Args:
            state_dim: Hidden state dimension.
            num_actions: Number of possible actions.
            support_size: Size of value categorical support.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # Value head (categorical)
        self.value_head = nn.Linear(hidden_dim, support_size)

        # Support for categorical value
        support = torch.linspace(-support_size // 2, support_size // 2, support_size)
        self.register_buffer("support", support)

    def forward(
        self,
        state: Float[Tensor, "batch state_dim"],
        temperature: float = 1.0,
    ) -> PredictionOutput:
        """Predict policy and value.

        Args:
            state: Hidden state.
            temperature: Policy temperature.

        Returns:
            PredictionOutput with policy and value.
        """
        features = self.trunk(state)

        # Policy
        policy_logits = self.policy_head(features)
        policy_probs = F.softmax(policy_logits / temperature, dim=-1)

        # Value (from categorical)
        value_logits = self.value_head(features)
        value_probs = F.softmax(value_logits, dim=-1)
        value = (value_probs * self.support).sum(dim=-1)

        return PredictionOutput(
            policy=PolicyOutput(logits=policy_logits, probs=policy_probs),
            value=value,
        )
