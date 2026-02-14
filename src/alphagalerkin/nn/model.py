"""Combined dual-head model for MCTS guidance."""
from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torch.nn as nn

from src.alphagalerkin.core.config import NetworkConfig
from src.alphagalerkin.nn.backbone import ElementBackbone
from src.alphagalerkin.nn.encoder import MeshEncoder
from src.alphagalerkin.nn.feature_norm import RunningNorm
from src.alphagalerkin.nn.galerkin_attention import GalerkinLinearAttention
from src.alphagalerkin.nn.policy_head import PolicyHead
from src.alphagalerkin.nn.value_head import ValueHead

logger = structlog.get_logger("nn.model")


class AlphaGalerkinNetwork(nn.Module):
    """Dual-head neural network for MCTS-guided discretization.

    Architecture:
        Input features -> RunningNorm -> MeshEncoder
        -> ElementBackbone
        -> PolicyHead (per-element action distribution)
        -> ValueHead (global quality estimate)
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        self.feature_norm = RunningNorm(config.input_features)
        self.encoder = MeshEncoder(
            input_features=config.input_features,
            hidden_dim=config.gnn.hidden_dim,
        )
        self.backbone = ElementBackbone(
            hidden_dim=config.gnn.hidden_dim,
            num_layers=config.gnn.num_layers,
            num_heads=config.gnn.attention_heads,
            dropout=config.gnn.dropout,
            activation=config.gnn.activation,
            architecture=config.gnn.architecture.value,
        )
        self.policy_head = PolicyHead(
            hidden_dim=config.gnn.hidden_dim,
            hidden_dims=config.policy_head.hidden_dims,
            dropout=config.policy_head.dropout,
        )
        self.value_head = ValueHead(
            hidden_dim=config.gnn.hidden_dim,
            hidden_dims=config.value_head.hidden_dims,
            pooling=config.value_head.pooling.value,
            dropout=config.value_head.dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: Per-element features,
                shape (batch, num_elements, input_features).
            action_mask: Optional action mask,
                shape (batch, num_elements, num_actions).

        Returns:
            Tuple of (policy_log_probs, value):
                - policy_log_probs:
                    shape (batch, num_elements, num_actions)
                - value: shape (batch, 1)

        """
        x = self.feature_norm(features)
        x = self.encoder(x)
        x = self.backbone(x)

        policy = self.policy_head(x, action_mask)
        value = self.value_head(x)

        return policy, value

    def compute_lbb_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute LBB regularization from Galerkin attention layers.

        Returns zero tensor if backbone doesn't use Galerkin attention.
        """
        from src.alphagalerkin.nn.stability_guard import StabilityGuard

        lbb_loss = torch.tensor(0.0, device=features.device)
        guard = StabilityGuard()

        for layer in self.backbone.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'compute_lbb_diagnostic'):
                # Get KTV matrix for stability check
                attn = layer.attention
                assert isinstance(attn, GalerkinLinearAttention)
                x = self.feature_norm(features)
                x = self.encoder(x)
                needs_batch = x.dim() == 2
                if needs_batch:
                    x = x.unsqueeze(0)
                batch, seq_len, _ = x.shape
                n_heads = attn.num_heads
                k_dim = attn.key_dim
                k = attn.k_proj(x).view(
                    batch, seq_len, n_heads, k_dim,
                )
                v = attn.v_proj(x).view(
                    batch, seq_len, n_heads, k_dim,
                )
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                ktv = torch.matmul(k.transpose(-2, -1), v) / seq_len
                lbb_loss = lbb_loss + guard(ktv)
                break  # Only need first layer for regularization

        return lbb_loss

    def predict(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference mode prediction (no gradients)."""
        self.eval()
        with torch.no_grad():
            return self.forward(features)

    def save(self, path: Path) -> None:
        """Save model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("nn.model.saved", path=str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        config: NetworkConfig,
    ) -> AlphaGalerkinNetwork:
        """Load model weights."""
        model = cls(config)
        state_dict = torch.load(
            path, map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state_dict)
        logger.info("nn.model.loaded", path=str(path))
        return model
