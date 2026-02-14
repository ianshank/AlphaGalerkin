"""Backbone architectures for element-level feature processing.

Supports three modes:
- ``"transformer"``: Standard multi-head self-attention (O(N^2))
- ``"galerkin"``: Galerkin linear attention (O(N))
- ``"fnet"``: FFT-based mixing (O(N log N))

The mode is selected via the GNN architecture config field.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

from src.alphagalerkin.nn.fnet_block import FNetBlock
from src.alphagalerkin.nn.galerkin_attention import (
    GalerkinLinearAttention,
)

logger = structlog.get_logger("nn.backbone")


class TransformerBlock(nn.Module):
    """Standard transformer block with softmax attention."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: shape (batch, num_elements, hidden_dim)
                or (num_elements, hidden_dim).

        """
        needs_batch = x.dim() == 2
        if needs_batch:
            x = x.unsqueeze(0)

        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        if needs_batch:
            x = x.squeeze(0)
        return x


class GalerkinBlock(nn.Module):
    """Galerkin attention block with O(N) complexity."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = GalerkinLinearAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Galerkin attention.

        Args:
            x: shape (batch, num_elements, hidden_dim)
                or (num_elements, hidden_dim).

        """
        needs_batch = x.dim() == 2
        if needs_batch:
            x = x.unsqueeze(0)

        normed = self.norm1(x)
        x = x + self.attention(normed)
        x = x + self.ffn(self.norm2(x))

        if needs_batch:
            x = x.squeeze(0)
        return x


def _build_block(
    mode: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    activation: str = "gelu",
) -> nn.Module:
    """Factory for backbone blocks.

    Args:
        mode: One of "gat" (transformer), "galerkin", "fnet".
        hidden_dim: Hidden dimension.
        num_heads: Attention heads (ignored for fnet).
        dropout: Dropout probability.
        activation: Activation function name.

    Returns:
        A single backbone block module.

    """
    if mode in ("galerkin", "custom"):
        return GalerkinBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    if mode == "fnet":
        return FNetBlock(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    # Default: standard transformer (for "gat", "gcn", "graphsage")
    return TransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation,
    )


class ElementBackbone(nn.Module):
    """Configurable backbone for element feature processing.

    Supports multiple architecture modes selected by the
    ``architecture`` parameter:

    - ``"gat"`` / ``"gcn"`` / ``"graphsage"``: Standard transformer
    - ``"galerkin"`` / ``"custom"``: Galerkin linear attention (O(N))
    - ``"fnet"``: FFT-based mixing (O(N log N))

    Parameters
    ----------
    hidden_dim:
        Hidden dimension per layer.
    num_layers:
        Number of stacked blocks.
    num_heads:
        Attention heads per block.
    dropout:
        Dropout probability.
    activation:
        Activation function name.
    architecture:
        Architecture mode string.

    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        architecture: str = "gat",
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [
                _build_block(
                    mode=architecture,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

        logger.info(
            "backbone.init",
            architecture=architecture,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process element features through backbone layers.

        Args:
            x: shape (batch, num_elements, hidden_dim).

        Returns:
            Processed features, same shape.

        """
        for layer in self.layers:
            x = layer(x)
        return x
