"""GNN-like backbone using standard PyTorch (self-attention over elements)."""
from __future__ import annotations

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.backbone")


class TransformerBlock(nn.Module):
    """Transformer block for element-level message passing."""

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
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
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

        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        if needs_batch:
            x = x.squeeze(0)

        return x


class ElementBackbone(nn.Module):
    """Transformer backbone for processing element features.

    Provides both per-element and global representations.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, dropout, activation,
            )
            for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process element features through transformer layers.

        Args:
            x: shape (batch, num_elements, hidden_dim).

        Returns:
            Processed features, same shape.

        """
        for layer in self.layers:
            x = layer(x)
        return x
