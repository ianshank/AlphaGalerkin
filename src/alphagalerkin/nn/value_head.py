"""Value head: global quality estimate."""
from __future__ import annotations

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.value_head")


class ValueHead(nn.Module):
    """Global quality estimate head.

    Uses global pooling followed by MLP to produce scalar value.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        hidden_dims: list[int] | None = None,
        pooling: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pooling = pooling

        if pooling == "attention":
            self.attention_weights = nn.Linear(hidden_dim, 1)

        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for h_dim in (hidden_dims or [128, 64]):
            if h_dim == 1:
                # Final output layer
                layers.append(nn.Linear(in_dim, 1))
                break
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        else:
            layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute value estimate.

        Args:
            x: Element embeddings,
                shape (batch, num_elements, hidden_dim).

        Returns:
            Value estimates, shape (batch, 1) or (1,).

        """
        needs_batch = x.dim() == 2
        if needs_batch:
            x = x.unsqueeze(0)

        # Pooling
        if self.pooling == "attention":
            weights = torch.softmax(
                self.attention_weights(x), dim=1,
            )  # (batch, N, 1)
            pooled = (weights * x).sum(dim=1)  # (batch, hidden)
        elif self.pooling == "max":
            pooled = x.max(dim=1)[0]  # (batch, hidden_dim)
        else:  # mean
            pooled = x.mean(dim=1)  # (batch, hidden_dim)

        value = self.mlp(pooled)  # (batch, 1)
        value = torch.tanh(value)  # Bound to [-1, 1]

        if needs_batch:
            value = value.squeeze(0)

        return value
