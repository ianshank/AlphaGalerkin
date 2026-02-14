"""Mesh graph encoder - converts discretization state to tensor features."""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.encoder")


class MeshEncoder(nn.Module):
    """Encodes per-element features into embeddings.

    Takes raw element features (poly_order, size, level, centroid,
    neighbors, etc.) and projects them to hidden dimension via a
    small MLP.
    """

    def __init__(
        self,
        input_features: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode element features.

        Args:
            x: Element features,
                shape (batch, num_elements, input_features)
                or (num_elements, input_features).

        Returns:
            Encoded features, same leading dims with
            last dim = hidden_dim.

        """
        result: torch.Tensor = self.encoder(x)
        return result
