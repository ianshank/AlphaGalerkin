"""Graph-based mesh encoder with message passing.

Implements a pure-PyTorch message-passing GNN that operates on the
mesh adjacency graph.  No ``torch_geometric`` dependency is required.

Classes:
    MessagePassingLayer: Single message-passing layer using adjacency.
    GraphEncoder: Multi-layer graph encoder with residual connections.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.graph_encoder")


class MessagePassingLayer(nn.Module):
    """Single message-passing layer using an adjacency matrix.

    For each node *i*, messages are computed from every neighbour *j*
    (where ``adjacency[i, j] = 1``) by concatenating features of *i*
    and *j*, passing through an MLP, and summing.  The aggregated
    message is then concatenated with the original node feature and
    passed through an update MLP.

    Parameters
    ----------
    hidden_dim:
        Feature dimension for both input and output.
    dropout:
        Dropout probability applied after message aggregation.

    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one round of message passing.

        Args:
            x: Node features, shape ``(batch, N, hidden_dim)``
                where *N* is the number of elements.
            adjacency: Float adjacency matrix,
                shape ``(batch, N, N)``.  Entry ``[b, i, j] = 1.0``
                if element *j* is a neighbour of element *i*.

        Returns:
            Updated node features, same shape as *x*.

        """
        batch, num_nodes, dim = x.shape

        # Expand x_i and x_j for all pairs.
        # x_i: (batch, N, 1, dim) -> (batch, N, N, dim)
        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        # x_j: (batch, 1, N, dim) -> (batch, N, N, dim)
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        # Concatenate features for each (i, j) pair.
        pair_features = torch.cat([x_i, x_j], dim=-1)
        # pair_features: (batch, N, N, 2*dim)

        # Compute messages for all pairs.
        messages = self.message_mlp(pair_features)
        # messages: (batch, N, N, dim)

        # Mask messages by adjacency.
        # adjacency: (batch, N, N) -> (batch, N, N, 1)
        mask = adjacency.unsqueeze(-1)
        messages = messages * mask

        # Aggregate: sum over neighbours (dim=2 is the source j).
        aggregated = messages.sum(dim=2)
        # aggregated: (batch, N, dim)

        aggregated = self.dropout(aggregated)

        # Update: combine aggregated messages with original features.
        combined = torch.cat([x, aggregated], dim=-1)
        updated: torch.Tensor = self.update_mlp(combined)
        return updated


class GraphEncoder(nn.Module):
    """Graph-based mesh encoder with message passing.

    Projects per-element input features to a hidden dimension, then
    applies multiple :class:`MessagePassingLayer` rounds with residual
    connections.  Handles both batched ``(B, N, F)`` and unbatched
    ``(N, F)`` inputs.

    Parameters
    ----------
    input_features:
        Number of per-element input features.
    hidden_dim:
        Hidden (and output) dimension.
    num_mp_layers:
        Number of message-passing rounds.
    dropout:
        Dropout probability.

    """

    def __init__(
        self,
        input_features: int = 8,
        hidden_dim: int = 128,
        num_mp_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.mp_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, dropout) for _ in range(num_mp_layers)]
        )

        logger.info(
            "graph_encoder.init",
            input_features=input_features,
            hidden_dim=hidden_dim,
            num_mp_layers=num_mp_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Process element features through message passing on graph.

        Args:
            x: Per-element features, shape ``(batch, N, input_features)``
                or ``(N, input_features)`` (unbatched).
            adjacency: Float adjacency matrix,
                shape ``(batch, N, N)`` or ``(N, N)`` (unbatched).

        Returns:
            Encoded features with last dimension ``hidden_dim``,
            same leading dimensions as *x*.

        """
        needs_batch = x.dim() == 2
        if needs_batch:
            x = x.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)

        h = self.input_proj(x)
        for layer in self.mp_layers:
            h = h + layer(h, adjacency)  # residual connection

        if needs_batch:
            h = h.squeeze(0)

        return h  # type: ignore[no-any-return]
