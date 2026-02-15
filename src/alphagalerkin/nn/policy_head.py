"""Policy head: per-element action distribution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alphagalerkin.core.types import ActionType

# Derived from the ActionType enum so it stays in sync automatically.
DEFAULT_NUM_ACTIONS = len(ActionType)


class PolicyHead(nn.Module):
    """Per-element action distribution head.

    Output: (batch, num_elements, num_actions) probability tensor.
    Uses masked softmax for invalid actions.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for h_dim in hidden_dims or [128, 64]:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute policy logits.

        Args:
            x: Element embeddings,
                shape (batch, num_elements, hidden_dim).
            action_mask: Binary mask,
                shape (batch, num_elements, num_actions).
                1 = valid, 0 = invalid.

        Returns:
            Log probabilities,
                shape (batch, num_elements, num_actions).

        """
        logits = self.mlp(x)  # (batch, num_elements, num_actions)

        if action_mask is not None:
            logits = logits.masked_fill(
                action_mask == 0,
                float("-inf"),
            )

        # Flatten and apply log_softmax over all element-action pairs
        batch_size = logits.shape[0] if logits.dim() == 3 else 1
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        flat = logits.view(batch_size, -1)  # (B, N * num_actions)
        log_probs = F.log_softmax(flat, dim=-1)

        return log_probs.view_as(logits)
