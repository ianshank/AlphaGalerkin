"""Online feature normalization for input features."""
from __future__ import annotations

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.feature_norm")


class RunningNorm(nn.Module):
    """Online normalization that tracks running mean and std.

    During training, updates running statistics.
    During eval, uses stored statistics.
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer(
            "running_mean", torch.zeros(num_features),
        )
        self.register_buffer(
            "running_var", torch.ones(num_features),
        )
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features.

        Args:
            x: Input tensor, shape (..., num_features).

        """
        if self.training:
            # Update running stats
            flat = x.reshape(-1, self.num_features)
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0, unbiased=False)

            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * batch_var
            )
            self.num_batches_tracked = self.num_batches_tracked + 1

        result: torch.Tensor = (x - self.running_mean) / torch.sqrt(
            self.running_var + self.eps,
        )
        return result
