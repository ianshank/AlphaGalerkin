"""Loss functions for neural operator training.

Provides standard losses for operator learning including
relative L2, H1 (Sobolev), and physics-informed variants.
"""

from __future__ import annotations

import structlog
import torch
from jaxtyping import Float
from torch import Tensor, nn

logger = structlog.get_logger(__name__)


class L2RelativeLoss(nn.Module):
    """Relative L2 loss for operator learning.
    
    Computes: ||pred - target||_2 / ||target||_2
    
    This is the standard loss for neural operators as it's
    scale-invariant across different output magnitudes.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        """Initialize L2 relative loss.

        Args:
            reduction: How to reduce batch dimension ('mean', 'sum', 'none').
            eps: Small constant to prevent division by zero.

        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: Float[Tensor, "batch ..."],
        target: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, ""]:
        """Compute relative L2 loss.

        Args:
            pred: Predicted field.
            target: Ground truth field.

        Returns:
            Scalar loss value.

        """
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Compute per-sample relative error
        diff_norm = torch.norm(pred_flat - target_flat, dim=-1)
        target_norm = torch.norm(target_flat, dim=-1)

        relative_error = diff_norm / (target_norm + self.eps)

        if self.reduction == "mean":
            return relative_error.mean()
        elif self.reduction == "sum":
            return relative_error.sum()
        else:
            return relative_error


class H1Loss(nn.Module):
    """H1 Sobolev loss including gradient penalty.
    
    Computes: L2(pred, target) + lambda * L2(grad(pred), grad(target))
    
    This encourages matching not just values but also spatial derivatives,
    useful for physics problems where smoothness matters.
    """

    def __init__(
        self,
        lambda_grad: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        """Initialize H1 loss.

        Args:
            lambda_grad: Weight for gradient term.
            reduction: How to reduce batch dimension.

        """
        super().__init__()
        self.lambda_grad = lambda_grad
        self.l2_loss = L2RelativeLoss(reduction=reduction)

    def _compute_gradient(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> tuple[Float[Tensor, "batch c h w"], Float[Tensor, "batch c h w"]]:
        """Compute spatial gradients using finite differences."""
        # Pad for boundary handling
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]

        # Pad to match original size
        dx = torch.nn.functional.pad(dx, (0, 0, 0, 1))
        dy = torch.nn.functional.pad(dy, (0, 1, 0, 0))

        return dx, dy

    def forward(
        self,
        pred: Float[Tensor, "batch c h w"],
        target: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, ""]:
        """Compute H1 loss.

        Args:
            pred: Predicted field.
            target: Ground truth field.

        Returns:
            Scalar loss value.

        """
        # L2 term
        l2_term = self.l2_loss(pred, target)

        # Gradient terms
        pred_dx, pred_dy = self._compute_gradient(pred)
        target_dx, target_dy = self._compute_gradient(target)

        grad_loss_x = self.l2_loss(pred_dx, target_dx)
        grad_loss_y = self.l2_loss(pred_dy, target_dy)

        return l2_term + self.lambda_grad * (grad_loss_x + grad_loss_y)


class MSELoss(nn.Module):
    """Standard MSE loss wrapper for compatibility."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        pred: Float[Tensor, "batch ..."],
        target: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, ""]:
        return self.mse(pred, target)


def get_loss(name: str, **kwargs) -> nn.Module:
    """Factory function to get loss by name.

    Args:
        name: Loss name ('l2_relative', 'h1', 'mse').
        **kwargs: Additional arguments for the loss.

    Returns:
        Loss module.

    """
    losses = {
        "l2_relative": L2RelativeLoss,
        "h1": H1Loss,
        "mse": MSELoss,
    }

    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")

    return losses[name](**kwargs)
