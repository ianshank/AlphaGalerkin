"""Intercept-specific training losses.

Provides loss functions for training neural evaluators on
engagement data:
- DynamicsResidualLoss: penalizes F=ma violations
- ConstraintViolationLoss: penalizes exceeding g-limits
- EngagementLoss: composite policy + value + physics loss
"""

from __future__ import annotations

import structlog
import torch
from torch import Tensor, nn

logger = structlog.get_logger(__name__)

G0 = 9.80665


class DynamicsResidualLoss(nn.Module):
    """Penalizes violation of Newton's second law in predicted trajectories.

    Residual: |m * a_actual - F_applied|^2 at each timestep.
    Also checks quaternion norm preservation.
    """

    def __init__(self, quat_weight: float = 1.0) -> None:
        super().__init__()
        self.quat_weight = quat_weight

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        forces: Tensor,
        masses: Tensor,
        dt: float,
        quaternions: Tensor | None = None,
    ) -> Tensor:
        """Compute dynamics residual loss.

        Args:
            positions: Trajectory positions (T, 3).
            velocities: Trajectory velocities (T, 3).
            forces: Applied forces at each step (T-1, 3).
            masses: Vehicle mass at each step (T-1,).
            dt: Time step in seconds.
            quaternions: Quaternion trajectory (T, 4) for norm check.

        Returns:
            Scalar loss value.

        """
        # Acceleration from finite differences
        accel = (velocities[1:] - velocities[:-1]) / dt

        # Expected acceleration from F=ma
        expected_accel = forces / masses.unsqueeze(-1)

        # Residual
        residual = accel - expected_accel
        dynamics_loss = torch.mean(residual**2)

        # Quaternion norm preservation
        quat_loss = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        if quaternions is not None:
            norms = torch.norm(quaternions, dim=-1)
            quat_loss = torch.mean((norms - 1.0) ** 2)

        return dynamics_loss + self.quat_weight * quat_loss


class ConstraintViolationLoss(nn.Module):
    """Penalizes exceeding structural and energy constraints.

    Soft penalty for g-loads above max and energy below zero.
    """

    def __init__(self, max_g: float = 30.0, g_weight: float = 1.0) -> None:
        super().__init__()
        self.max_g = max_g
        self.g_weight = g_weight

    def forward(
        self,
        accelerations: Tensor,
    ) -> Tensor:
        """Compute constraint violation loss.

        Args:
            accelerations: Commanded accelerations (T, 3).

        Returns:
            Scalar loss value.

        """
        g_loads = torch.norm(accelerations, dim=-1) / G0
        violations = torch.clamp(g_loads - self.max_g, min=0.0)
        return self.g_weight * torch.mean(violations**2)


class EngagementLoss(nn.Module):
    """Composite loss for engagement training.

    Combines policy loss (cross-entropy on MCTS visit distribution),
    value loss (MSE on engagement outcome), and optional physics losses.
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        dynamics_weight: float = 0.1,
        constraint_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.dynamics_weight = dynamics_weight
        self.constraint_weight = constraint_weight
        self.dynamics_loss = DynamicsResidualLoss()
        self.constraint_loss = ConstraintViolationLoss()

    def forward(
        self,
        policy_logits: Tensor,
        value_pred: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute composite engagement loss.

        Args:
            policy_logits: Predicted action logits (B, A).
            value_pred: Predicted engagement value (B, 1) or (B,).
            target_policy: MCTS visit distribution (B, A).
            target_value: Actual outcome (B, 1) or (B,).

        Returns:
            (total_loss, metrics_dict).

        """
        # Policy loss: cross-entropy with soft targets
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.mean(torch.sum(target_policy * log_probs, dim=-1))

        # Value loss: MSE
        value_pred_flat = value_pred.view(-1)
        target_value_flat = target_value.view(-1)
        value_loss = torch.mean((value_pred_flat - target_value_flat) ** 2)

        total = self.policy_weight * policy_loss + self.value_weight * value_loss

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total.item(),
        }

        return total, metrics
