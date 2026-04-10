"""AlphaGalerkin composite loss (policy + value + LBB).

Implements the main training loss combining:
- Policy loss (cross-entropy with MCTS target distribution)
- Value loss (MSE with game outcome)
- LBB regularization (ensures Galerkin stability)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.constants import DEFAULT_LBB_EPS, DEFAULT_LBB_TARGET, DEFAULT_LBB_WEIGHT, LOG_PROB_MIN
from src.training.losses.base import LossOutput, register_loss

if TYPE_CHECKING:
    from config.schemas import TrainingConfig

logger = structlog.get_logger(__name__)


@register_loss("alphagalerkin")
class AlphaGalerkinLoss(nn.Module):
    """Composite loss for AlphaGalerkin training.

    Loss = w_policy * L_policy + w_value * L_value + w_lbb * L_lbb

    Where:
    - L_policy = CrossEntropy(policy_logits, target_policy)
    - L_value = MSE(value, target_value)
    - L_lbb = -log(lbb_constant + eps) (encourages stability)
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        lbb_weight: float = DEFAULT_LBB_WEIGHT,
        lbb_eps: float = DEFAULT_LBB_EPS,
        lbb_target: float = DEFAULT_LBB_TARGET,
        label_smoothing: float = 0.0,
        log_barrier_weight: float = DEFAULT_LBB_TARGET,
    ) -> None:
        """Initialize loss function.

        Args:
            policy_weight: Weight for policy loss.
            value_weight: Weight for value loss.
            lbb_weight: Weight for LBB regularization.
            lbb_eps: Small constant for numerical stability in LBB loss.
            lbb_target: Target minimum for LBB constant (soft constraint).
            label_smoothing: Label smoothing for policy cross-entropy.
            log_barrier_weight: Weight for log-barrier term in LBB loss.

        """
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.lbb_weight = lbb_weight
        self.lbb_eps = lbb_eps
        self.lbb_target = lbb_target
        self.label_smoothing = label_smoothing
        self.log_barrier_weight = log_barrier_weight

        # Track running statistics for logging
        self._running_policy_loss = 0.0
        self._running_value_loss = 0.0
        self._running_lbb_loss = 0.0
        self._n_updates = 0

    @classmethod
    def from_config(cls, config: TrainingConfig) -> AlphaGalerkinLoss:
        """Create loss function from training config.

        Args:
            config: Training configuration.

        Returns:
            Configured loss function.

        """
        return cls(
            policy_weight=config.policy_loss_weight,
            value_weight=config.value_loss_weight,
            lbb_weight=config.lbb_loss_weight,
            lbb_eps=config.lbb_eps,
            lbb_target=config.lbb_target,
            label_smoothing=config.label_smoothing,
            log_barrier_weight=config.log_barrier_weight,
        )

    def compute_policy_loss(
        self,
        policy_logits: Float[Tensor, "batch actions"],
        target_policy: Float[Tensor, "batch actions"],
        mask: Float[Tensor, "batch actions"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute policy loss (cross-entropy with soft targets).

        Args:
            policy_logits: Predicted policy logits.
            target_policy: Target policy distribution from MCTS.
            mask: Optional mask for valid actions (1 = valid, 0 = invalid).

        Returns:
            Policy cross-entropy loss.

        """
        # Apply mask if provided
        if mask is not None:
            # Set invalid actions to large negative logits
            policy_logits = policy_logits.masked_fill(mask == 0, float("-inf"))
            # Zero out target probability for invalid actions
            target_policy = target_policy * mask
            # Renormalize target
            target_sum = target_policy.sum(dim=-1, keepdim=True).clamp(min=self.lbb_eps)
            target_policy = target_policy / target_sum

        # Compute log softmax with clamping to prevent -inf * 0 = NaN
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        log_probs = log_probs.clamp(min=LOG_PROB_MIN)

        # Apply label smoothing if configured
        if self.label_smoothing > 0:
            n_classes = target_policy.size(-1)
            smooth_target = (
                1.0 - self.label_smoothing
            ) * target_policy + self.label_smoothing / n_classes
            target_policy = smooth_target

        # Cross-entropy with soft targets: -sum(target * log_probs)
        loss = -(target_policy * log_probs).sum(dim=-1).mean()

        return loss

    def compute_value_loss(
        self,
        value: Float[Tensor, "batch 1"],
        target_value: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, ""]:
        """Compute value loss (MSE).

        Args:
            value: Predicted value in [-1, 1].
            target_value: Target value (game outcome).

        Returns:
            Value MSE loss.

        """
        return torch.nn.functional.mse_loss(value, target_value)

    def compute_lbb_loss(
        self,
        lbb_constant: Float[Tensor, batch] | None,
    ) -> Float[Tensor, ""]:
        """Compute LBB regularization loss.

        Encourages the LBB constant (minimum singular value) to stay above
        a threshold for numerical stability of Galerkin attention.

        Loss = max(0, lbb_target - lbb_constant)^2 + (-log(lbb_constant + eps))

        Args:
            lbb_constant: LBB stability constant per sample.

        Returns:
            LBB regularization loss.

        """
        if lbb_constant is None:
            return torch.tensor(0.0)

        # Ensure positive LBB constant
        lbb_clamped = lbb_constant.clamp(min=self.lbb_eps)

        # Soft threshold penalty: penalize if below target
        threshold_penalty = torch.relu(self.lbb_target - lbb_clamped).pow(2).mean()

        # Log barrier: smooth penalty encouraging larger LBB constant
        log_penalty = (-torch.log(lbb_clamped)).mean()

        # Combined loss (threshold penalty is primary, log barrier is secondary)
        loss = threshold_penalty + self.log_barrier_weight * log_penalty

        return loss

    def forward(
        self,
        policy_logits: Float[Tensor, "batch actions"],
        value: Float[Tensor, "batch 1"],
        target_policy: Float[Tensor, "batch actions"],
        target_value: Float[Tensor, "batch 1"],
        lbb_constant: Float[Tensor, batch] | None = None,
        action_mask: Float[Tensor, "batch actions"] | None = None,
    ) -> LossOutput:
        """Compute total loss.

        Args:
            policy_logits: Predicted policy logits.
            value: Predicted value.
            target_policy: Target policy from MCTS.
            target_value: Target value (game outcome).
            lbb_constant: Optional LBB stability constant.
            action_mask: Optional mask for valid actions.

        Returns:
            LossOutput with total and component losses.

        """
        # Compute individual losses
        policy_loss = self.compute_policy_loss(policy_logits, target_policy, action_mask)
        value_loss = self.compute_value_loss(value, target_value)
        lbb_loss = self.compute_lbb_loss(lbb_constant)

        # Weighted sum
        total_loss = (
            self.policy_weight * policy_loss
            + self.value_weight * value_loss
            + self.lbb_weight * lbb_loss
        )

        # Update running statistics
        self._running_policy_loss += policy_loss.item()
        self._running_value_loss += value_loss.item()
        self._running_lbb_loss += lbb_loss.item()
        self._n_updates += 1

        return LossOutput(
            total=total_loss,
            policy=policy_loss,
            value=value_loss,
            lbb=lbb_loss,
        )

    def get_running_stats(self) -> dict[str, float]:
        """Get running average of loss components.

        Returns:
            Dictionary with running averages.

        """
        if self._n_updates == 0:
            return {"policy": 0.0, "value": 0.0, "lbb": 0.0}

        return {
            "policy": self._running_policy_loss / self._n_updates,
            "value": self._running_value_loss / self._n_updates,
            "lbb": self._running_lbb_loss / self._n_updates,
        }

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self._running_policy_loss = 0.0
        self._running_value_loss = 0.0
        self._running_lbb_loss = 0.0
        self._n_updates = 0


@register_loss("entropy_regularizer")
class EntropyRegularizer(nn.Module):
    """Entropy regularization to prevent policy collapse.

    Adds a bonus for maintaining exploration in the policy.
    """

    def __init__(
        self,
        weight: float = 0.01,
        min_entropy_ratio: float = 0.1,
    ) -> None:
        """Initialize entropy regularizer.

        Args:
            weight: Weight for entropy bonus.
            min_entropy_ratio: Minimum entropy as ratio of maximum.

        """
        super().__init__()
        self.weight = weight
        self.min_entropy_ratio = min_entropy_ratio

    def forward(
        self,
        policy_logits: Float[Tensor, "batch actions"],
        mask: Float[Tensor, "batch actions"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute entropy bonus (negative loss).

        Args:
            policy_logits: Policy logits.
            mask: Optional action mask (1 = valid, 0 = invalid).

        Returns:
            Negative entropy (to be added to loss for entropy bonus).

        """
        batch_size = policy_logits.size(0)
        n_actions = policy_logits.size(-1)
        device = policy_logits.device

        if mask is not None:
            # Count valid actions per sample
            n_valid = mask.sum(dim=-1)

            # Identify degenerate samples (no valid actions)
            degenerate_mask = n_valid < 1

            if degenerate_mask.any():
                logger.debug(
                    "degenerate_action_mask_detected",
                    n_degenerate=degenerate_mask.sum().item(),
                    batch_size=batch_size,
                )

            # Create safe mask to prevent NaN in softmax
            # For degenerate samples, set first action as valid (we'll zero out later)
            safe_mask = mask.clone()
            safe_mask[degenerate_mask, 0] = 1.0

            # Apply safe mask to logits
            masked_logits = policy_logits.masked_fill(safe_mask == 0, float("-inf"))

            # Compute softmax safely (no all-inf rows now)
            probs = torch.softmax(masked_logits, dim=-1)
            log_probs = torch.log_softmax(masked_logits, dim=-1)

            # Clamp log_probs to prevent -inf * 0 = NaN edge cases
            log_probs = log_probs.clamp(min=LOG_PROB_MIN)

            # Compute entropy: -sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1)

            # Zero out entropy for degenerate samples (they contribute nothing)
            entropy = entropy.masked_fill(degenerate_mask, 0.0)

            # Maximum entropy based on valid actions (clamped to avoid log(0))
            max_entropy = torch.log(n_valid.float().clamp(min=1.0))
            # For degenerate samples, set max_entropy to 1 to avoid division issues
            max_entropy = max_entropy.masked_fill(degenerate_mask, 1.0)

        else:
            # No mask - compute normally
            probs = torch.softmax(policy_logits, dim=-1)
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            log_probs = log_probs.clamp(min=LOG_PROB_MIN)

            entropy = -(probs * log_probs).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(n_actions, dtype=torch.float32, device=device))

        # Normalized entropy
        normalized_entropy = entropy / max_entropy.clamp(min=1e-8)

        # Return negative entropy (so adding to loss encourages higher entropy)
        return -self.weight * normalized_entropy.mean()
