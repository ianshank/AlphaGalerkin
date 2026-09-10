"""Combined AlphaGalerkin + physics loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.training.loss_balancing import LossBalancingConfig, create_loss_balancer
from src.training.losses.base import register_loss
from src.training.losses.physics.config import PhysicsLossConfig
from src.training.losses.physics.informed import PhysicsInformedLoss

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator


@register_loss("combined_alphagalerkin_physics")
class CombinedAlphaGalerkinPhysicsLoss(nn.Module):
    """Combined AlphaGalerkin + Physics loss.

    Integrates standard AlphaGalerkin losses (policy, value, LBB)
    with physics-informed losses (residual, boundary, IC).

    This enables:
    - Denser training signal from physics
    - Regularization through physical constraints
    - Better generalization through physical inductive bias
    """

    def __init__(
        self,
        pde_operator: PDEOperator | None = None,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        lbb_weight: float = 0.01,
        physics_weight: float = 0.1,
        physics_config: PhysicsLossConfig | None = None,
    ) -> None:
        """Initialize combined loss.

        Args:
            pde_operator: PDE operator (optional - physics losses disabled if None).
            policy_weight: Weight for policy loss.
            value_weight: Weight for value loss.
            lbb_weight: Weight for LBB regularization.
            physics_weight: Overall weight for physics losses.
            physics_config: Configuration for physics losses.

        """
        super().__init__()

        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.lbb_weight = lbb_weight
        self.physics_weight = physics_weight

        from src.training.losses.alphagalerkin import AlphaGalerkinLoss

        # Standard AlphaGalerkin loss
        self.alphagalerkin_loss = AlphaGalerkinLoss(
            policy_weight=1.0,  # Weights applied externally
            value_weight=1.0,
            lbb_weight=1.0,
        )

        # Physics loss (optional)
        if pde_operator is not None:
            physics_config = physics_config or PhysicsLossConfig(name="physics")
            self.physics_loss = PhysicsInformedLoss(pde_operator, physics_config)
        else:
            self.physics_loss = None

        # Combined loss balancing
        loss_names = ["policy", "value", "lbb"]
        if self.physics_loss is not None:
            loss_names.append("physics")

        # Use adaptive balancing for all terms
        balancing_config = LossBalancingConfig(name="combined_balancing")
        self.balancer = create_loss_balancer(balancing_config, loss_names)

    def forward(
        self,
        policy_logits: Tensor,
        value: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
        lbb_constant: Tensor | None = None,
        action_mask: Tensor | None = None,
        model: nn.Module | None = None,
        coords: Tensor | None = None,
    ) -> dict[str, Tensor | float]:
        """Compute combined loss.

        Args:
            policy_logits: Predicted policy logits.
            value: Predicted value.
            target_policy: Target policy from MCTS.
            target_value: Target value (game outcome).
            lbb_constant: LBB stability constant.
            action_mask: Valid action mask.
            model: Model for physics loss (if using).
            coords: Coordinates for physics loss.

        Returns:
            Dictionary with all loss components and total.

        """
        # Compute AlphaGalerkin losses
        ag_loss = self.alphagalerkin_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            lbb_constant=lbb_constant,
            action_mask=action_mask,
        )

        losses = {
            "policy": ag_loss.policy * self.policy_weight,
            "value": ag_loss.value * self.value_weight,
            "lbb": ag_loss.lbb * self.lbb_weight,
        }

        # Physics loss (if available)
        if self.physics_loss is not None and model is not None:
            physics_output = self.physics_loss(model, coords_interior=coords)
            losses["physics"] = physics_output.total * self.physics_weight

        # Apply balancing
        result = self.balancer.compute_weighted_loss(losses)

        # Get device from first loss for default physics tensor
        device = next(iter(losses.values())).device

        return {
            "total": result.weighted_sum,
            "policy": losses["policy"],
            "value": losses["value"],
            "lbb": losses["lbb"],
            "physics": losses.get("physics", torch.tensor(0.0, device=device)),
            "weights": result.weights,
        }
