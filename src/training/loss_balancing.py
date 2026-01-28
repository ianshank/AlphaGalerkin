"""Adaptive loss balancing for multi-objective training.

This module implements loss balancing strategies for combining
multiple loss terms (policy, value, physics, LBB) during training.

Strategies:
- ReLoBRaLo: Relative Loss Balancing with Random Lookback
- GradNorm: Gradient normalization
- Uncertainty: Homoscedastic uncertainty weighting
- Static: Fixed weights

ReLoBRaLo (Bischof et al., 2022) is particularly effective for
physics-informed neural networks where loss terms have different
magnitudes and convergence rates.

Reference:
    Bischof, R., & Kraus, M. (2022). Multi-Objective Loss Balancing
    for Physics-Informed Deep Learning.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor, nn

from src.templates.config import BaseModuleConfig

logger = structlog.get_logger(__name__)

# Default numerical stability constant
DEFAULT_EPSILON = 1e-10
# Default maximum exponential clipping value
DEFAULT_MAX_EXP = 1e6


class BalancingStrategy(str, Enum):
    """Available loss balancing strategies."""

    STATIC = "static"
    RELOBRALO = "relobralo"
    GRADNORM = "gradnorm"
    UNCERTAINTY = "uncertainty"
    SOFTADAPT = "softadapt"


class LossBalancingConfig(BaseModuleConfig):
    """Configuration for loss balancing.

    Attributes:
        strategy: Balancing strategy to use.
        beta: EMA decay for running loss averages.
        tau: Temperature for softmax weighting.
        alpha: GradNorm asymmetry hyperparameter.
        update_frequency: How often to update weights.
        min_weight: Minimum allowed weight per loss term.
        max_weight: Maximum allowed weight per loss term.
    """

    strategy: BalancingStrategy = Field(
        default=BalancingStrategy.RELOBRALO,
        description="Loss balancing strategy",
    )

    # ReLoBRaLo parameters
    beta: float = Field(
        default=0.99,
        ge=0.0,
        lt=1.0,
        description="EMA decay for running loss averages",
    )
    tau: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature for softmax weighting",
    )
    random_lookback: bool = Field(
        default=True,
        description="Use random lookback (ReLoBRaLo) vs fixed (ReLoBRaLo-F)",
    )

    # GradNorm parameters
    alpha: float = Field(
        default=1.5,
        gt=0.0,
        le=3.0,
        description="GradNorm asymmetry hyperparameter",
    )

    # General parameters
    update_frequency: int = Field(
        default=1,
        ge=1,
        description="Steps between weight updates",
    )
    min_weight: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Minimum weight per loss term",
    )
    max_weight: float = Field(
        default=10.0,
        gt=1.0,
        description="Maximum weight per loss term",
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Steps before starting adaptation",
    )

    # Numerical stability parameters
    epsilon: float = Field(
        default=DEFAULT_EPSILON,
        gt=0.0,
        lt=1e-3,
        description="Numerical stability constant for division",
    )
    max_exp_clip: float = Field(
        default=DEFAULT_MAX_EXP,
        gt=0.0,
        description="Maximum value for exponential clipping",
    )
    history_buffer_size: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum history buffer size for lookback",
    )

    # SoftAdapt specific
    softadapt_window_size: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Window size for SoftAdapt rate computation",
    )


@dataclass
class LossTerms:
    """Container for multiple loss terms.

    Attributes:
        losses: Dictionary mapping loss names to values.
        weights: Current weights per loss term.
        weighted_sum: Total weighted loss.
    """

    losses: dict[str, Float[Tensor, ""]]
    weights: dict[str, float]
    weighted_sum: Float[Tensor, ""]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of scalar values."""
        result = {
            f"loss_{name}": loss.item()
            for name, loss in self.losses.items()
        }
        result.update({
            f"weight_{name}": weight
            for name, weight in self.weights.items()
        })
        result["total"] = self.weighted_sum.item()
        return result


class LossBalancer(ABC):
    """Abstract base class for loss balancing strategies."""

    def __init__(self, config: LossBalancingConfig, loss_names: list[str]) -> None:
        """Initialize balancer.

        Args:
            config: Balancing configuration.
            loss_names: Names of loss terms to balance.
        """
        self.config = config
        self.loss_names = loss_names
        self.n_losses = len(loss_names)
        self._step = 0

        # Initialize weights uniformly
        self._weights = {name: 1.0 for name in loss_names}

    @property
    def weights(self) -> dict[str, float]:
        """Current loss weights."""
        return dict(self._weights)

    @abstractmethod
    def update(self, losses: dict[str, Tensor]) -> dict[str, float]:
        """Update weights based on current losses.

        Args:
            losses: Current loss values per term.

        Returns:
            Updated weights.
        """
        raise NotImplementedError

    def compute_weighted_loss(self, losses: dict[str, Tensor]) -> LossTerms:
        """Compute weighted sum of losses.

        Args:
            losses: Loss values per term.

        Returns:
            LossTerms with weighted sum and current weights.

        Raises:
            ValueError: If no valid loss terms are provided.
        """
        # Check for missing loss terms and log warning
        missing_terms = [name for name in self.loss_names if name not in losses]
        if missing_terms:
            logger.warning(
                "missing_loss_terms",
                missing=missing_terms,
                provided=list(losses.keys()),
            )

        # Ensure we have at least one valid loss
        valid_losses = [name for name in self.loss_names if name in losses]
        if not valid_losses:
            raise ValueError(
                f"No valid loss terms provided. Expected: {self.loss_names}, "
                f"got: {list(losses.keys())}"
            )

        # Update weights if needed
        if self._step % self.config.update_frequency == 0:
            if self._step >= self.config.warmup_steps:
                self._weights = self.update(losses)
                logger.debug(
                    "weights_updated",
                    step=self._step,
                    weights=self._weights,
                )

        self._step += 1

        # Compute weighted sum (only for present losses)
        total = sum(
            self._weights[name] * losses[name]
            for name in self.loss_names
            if name in losses
        )

        return LossTerms(
            losses=losses,
            weights=dict(self._weights),
            weighted_sum=total,
        )

    def reset(self) -> None:
        """Reset balancer state."""
        self._step = 0
        self._weights = {name: 1.0 for name in self.loss_names}


class ReLoBRaLo(LossBalancer):
    """Relative Loss Balancing with Random Lookback.

    This strategy balances losses by comparing current loss values
    to historical averages, using random lookback for robustness.

    The weight for loss i at step t is:
        w_i(t) = softmax(L_i(t) / (L_i(τ) * temperature))

    where τ is a random lookback step (or exponential moving average).

    Reference:
        Bischof, R., & Kraus, M. (2022). Multi-Objective Loss Balancing
        for Physics-Informed Deep Learning.
    """

    def __init__(self, config: LossBalancingConfig, loss_names: list[str]) -> None:
        """Initialize ReLoBRaLo balancer."""
        super().__init__(config, loss_names)

        # Running loss averages
        self._running_losses: dict[str, float] = {}
        self._loss_history: dict[str, list[float]] = {
            name: [] for name in loss_names
        }

    def update(self, losses: dict[str, Tensor]) -> dict[str, float]:
        """Update weights using ReLoBRaLo formula."""
        eps = self.config.epsilon
        max_exp = self.config.max_exp_clip
        buffer_size = self.config.history_buffer_size

        # Update running averages
        for name in self.loss_names:
            if name not in losses:
                continue

            loss_val = losses[name].detach().item()

            # Store history for random lookback
            self._loss_history[name].append(loss_val)
            if len(self._loss_history[name]) > buffer_size:
                self._loss_history[name] = self._loss_history[name][-buffer_size:]

            # Update EMA
            if name not in self._running_losses:
                self._running_losses[name] = loss_val
            else:
                self._running_losses[name] = (
                    self.config.beta * self._running_losses[name]
                    + (1 - self.config.beta) * loss_val
                )

        # Compute lookback values
        lookback_losses = {}
        for name in self.loss_names:
            if name not in losses:
                continue

            if self.config.random_lookback and len(self._loss_history[name]) > 1:
                # Random lookback: sample from history
                idx = random.randint(0, len(self._loss_history[name]) - 1)
                lookback_losses[name] = self._loss_history[name][idx]
            else:
                # Fixed: use running average
                lookback_losses[name] = self._running_losses.get(name, 1.0)

        # Compute relative losses
        relative_losses = {}
        for name in self.loss_names:
            if name not in losses:
                continue

            loss_val = losses[name].detach().item()
            lookback = max(lookback_losses[name], eps)  # Avoid division by zero

            relative_losses[name] = loss_val / (lookback * self.config.tau + eps)

        # Softmax to get weights
        if relative_losses:
            max_rel = max(relative_losses.values())
            exp_rel = {}
            for name, val in relative_losses.items():
                # Use math.exp for scalar computation (avoid torch overhead)
                try:
                    exp_val = min(math.exp(val - max_rel), max_exp)
                except OverflowError:
                    exp_val = max_exp
                exp_rel[name] = exp_val

            sum_exp = sum(exp_rel.values())

            weights = {}
            for name in self.loss_names:
                if name in exp_rel:
                    raw_weight = exp_rel[name] / sum_exp * self.n_losses
                    weights[name] = max(
                        self.config.min_weight,
                        min(self.config.max_weight, raw_weight)
                    )
                else:
                    weights[name] = 1.0

            return weights

        return self._weights

    def reset(self) -> None:
        """Reset balancer state."""
        super().reset()
        self._running_losses = {}
        self._loss_history = {name: [] for name in self.loss_names}


class GradNorm(LossBalancer):
    """Gradient Normalization loss balancing.

    Balances losses by normalizing gradient magnitudes across
    different loss terms, ensuring equal training signal.

    Reference:
        Chen, Z., et al. (2018). GradNorm: Gradient Normalization
        for Adaptive Loss Balancing in Deep Multitask Networks.
    """

    def __init__(
        self,
        config: LossBalancingConfig,
        loss_names: list[str],
        model: nn.Module | None = None,
    ) -> None:
        """Initialize GradNorm balancer.

        Args:
            config: Balancing configuration.
            loss_names: Names of loss terms.
            model: Model for gradient computation (optional).
        """
        super().__init__(config, loss_names)
        self.model = model

        # Learnable weights (log-space for stability)
        self._log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in loss_names
        })

        # Initial loss values for relative training rate
        self._initial_losses: dict[str, float] = {}

    def update(self, losses: dict[str, Tensor]) -> dict[str, float]:
        """Update weights using gradient normalization."""
        # Store initial losses
        for name in self.loss_names:
            if name in losses and name not in self._initial_losses:
                self._initial_losses[name] = losses[name].detach().item()

        # Compute current weights from log weights
        weights = {}
        for name in self.loss_names:
            if name in self._log_weights:
                raw_weight = torch.exp(self._log_weights[name]).item()
                weights[name] = max(
                    self.config.min_weight,
                    min(self.config.max_weight, raw_weight)
                )
            else:
                weights[name] = 1.0

        return weights

    def compute_gradnorm_loss(
        self,
        losses: dict[str, Tensor],
        shared_layer: nn.Module,
    ) -> Tensor:
        """Compute GradNorm balancing loss for weight updates.

        Args:
            losses: Individual loss values.
            shared_layer: Shared layer for gradient computation.

        Returns:
            GradNorm loss for updating weights.
        """
        # Determine device from first loss tensor
        device = next(iter(losses.values())).device if losses else torch.device("cpu")

        # Compute gradient norms for each task
        grad_norms = {}
        for name, loss in losses.items():
            if name not in self._log_weights:
                continue

            # Get gradients w.r.t. shared layer
            try:
                grads = torch.autograd.grad(
                    loss, shared_layer.parameters(),
                    retain_graph=True, allow_unused=True
                )
                grad_norm = sum(
                    g.norm() ** 2 for g in grads if g is not None
                ) ** 0.5
                grad_norms[name] = grad_norm
            except RuntimeError as e:
                logger.warning(
                    "gradient_computation_failed",
                    loss_name=name,
                    error=str(e),
                )
                continue

        if not grad_norms:
            return torch.tensor(0.0, device=device)

        # Average gradient norm
        avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)

        # Compute relative training rates
        rel_rates = {}
        for name in grad_norms:
            if name in self._initial_losses and self._initial_losses[name] > 0:
                current = losses[name].detach().item()
                initial = self._initial_losses[name]
                rel_rates[name] = current / initial
            else:
                rel_rates[name] = 1.0

        avg_rel_rate = sum(rel_rates.values()) / len(rel_rates) if rel_rates else 1.0

        # GradNorm loss: make all gradient norms equal to avg * relative rate
        gradnorm_loss = torch.tensor(0.0, device=device)
        for name, grad_norm in grad_norms.items():
            target_norm = avg_grad_norm * (
                (rel_rates.get(name, 1.0) / avg_rel_rate) ** self.config.alpha
            )
            weight = torch.exp(self._log_weights[name])
            gradnorm_loss = gradnorm_loss + torch.abs(
                weight * grad_norm - target_norm
            )

        return gradnorm_loss


class UncertaintyWeighting(LossBalancer):
    """Homoscedastic uncertainty weighting.

    Models task-dependent uncertainty and uses it to weight losses.
    Each loss term i has learnable log-variance σ_i, and the
    effective weight is 1/(2σ_i²).

    Reference:
        Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning
        Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.
    """

    def __init__(self, config: LossBalancingConfig, loss_names: list[str]) -> None:
        """Initialize uncertainty weighting."""
        super().__init__(config, loss_names)

        # Learnable log-variance parameters
        self._log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in loss_names
        })

    @property
    def log_vars(self) -> nn.ParameterDict:
        """Access learnable log-variance parameters."""
        return self._log_vars

    def update(self, losses: dict[str, Tensor]) -> dict[str, float]:
        """Compute weights from learned uncertainties."""
        weights = {}
        for name in self.loss_names:
            if name in self._log_vars:
                # Weight = 1 / (2 * exp(log_var)) = exp(-log_var) / 2
                log_var = self._log_vars[name]
                raw_weight = (0.5 * torch.exp(-log_var)).item()
                weights[name] = max(
                    self.config.min_weight,
                    min(self.config.max_weight, raw_weight)
                )
            else:
                weights[name] = 1.0

        return weights

    def compute_regularized_loss(self, losses: dict[str, Tensor]) -> Tensor:
        """Compute loss with uncertainty regularization.

        The uncertainty-weighted loss includes a regularization term
        to prevent uncertainties from growing unboundedly:

        L = Σ (1/(2σ²) * L_i + log(σ))

        Args:
            losses: Individual loss values.

        Returns:
            Total regularized loss.
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)

        for name, loss in losses.items():
            if name in self._log_vars:
                log_var = self._log_vars[name]
                precision = torch.exp(-log_var)
                total = total + precision * loss + log_var
            else:
                total = total + loss

        return total


class SoftAdapt(LossBalancer):
    """SoftAdapt loss balancing.

    Adapts weights based on rate of change of each loss term.
    Losses that are improving faster get lower weights.

    Reference:
        Heydari, A. A., et al. (2019). SoftAdapt: Techniques for
        Adaptive Loss Weighting of Neural Networks.
    """

    def __init__(self, config: LossBalancingConfig, loss_names: list[str]) -> None:
        """Initialize SoftAdapt balancer."""
        super().__init__(config, loss_names)

        # Loss history for rate computation (use configurable window size)
        self._loss_history: dict[str, list[float]] = {
            name: [] for name in loss_names
        }
        self._window_size = config.softadapt_window_size

    def update(self, losses: dict[str, Tensor]) -> dict[str, float]:
        """Update weights based on loss improvement rates."""
        eps = self.config.epsilon

        # Update history
        for name in self.loss_names:
            if name not in losses:
                continue

            loss_val = losses[name].detach().item()
            self._loss_history[name].append(loss_val)
            if len(self._loss_history[name]) > self._window_size:
                self._loss_history[name] = self._loss_history[name][-self._window_size:]

        # Compute improvement rates
        rates = {}
        for name in self.loss_names:
            history = self._loss_history.get(name, [])
            if len(history) >= 2:
                # Rate of improvement (negative = improving)
                denominator = len(history) * max(history[0], eps)
                rate = (history[-1] - history[0]) / denominator
                rates[name] = rate
            else:
                rates[name] = 0.0

        # Softmax over rates (higher rate = higher weight)
        if rates:
            max_rate = max(rates.values())
            exp_rates = {}
            for name, rate in rates.items():
                try:
                    exp_val = math.exp((rate - max_rate) / self.config.tau)
                except OverflowError:
                    exp_val = self.config.max_exp_clip
                exp_rates[name] = exp_val

            sum_exp = sum(exp_rates.values())

            weights = {}
            for name in self.loss_names:
                if name in exp_rates:
                    raw_weight = exp_rates[name] / sum_exp * self.n_losses
                    weights[name] = max(
                        self.config.min_weight,
                        min(self.config.max_weight, raw_weight)
                    )
                else:
                    weights[name] = 1.0

            return weights

        return self._weights

    def reset(self) -> None:
        """Reset balancer state."""
        super().reset()
        self._loss_history = {name: [] for name in self.loss_names}


class StaticWeighting(LossBalancer):
    """Static (fixed) loss weighting.

    Simply uses fixed weights without adaptation.
    Useful as a baseline or when loss scales are known.
    """

    def __init__(
        self,
        config: LossBalancingConfig,
        loss_names: list[str],
        initial_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize static weighting.

        Args:
            config: Balancing configuration.
            loss_names: Names of loss terms.
            initial_weights: Fixed weights (defaults to uniform).
        """
        super().__init__(config, loss_names)

        if initial_weights:
            self._weights = {
                name: initial_weights.get(name, 1.0)
                for name in loss_names
            }

    def update(self, losses: dict[str, Tensor]) -> dict[str, float]:
        """No update for static weighting."""
        return self._weights


def create_loss_balancer(
    config: LossBalancingConfig,
    loss_names: list[str],
    model: nn.Module | None = None,
    initial_weights: dict[str, float] | None = None,
) -> LossBalancer:
    """Factory function to create a loss balancer.

    Args:
        config: Balancing configuration.
        loss_names: Names of loss terms.
        model: Model for gradient-based methods (optional).
        initial_weights: Initial weights for static method.

    Returns:
        Configured LossBalancer instance.
    """
    if config.strategy == BalancingStrategy.STATIC:
        return StaticWeighting(config, loss_names, initial_weights)
    elif config.strategy == BalancingStrategy.RELOBRALO:
        return ReLoBRaLo(config, loss_names)
    elif config.strategy == BalancingStrategy.GRADNORM:
        return GradNorm(config, loss_names, model)
    elif config.strategy == BalancingStrategy.UNCERTAINTY:
        return UncertaintyWeighting(config, loss_names)
    elif config.strategy == BalancingStrategy.SOFTADAPT:
        return SoftAdapt(config, loss_names)
    else:
        raise ValueError(f"Unknown balancing strategy: {config.strategy}")
