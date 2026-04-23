"""Stability monitoring for LBB condition in Galerkin attention.

The Ladyzhenskaya-Babuska-Brezzi (LBB) condition ensures numerical
stability of the Galerkin projection. It requires:

    inf_{v in V} sup_{u in U} b(u,v) / (||u|| ||v||) >= beta > 0

For our attention mechanism, this translates to ensuring the minimum
singular value of the Key-to-Value projection is bounded away from zero.
"""

from __future__ import annotations

from typing import cast

import structlog
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

logger = structlog.get_logger(__name__)


class StabilityGuard(nn.Module):
    """Monitors and enforces LBB stability condition.

    This module:
    1. Computes the LBB constant (minimum singular value)
    2. Logs stability metrics
    3. Optionally applies regularization to prevent collapse
    """

    def __init__(
        self,
        beta_threshold: float = 1e-6,
        regularization_strength: float = 0.01,
        log_interval: int = 100,
        margin_multiplier: float = 10.0,
    ) -> None:
        """Initialize stability guard.

        Args:
            beta_threshold: Minimum acceptable LBB constant.
            regularization_strength: Strength of regularization term.
            log_interval: Steps between stability logging.
            margin_multiplier: Multiplier on ``beta_threshold`` used as the
                soft-margin target in :meth:`regularization_loss`. Default
                ``10.0`` preserves prior behaviour.

        Raises:
            ValueError: If ``beta_threshold`` or ``margin_multiplier`` is
                not strictly positive, if ``regularization_strength`` is
                negative, or if ``log_interval`` is less than 1.

        """
        super().__init__()
        if beta_threshold <= 0:
            raise ValueError(f"beta_threshold must be > 0, got {beta_threshold}")
        if regularization_strength < 0:
            raise ValueError(f"regularization_strength must be >= 0, got {regularization_strength}")
        if log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {log_interval}")
        if margin_multiplier <= 0:
            raise ValueError(f"margin_multiplier must be > 0, got {margin_multiplier}")

        self.beta_threshold = beta_threshold
        self.regularization_strength = regularization_strength
        self.log_interval = log_interval
        self.margin_multiplier = margin_multiplier

        # Tracking
        self.step_counter: Tensor
        self.min_beta_seen: Tensor
        self.max_beta_seen: Tensor
        self.register_buffer("step_counter", torch.tensor(0))
        self.register_buffer("min_beta_seen", torch.tensor(float("inf")))
        self.register_buffer("max_beta_seen", torch.tensor(0.0))

        # History for logging
        self._beta_history: list[float] = []

    def compute_lbb_constant(
        self,
        keys: Float[Tensor, "batch n d_key"],
    ) -> Float[Tensor, batch]:
        """Compute LBB stability constant from key matrix.

        The LBB constant is the minimum singular value of K^T K / n.

        Args:
            keys: Key tensor from attention.

        Returns:
            LBB constant (beta) for each batch element.

        """
        n = keys.shape[1]

        # Compute Gram matrix: K^T K / n
        gram = einsum(keys, keys, "b n i, b n j -> b i j") / n

        # Compute singular values
        try:
            singular_values = torch.linalg.svdvals(gram)
            beta = singular_values.min(dim=-1).values
        except RuntimeError:
            # Fallback for numerical issues
            beta = torch.zeros(keys.shape[0], device=keys.device)

        return beta

    def compute_multihead_lbb(
        self,
        keys: Float[Tensor, "batch heads n d_key"],
    ) -> tuple[Float[Tensor, batch], Float[Tensor, "batch heads"]]:
        """Compute LBB constant for multi-head attention.

        Args:
            keys: Multi-head key tensor.

        Returns:
            Tuple of (minimum beta across heads, beta per head).

        """
        batch, heads, n, d_key = keys.shape

        # Flatten batch and heads for vectorized computation
        keys_flat = rearrange(keys, "b h n d -> (b h) n d")

        # Compute beta for each head
        beta_flat = self.compute_lbb_constant(keys_flat)
        beta_per_head = rearrange(beta_flat, "(b h) -> b h", b=batch, h=heads)

        # Minimum across heads (worst-case stability)
        beta_min = beta_per_head.min(dim=-1).values

        return beta_min, beta_per_head

    def check_stability(
        self,
        keys: Float[Tensor, "batch n d_key"] | Float[Tensor, "batch heads n d_key"],
        multihead: bool = False,
    ) -> tuple[bool, Float[Tensor, batch]]:
        """Check if LBB condition is satisfied.

        Args:
            keys: Key tensor (single or multi-head).
            multihead: Whether keys are multi-head format.

        Returns:
            Tuple of (is_stable, beta_values).

        """
        if multihead:
            beta, _ = self.compute_multihead_lbb(keys)
        else:
            beta = self.compute_lbb_constant(keys)

        # Check threshold
        is_stable: bool = bool((beta > self.beta_threshold).all().item())

        # Update tracking
        self.step_counter += 1
        self.min_beta_seen = torch.minimum(self.min_beta_seen, beta.min())
        self.max_beta_seen = torch.maximum(self.max_beta_seen, beta.max())

        # Log periodically
        if self.step_counter % self.log_interval == 0:
            self._log_stability(beta, is_stable)

        return is_stable, beta

    def _log_stability(
        self,
        beta: Float[Tensor, batch],
        is_stable: bool,
    ) -> None:
        """Log stability metrics.

        Args:
            beta: Current LBB constants.
            is_stable: Whether condition is satisfied.

        """
        beta_mean = beta.mean().item()
        beta_min = beta.min().item()
        beta_max = beta.max().item()

        self._beta_history.append(beta_mean)

        logger.info(
            "lbb_stability_check",
            step=self.step_counter.item(),
            is_stable=is_stable,
            beta_mean=beta_mean,
            beta_min=beta_min,
            beta_max=beta_max,
            threshold=self.beta_threshold,
            min_beta_seen=self.min_beta_seen.item(),
        )

    def regularization_loss(
        self,
        keys: Float[Tensor, "batch n d_key"] | Float[Tensor, "batch heads n d_key"],
        multihead: bool = False,
    ) -> Float[Tensor, ""]:
        """Compute regularization loss to encourage stability.

        Penalizes small singular values to prevent LBB violation.

        Args:
            keys: Key tensor.
            multihead: Whether keys are multi-head format.

        Returns:
            Regularization loss scalar.

        """
        if multihead:
            beta, _ = self.compute_multihead_lbb(keys)
        else:
            beta = self.compute_lbb_constant(keys)

        # Penalize when beta is below threshold (soft margin targets larger values).
        margin = self.beta_threshold * self.margin_multiplier
        violation = torch.relu(margin - beta)

        loss = self.regularization_strength * violation.mean()

        return loss

    def forward(
        self,
        keys: Float[Tensor, ...],
        multihead: bool = False,
    ) -> tuple[bool, Float[Tensor, batch], Float[Tensor, ""]]:
        """Forward pass: check stability and compute regularization.

        Args:
            keys: Key tensor.
            multihead: Whether keys are multi-head format.

        Returns:
            Tuple of (is_stable, beta, regularization_loss).

        """
        is_stable, beta = self.check_stability(keys, multihead)
        reg_loss = self.regularization_loss(keys, multihead)

        return is_stable, beta, reg_loss

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics.

        Returns:
            Dictionary of stability statistics.

        """
        return {
            "total_steps": self.step_counter.item(),
            "min_beta_seen": self.min_beta_seen.item(),
            "max_beta_seen": self.max_beta_seen.item(),
            "beta_threshold": self.beta_threshold,
            "recent_beta_mean": (
                sum(self._beta_history[-100:]) / len(self._beta_history[-100:])
                if self._beta_history
                else 0.0
            ),
        }


class StableGalerkinInitializer:
    """Initializer that ensures LBB stability from the start.

    Proper initialization is crucial for maintaining the inf-sup condition.
    This class provides initialization schemes that satisfy LBB from t=0.
    """

    # Default numerical safety constants (made parameters for testability and
    # for downstream consumers that need to tune them; defaults preserve prior
    # behaviour bit-for-bit).
    DEFAULT_GUARD_THRESHOLD_RATIO: float = 10.0
    DEFAULT_SCALE_EPSILON: float = 1e-8
    DEFAULT_SCALE_CLAMP: tuple[float, float] = (1.0, 2.0)

    def __init__(
        self,
        beta_target: float = 0.1,
        max_iterations: int = 10,
        scale_epsilon: float = DEFAULT_SCALE_EPSILON,
        scale_clamp: tuple[float, float] = DEFAULT_SCALE_CLAMP,
        guard_threshold_ratio: float = DEFAULT_GUARD_THRESHOLD_RATIO,
    ) -> None:
        """Initialize the stable initializer.

        Args:
            beta_target: Target LBB constant.
            max_iterations: Maximum initialization attempts.
            scale_epsilon: Small constant added to the denominator when
                computing the adjustment scale factor to avoid division by
                zero. Default ``1e-8`` preserves prior behaviour.
            scale_clamp: ``(min, max)`` clamp range applied to the
                per-iteration adjustment scale factor. Bounds the scale
                factor to this range on every adjustment step. With the
                default ``(1.0, 2.0)`` the clamp prevents both shrinking
                weights (``min >= 1.0``) and pathologically large boosts
                (``max``-capped); callers may pass ``min < 1.0`` if
                shrinking during adjustment is acceptable for their use
                case.
            guard_threshold_ratio: Ratio used to derive the internal
                :class:`StabilityGuard` threshold from ``beta_target``
                (``threshold = beta_target / ratio``). Default ``10.0``
                preserves prior behaviour.

        Raises:
            ValueError: If any numeric argument is invalid.

        """
        if beta_target <= 0:
            raise ValueError(f"beta_target must be > 0, got {beta_target}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if scale_epsilon <= 0:
            raise ValueError(f"scale_epsilon must be > 0, got {scale_epsilon}")
        if guard_threshold_ratio <= 0:
            raise ValueError(f"guard_threshold_ratio must be > 0, got {guard_threshold_ratio}")
        if len(scale_clamp) != 2:
            raise ValueError(
                f"scale_clamp must have exactly 2 elements (min, max), got {len(scale_clamp)}"
            )
        if scale_clamp[0] <= 0 or scale_clamp[1] <= 0:
            raise ValueError(f"scale_clamp values must be > 0, got {scale_clamp}")
        if scale_clamp[0] > scale_clamp[1]:
            raise ValueError(f"scale_clamp[0] must be <= scale_clamp[1], got {scale_clamp}")

        self.beta_target = beta_target
        self.max_iterations = max_iterations
        self.scale_epsilon = scale_epsilon
        self.scale_clamp = scale_clamp
        self.guard_threshold_ratio = guard_threshold_ratio
        self.stability_guard = StabilityGuard(beta_threshold=beta_target / guard_threshold_ratio)

    def initialize_projection(
        self,
        weight: Tensor,
        d_key: int,
    ) -> None:
        """Initialize a projection weight matrix for LBB stability.

        Uses orthogonal initialization with proper scaling.

        Args:
            weight: Weight tensor to initialize.
            d_key: Key dimension (for computing variance).

        """
        # Start with orthogonal initialization
        nn.init.orthogonal_(weight)

        # Scale to target variance
        scale = 1.0 / (d_key**0.5)
        weight.data.mul_(scale)

    def verify_and_adjust(
        self,
        module: nn.Module,
        sample_input: Tensor,
    ) -> bool:
        """Verify LBB stability and adjust if needed.

        Args:
            module: Module with key projection.
            sample_input: Sample input for verification.

        Returns:
            True if stable, False if adjustment failed.

        """
        for i in range(self.max_iterations):
            # Get keys from module
            with torch.no_grad():
                if hasattr(module, "to_k"):
                    keys = cast(nn.Linear, module.to_k)(sample_input)
                elif hasattr(module, "to_key"):
                    keys = cast(nn.Linear, module.to_key)(sample_input)
                else:
                    raise ValueError("Module must have 'to_k' or 'to_key' attribute")

            # Check stability
            is_stable, beta = self.stability_guard.check_stability(keys)

            if is_stable and beta.min() >= self.beta_target:
                logger.info(
                    "initialization_stable",
                    iteration=i,
                    beta_min=beta.min().item(),
                    beta_target=self.beta_target,
                )
                return True

            # Adjust weights
            self._adjust_for_stability(module, beta)

        logger.warning(
            "initialization_unstable",
            final_beta=beta.min().item(),
            beta_target=self.beta_target,
        )
        return False

    def _adjust_for_stability(
        self,
        module: nn.Module,
        current_beta: Tensor,
    ) -> None:
        """Adjust module weights to improve stability.

        Args:
            module: Module to adjust.
            current_beta: Current LBB constants.

        """
        # Add small perturbation to increase singular values
        if hasattr(module, "to_k"):
            weight = cast(nn.Linear, module.to_k).weight
        elif hasattr(module, "to_key"):
            weight = cast(nn.Linear, module.to_key).weight
        else:
            return

        # Scale up slightly to increase singular values. Wrap the in-place
        # parameter mutation in ``torch.no_grad()`` so that if this method is
        # ever called during a training loop, autograd state is not polluted.
        scale_factor = (self.beta_target / (current_beta.min() + self.scale_epsilon)).sqrt()
        scale_factor = torch.clamp(scale_factor, *self.scale_clamp)
        with torch.no_grad():
            weight.mul_(scale_factor.item())
