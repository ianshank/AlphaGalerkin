"""StabilityGuard: LBB condition monitor for Galerkin attention.

Monitors the singular values of the Key-to-Value projection
during training and provides a regularization loss term to
maintain numerical stability.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

from src.alphagalerkin.core.constants import DIVISION_GUARD, MIN_SINGULAR_VALUE

logger = structlog.get_logger("nn.stability_guard")


class StabilityGuard(nn.Module):
    """Monitors and regularizes LBB stability of Galerkin attention.

    Computes the singular values of the Key-to-Value projection
    matrix and produces:
    1. A diagnostic dict (sigma_min, sigma_max, condition_number).
    2. A regularization loss that penalizes small sigma_min.

    The regularization loss is:
        loss = max(0, beta - sigma_min)^2

    where beta is the stability threshold.

    Parameters
    ----------
    beta:
        Minimum acceptable singular value threshold.
        Defaults to ``MIN_SINGULAR_VALUE`` from constants.
    penalty_weight:
        Scaling factor for the regularization loss.

    """

    def __init__(
        self,
        beta: float = MIN_SINGULAR_VALUE,
        penalty_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.penalty_weight = penalty_weight
        self._last_diagnostics: dict[str, float] = {}

    def forward(
        self,
        ktv_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LBB regularization loss.

        Args:
            ktv_matrix: The K^T @ V matrix from Galerkin attention,
                shape (..., d, d). Can be batched.

        Returns:
            Scalar regularization loss.

        """
        # Flatten batch dims if needed
        if ktv_matrix.dim() > 2:
            flat = ktv_matrix.reshape(-1, ktv_matrix.shape[-2], ktv_matrix.shape[-1])
            # Average over batch for stability
            avg_matrix = flat.mean(dim=0)
        else:
            avg_matrix = ktv_matrix

        singular_values = torch.linalg.svdvals(avg_matrix)
        sigma_min = singular_values.min()
        sigma_max = singular_values.max()

        # Store diagnostics
        self._last_diagnostics = {
            "sigma_min": float(sigma_min.item()),
            "sigma_max": float(sigma_max.item()),
            "condition_number": float((sigma_max / sigma_min.clamp(min=DIVISION_GUARD)).item()),
            "num_singular_values": int(singular_values.numel()),
        }

        # Penalty: penalize when sigma_min drops below beta
        violation = torch.clamp(self.beta - sigma_min, min=0.0)
        loss = self.penalty_weight * violation**2

        if float(sigma_min.item()) < self.beta:
            logger.warning(
                "stability.lbb_violation",
                sigma_min=self._last_diagnostics["sigma_min"],
                beta=self.beta,
            )

        return loss

    @property
    def diagnostics(self) -> dict[str, float]:
        """Most recent stability diagnostics."""
        return dict(self._last_diagnostics)

    def is_stable(self) -> bool:
        """Check if the last measurement satisfies LBB."""
        sigma_min = self._last_diagnostics.get("sigma_min", 0.0)
        return sigma_min >= self.beta
