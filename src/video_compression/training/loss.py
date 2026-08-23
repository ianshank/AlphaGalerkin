"""Loss functions for video compression training.

Implements rate-distortion loss with multiple distortion metrics:
    L = D + λR

where:
- D: Distortion (MSE, MS-SSIM, or mixed)
- R: Rate (bits per pixel)
- λ: Rate-distortion tradeoff parameter
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.metrics.quality import (
    MSSSIM,
    PerceptualLoss,
    compute_psnr,
)


class LossOutput(NamedTuple):
    """Output from loss computation."""

    total: Tensor
    rate: Tensor
    distortion: Tensor
    mse: Tensor
    psnr: Tensor
    ms_ssim: Tensor | None


class DistortionLoss(nn.Module):
    """Distortion loss supporting multiple metrics.

    Supports:
    - MSE: Mean Squared Error
    - MS-SSIM: Multi-Scale Structural Similarity
    - Mixed: Weighted combination of MSE and MS-SSIM
    """

    def __init__(
        self,
        metric: Literal["mse", "ms_ssim", "mixed"] = "mixed",
        ms_ssim_weight: float = 0.84,
    ) -> None:
        """Initialize distortion loss.

        Args:
            metric: Distortion metric type.
            ms_ssim_weight: Weight for MS-SSIM in mixed mode.

        """
        super().__init__()
        self.metric = metric
        self.ms_ssim_weight = ms_ssim_weight

        if metric in ["ms_ssim", "mixed"]:
            self.ms_ssim = MSSSIM(as_loss=True)

    def forward(
        self,
        pred: Float[Tensor, "batch 3 height width"],
        target: Float[Tensor, "batch 3 height width"],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute distortion loss.

        Args:
            pred: Predicted images.
            target: Target images.

        Returns:
            Tuple of (distortion_loss, mse, ms_ssim_loss).

        """
        # MSE
        mse = torch.mean((pred - target) ** 2)

        # MS-SSIM if needed
        ms_ssim_loss = None
        if self.metric in ["ms_ssim", "mixed"]:
            ms_ssim_loss = self.ms_ssim(pred, target)

        # Combine based on metric
        if self.metric == "mse":
            distortion = mse
        elif self.metric == "ms_ssim":
            assert ms_ssim_loss is not None, "MS-SSIM loss required for ms_ssim metric"
            distortion = ms_ssim_loss
        else:  # mixed
            # Combined loss: balance MSE and MS-SSIM
            # MS-SSIM is scale-invariant, MSE is not
            assert ms_ssim_loss is not None, "MS-SSIM loss required for mixed metric"
            distortion = self.ms_ssim_weight * ms_ssim_loss + (1 - self.ms_ssim_weight) * mse

        return distortion, mse, ms_ssim_loss


class RDLoss(nn.Module):
    """Rate-Distortion loss for compression training.

    L = D + λR

    where D is distortion and R is rate in bits per pixel.
    """

    def __init__(
        self,
        lambda_rd: float = 0.01,
        distortion_metric: Literal["mse", "ms_ssim", "mixed"] = "mixed",
        ms_ssim_weight: float = 0.84,
    ) -> None:
        """Initialize R-D loss.

        Args:
            lambda_rd: Rate-distortion tradeoff parameter.
            distortion_metric: Distortion metric type.
            ms_ssim_weight: MS-SSIM weight in mixed mode.

        """
        super().__init__()
        self.lambda_rd = lambda_rd
        self.distortion = DistortionLoss(distortion_metric, ms_ssim_weight)

    def forward(
        self,
        pred: Float[Tensor, "batch 3 height width"],
        target: Float[Tensor, "batch 3 height width"],
        rate: Float[Tensor, batch],
    ) -> LossOutput:
        """Compute R-D loss.

        Args:
            pred: Reconstructed images.
            target: Original images.
            rate: Estimated rate in bits.

        Returns:
            LossOutput with all loss components.

        """
        batch, _, h, w = pred.shape

        # Distortion
        distortion_loss, mse, ms_ssim_loss = self.distortion(pred, target)

        # Rate in bits per pixel
        rate_bpp = rate / (h * w)
        rate_loss = rate_bpp.mean()

        # R-D loss
        total_loss = distortion_loss + self.lambda_rd * rate_loss

        # Metrics
        psnr = compute_psnr(pred, target)

        return LossOutput(
            total=total_loss,
            rate=rate_loss,
            distortion=distortion_loss,
            mse=mse,
            psnr=psnr,
            ms_ssim=ms_ssim_loss,
        )


class CompressionLoss(nn.Module):
    """Complete compression loss with optional perceptual component.

    L = D + λR + α * L_perceptual

    where:
    - D: Distortion (MSE or MS-SSIM)
    - R: Rate
    - L_perceptual: VGG perceptual loss
    """

    def __init__(
        self,
        lambda_rd: float = 0.01,
        distortion_metric: Literal["mse", "ms_ssim", "mixed"] = "mixed",
        ms_ssim_weight: float = 0.84,
        use_perceptual: bool = True,
        perceptual_weight: float = 0.1,
    ) -> None:
        """Initialize compression loss.

        Args:
            lambda_rd: Rate-distortion tradeoff.
            distortion_metric: Distortion metric type.
            ms_ssim_weight: MS-SSIM weight in mixed mode.
            use_perceptual: Whether to use perceptual loss.
            perceptual_weight: Weight for perceptual loss.

        """
        super().__init__()
        self.lambda_rd = lambda_rd
        self.perceptual_weight = perceptual_weight

        self.rd_loss = RDLoss(lambda_rd, distortion_metric, ms_ssim_weight)

        self.perceptual_loss: PerceptualLoss | None = PerceptualLoss() if use_perceptual else None

    def forward(
        self,
        pred: Float[Tensor, "batch 3 height width"],
        target: Float[Tensor, "batch 3 height width"],
        rate: Float[Tensor, batch],
    ) -> dict[str, Tensor]:
        """Compute compression loss.

        Args:
            pred: Reconstructed images.
            target: Original images.
            rate: Estimated rate in bits.

        Returns:
            Dictionary of loss components.

        """
        # R-D loss
        rd_output = self.rd_loss(pred, target, rate)

        losses = {
            "total": rd_output.total,
            "rate": rd_output.rate,
            "distortion": rd_output.distortion,
            "mse": rd_output.mse,
            "psnr": rd_output.psnr,
        }

        if rd_output.ms_ssim is not None:
            losses["ms_ssim_loss"] = rd_output.ms_ssim

        # Perceptual loss
        if self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(pred, target)
            losses["perceptual"] = p_loss
            losses["total"] = losses["total"] + self.perceptual_weight * p_loss

        return losses
