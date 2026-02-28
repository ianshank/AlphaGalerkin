"""Image and video quality metrics.

Implements differentiable versions of standard quality metrics:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- MS-SSIM: Multi-Scale Structural Similarity

All metrics support batch computation and can be used as loss functions.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


def compute_psnr(
    pred: Float[Tensor, "batch channels height width"],
    target: Float[Tensor, "batch channels height width"],
    max_val: float = 1.0,
    reduction: Literal["mean", "none"] = "mean",
) -> Float[Tensor, ...]:
    """Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        pred: Predicted images.
        target: Target images.
        max_val: Maximum pixel value (1.0 for normalized, 255.0 for uint8).
        reduction: "mean" for scalar, "none" for per-image.

    Returns:
        PSNR value(s) in dB.

    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])

    # Avoid log(0)
    mse = torch.clamp(mse, min=1e-10)

    psnr = 10 * torch.log10(max_val**2 / mse)

    if reduction == "mean":
        return psnr.mean()
    return psnr


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> Tensor:
    """Create 1D Gaussian kernel.

    Args:
        size: Kernel size.
        sigma: Gaussian standard deviation.
        device: Target device.

    Returns:
        1D Gaussian kernel.

    """
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_kernel_2d(
    size: int,
    sigma: float,
    channels: int,
    device: torch.device,
) -> Tensor:
    """Create 2D Gaussian kernel for convolution.

    Args:
        size: Kernel size.
        sigma: Gaussian standard deviation.
        channels: Number of channels.
        device: Target device.

    Returns:
        2D Gaussian kernel (channels, 1, size, size).

    """
    kernel_1d = _gaussian_kernel_1d(size, sigma, device)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel = kernel_2d.expand(channels, 1, size, size)
    return kernel


def compute_ssim(
    pred: Float[Tensor, "batch channels height width"],
    target: Float[Tensor, "batch channels height width"],
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    max_val: float = 1.0,
    reduction: Literal["mean", "none"] = "mean",
) -> Float[Tensor, ...]:
    """Compute Structural Similarity Index.

    SSIM compares luminance, contrast, and structure between images.

    Args:
        pred: Predicted images.
        target: Target images.
        window_size: Size of Gaussian window.
        sigma: Gaussian standard deviation.
        k1: Stability constant for luminance.
        k2: Stability constant for contrast.
        max_val: Maximum pixel value.
        reduction: "mean" for scalar, "none" for per-image.

    Returns:
        SSIM value(s) in [0, 1].

    """
    batch, channels, height, width = pred.shape
    device = pred.device

    # Stability constants
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Create Gaussian window
    kernel = _gaussian_kernel_2d(window_size, sigma, channels, device)
    padding = window_size // 2

    # Compute local means
    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)

    # Compute local variances and covariance
    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred**2, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target**2, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred_target
    )

    # SSIM formula
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    ssim_map = numerator / (denominator + 1e-8)

    # Reduce to scalar
    ssim_val = ssim_map.mean(dim=[1, 2, 3])

    if reduction == "mean":
        return ssim_val.mean()
    return ssim_val


def compute_ms_ssim(
    pred: Float[Tensor, "batch channels height width"],
    target: Float[Tensor, "batch channels height width"],
    window_size: int = 11,
    sigma: float = 1.5,
    weights: list[float] | None = None,
    max_val: float = 1.0,
    reduction: Literal["mean", "none"] = "mean",
) -> Float[Tensor, ...]:
    """Compute Multi-Scale Structural Similarity.

    MS-SSIM computes SSIM at multiple scales and combines them.

    Args:
        pred: Predicted images.
        target: Target images.
        window_size: Size of Gaussian window.
        sigma: Gaussian standard deviation.
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
        max_val: Maximum pixel value.
        reduction: "mean" for scalar, "none" for per-image.

    Returns:
        MS-SSIM value(s) in [0, 1].

    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    num_scales = len(weights)

    # Check minimum size
    min_size = 2 ** (num_scales - 1) * window_size
    _, _, h, w = pred.shape
    if h < min_size or w < min_size:
        # Fall back to fewer scales
        num_scales = max(1, min(int(math.log2(min(h, w) / window_size)) + 1, num_scales))
        weights = weights[:num_scales]
        # Renormalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Compute SSIM at each scale
    cs_products = []
    ssim_last = None

    for scale in range(num_scales):
        if scale > 0:
            # Downsample
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)

        # Compute SSIM components
        ssim_val = compute_ssim(pred, target, window_size, sigma, max_val=max_val, reduction="none")

        if scale < num_scales - 1:
            # Extract contrast-structure component (CS)
            cs = ssim_val  # Simplified: use full SSIM as CS proxy
            cs_products.append(cs)
        else:
            ssim_last = ssim_val

    # Combine scales
    assert ssim_last is not None, "ssim_last should be set for at least 1 scale"
    ms_ssim = ssim_last ** weights[-1]
    for i, cs in enumerate(cs_products):
        ms_ssim = ms_ssim * (cs ** weights[i])

    if reduction == "mean":
        return ms_ssim.mean()
    return ms_ssim


class PSNR(nn.Module):
    """PSNR as a PyTorch module."""

    def __init__(self, max_val: float = 1.0) -> None:
        """Initialize PSNR module.

        Args:
            max_val: Maximum pixel value.

        """
        super().__init__()
        self.max_val = max_val

    def forward(
        self,
        pred: Float[Tensor, "batch channels height width"],
        target: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """Compute PSNR.

        Args:
            pred: Predicted images.
            target: Target images.

        Returns:
            Mean PSNR in dB.

        """
        return compute_psnr(pred, target, self.max_val)


class SSIM(nn.Module):
    """SSIM as a PyTorch module (can be used as loss)."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        max_val: float = 1.0,
        as_loss: bool = False,
    ) -> None:
        """Initialize SSIM module.

        Args:
            window_size: Size of Gaussian window.
            sigma: Gaussian standard deviation.
            max_val: Maximum pixel value.
            as_loss: If True, returns 1 - SSIM for use as loss.

        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.max_val = max_val
        self.as_loss = as_loss

    def forward(
        self,
        pred: Float[Tensor, "batch channels height width"],
        target: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """Compute SSIM.

        Args:
            pred: Predicted images.
            target: Target images.

        Returns:
            SSIM value (or 1 - SSIM if as_loss=True).

        """
        ssim_val = compute_ssim(pred, target, self.window_size, self.sigma, max_val=self.max_val)
        if self.as_loss:
            return 1 - ssim_val
        return ssim_val


class MSSSIM(nn.Module):
    """MS-SSIM as a PyTorch module (can be used as loss)."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        weights: list[float] | None = None,
        max_val: float = 1.0,
        as_loss: bool = False,
    ) -> None:
        """Initialize MS-SSIM module.

        Args:
            window_size: Size of Gaussian window.
            sigma: Gaussian standard deviation.
            weights: Weights for each scale.
            max_val: Maximum pixel value.
            as_loss: If True, returns 1 - MS-SSIM for use as loss.

        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.weights = weights
        self.max_val = max_val
        self.as_loss = as_loss

    def forward(
        self,
        pred: Float[Tensor, "batch channels height width"],
        target: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """Compute MS-SSIM.

        Args:
            pred: Predicted images.
            target: Target images.

        Returns:
            MS-SSIM value (or 1 - MS-SSIM if as_loss=True).

        """
        ms_ssim_val = compute_ms_ssim(
            pred, target, self.window_size, self.sigma, self.weights, self.max_val
        )
        if self.as_loss:
            return 1 - ms_ssim_val
        return ms_ssim_val


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Computes feature-space loss using pretrained VGG network.
    """

    def __init__(
        self,
        layers: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize perceptual loss.

        Args:
            layers: VGG layers to use.
            weights: Loss weights per layer.

        """
        super().__init__()
        self.layers = layers or ["relu1_2", "relu2_2", "relu3_3"]
        self.weights = weights or {"relu1_2": 0.1, "relu2_2": 0.1, "relu3_3": 0.05}

        # Lazy initialization of VGG
        self._vgg = None

    def _get_vgg(self, device: torch.device) -> nn.Module:
        """Get or create VGG feature extractor.

        Args:
            device: Target device.

        Returns:
            VGG feature extractor module.

        """
        if self._vgg is None:
            try:
                from torchvision.models import VGG16_Weights, vgg16

                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            except ImportError:
                # Fallback for older torchvision
                from torchvision.models import vgg16

                vgg = vgg16(pretrained=True).features

            # Freeze VGG
            for param in vgg.parameters():
                param.requires_grad = False
            vgg.eval()
            self._vgg = vgg

        assert self._vgg is not None, "VGG model failed to initialize"
        return self._vgg.to(device)

    def forward(
        self,
        pred: Float[Tensor, "batch 3 height width"],
        target: Float[Tensor, "batch 3 height width"],
    ) -> Float[Tensor, ""]:
        """Compute perceptual loss.

        Args:
            pred: Predicted images (normalized to [0, 1]).
            target: Target images (normalized to [0, 1]).

        Returns:
            Perceptual loss value.

        """
        vgg = self._get_vgg(pred.device)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        # Extract features at each layer
        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred_norm
        x_target = target_norm

        layer_names = {
            "relu1_2": 4,
            "relu2_2": 9,
            "relu3_3": 16,
            "relu4_3": 23,
            "relu5_3": 30,
        }

        current_idx = 0
        for layer_name in self.layers:
            target_idx = layer_names.get(layer_name, 0)

            # Forward through VGG layers
            for i in range(current_idx, target_idx):
                x_pred = vgg[i](x_pred)
                x_target = vgg[i](x_target)
            current_idx = target_idx

            # Compute loss at this layer
            weight = self.weights.get(layer_name, 1.0)
            loss = loss + weight * F.mse_loss(x_pred, x_target)

        return loss
