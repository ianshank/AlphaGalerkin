"""Data transforms for video compression training.

Provides composable transforms for image and video data
augmentation during compression training.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class RandomCrop(nn.Module):
    """Random crop transform."""

    def __init__(self, size: int) -> None:
        """Initialize random crop.

        Args:
            size: Crop size (square).

        """
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        """Apply random crop.

        Args:
            x: Input tensor (C, H, W) or (T, C, H, W).

        Returns:
            Cropped tensor.

        """
        if x.dim() == 3:
            _, h, w = x.shape
        else:
            _, _, h, w = x.shape

        if h < self.size or w < self.size:
            # Resize if too small
            scale = max(self.size / h, self.size / w) * 1.1
            new_size = (int(h * scale), int(w * scale))
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0) if x.dim() == 3 else x,
                size=new_size,
                mode="bilinear",
                align_corners=False,
            )
            if x.dim() == 4 and x.shape[0] == 1:
                x = x.squeeze(0)
            h, w = new_size

        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)

        return x[..., top : top + self.size, left : left + self.size]


class CenterCrop(nn.Module):
    """Center crop transform."""

    def __init__(self, size: int) -> None:
        """Initialize center crop.

        Args:
            size: Crop size (square).

        """
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        """Apply center crop.

        Args:
            x: Input tensor (C, H, W) or (T, C, H, W).

        Returns:
            Cropped tensor.

        """
        h, w = x.shape[-2:]

        if h < self.size or w < self.size:
            scale = max(self.size / h, self.size / w) * 1.1
            new_size = (int(h * scale), int(w * scale))
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0) if x.dim() == 3 else x,
                size=new_size,
                mode="bilinear",
                align_corners=False,
            )
            if x.dim() == 4 and x.shape[0] == 1:
                x = x.squeeze(0)
            h, w = new_size

        top = (h - self.size) // 2
        left = (w - self.size) // 2

        return x[..., top : top + self.size, left : left + self.size]


class RandomFlip(nn.Module):
    """Random horizontal flip transform."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize random flip.

        Args:
            p: Probability of flipping.

        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply random horizontal flip.

        Args:
            x: Input tensor.

        Returns:
            Possibly flipped tensor.

        """
        if random.random() < self.p:
            return torch.flip(x, dims=[-1])
        return x


class ColorJitter(nn.Module):
    """Color jittering transform for robustness.

    Applies random brightness, contrast, and saturation adjustments.
    """

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
    ) -> None:
        """Initialize color jitter.

        Args:
            brightness: Brightness jitter range.
            contrast: Contrast jitter range.
            saturation: Saturation jitter range.

        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def forward(self, x: Tensor) -> Tensor:
        """Apply color jittering.

        Args:
            x: Input tensor (C, H, W) with C=3.

        Returns:
            Jittered tensor.

        """
        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            x = x * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * factor + mean

        # Clamp to [0, 1]
        return torch.clamp(x, 0.0, 1.0)


class Normalize(nn.Module):
    """Normalize tensor to specific range or distribution."""

    def __init__(
        self,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        """Initialize normalize transform.

        Args:
            mean: Per-channel mean.
            std: Per-channel std.

        """
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply normalization.

        Args:
            x: Input tensor (C, H, W) in [0, 1].

        Returns:
            Normalized tensor.

        """
        return (x - self.mean) / self.std

    def inverse(self, x: Tensor) -> Tensor:
        """Inverse normalization.

        Args:
            x: Normalized tensor.

        Returns:
            Tensor in [0, 1].

        """
        return x * self.std + self.mean


class CompressionTransforms(nn.Module):
    """Composed transforms for compression training.

    Default configuration suitable for learned image/video compression.
    """

    def __init__(
        self,
        patch_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        color_jitter: bool = False,
        training: bool = True,
    ) -> None:
        """Initialize compression transforms.

        Args:
            patch_size: Crop size for training.
            random_crop: Use random cropping (vs center crop).
            random_flip: Use random horizontal flip.
            color_jitter: Apply color jittering.
            training: Training mode (enables augmentation).

        """
        super().__init__()
        self.training_mode = training

        transforms = []

        # Spatial cropping
        if training and random_crop:
            transforms.append(RandomCrop(patch_size))
        else:
            transforms.append(CenterCrop(patch_size))

        # Augmentation (training only)
        if training:
            if random_flip:
                transforms.append(RandomFlip())
            if color_jitter:
                transforms.append(ColorJitter())

        self.transforms = nn.Sequential(*transforms)

    def forward(self, x: Tensor) -> Tensor:
        """Apply transforms.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.

        """
        return self.transforms(x)
