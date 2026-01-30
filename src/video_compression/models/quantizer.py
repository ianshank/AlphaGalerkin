"""Differentiable quantization for learned compression.

Provides three quantization strategies:
1. NoiseQuantizer: Add uniform noise during training (Ballé et al.)
2. STEQuantizer: Straight-through estimator
3. SoftQuantizer: Temperature-based soft quantization

All quantizers:
- Round to nearest integer during inference
- Provide differentiable approximation during training
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.config import QuantizerConfig, QuantizationMode


class Quantizer(nn.Module, ABC):
    """Base class for differentiable quantization."""

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, "..."],
        training: bool | None = None,
    ) -> Float[Tensor, "..."]:
        """Quantize input.

        Args:
            x: Input tensor to quantize.
            training: Override training mode (uses module training mode if None).

        Returns:
            Quantized tensor.
        """
        pass

    def encode(
        self,
        x: Float[Tensor, "..."],
    ) -> Tensor:
        """Quantize for encoding (always uses hard quantization).

        Args:
            x: Input tensor.

        Returns:
            Integer-valued quantized tensor.
        """
        return torch.round(x).to(torch.int32)

    def decode(
        self,
        x: Tensor,
    ) -> Float[Tensor, "..."]:
        """Dequantize (convert back to float).

        Args:
            x: Quantized tensor.

        Returns:
            Float tensor.
        """
        return x.float()


class NoiseQuantizer(Quantizer):
    """Uniform noise quantization for training.

    During training, adds uniform noise U(-0.5, 0.5) scaled by noise_scale.
    During inference, rounds to nearest integer.

    This provides an unbiased gradient estimator for the quantization operation.
    Reference: Ballé et al., "End-to-end optimized image compression" (2017)
    """

    def __init__(self, noise_scale: float = 0.5) -> None:
        """Initialize noise quantizer.

        Args:
            noise_scale: Scale of uniform noise (relative to bin width).
        """
        super().__init__()
        self.noise_scale = noise_scale

    def forward(
        self,
        x: Float[Tensor, "..."],
        training: bool | None = None,
    ) -> Float[Tensor, "..."]:
        """Quantize with noise or rounding.

        Args:
            x: Input tensor.
            training: Override training mode.

        Returns:
            Quantized tensor.
        """
        is_training = training if training is not None else self.training

        if is_training:
            # Add uniform noise during training
            noise = torch.empty_like(x).uniform_(-0.5, 0.5) * self.noise_scale
            return x + noise
        else:
            # Round during inference
            return torch.round(x)


class STEQuantizer(Quantizer):
    """Straight-through estimator quantization.

    Forward pass: Round to nearest integer
    Backward pass: Identity (gradient passes through unchanged)

    Simple and effective, but can have bias issues for very small gradients.
    Reference: Bengio et al., "Estimating or propagating gradients through
    stochastic neurons for conditional computation" (2013)
    """

    def forward(
        self,
        x: Float[Tensor, "..."],
        training: bool | None = None,
    ) -> Float[Tensor, "..."]:
        """Quantize with straight-through estimator.

        Args:
            x: Input tensor.
            training: Override training mode (not used, STE always applies).

        Returns:
            Quantized tensor with STE gradient.
        """
        return x + (torch.round(x) - x).detach()


class SoftQuantizer(Quantizer):
    """Soft quantization with temperature annealing.

    Uses a soft argmax over quantization bins that approximates
    hard quantization as temperature decreases.

    Formula:
        q(x) = sum_k (k * softmax(-|x - k| / tau))

    where k are integer bin centers and tau is temperature.
    """

    def __init__(
        self,
        num_bins: int = 256,
        temperature: float = 1.0,
        min_temperature: float = 0.5,
    ) -> None:
        """Initialize soft quantizer.

        Args:
            num_bins: Number of quantization bins.
            temperature: Initial temperature for softmax.
            min_temperature: Minimum temperature after annealing.
        """
        super().__init__()
        self.num_bins = num_bins
        self.min_temperature = min_temperature

        # Learnable or fixed temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Register bin centers
        bins = torch.arange(-(num_bins // 2), num_bins // 2 + 1).float()
        self.register_buffer("bins", bins)

    def forward(
        self,
        x: Float[Tensor, "..."],
        training: bool | None = None,
    ) -> Float[Tensor, "..."]:
        """Quantize with soft bins.

        Args:
            x: Input tensor.
            training: Override training mode.

        Returns:
            Soft-quantized tensor.
        """
        is_training = training if training is not None else self.training

        if not is_training:
            # Hard quantization during inference
            return torch.round(x)

        # Clamp temperature
        tau = torch.clamp(self.temperature, min=self.min_temperature)

        # Compute distances to bin centers: (*, num_bins)
        x_expanded = x.unsqueeze(-1)
        distances = -torch.abs(x_expanded - self.bins) / tau

        # Soft assignment: (*, num_bins)
        soft_assignment = torch.softmax(distances, dim=-1)

        # Soft quantized value: sum over bins
        return (soft_assignment * self.bins).sum(dim=-1)

    def anneal_temperature(self, factor: float = 0.99) -> float:
        """Anneal temperature by a factor.

        Args:
            factor: Multiplicative factor for annealing.

        Returns:
            New temperature value.
        """
        with torch.no_grad():
            self.temperature.mul_(factor)
            self.temperature.clamp_(min=self.min_temperature)
        return self.temperature.item()


def create_quantizer(config: QuantizerConfig) -> Quantizer:
    """Factory function to create quantizer from config.

    Args:
        config: Quantizer configuration.

    Returns:
        Configured quantizer instance.
    """
    match config.mode:
        case QuantizationMode.NOISE:
            return NoiseQuantizer(noise_scale=config.noise_scale)
        case QuantizationMode.STE:
            return STEQuantizer()
        case QuantizationMode.SOFT:
            return SoftQuantizer(
                temperature=config.temperature,
                min_temperature=config.min_temperature,
            )
        case _:
            raise ValueError(f"Unknown quantization mode: {config.mode}")


class LearnedQuantizer(Quantizer):
    """Quantizer with learned scale and offset per channel.

    Learns optimal quantization parameters for each channel to minimize
    rate-distortion cost.

    Parameters:
        scale: Per-channel scaling (default 1.0)
        offset: Per-channel offset (default 0.0)

    Quantization:
        q(x) = round((x - offset) / scale) * scale + offset
    """

    def __init__(
        self,
        num_channels: int,
        init_scale: float = 1.0,
        base_quantizer: Quantizer | None = None,
    ) -> None:
        """Initialize learned quantizer.

        Args:
            num_channels: Number of channels to quantize.
            init_scale: Initial scale value.
            base_quantizer: Base quantizer for the round operation.
        """
        super().__init__()
        self.base_quantizer = base_quantizer or STEQuantizer()

        # Learnable parameters (in log space for scale)
        self.log_scale = nn.Parameter(torch.full((num_channels,), math.log(init_scale)))
        self.offset = nn.Parameter(torch.zeros(num_channels))

    @property
    def scale(self) -> Tensor:
        """Get scale from log_scale."""
        return torch.exp(self.log_scale)

    def forward(
        self,
        x: Float[Tensor, "batch channels ..."],
        training: bool | None = None,
    ) -> Float[Tensor, "batch channels ..."]:
        """Quantize with learned scale and offset.

        Args:
            x: Input tensor (B, C, ...).
            training: Override training mode.

        Returns:
            Quantized tensor.
        """
        # Get parameters with proper shape
        scale = self.scale.view(1, -1, *([1] * (x.ndim - 2)))
        offset = self.offset.view(1, -1, *([1] * (x.ndim - 2)))

        # Normalize, quantize, denormalize
        x_norm = (x - offset) / (scale + 1e-8)
        x_quant = self.base_quantizer(x_norm, training)
        return x_quant * scale + offset


# Import math for LearnedQuantizer
import math
