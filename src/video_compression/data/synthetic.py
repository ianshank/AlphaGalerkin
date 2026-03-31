"""Synthetic video data generator for demos and testing.

Generates deterministic video sequences using pure PyTorch operations.
No external dependencies (PIL, OpenCV, numpy) required.

Patterns:
- GRADIENT: Smooth color gradients with temporal shift
- MOTION: Moving geometric shapes (circles, rectangles)
- CHECKERBOARD: Alternating pattern with temporal variation
- WAVES: Sinusoidal waves testing frequency content
- NOISE: Controlled random noise for stress testing

Usage:
    from src.video_compression.data.synthetic import (
        SyntheticVideoGenerator,
        SyntheticVideoConfig,
        SyntheticPattern,
        create_test_sequence,
    )

    # Via config
    config = SyntheticVideoConfig(pattern=SyntheticPattern.GRADIENT, num_frames=8)
    generator = SyntheticVideoGenerator(config)
    frames = generator.generate()  # (8, 3, 64, 64) in [0, 1]

    # Via factory
    frames = create_test_sequence(SyntheticPattern.WAVES, num_frames=16)
"""

from __future__ import annotations

import logging
import math
from enum import Enum

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

logger = logging.getLogger(__name__)


class SyntheticPattern(str, Enum):
    """Available synthetic video patterns."""

    GRADIENT = "gradient"
    MOTION = "motion"
    CHECKERBOARD = "checkerboard"
    WAVES = "waves"
    NOISE = "noise"


class SyntheticVideoConfig(BaseModel):
    """Configuration for synthetic video generation.

    All parameters are configurable with sensible defaults.
    No hardcoded values in generation logic.
    """

    model_config = ConfigDict(extra="forbid")

    # Pattern selection
    pattern: SyntheticPattern = Field(
        default=SyntheticPattern.GRADIENT,
        description="Video pattern type to generate",
    )

    # Dimensions
    num_frames: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Number of frames to generate",
    )
    height: int = Field(
        default=64,
        ge=16,
        le=2048,
        description="Frame height in pixels",
    )
    width: int = Field(
        default=64,
        ge=16,
        le=2048,
        description="Frame width in pixels",
    )
    channels: int = Field(
        default=3,
        ge=1,
        le=4,
        description="Number of color channels",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for deterministic generation",
    )

    # Pattern-specific parameters
    motion_speed: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Speed of motion in motion pattern (fraction of frame per step)",
    )
    wave_frequency: float = Field(
        default=4.0,
        gt=0.0,
        le=64.0,
        description="Base frequency for wave pattern (cycles per frame)",
    )
    checkerboard_size: int = Field(
        default=8,
        ge=2,
        le=128,
        description="Size of each checkerboard square in pixels",
    )
    noise_std: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Standard deviation for noise pattern",
    )
    temporal_variation: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Amount of temporal change between frames",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_checkerboard_size(self) -> SyntheticVideoConfig:
        """Ensure checkerboard size doesn't exceed frame dimensions."""
        if self.checkerboard_size > min(self.height, self.width):
            raise ValueError(
                f"checkerboard_size ({self.checkerboard_size}) must not exceed "
                f"min(height, width) = {min(self.height, self.width)}"
            )
        return self


class SyntheticVideoGenerator:
    """Generates synthetic video sequences using pure PyTorch.

    All operations use torch tensors for GPU compatibility and
    deterministic output via seed control.
    """

    def __init__(self, config: SyntheticVideoConfig) -> None:
        """Initialize generator with configuration.

        Args:
            config: Generation configuration.

        """
        self.config = config
        logger.debug(
            "SyntheticVideoGenerator initialized: pattern=%s, frames=%d, resolution=%dx%d, seed=%d",
            config.pattern.value,
            config.num_frames,
            config.height,
            config.width,
            config.seed,
        )

    def generate(self) -> Tensor:
        """Generate complete video sequence.

        Returns:
            Tensor of shape (T, C, H, W) with values in [0, 1].

        """
        frames = []
        for t in range(self.config.num_frames):
            frame = self.generate_frame(t)
            frames.append(frame)

        sequence = torch.stack(frames)
        logger.debug(
            "Generated %s sequence: shape=%s, range=[%.3f, %.3f]",
            self.config.pattern.value,
            list(sequence.shape),
            sequence.min().item(),
            sequence.max().item(),
        )
        return sequence

    def generate_frame(self, frame_idx: int) -> Tensor:
        """Generate a single frame.

        Uses seed + frame_idx for deterministic per-frame generation
        while ensuring temporal variation.

        Args:
            frame_idx: Frame index (0-based).

        Returns:
            Tensor of shape (C, H, W) with values in [0, 1].

        """
        # Deterministic seed per frame
        torch.manual_seed(self.config.seed + frame_idx)

        pattern_generators = {
            SyntheticPattern.GRADIENT: self._generate_gradient,
            SyntheticPattern.MOTION: self._generate_motion,
            SyntheticPattern.CHECKERBOARD: self._generate_checkerboard,
            SyntheticPattern.WAVES: self._generate_waves,
            SyntheticPattern.NOISE: self._generate_noise,
        }

        generator = pattern_generators[self.config.pattern]
        frame = generator(frame_idx)

        # Ensure valid range
        frame = torch.clamp(frame, 0.0, 1.0)
        return frame

    def _generate_gradient(self, frame_idx: int) -> Tensor:
        """Generate smooth color gradient with temporal shift.

        Creates RGB gradients that shift over time, producing smooth
        temporal transitions useful for testing color reproduction.
        Seed influences the initial angle offset for reproducibility.

        Args:
            frame_idx: Frame index for temporal offset.

        Returns:
            Tensor of shape (C, H, W).

        """
        c = self.config
        t_offset = frame_idx * c.temporal_variation
        # Incorporate seed into base angle for seed-dependent variation
        seed_offset = (c.seed % 360) * math.pi / 180.0

        # Spatial coordinates [0, 1]
        y_coord = torch.linspace(0.0, 1.0, c.height).unsqueeze(1).expand(c.height, c.width)
        x_coord = torch.linspace(0.0, 1.0, c.width).unsqueeze(0).expand(c.height, c.width)

        channels = []
        for ch in range(c.channels):
            # Each channel gets a different gradient direction with temporal + seed shift
            angle = (ch / max(c.channels, 1)) * math.pi + t_offset + seed_offset
            gradient = (
                torch.cos(torch.tensor(angle)) * x_coord + torch.sin(torch.tensor(angle)) * y_coord
            )
            # Normalize to [0, 1]
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
            channels.append(gradient)

        return torch.stack(channels)

    def _generate_motion(self, frame_idx: int) -> Tensor:
        """Generate moving geometric shapes.

        Creates circles that move across the frame, useful for testing
        temporal prediction and motion compensation.

        Args:
            frame_idx: Frame index for position calculation.

        Returns:
            Tensor of shape (C, H, W).

        """
        c = self.config
        frame = torch.zeros(c.channels, c.height, c.width)

        # Background gradient
        y_coord = torch.linspace(0.0, 0.3, c.height).unsqueeze(1).expand(c.height, c.width)
        for ch in range(c.channels):
            frame[ch] = y_coord * (ch + 1) / c.channels

        # Spatial coordinate grids
        yy = torch.arange(c.height, dtype=torch.float32).unsqueeze(1).expand(c.height, c.width)
        xx = torch.arange(c.width, dtype=torch.float32).unsqueeze(0).expand(c.height, c.width)

        # Moving circle
        radius = min(c.height, c.width) * 0.15
        center_x = c.width * (0.2 + 0.6 * ((frame_idx * c.motion_speed) % 1.0))
        center_y = c.height * (0.3 + 0.4 * math.sin(frame_idx * c.motion_speed * 2 * math.pi))

        dist = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        circle_mask = (dist < radius).float()

        # Apply circle with color
        for ch in range(c.channels):
            color_val = 0.5 + 0.5 * math.sin(frame_idx * c.temporal_variation + ch * 2.0)
            frame[ch] = frame[ch] * (1 - circle_mask) + color_val * circle_mask

        return frame

    def _generate_checkerboard(self, frame_idx: int) -> Tensor:
        """Generate checkerboard pattern with temporal variation.

        Creates alternating black/white squares that shift over time,
        useful for testing spatial frequency response.

        Args:
            frame_idx: Frame index for temporal shift.

        Returns:
            Tensor of shape (C, H, W).

        """
        c = self.config

        # Pixel coordinates
        yy = torch.arange(c.height, dtype=torch.float32)
        xx = torch.arange(c.width, dtype=torch.float32)

        # Temporal shift (incorporate seed for seed-dependent output)
        shift = frame_idx * c.temporal_variation * c.checkerboard_size + (c.seed % 50)

        # Checkerboard via integer division parity
        y_grid = ((yy + shift) / c.checkerboard_size).long()
        x_grid = ((xx + shift) / c.checkerboard_size).long()

        # XOR pattern for checkerboard
        pattern = ((y_grid.unsqueeze(1) + x_grid.unsqueeze(0)) % 2).float()

        # Add slight per-channel color variation
        channels = []
        for ch in range(c.channels):
            ch_offset = ch * 0.1
            channels.append(torch.clamp(pattern + ch_offset, 0.0, 1.0))

        return torch.stack(channels)

    def _generate_waves(self, frame_idx: int) -> Tensor:
        """Generate sinusoidal wave patterns.

        Creates overlapping sine waves at multiple frequencies,
        useful for testing frequency domain processing (FFT).

        Args:
            frame_idx: Frame index for phase shift.

        Returns:
            Tensor of shape (C, H, W).

        """
        c = self.config

        # Spatial coordinates normalized to [0, 2*pi * frequency]
        y_coord = torch.linspace(0.0, 2.0 * math.pi * c.wave_frequency, c.height)
        x_coord = torch.linspace(0.0, 2.0 * math.pi * c.wave_frequency, c.width)
        yy = y_coord.unsqueeze(1).expand(c.height, c.width)
        xx = x_coord.unsqueeze(0).expand(c.height, c.width)

        # Temporal phase shift (incorporate seed for seed-dependent output)
        phase = frame_idx * c.temporal_variation * 2.0 * math.pi + (c.seed % 100) * 0.1

        channels = []
        for ch in range(c.channels):
            # Different frequency harmonics per channel
            harmonic = ch + 1
            wave = (
                0.5 * torch.sin(xx * harmonic + phase)
                + 0.3 * torch.cos(yy * harmonic + phase * 0.7)
                + 0.2 * torch.sin((xx + yy) * 0.5 + phase * 1.3)
            )
            # Normalize to [0, 1]
            wave = (wave + 1.0) / 2.0
            channels.append(wave)

        return torch.stack(channels)

    def _generate_noise(self, frame_idx: int) -> Tensor:
        """Generate controlled noise pattern.

        Creates noise with configurable standard deviation and
        temporal correlation, useful for stress-testing the codec.

        Args:
            frame_idx: Frame index for temporal correlation.

        Returns:
            Tensor of shape (C, H, W).

        """
        c = self.config

        # Base noise (seed already set per-frame in generate_frame)
        noise = torch.randn(c.channels, c.height, c.width) * c.noise_std

        # Add structured base (prevents all-noise which is incompressible)
        y_coord = torch.linspace(0.3, 0.7, c.height).unsqueeze(1).expand(c.height, c.width)
        base = y_coord.unsqueeze(0).expand(c.channels, c.height, c.width)

        frame = base + noise
        return torch.clamp(frame, 0.0, 1.0)


def create_test_sequence(
    pattern: SyntheticPattern = SyntheticPattern.GRADIENT,
    num_frames: int = 8,
    height: int = 64,
    width: int = 64,
    seed: int = 42,
) -> Tensor:
    """Factory function for quick synthetic video creation.

    Args:
        pattern: Pattern type to generate.
        num_frames: Number of frames.
        height: Frame height.
        width: Frame width.
        seed: Random seed.

    Returns:
        Tensor of shape (T, 3, H, W) with values in [0, 1].

    """
    config = SyntheticVideoConfig(
        pattern=pattern,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )
    generator = SyntheticVideoGenerator(config)
    return generator.generate()
