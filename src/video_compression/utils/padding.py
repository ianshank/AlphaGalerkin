"""Frame padding utilities for arbitrary resolution support.

The encoder requires dimensions divisible by the downsample factor.
These utilities handle padding before encoding and cropping after decoding.

Design principles:
- All padding parameters are configurable via Pydantic
- Padding is symmetric by default for better edge handling
- Original dimensions are preserved in metadata for exact reconstruction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class PaddingMode(str, Enum):
    """Padding strategies for frame alignment."""

    CONSTANT = "constant"  # Pad with constant value (default 0)
    REFLECT = "reflect"  # Reflect at boundary
    REPLICATE = "replicate"  # Replicate edge values
    CIRCULAR = "circular"  # Wrap around


class PaddingConfig(BaseModel):
    """Configuration for frame padding."""

    mode: PaddingMode = Field(
        default=PaddingMode.REFLECT,
        description="Padding mode for boundary handling",
    )
    constant_value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Constant value for constant padding mode",
    )
    align_to: int = Field(
        default=16,
        ge=1,
        description="Alignment factor (typically downsample_factor)",
    )
    symmetric: bool = Field(
        default=True,
        description="Use symmetric padding (equal on both sides)",
    )

    model_config = ConfigDict(extra="forbid")


@dataclass
class PaddingInfo:
    """Information about applied padding for reconstruction."""

    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for serialization."""
        return {
            "original_height": self.original_height,
            "original_width": self.original_width,
            "padded_height": self.padded_height,
            "padded_width": self.padded_width,
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> PaddingInfo:
        """Create from dictionary."""
        return cls(**data)


def compute_padding(
    height: int,
    width: int,
    align_to: int,
    symmetric: bool = True,
) -> PaddingInfo:
    """Compute padding needed to align dimensions.

    Args:
        height: Original height.
        width: Original width.
        align_to: Alignment factor.
        symmetric: Use symmetric padding.

    Returns:
        PaddingInfo with computed padding values.

    """
    # Compute padded dimensions
    padded_height = ((height + align_to - 1) // align_to) * align_to
    padded_width = ((width + align_to - 1) // align_to) * align_to

    total_pad_h = padded_height - height
    total_pad_w = padded_width - width

    if symmetric:
        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left
    else:
        # Right/bottom padding only
        pad_top = 0
        pad_bottom = total_pad_h
        pad_left = 0
        pad_right = total_pad_w

    return PaddingInfo(
        original_height=height,
        original_width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def pad_to_multiple(
    x: Float[Tensor, "batch channels height width"],
    align_to: int,
    mode: PaddingMode = PaddingMode.REFLECT,
    constant_value: float = 0.0,
    symmetric: bool = True,
) -> tuple[Float[Tensor, "batch channels h_pad w_pad"], PaddingInfo]:
    """Pad tensor to dimensions divisible by align_to.

    Args:
        x: Input tensor (B, C, H, W).
        align_to: Alignment factor.
        mode: Padding mode.
        constant_value: Value for constant padding.
        symmetric: Use symmetric padding.

    Returns:
        Tuple of (padded tensor, padding info).

    """
    _, _, height, width = x.shape

    # Compute padding
    pad_info = compute_padding(height, width, align_to, symmetric)

    # No padding needed
    if pad_info.padded_height == height and pad_info.padded_width == width:
        logger.debug(f"No padding needed for {height}x{width}")
        return x, pad_info

    # Apply padding
    # F.pad uses (left, right, top, bottom) order
    padding = (
        pad_info.pad_left,
        pad_info.pad_right,
        pad_info.pad_top,
        pad_info.pad_bottom,
    )

    if mode == PaddingMode.CONSTANT:
        x_padded = F.pad(x, padding, mode="constant", value=constant_value)
    else:
        x_padded = F.pad(x, padding, mode=mode.value)

    logger.debug(
        f"Padded {height}x{width} -> {pad_info.padded_height}x{pad_info.padded_width} "
        f"(mode={mode.value})"
    )

    return x_padded, pad_info


def crop_to_original(
    x: Float[Tensor, "batch channels h_pad w_pad"],
    pad_info: PaddingInfo,
) -> Float[Tensor, "batch channels height width"]:
    """Crop tensor back to original dimensions.

    Args:
        x: Padded tensor.
        pad_info: Padding information from pad_to_multiple.

    Returns:
        Cropped tensor with original dimensions.

    """
    _, _, h, w = x.shape

    # Validate dimensions
    if h != pad_info.padded_height or w != pad_info.padded_width:
        raise ValueError(
            f"Tensor dimensions ({h}, {w}) don't match padding info "
            f"({pad_info.padded_height}, {pad_info.padded_width})"
        )

    # No cropping needed
    if (
        pad_info.pad_top == 0
        and pad_info.pad_bottom == 0
        and pad_info.pad_left == 0
        and pad_info.pad_right == 0
    ):
        return x

    # Compute crop bounds
    top = pad_info.pad_top
    bottom = h - pad_info.pad_bottom
    left = pad_info.pad_left
    right = w - pad_info.pad_right

    x_cropped = x[:, :, top:bottom, left:right]

    logger.debug(f"Cropped {h}x{w} -> {pad_info.original_height}x{pad_info.original_width}")

    return x_cropped


class PadToMultiple(nn.Module):
    """PyTorch module for frame padding.

    Wraps padding functions as a stateful module that tracks
    padding info for later cropping.
    """

    def __init__(self, config: PaddingConfig | None = None) -> None:
        """Initialize padding module.

        Args:
            config: Padding configuration. Uses defaults if None.

        """
        super().__init__()
        self.config = config or PaddingConfig()
        self._last_padding_info: PaddingInfo | None = None

    @property
    def last_padding_info(self) -> PaddingInfo | None:
        """Get padding info from last forward pass."""
        return self._last_padding_info

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels h_pad w_pad"]:
        """Pad input tensor.

        Args:
            x: Input tensor.

        Returns:
            Padded tensor.

        """
        x_padded, self._last_padding_info = pad_to_multiple(
            x,
            align_to=self.config.align_to,
            mode=self.config.mode,
            constant_value=self.config.constant_value,
            symmetric=self.config.symmetric,
        )
        return x_padded

    def inverse(
        self,
        x: Float[Tensor, "batch channels h_pad w_pad"],
        pad_info: PaddingInfo | None = None,
    ) -> Float[Tensor, "batch channels height width"]:
        """Crop tensor back to original size.

        Args:
            x: Padded tensor.
            pad_info: Padding info (uses last_padding_info if None).

        Returns:
            Cropped tensor.

        """
        info = pad_info or self._last_padding_info
        if info is None:
            raise ValueError("No padding info available. Call forward() first or provide pad_info.")
        return crop_to_original(x, info)


class DynamicPadding(nn.Module):
    """Adaptive padding that infers alignment from model config.

    Automatically determines required alignment based on encoder
    downsample factor.
    """

    def __init__(
        self,
        downsample_factor: int,
        mode: PaddingMode = PaddingMode.REFLECT,
    ) -> None:
        """Initialize dynamic padding.

        Args:
            downsample_factor: Encoder downsample factor.
            mode: Padding mode.

        """
        super().__init__()
        self.config = PaddingConfig(
            align_to=downsample_factor,
            mode=mode,
        )
        self.padder = PadToMultiple(self.config)

    def pad(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> tuple[Float[Tensor, "batch channels h_pad w_pad"], PaddingInfo]:
        """Pad frame for encoding.

        Args:
            x: Input frame.

        Returns:
            Tuple of (padded frame, padding info).

        """
        x_padded = self.padder(x)
        pad_info = self.padder.last_padding_info
        if pad_info is None:
            raise RuntimeError("Padding info not available after padding")
        return x_padded, pad_info

    def unpad(
        self,
        x: Float[Tensor, "batch channels h_pad w_pad"],
        pad_info: PaddingInfo,
    ) -> Float[Tensor, "batch channels height width"]:
        """Remove padding after decoding.

        Args:
            x: Decoded padded frame.
            pad_info: Padding info from encoding.

        Returns:
            Original-sized frame.

        """
        return self.padder.inverse(x, pad_info)
