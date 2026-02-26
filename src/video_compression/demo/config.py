"""Configuration for video compression MVP demo.

Provides a single DemoConfig that controls all demo behavior:
- Synthetic video generation parameters
- Codec architecture (small defaults for CPU execution)
- Rate-distortion sweep parameters
- Resolution independence test parameters
- Output and runtime settings

All values are configurable with no hardcoded constants.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.video_compression.data.synthetic import SyntheticPattern


class DemoConfig(BaseModel):
    """Complete configuration for the compression demo.

    Defaults are chosen for fast CPU execution with a small model.
    Override for production-quality evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Video generation ──────────────────────────────────────────────
    num_frames: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Number of frames to generate per pattern",
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
    patterns: list[SyntheticPattern] = Field(
        default=[SyntheticPattern.GRADIENT, SyntheticPattern.WAVES],
        min_length=1,
        description="Patterns to demo",
    )

    # ── Codec architecture (small for CPU demo) ──────────────────────
    latent_channels: int = Field(
        default=64,
        ge=32,
        le=512,
        description="Latent representation channels",
    )
    d_model: int = Field(
        default=128,
        ge=64,
        le=1024,
        description="Model dimension for attention layers",
    )
    n_heads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of attention heads",
    )
    d_ffn: int = Field(
        default=256,
        ge=128,
        le=4096,
        description="Feed-forward network dimension",
    )
    n_layers: int = Field(
        default=2,
        ge=1,
        le=12,
        description="Number of encoder/decoder layers",
    )
    downsample_factor: int = Field(
        default=8,
        ge=4,
        le=64,
        description="Spatial downsampling factor (must be power of 2)",
    )

    # ── Rate-distortion sweep ─────────────────────────────────────────
    lambda_values: list[float] = Field(
        default=[0.005, 0.01, 0.02, 0.05],
        min_length=1,
        description="Lambda values for R-D curve sweep",
    )

    # ── Resolution independence test ──────────────────────────────────
    resolution_sizes: list[tuple[int, int]] = Field(
        default=[(64, 64), (128, 128)],
        min_length=1,
        description="(height, width) pairs for resolution independence test",
    )
    resolution_lambda: float = Field(
        default=0.01,
        gt=0.0,
        description="Lambda value used for resolution independence test",
    )

    # ── Runtime settings ──────────────────────────────────────────────
    device: str = Field(
        default="cpu",
        description="Device for computation ('cpu', 'cuda', 'auto')",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    output_dir: str = Field(
        default="outputs/demo_compression",
        description="Directory for output files (JSON, bitstreams)",
    )
    write_bitstream: bool = Field(
        default=True,
        description="Write .agk bitstream files to output_dir",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )

    # ── Validators ────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_architecture(self) -> DemoConfig:
        """Validate codec architecture constraints."""
        # Downsample factor must be power of 2
        if not math.log2(self.downsample_factor).is_integer():
            raise ValueError(
                f"downsample_factor ({self.downsample_factor}) must be a power of 2"
            )
        # d_model must be divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        # Video dimensions must be divisible by downsample factor
        if self.height % self.downsample_factor != 0:
            raise ValueError(
                f"height ({self.height}) must be divisible by "
                f"downsample_factor ({self.downsample_factor})"
            )
        if self.width % self.downsample_factor != 0:
            raise ValueError(
                f"width ({self.width}) must be divisible by "
                f"downsample_factor ({self.downsample_factor})"
            )
        # Validate resolution test sizes
        for h, w in self.resolution_sizes:
            if h % self.downsample_factor != 0 or w % self.downsample_factor != 0:
                raise ValueError(
                    f"Resolution ({h}, {w}) must be divisible by "
                    f"downsample_factor ({self.downsample_factor})"
                )
        return self

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a summary dict suitable for JSON serialization."""
        return {
            "video": {
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "patterns": [p.value for p in self.patterns],
            },
            "codec": {
                "latent_channels": self.latent_channels,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "d_ffn": self.d_ffn,
                "n_layers": self.n_layers,
                "downsample_factor": self.downsample_factor,
            },
            "rd_sweep": {
                "lambda_values": self.lambda_values,
            },
            "resolution_test": {
                "sizes": self.resolution_sizes,
                "lambda": self.resolution_lambda,
            },
            "runtime": {
                "device": self.device,
                "seed": self.seed,
                "output_dir": self.output_dir,
            },
        }
