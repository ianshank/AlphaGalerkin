"""Configuration schemas for AlphaGalerkin video compression.

All configurations use Pydantic for validation with no hardcoded values.
Follows the constraint programming paradigm from CLAUDE.md.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig, TrainableModuleConfig


class QuantizationMode(str, Enum):
    """Quantization strategies for training and inference."""

    NOISE = "noise"  # Add uniform noise during training
    STE = "ste"  # Straight-through estimator
    SOFT = "soft"  # Soft quantization with temperature


class EntropyModelType(str, Enum):
    """Entropy model architectures."""

    FACTORIZED = "factorized"  # Factorized prior (fast, training stable)
    HYPERPRIOR = "hyperprior"  # Scale hyperprior (Ballé et al.)
    AUTOREGRESSIVE = "autoregressive"  # Context model (slow but better)


class RateControlMode(str, Enum):
    """Rate control strategies."""

    CBR = "cbr"  # Constant bitrate
    VBR = "vbr"  # Variable bitrate
    CRF = "crf"  # Constant rate factor (quality-based)


class EncoderConfig(BaseModuleConfig):
    """Configuration for the analysis transform (encoder).

    The encoder uses FNet mixing and Galerkin attention for
    resolution-independent feature extraction.
    """

    # Input/output channels
    in_channels: int = Field(
        default=3,
        ge=1,
        le=16,
        description="Input channels (3 for RGB)",
    )
    latent_channels: int = Field(
        default=192,
        ge=32,
        le=512,
        description="Latent representation channels",
    )

    # Architecture
    n_layers: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of encoder blocks",
    )
    d_model: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Model dimension",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of attention heads",
    )
    d_ffn: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="Feed-forward network dimension",
    )

    # FNet configuration
    use_fnet_mixing: bool = Field(
        default=True,
        description="Use FNet FFT mixing layers",
    )
    fnet_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio of FNet to Galerkin in hybrid blocks",
    )

    # Galerkin attention
    normalize_features: bool = Field(
        default=True,
        description="Normalize Q/K in Galerkin attention",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate",
    )

    # Downsampling
    downsample_factor: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Spatial downsampling factor",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_downsample(self) -> EncoderConfig:
        """Ensure downsample factor is a power of 2."""
        import math

        if not math.log2(self.downsample_factor).is_integer():
            raise ValueError(f"downsample_factor ({self.downsample_factor}) must be a power of 2")
        return self


class DecoderConfig(BaseModuleConfig):
    """Configuration for the synthesis transform (decoder).

    Mirrors encoder architecture for reconstruction.
    """

    # Input/output channels
    latent_channels: int = Field(
        default=192,
        ge=32,
        le=512,
        description="Latent representation channels",
    )
    out_channels: int = Field(
        default=3,
        ge=1,
        le=16,
        description="Output channels (3 for RGB)",
    )

    # Architecture
    n_layers: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of decoder blocks",
    )
    d_model: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Model dimension",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of attention heads",
    )
    d_ffn: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="Feed-forward network dimension",
    )

    # FNet configuration
    use_fnet_mixing: bool = Field(
        default=True,
        description="Use FNet FFT mixing layers",
    )
    fnet_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio of FNet to Galerkin in hybrid blocks",
    )

    # Galerkin attention
    normalize_features: bool = Field(
        default=True,
        description="Normalize Q/K in Galerkin attention",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate",
    )

    # Upsampling
    upsample_factor: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Spatial upsampling factor",
    )


class QuantizerConfig(BaseModuleConfig):
    """Configuration for differentiable quantization."""

    mode: QuantizationMode = Field(
        default=QuantizationMode.NOISE,
        description="Quantization strategy",
    )

    # For soft quantization
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Temperature for soft quantization",
    )
    temperature_annealing: bool = Field(
        default=True,
        description="Anneal temperature during training",
    )
    min_temperature: float = Field(
        default=0.5,
        gt=0.0,
        le=10.0,
        description="Minimum temperature after annealing",
    )

    # Noise injection parameters
    noise_scale: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Scale of uniform noise (relative to quantization bin)",
    )


class EntropyConfig(BaseModuleConfig):
    """Configuration for entropy model and coding."""

    model_type: EntropyModelType = Field(
        default=EntropyModelType.HYPERPRIOR,
        description="Entropy model architecture",
    )

    # Hyperprior configuration
    hyper_channels: int = Field(
        default=192,
        ge=32,
        le=512,
        description="Hyperprior latent channels",
    )
    hyper_layers: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Number of hyperprior encoder/decoder layers",
    )

    # Factorized prior
    num_filters: int = Field(
        default=192,
        ge=32,
        le=512,
        description="Number of filters in factorized prior",
    )

    # Context model (for autoregressive)
    context_size: int = Field(
        default=5,
        ge=1,
        le=11,
        description="Context window size for autoregressive model",
    )

    # Entropy coding
    use_range_coder: bool = Field(
        default=True,
        description="Use range coder (True) or arithmetic coder (False)",
    )
    precision: int = Field(
        default=16,
        ge=8,
        le=32,
        description="Precision bits for entropy coding",
    )


class MCTSRateControlConfig(BaseModuleConfig):
    """Configuration for MCTS-based rate control.

    Uses MuZero-style learned models for GOP-level bit allocation.
    """

    # MCTS parameters
    num_simulations: int = Field(
        default=50,
        ge=1,
        le=500,
        description="MCTS simulations per decision",
    )
    c_puct: float = Field(
        default=1.25,
        gt=0.0,
        le=10.0,
        description="Exploration constant",
    )
    dirichlet_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Dirichlet noise alpha",
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Dirichlet noise weight",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Action selection temperature",
    )
    discount: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="Future reward discount factor",
    )

    # Value network
    value_support_size: int = Field(
        default=51,
        ge=11,
        le=201,
        description="Categorical value distribution size",
    )

    # GOP structure
    gop_size: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Group of Pictures size",
    )
    i_frame_interval: int = Field(
        default=16,
        ge=1,
        le=256,
        description="I-frame interval",
    )
    use_b_frames: bool = Field(
        default=True,
        description="Use B-frames for temporal compression",
    )
    b_frame_count: int = Field(
        default=3,
        ge=0,
        le=7,
        description="Number of B-frames between references",
    )

    # Rate control
    rate_control_mode: RateControlMode = Field(
        default=RateControlMode.VBR,
        description="Rate control strategy",
    )
    target_bitrate_kbps: float = Field(
        default=2000.0,
        gt=0.0,
        le=100000.0,
        description="Target bitrate in kbps for CBR mode",
    )
    crf_value: int = Field(
        default=23,
        ge=0,
        le=51,
        description="CRF value (0-51, lower = higher quality)",
    )
    bitrate_tolerance: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Allowed bitrate deviation (fraction)",
    )

    # QP range
    qp_min: int = Field(
        default=0,
        ge=0,
        le=51,
        description="Minimum QP value",
    )
    qp_max: int = Field(
        default=51,
        ge=0,
        le=51,
        description="Maximum QP value",
    )

    # Frame rate
    fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Video frame rate for bitrate calculations",
    )

    # Frame-type QP offsets
    qp_offset_i: int = Field(
        default=-2,
        ge=-10,
        le=10,
        description="QP offset for I-frames (negative = higher quality)",
    )
    qp_offset_p: int = Field(
        default=0,
        ge=-10,
        le=10,
        description="QP offset for P-frames",
    )
    qp_offset_b: int = Field(
        default=2,
        ge=-10,
        le=10,
        description="QP offset for B-frames (positive = lower quality)",
    )

    # Frame-type bit allocation weights
    weight_i: float = Field(
        default=2.0,
        gt=0.0,
        le=10.0,
        description="Bit allocation weight for I-frames",
    )
    weight_p: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Bit allocation weight for P-frames",
    )
    weight_b: float = Field(
        default=0.5,
        gt=0.0,
        le=10.0,
        description="Bit allocation weight for B-frames",
    )

    # MCTS network state dimension
    state_dim: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Hidden state dimension for MCTS networks",
    )

    # Heuristic model parameters
    bit_estimation_slope: float = Field(
        default=0.1,
        gt=0.0,
        le=10.0,
        description="Slope for bit estimation model",
    )
    quality_estimation_intercept: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Intercept for quality estimation model",
    )
    quality_estimation_slope: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Slope for quality estimation model",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_qp_range(self) -> MCTSRateControlConfig:
        """Ensure QP min <= max."""
        if self.qp_min > self.qp_max:
            raise ValueError(f"qp_min ({self.qp_min}) must be <= qp_max ({self.qp_max})")
        return self


class TrainingConfig(TrainableModuleConfig):
    """Configuration for codec training."""

    # Rate-distortion tradeoff
    lambda_rd: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Rate-distortion lambda (higher = more compression)",
    )
    lambda_values: list[float] = Field(
        default_factory=lambda: [0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045, 0.09, 0.18],
        min_length=1,
        description="Lambda values for R-D curve training",
    )

    # Distortion metric
    distortion_metric: Literal["mse", "ms_ssim", "mixed"] = Field(
        default="mixed",
        description="Distortion metric for training",
    )
    ms_ssim_weight: float = Field(
        default=0.84,
        ge=0.0,
        le=1.0,
        description="MS-SSIM weight when using mixed distortion",
    )

    # Perceptual loss
    use_perceptual_loss: bool = Field(
        default=True,
        description="Add VGG perceptual loss",
    )
    perceptual_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for perceptual loss",
    )

    # Training data
    patch_size: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Training patch size",
    )
    variable_resolution: bool = Field(
        default=True,
        description="Train on variable resolutions",
    )
    min_resolution: int = Field(
        default=128,
        ge=64,
        le=512,
        description="Minimum training resolution",
    )
    max_resolution: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Maximum training resolution",
    )

    # Quality constraints
    vmaf_min: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable VMAF",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_resolution_range(self) -> TrainingConfig:
        """Ensure resolution range is valid."""
        if self.min_resolution > self.max_resolution:
            raise ValueError(
                f"min_resolution ({self.min_resolution}) must be <= "
                f"max_resolution ({self.max_resolution})"
            )
        return self


class CodecConfig(BaseModuleConfig):
    """Complete codec configuration combining all components."""

    encoder: EncoderConfig = Field(
        default_factory=lambda: EncoderConfig(name="encoder"),
        description="Encoder configuration",
    )
    decoder: DecoderConfig = Field(
        default_factory=lambda: DecoderConfig(name="decoder"),
        description="Decoder configuration",
    )
    quantizer: QuantizerConfig = Field(
        default_factory=lambda: QuantizerConfig(name="quantizer"),
        description="Quantizer configuration",
    )
    entropy: EntropyConfig = Field(
        default_factory=lambda: EntropyConfig(name="entropy"),
        description="Entropy model configuration",
    )
    mcts: MCTSRateControlConfig = Field(
        default_factory=lambda: MCTSRateControlConfig(name="mcts"),
        description="MCTS rate control configuration",
    )
    training: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(name="training"),
        description="Training configuration",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_channel_consistency(self) -> CodecConfig:
        """Ensure encoder/decoder channel compatibility."""
        if self.encoder.latent_channels != self.decoder.latent_channels:
            raise ValueError(
                f"Encoder latent_channels ({self.encoder.latent_channels}) must match "
                f"decoder latent_channels ({self.decoder.latent_channels})"
            )
        if self.encoder.in_channels != self.decoder.out_channels:
            raise ValueError(
                f"Encoder in_channels ({self.encoder.in_channels}) must match "
                f"decoder out_channels ({self.decoder.out_channels})"
            )
        return self
