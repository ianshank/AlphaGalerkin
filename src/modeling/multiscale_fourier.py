"""Multi-scale Fourier features for overcoming spectral bias.

Neural networks have an inherent spectral bias towards learning
low-frequency functions first. This module provides multi-scale
Fourier feature encodings to help networks learn high-frequency
components efficiently.

Key components:
- MultiScaleFourierFeatures: Multiple frequency bands
- AdaptiveFourierFeatures: Learnable frequency selection
- ProgressiveFourierFeatures: Curriculum-based frequency introduction

Reference:
    Tancik, M., et al. (2020). Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains.

    Wang, S., et al. (2021). On the Eigenvector Bias of Fourier
    Feature Networks: From Regression to Solving Multi-Scale PDEs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog
import torch
from einops import rearrange
from jaxtyping import Float
from pydantic import Field
from torch import Tensor, nn

from src.templates.config import BaseModuleConfig

# Module-level logger
logger = structlog.get_logger(__name__)

# Constants
TWO_PI = 2 * np.pi
DEFAULT_SCALES = [1.0, 2.0, 4.0, 8.0, 16.0]

__all__ = [
    "FourierFeaturesConfig",
    "MultiScaleFourierFeatures",
    "AdaptiveFourierFeatures",
    "ProgressiveFourierFeatures",
    "PositionalEncoding",
    "SpatialPositionalEncoding",
]


class FourierFeaturesConfig(BaseModuleConfig):
    """Configuration for Fourier feature encoding.

    Attributes:
        n_features: Number of Fourier features per scale.
        scales: Frequency scales (sigma values for random Fourier).
        learnable: Whether frequencies are learnable.
        include_input: Whether to include raw coordinates.
        encoding_type: Type of encoding ('positional', 'random', 'gaussian').

    """

    n_features: int = Field(
        default=128,
        ge=1,
        le=4096,
        description="Number of Fourier features per scale",
    )
    scales: list[float] = Field(
        default_factory=lambda: DEFAULT_SCALES.copy(),
        description="Frequency scales (sigma for random Fourier)",
    )
    learnable: bool = Field(
        default=True,
        description="Whether frequencies are learnable",
    )
    include_input: bool = Field(
        default=True,
        description="Concatenate raw input coordinates",
    )
    encoding_type: Literal["positional", "random", "gaussian"] = Field(
        default="random",
        description="Type of Fourier encoding",
    )
    max_frequency: int = Field(
        default=100,
        ge=1,
        description="Maximum frequency for positional encoding",
    )


class MultiScaleFourierFeatures(nn.Module):
    """Multi-scale Fourier feature encoding.

    Projects input coordinates into multiple frequency bands to
    overcome spectral bias. Each scale captures different levels
    of detail:
    - Low scales (σ < 1): Large-scale structure
    - Medium scales (σ ~ 1-10): Intermediate features
    - High scales (σ > 10): Fine details and sharp gradients

    The output is:
        γ(x) = [x, sin(2πB₁x), cos(2πB₁x), ..., sin(2πBₖx), cos(2πBₖx)]

    where Bᵢ are frequency matrices at different scales.
    """

    def __init__(
        self,
        input_dim: int,
        config: FourierFeaturesConfig | None = None,
        n_features: int = 128,
        scales: list[float] | None = None,
        learnable: bool = True,
        include_input: bool = True,
    ) -> None:
        """Initialize multi-scale Fourier features.

        Args:
            input_dim: Dimension of input coordinates.
            config: Configuration object (overrides other args).
            n_features: Features per scale (if config not provided).
            scales: Frequency scales (if config not provided).
            learnable: Whether frequencies are learnable.
            include_input: Whether to include raw coordinates.

        """
        super().__init__()

        if config is not None:
            n_features = config.n_features
            scales = config.scales
            learnable = config.learnable
            include_input = config.include_input

        if scales is None:
            scales = DEFAULT_SCALES.copy()

        self.input_dim = input_dim
        self.n_features = n_features
        self.scales = scales
        self.n_scales = len(scales)
        self.include_input = include_input

        # Initialize frequency matrices for each scale
        # B ~ N(0, σ²) where σ is the scale
        self.frequency_matrices: nn.ParameterList | None
        if learnable:
            self.frequency_matrices = nn.ParameterList(
                [nn.Parameter(torch.randn(input_dim, n_features) * scale) for scale in scales]
            )
        else:
            for i, scale in enumerate(scales):
                B = torch.randn(input_dim, n_features) * scale
                self.register_buffer(f"B_{i}", B)
            self.frequency_matrices = None

    @property
    def output_dim(self) -> int:
        """Total output dimension."""
        dim = 2 * self.n_features * self.n_scales  # cos + sin per scale
        if self.include_input:
            dim += self.input_dim
        return dim

    def forward(
        self,
        x: Float[Tensor, "... d"],
    ) -> Float[Tensor, "... features"]:
        """Encode coordinates with multi-scale Fourier features.

        Args:
            x: Input coordinates of any shape ending in input_dim.

        Returns:
            Fourier feature encoding.

        Raises:
            ValueError: If input dimension doesn't match expected.

        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)

        features = []

        # Optionally include raw coordinates
        if self.include_input:
            features.append(x_flat)

        # Compute features at each scale
        for i in range(self.n_scales):
            if self.frequency_matrices is not None:
                B = self.frequency_matrices[i]
            else:
                B = getattr(self, f"B_{i}")

            # Project: x @ B
            projected = x_flat @ B  # (batch, n_features)

            # Fourier features: sin and cos
            features.append(torch.sin(TWO_PI * projected))
            features.append(torch.cos(TWO_PI * projected))

        # Concatenate all features
        output = torch.cat(features, dim=-1)

        # Restore original shape
        output = output.reshape(*original_shape, self.output_dim)

        return output

    def get_scale_features(
        self,
        x: Float[Tensor, "... d"],
        scale_idx: int,
    ) -> Float[Tensor, "... 2*n_features"]:
        """Get features from a specific scale only.

        Args:
            x: Input coordinates.
            scale_idx: Index of scale to use.

        Returns:
            Features from specified scale.

        """
        if scale_idx < 0 or scale_idx >= self.n_scales:
            raise ValueError(f"Invalid scale index: {scale_idx}")

        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)

        if self.frequency_matrices is not None:
            B = self.frequency_matrices[scale_idx]
        else:
            B = getattr(self, f"B_{scale_idx}")

        projected = x_flat @ B
        features = torch.cat(
            [
                torch.sin(TWO_PI * projected),
                torch.cos(TWO_PI * projected),
            ],
            dim=-1,
        )

        return features.reshape(*original_shape, 2 * self.n_features)


class AdaptiveFourierFeatures(nn.Module):
    """Adaptive Fourier features with learnable frequency selection.

    Uses an attention mechanism to weight different frequency
    components based on the input, allowing the network to
    adaptively focus on relevant frequencies.

    The output is:
        γ(x) = Σᵢ αᵢ(x) * [sin(2πBᵢx), cos(2πBᵢx)]

    where αᵢ(x) are learned attention weights.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 128,
        n_frequency_banks: int = 8,
        scale_range: tuple[float, float] = (0.1, 100.0),
        use_attention: bool = True,
        attention_hidden_dim: int = 64,
    ) -> None:
        """Initialize adaptive Fourier features.

        Args:
            input_dim: Input coordinate dimension.
            n_features: Features per frequency bank.
            n_frequency_banks: Number of frequency banks.
            scale_range: Range of frequency scales (log-uniform spacing).
                First value is minimum scale, second is maximum scale.
            use_attention: Whether to use attention weighting.
            attention_hidden_dim: Hidden dimension for attention network.

        Raises:
            ValueError: If scale_range is invalid.

        """
        super().__init__()

        # Validate scale_range
        if scale_range[0] >= scale_range[1]:
            raise ValueError(f"scale_range[0] must be < scale_range[1], got {scale_range}")
        if scale_range[0] <= 0:
            raise ValueError(f"scale_range values must be positive, got {scale_range}")

        self.input_dim = input_dim
        self.n_features = n_features
        self.n_banks = n_frequency_banks
        self.use_attention = use_attention

        # Initialize frequency banks at log-uniform scales
        log_scales = torch.linspace(
            np.log(scale_range[0]),
            np.log(scale_range[1]),
            n_frequency_banks,
        )
        scales = torch.exp(log_scales)

        # Learnable frequency matrices
        self.frequency_matrices = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, n_features) * scale) for scale in scales]
        )

        # Attention network for weighting frequencies
        if use_attention:
            self.attention_net = nn.Sequential(
                nn.Linear(input_dim, attention_hidden_dim),
                nn.GELU(),
                nn.Linear(attention_hidden_dim, n_frequency_banks),
                nn.Softmax(dim=-1),
            )

        # Output projection (allows learning cross-frequency interactions)
        self.output_proj = nn.Linear(2 * n_features, 2 * n_features)

        logger.debug(
            "initialized_adaptive_fourier",
            input_dim=input_dim,
            n_features=n_features,
            n_banks=n_frequency_banks,
            scale_range=scale_range,
        )

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return 2 * self.n_features + self.input_dim  # Features + raw coords

    def forward(
        self,
        x: Float[Tensor, "... d"],
    ) -> Float[Tensor, "... features"]:
        """Encode with adaptive frequency selection.

        Args:
            x: Input coordinates of any shape ending in input_dim.

        Returns:
            Encoded features.

        Raises:
            ValueError: If input dimension doesn't match expected.

        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        # Handle arbitrary input shapes
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)  # (N, input_dim)

        # Compute features from each frequency bank
        bank_features = []
        for B in self.frequency_matrices:
            projected = torch.matmul(x_flat, B)  # (N, n_features)
            features = torch.cat(
                [
                    torch.sin(TWO_PI * projected),
                    torch.cos(TWO_PI * projected),
                ],
                dim=-1,
            )  # (N, 2*n_features)
            bank_features.append(features)

        # Stack: (N, n_banks, 2*n_features)
        stacked_features: torch.Tensor = torch.stack(bank_features, dim=1)

        if self.use_attention:
            # Compute attention weights
            attention = self.attention_net(x_flat)  # (N, n_banks)
            attention = attention.unsqueeze(-1)  # (N, n_banks, 1)

            # Weighted sum across banks
            combined = (stacked_features * attention).sum(dim=1)  # (N, 2*n_features)
        else:
            # Simple average
            combined = stacked_features.mean(dim=1)

        # Project and combine with raw coordinates
        combined = self.output_proj(combined)
        output = torch.cat([x_flat, combined], dim=-1)  # (N, output_dim)

        # Restore original shape
        output = output.reshape(*original_shape, self.output_dim)

        return output


class ProgressiveFourierFeatures(nn.Module):
    """Progressive Fourier features with curriculum-based frequency introduction.

    Gradually introduces higher frequencies during training to avoid
    the network getting stuck in local minima. This follows the
    intuition that networks should first learn low-frequency structure
    before adding high-frequency details.

    Usage:
        model.fourier.set_progress(0.0)  # Start with low frequencies
        for step in training:
            progress = step / total_steps
            model.fourier.set_progress(progress)  # Gradually add high freq
    """

    # Default scales for progressive curriculum
    DEFAULT_PROGRESSIVE_SCALES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

    def __init__(
        self,
        input_dim: int,
        n_features: int = 128,
        scales: list[float] | None = None,
        learnable: bool = True,
        gate_steepness: float = 10.0,
    ) -> None:
        """Initialize progressive Fourier features.

        Args:
            input_dim: Input coordinate dimension.
            n_features: Features per scale.
            scales: Frequency scales (low to high).
            learnable: Whether frequencies are learnable.
            gate_steepness: Steepness of sigmoid gate activation (higher = sharper).
                Controls how quickly frequencies are activated during curriculum.

        """
        super().__init__()

        if scales is None:
            scales = self.DEFAULT_PROGRESSIVE_SCALES.copy()

        self.input_dim = input_dim
        self.n_features = n_features
        self.scales = scales
        self.n_scales = len(scales)
        self.gate_steepness = gate_steepness

        # Initialize frequency matrices
        self.frequency_matrices: nn.ParameterList | None
        if learnable:
            self.frequency_matrices = nn.ParameterList(
                [nn.Parameter(torch.randn(input_dim, n_features) * scale) for scale in scales]
            )
        else:
            for i, scale in enumerate(scales):
                self.register_buffer(f"B_{i}", torch.randn(input_dim, n_features) * scale)
            self.frequency_matrices = None

        # Progress tracking (0 = only lowest freq, 1 = all frequencies)
        self.register_buffer("_progress", torch.tensor(1.0))

        # Scale-wise gating weights (interpolated based on progress)
        self.register_buffer("_gate_weights", torch.ones(len(scales)))

        logger.debug(
            "initialized_progressive_fourier",
            input_dim=input_dim,
            n_features=n_features,
            n_scales=len(scales),
            gate_steepness=gate_steepness,
        )

    @property
    def output_dim(self) -> int:
        """Output dimension (varies with progress)."""
        return 2 * self.n_features * self.n_scales + self.input_dim

    @property
    def progress(self) -> float:
        """Current training progress."""
        assert isinstance(self._progress, torch.Tensor)
        return self._progress.item()

    def set_progress(self, progress: float) -> None:
        """Set training progress to control frequency activation.

        Args:
            progress: Value in [0, 1] indicating training progress.
                     0 = only lowest frequency, 1 = all frequencies.

        """
        progress = max(0.0, min(1.0, progress))
        assert isinstance(self._progress, torch.Tensor)
        assert isinstance(self._gate_weights, torch.Tensor)
        self._progress.fill_(progress)

        # Get device and dtype from existing buffer to ensure consistency
        device = self._gate_weights.device
        dtype = self._gate_weights.dtype

        # Vectorized computation on the correct device to avoid device mismatch
        # Thresholds: [0/n, 1/n, 2/n, ...] for each scale
        thresholds = torch.arange(self.n_scales, device=device, dtype=dtype) / float(self.n_scales)

        # Compute sigmoid gate values: 1 / (1 + exp(-k * (progress - threshold)))
        delta = progress - thresholds
        gate_values = 1.0 / (1.0 + torch.exp(-self.gate_steepness * delta))

        # Zero out gates for scales that haven't been activated yet
        mask = progress >= thresholds
        gate_values = torch.where(mask, gate_values, torch.zeros_like(gate_values))

        # Update buffer in-place
        self._gate_weights.copy_(gate_values)

        logger.debug(
            "progress_updated",
            progress=progress,
            active_scales=int((self._gate_weights > 0.5).sum().item()),
            total_scales=self.n_scales,
        )

    def forward(
        self,
        x: Float[Tensor, "... d"],
    ) -> Float[Tensor, "... features"]:
        """Encode with progressive frequency activation.

        Args:
            x: Input coordinates.

        Returns:
            Fourier features with gated high-frequency components.

        Raises:
            ValueError: If input dimension doesn't match expected.

        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)

        features = [x_flat]  # Start with raw coordinates

        assert isinstance(self._gate_weights, torch.Tensor)
        for i in range(self.n_scales):
            if self.frequency_matrices is not None:
                B = self.frequency_matrices[i]
            else:
                B = getattr(self, f"B_{i}")

            projected = x_flat @ B

            # Apply gate weight to scale
            gate = self._gate_weights[i]
            sin_features = gate * torch.sin(TWO_PI * projected)
            cos_features = gate * torch.cos(TWO_PI * projected)

            features.append(sin_features)
            features.append(cos_features)

        output = torch.cat(features, dim=-1)
        output = output.reshape(*original_shape, self.output_dim)

        return output


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Classic Transformer-style positional encoding using fixed
    sine and cosine functions at different frequencies.

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        temperature: float = 10000.0,
    ) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Encoding dimension.
            max_len: Maximum sequence length.
            temperature: Base for frequency computation.

        """
        super().__init__()

        self.d_model = d_model

        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(temperature) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self,
        x: Float[Tensor, "batch seq d"],
    ) -> Float[Tensor, "batch seq d"]:
        """Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Input with positional encoding added.

        Raises:
            ValueError: If input dimension doesn't match d_model.

        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {x.size(-1)}")
        assert isinstance(self.pe, torch.Tensor)
        return x + self.pe[:, : x.size(1)]


class SpatialPositionalEncoding(nn.Module):
    """2D spatial positional encoding for grid data.

    Extends positional encoding to 2D grids, suitable for
    image-like or grid-based PDE data.
    """

    def __init__(
        self,
        d_model: int,
        max_size: int = 256,
        temperature: float = 10000.0,
    ) -> None:
        """Initialize spatial positional encoding.

        Args:
            d_model: Encoding dimension (must be divisible by 4).
            max_size: Maximum grid size.
            temperature: Base for frequency computation.

        """
        super().__init__()

        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")

        self.d_model = d_model
        d_half = d_model // 2

        # Compute 1D encodings for x and y
        position = torch.arange(max_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_half, 2) * (-np.log(temperature) / d_half))

        pe_x = torch.zeros(max_size, d_half)
        pe_x[:, 0::2] = torch.sin(position * div_term)
        pe_x[:, 1::2] = torch.cos(position * div_term)

        pe_y = torch.zeros(max_size, d_half)
        pe_y[:, 0::2] = torch.sin(position * div_term)
        pe_y[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe_x", pe_x)
        self.register_buffer("pe_y", pe_y)

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch d h w"]:
        """Add spatial positional encoding.

        Args:
            x: Input tensor (batch, channels, height, width).

        Returns:
            Input with spatial positional encoding.

        Raises:
            ValueError: If input is not 4D tensor.

        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")
        _, _, h, w = x.shape

        # Get encodings for this size
        assert isinstance(self.pe_x, torch.Tensor)
        assert isinstance(self.pe_y, torch.Tensor)
        pe_x = self.pe_x[:w].unsqueeze(0)  # (1, w, d/2)
        pe_y = self.pe_y[:h].unsqueeze(1)  # (h, 1, d/2)

        # Broadcast to (h, w, d)
        pe = torch.cat(
            [
                pe_x.expand(h, -1, -1),
                pe_y.expand(-1, w, -1),
            ],
            dim=-1,
        )

        # Rearrange to (1, d, h, w) and broadcast to batch
        pe = rearrange(pe, "h w d -> 1 d h w")

        # Truncate to match input channels or expand
        if pe.size(1) > x.size(1):
            pe = pe[:, : x.size(1)]
        elif pe.size(1) < x.size(1):
            # Pad with zeros
            padding = torch.zeros(1, x.size(1) - pe.size(1), h, w, device=pe.device, dtype=pe.dtype)
            pe = torch.cat([pe, padding], dim=1)

        return x + pe
