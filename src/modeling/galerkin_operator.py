"""Galerkin Neural Operator for resolution-independent PDE solving.

Implements a Petrov-Galerkin projection based neural operator using:
- Multi-scale Fourier features for coordinate encoding
- O(N) linear Galerkin attention for global influence modeling
- LBB stability monitoring for numerical robustness

The Galerkin backend provides an alternative to FNO with mathematically
grounded attention mechanisms from the AlphaGalerkin architecture.
"""

from __future__ import annotations

from typing import cast

import structlog
import torch
from einops import rearrange
from pydantic import Field, model_validator
from torch import Tensor, nn

from src.modeling.attention import GalerkinAttention
from src.modeling.multiscale_fourier import MultiScaleFourierFeatures
from src.templates.config import BaseModuleConfig

logger = structlog.get_logger(__name__)


# Default frequency scales for positional encoding
DEFAULT_FOURIER_SCALES: list[float] = [1.0, 2.0, 4.0, 8.0]


class GalerkinOperatorConfig(BaseModuleConfig):
    """Configuration for Galerkin neural operator.

    All parameters are validated with Pydantic constraints.
    No hardcoded values - everything is configurable.

    Example:
        >>> config = GalerkinOperatorConfig(
        ...     name="poisson_solver",
        ...     width=64,
        ...     n_layers=4,
        ... )

    """

    # Input/Output channels
    in_channels: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Input field channels",
    )
    out_channels: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Output field channels",
    )

    # Model dimensions
    width: int = Field(
        default=64,
        ge=8,
        le=1024,
        description="Hidden dimension width",
    )
    n_layers: int = Field(
        default=4,
        ge=1,
        le=24,
        description="Number of Galerkin blocks",
    )
    n_heads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of attention heads",
    )
    d_ffn: int | None = Field(
        default=None,
        description="FFN hidden dimension (default: 4 * width)",
    )

    # Fourier feature encoding
    fourier_features: int = Field(
        default=64,
        ge=8,
        le=512,
        description="Fourier features per frequency scale",
    )
    fourier_scales: list[float] = Field(
        default_factory=lambda: DEFAULT_FOURIER_SCALES.copy(),
        description="Frequency scales for positional encoding",
    )
    include_coords: bool = Field(
        default=True,
        description="Include raw coordinates in Fourier encoding",
    )
    learnable_frequencies: bool = Field(
        default=True,
        description="Whether Fourier frequencies are learnable",
    )

    # Attention settings
    normalize_features: bool = Field(
        default=True,
        description="Normalize Q/K features before attention",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout rate",
    )

    # LBB stability
    lbb_threshold: float = Field(
        default=1e-6,
        gt=0,
        description="LBB stability threshold for warnings",
    )
    lbb_regularization: float = Field(
        default=0.01,
        ge=0,
        description="LBB regularization strength in loss",
    )

    # Normalization
    activation: str = Field(
        default="gelu",
        description="Activation function (gelu, relu, silu)",
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> GalerkinOperatorConfig:
        """Validate that width is compatible with n_heads."""
        if self.width < self.n_heads:
            raise ValueError(
                f"width ({self.width}) must be >= n_heads ({self.n_heads})"
            )
        if self.width % self.n_heads != 0:
            raise ValueError(
                f"width ({self.width}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


def _get_activation(name: str) -> nn.Module:
    """Get activation module by name.

    Args:
        name: Activation name (gelu, relu, silu).

    Returns:
        Activation module instance.

    Raises:
        ValueError: If activation name is unknown.

    """
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(activations.keys())}"
        )
    return activations[name.lower()]()


class GalerkinOperatorBlock(nn.Module):
    """Single Galerkin operator block with attention and FFN.

    Structure (Pre-LN Transformer style):
        x -> LayerNorm -> GalerkinAttention -> Residual
        x -> LayerNorm -> FFN -> Residual

    The Galerkin attention provides O(N) complexity for global
    influence modeling while the FFN adds local non-linearity.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        normalize_features: bool = True,
        activation: str = "gelu",
    ) -> None:
        """Initialize Galerkin operator block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ffn: FFN hidden dimension (default: 4 * d_model).
            dropout: Dropout rate.
            normalize_features: Whether to normalize Q/K in attention.
            activation: Activation function name.

        """
        super().__init__()

        # Validate dimensions for proper attention head splitting
        if d_model < n_heads:
            raise ValueError(
                f"d_model ({d_model}) must be >= n_heads ({n_heads})"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        d_ffn = d_ffn or 4 * d_model

        # Attention block
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GalerkinAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            normalize_features=normalize_features,
        )

        # FFN block
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        return_lbb: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through Galerkin block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            return_lbb: Whether to return LBB stability constant.

        Returns:
            Output tensor, optionally with LBB constant.

        """
        # Attention with residual
        normed = self.norm1(x)
        if return_lbb:
            attn_out, lbb = self.attention(normed, return_lbb=True)
            x = x + attn_out
        else:
            attn_out = self.attention(normed, return_lbb=False)
            x = x + cast(Tensor, attn_out)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        if return_lbb:
            return x, lbb
        return x


class Galerkin2d(nn.Module):
    """Galerkin Neural Operator for 2D problems.

    Resolution-independent operator using Petrov-Galerkin projection.
    Maps input field a(x) -> output field u(x).

    Key features:
    - O(N) complexity via linear Galerkin attention
    - Resolution independence via Fourier coordinate encoding
    - LBB stability monitoring for numerical robustness

    Example:
        >>> model = Galerkin2d(in_channels=1, out_channels=1, width=64)
        >>> x = torch.randn(4, 1, 16, 16)  # Train resolution
        >>> y = model(x)
        >>> # Zero-shot transfer to higher resolution
        >>> x_hi = torch.randn(4, 1, 64, 64)
        >>> y_hi = model(x_hi)  # Works without retraining!

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ffn: int | None = None,
        fourier_features: int = 64,
        fourier_scales: list[float] | None = None,
        include_coords: bool = True,
        learnable_frequencies: bool = True,
        dropout: float = 0.0,
        normalize_features: bool = True,
        lbb_threshold: float = 1e-6,
        lbb_regularization: float = 0.01,
        activation: str = "gelu",
        config: GalerkinOperatorConfig | None = None,
    ) -> None:
        """Initialize Galerkin2d operator.

        Args:
            in_channels: Input field channels.
            out_channels: Output field channels.
            width: Hidden dimension.
            n_layers: Number of Galerkin blocks.
            n_heads: Number of attention heads.
            d_ffn: FFN hidden dimension (default: 4 * width).
            fourier_features: Features per frequency scale.
            fourier_scales: Frequency scales for positional encoding.
            include_coords: Include raw coordinates in encoding.
            learnable_frequencies: Whether Fourier frequencies are learnable.
            dropout: Dropout rate.
            normalize_features: Normalize Q/K in attention.
            lbb_threshold: LBB stability warning threshold.
            lbb_regularization: LBB regularization strength.
            activation: Activation function name.
            config: Optional config object (overrides other args).

        """
        super().__init__()

        # Use config if provided
        if config is not None:
            in_channels = config.in_channels
            out_channels = config.out_channels
            width = config.width
            n_layers = config.n_layers
            n_heads = config.n_heads
            d_ffn = config.d_ffn
            fourier_features = config.fourier_features
            fourier_scales = config.fourier_scales
            include_coords = config.include_coords
            learnable_frequencies = config.learnable_frequencies
            dropout = config.dropout
            normalize_features = config.normalize_features
            lbb_threshold = config.lbb_threshold
            lbb_regularization = config.lbb_regularization
            activation = config.activation

        if fourier_scales is None:
            fourier_scales = DEFAULT_FOURIER_SCALES.copy()

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.lbb_threshold = lbb_threshold
        self.lbb_regularization = lbb_regularization

        # Fourier coordinate encoder
        self.coord_encoder = MultiScaleFourierFeatures(
            input_dim=2,  # 2D coordinates
            n_features=fourier_features,
            scales=fourier_scales,
            learnable=learnable_frequencies,
            include_input=include_coords,
        )
        fourier_dim = self.coord_encoder.output_dim

        # Input lifting: (in_channels + fourier_dim) -> width
        self.lifting = nn.Linear(in_channels + fourier_dim, width)

        # Galerkin blocks
        self.blocks = nn.ModuleList([
            GalerkinOperatorBlock(
                d_model=width,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                normalize_features=normalize_features,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(width)

        # Output projection: width -> out_channels
        self.projection = nn.Sequential(
            nn.Linear(width, width),
            _get_activation(activation),
            nn.Linear(width, out_channels),
        )

        # Track LBB constants for regularization
        self._last_lbb_constants: list[Tensor] | None = None

        logger.info(
            "galerkin2d_initialized",
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            n_layers=n_layers,
            n_heads=n_heads,
            fourier_dim=fourier_dim,
            total_params=sum(p.numel() for p in self.parameters()),
        )

    def _create_coordinate_encoding(
        self,
        h: int,
        w: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Create Fourier-encoded coordinates for the grid.

        Args:
            h: Grid height.
            w: Grid width.
            batch_size: Batch size.
            device: Device for tensors.

        Returns:
            Fourier-encoded coordinates of shape (batch, h*w, fourier_dim).

        """
        # Generate normalized grid coordinates [0, 1]^2
        grid_x = torch.linspace(0, 1, h, device=device)
        grid_y = torch.linspace(0, 1, w, device=device)
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)  # (h, w, 2)

        # Flatten and encode
        coords_flat = coords.view(-1, 2)  # (h*w, 2)
        encoded = self.coord_encoder(coords_flat)  # (h*w, fourier_dim)

        # Expand for batch
        return encoded.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        x: Tensor,
        coords: Tensor | None = None,
        return_lbb: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Forward pass through Galerkin operator.

        Args:
            x: Input field of shape (batch, in_channels, h, w).
            coords: Optional explicit coordinates. If None, uses uniform grid.
            return_lbb: Whether to return LBB constants for each layer.

        Returns:
            Output field of shape (batch, out_channels, h, w).
            Optionally returns list of LBB constants per layer.

        """
        batch_size, c, h, w = x.shape
        device = x.device

        logger.debug(
            "galerkin2d_forward",
            batch_size=batch_size,
            resolution=(h, w),
            return_lbb=return_lbb,
        )

        # Get coordinate encoding
        if coords is not None:
            # Use provided coordinates
            coords_flat = coords.view(batch_size, -1, 2)  # (batch, h*w, 2)
            coord_features = self.coord_encoder(coords_flat)  # (batch, h*w, fourier)
        else:
            # Generate uniform grid encoding
            coord_features = self._create_coordinate_encoding(h, w, batch_size, device)

        # Flatten spatial dimensions: (batch, c, h, w) -> (batch, h*w, c)
        x_flat = rearrange(x, "b c h w -> b (h w) c")

        # Concatenate input features with coordinate encoding
        # (batch, h*w, c + fourier_dim)
        combined = torch.cat([x_flat, coord_features], dim=-1)

        # Lift to model dimension
        features = self.lifting(combined)  # (batch, h*w, width)

        # Process through Galerkin blocks
        lbb_constants: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            if return_lbb:
                features, lbb = block(features, return_lbb=True)
                lbb_constants.append(lbb)

                # Log LBB warnings
                lbb_min = lbb.min().item()
                if lbb_min < self.lbb_threshold:
                    logger.warning(
                        "lbb_threshold_violation",
                        layer_idx=i,
                        lbb_min=lbb_min,
                        threshold=self.lbb_threshold,
                    )
            else:
                features = cast(Tensor, block(features, return_lbb=False))

        # Store LBB constants for regularization
        if return_lbb:
            self._last_lbb_constants = lbb_constants

        # Final normalization
        features = self.final_norm(features)

        # Project to output channels
        output = self.projection(features)  # (batch, h*w, out_channels)

        # Reshape back to spatial: (batch, out_channels, h, w)
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)

        if return_lbb:
            return output, lbb_constants
        return output

    def get_lbb_regularization(self) -> Tensor:
        """Compute LBB regularization loss for training stability.

        Should be called after a forward pass with return_lbb=True.
        Returns zero if no LBB constants are cached.

        Returns:
            Scalar regularization loss penalizing small LBB constants,
            or zero tensor on the model's device if no constants are cached.

        """
        if self._last_lbb_constants is None:
            # Get device from model parameters to ensure device consistency
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=self._last_lbb_constants[0].device)
        for lbb in self._last_lbb_constants:
            # Penalize LBB constants below threshold * 10
            # Encourages stability margin
            violation = torch.relu(self.lbb_threshold * 10 - lbb)
            total_loss = total_loss + violation.mean()

        return total_loss * self.lbb_regularization

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
