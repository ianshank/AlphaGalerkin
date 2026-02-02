"""Analysis transform (encoder) for video compression.

Uses FNet mixing for O(N log N) frequency analysis combined with
Galerkin attention for resolution-independent feature extraction.

Architecture:
    Input (B, C, H, W) -> Patch Embed -> Encoder Blocks -> Latent (B, M, H', W')

The encoder is fully resolution-independent, accepting any (H, W) divisible by
the downsample factor.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.config import EncoderConfig


class GDN(nn.Module):
    """Generalized Divisive Normalization.

    Implements GDN for learned image compression (Ballé et al., 2016).
    Provides local gain control that adapts to signal statistics.

    Formula:
        y_i = x_i / sqrt(beta_i + sum_j(gamma_ij * x_j^2))
    """

    def __init__(
        self,
        channels: int,
        inverse: bool = False,
        gamma_init: float = 0.1,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize GDN.

        Args:
            channels: Number of input/output channels.
            inverse: If True, compute inverse GDN (IGDN).
            gamma_init: Initial value for gamma parameters.
            epsilon: Small constant for numerical stability.

        """
        super().__init__()
        self.channels = channels
        self.inverse = inverse
        self.epsilon = epsilon

        # Learnable parameters
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.eye(channels) * gamma_init)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Tensor:
        """Apply GDN or IGDN.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Normalized tensor (B, C, H, W).

        """
        # Compute norm: beta + gamma * x^2
        x_sq = x**2
        # Sum over channels: (B, C, H, W) -> (B, C, H, W)
        norm = self.beta.view(1, -1, 1, 1) + torch.einsum("cd,bdhw->bchw", self.gamma, x_sq)
        norm = torch.sqrt(norm + self.epsilon)

        if self.inverse:
            return x * norm
        else:
            return x / norm


class FNetGalerkinBlock(nn.Module):
    """Hybrid block combining FNet mixing and Galerkin attention.

    Uses FFT for efficient frequency mixing and Galerkin attention
    for resolution-independent global context modeling.

    Structure:
        x -> LayerNorm -> FFT Mix (no learnable params) -> Residual
        x -> LayerNorm -> Galerkin Attention -> Residual
        x -> LayerNorm -> FFN -> Residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        fnet_ratio: float = 0.5,
        dropout: float = 0.1,
        normalize_features: bool = True,
    ) -> None:
        """Initialize FNet-Galerkin block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ffn: Feed-forward network dimension.
            fnet_ratio: Weight of FNet path vs Galerkin path.
            dropout: Dropout rate.
            normalize_features: Normalize Q/K in Galerkin attention.

        """
        super().__init__()
        self.d_model = d_model
        self.fnet_ratio = fnet_ratio

        # Layer norms
        self.norm_fft = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # Galerkin attention (O(N) linear attention)
        self.galerkin_attn = GalerkinEncoderAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            normalize_features=normalize_features,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        height: int,
        width: int,
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass through hybrid block.

        Args:
            x: Input tensor (B, H*W, D).
            height: Spatial height for 2D FFT.
            width: Spatial width for 2D FFT.

        Returns:
            Output tensor (B, H*W, D).

        """
        # FNet path: FFT mixing (no learnable parameters)
        x_norm = self.norm_fft(x)
        x_fft = self._fft_mixing(x_norm, height, width)
        x = x + self.dropout(x_fft) * self.fnet_ratio

        # Galerkin attention path
        x_norm = self.norm_attn(x)
        x_attn = self.galerkin_attn(x_norm)
        x = x + self.dropout(x_attn) * (1 - self.fnet_ratio)

        # FFN
        x_norm = self.norm_ffn(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn

        return x

    def _fft_mixing(
        self,
        x: Float[Tensor, "batch n d"],
        height: int,
        width: int,
    ) -> Float[Tensor, "batch n d"]:
        """Apply 2D FFT mixing.

        FNet uses real FFT for O(N log N) token mixing.
        No learnable parameters - pure frequency domain mixing.

        Args:
            x: Input tensor (B, H*W, D).
            height: Spatial height.
            width: Spatial width.

        Returns:
            Mixed tensor (B, H*W, D).

        """
        batch, n, d = x.shape

        # Reshape to 2D spatial
        x_2d = rearrange(x, "b (h w) d -> b d h w", h=height, w=width)

        # 2D real FFT with orthonormal normalization
        x_freq = torch.fft.rfft2(x_2d, dim=(-2, -1), norm="ortho")

        # Take real part for mixing (equivalent to keeping symmetric components)
        x_mixed = torch.fft.irfft2(x_freq, s=(height, width), dim=(-2, -1), norm="ortho")

        # Reshape back
        return rearrange(x_mixed, "b d h w -> b (h w) d")


class GalerkinEncoderAttention(nn.Module):
    """Galerkin attention for encoder.

    Implements linear attention Q(K^T V) with O(N) complexity.
    Uses Monte Carlo normalization (1/n) instead of 1/sqrt(d).

    Critical: Does NOT use softmax - this is what makes it
    resolution-independent and O(N).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_key: int | None = None,
        dropout: float = 0.0,
        normalize_features: bool = True,
    ) -> None:
        """Initialize Galerkin attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_key: Key/value dimension per head.
            dropout: Dropout rate.
            normalize_features: Normalize K and V (Galerkin convention).

        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key or d_model // n_heads
        self.normalize_features = normalize_features

        # Projections
        self.to_q = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_k = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_v = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_out = nn.Linear(n_heads * self.d_key, d_model)

        # Layer norms for K and V (Galerkin convention)
        if normalize_features:
            self.norm_k = nn.LayerNorm(self.d_key)
            self.norm_v = nn.LayerNorm(self.d_key)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Apply Galerkin attention.

        Formula: Output = Q @ (K^T @ V) / n

        This is O(N) because we compute K^T @ V first (d_k x d_k matrix),
        then multiply by Q.

        Args:
            x: Input tensor (B, N, D).

        Returns:
            Output tensor (B, N, D).

        """
        batch, n, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        # Layer norm on K and V (Galerkin convention)
        if self.normalize_features:
            k = self.norm_k(k)
            v = self.norm_v(v)

        # Galerkin: Q @ (K^T @ V) / n
        # Step 1: K^T @ V (Monte Carlo integral approximation)
        # Shape: (B, H, d_k, d_k)
        kv = torch.matmul(k.transpose(-2, -1), v) / n

        # Step 2: Q @ KV
        # Shape: (B, H, N, d_k)
        out = torch.matmul(q, kv)

        # Reshape and project
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.dropout(out)

        return out


class DownsampleBlock(nn.Module):
    """Spatial downsampling with channel expansion.

    Uses strided convolution for downsampling with GDN normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        """Initialize downsample block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size.
            stride: Downsampling stride.

        """
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.gdn = GDN(out_channels)

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch c_out h//s w//s"]:
        """Apply downsampling.

        Args:
            x: Input tensor.

        Returns:
            Downsampled tensor.

        """
        return self.gdn(self.conv(x))


class EncoderBlock(nn.Module):
    """Complete encoder block with downsampling and attention.

    Structure:
        Downsample -> Patch to Sequence -> FNet-Galerkin Blocks -> Sequence to Patch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        n_attention_layers: int = 2,
        fnet_ratio: float = 0.5,
        dropout: float = 0.1,
        downsample_stride: int = 2,
    ) -> None:
        """Initialize encoder block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            d_model: Attention model dimension.
            n_heads: Number of attention heads.
            d_ffn: FFN dimension.
            n_attention_layers: Number of attention layers.
            fnet_ratio: FNet to Galerkin ratio.
            dropout: Dropout rate.
            downsample_stride: Spatial downsampling factor.

        """
        super().__init__()

        # Spatial downsampling
        self.downsample = DownsampleBlock(in_channels, out_channels, stride=downsample_stride)

        # Channel to model dimension projection
        self.channel_proj = nn.Linear(out_channels, d_model)
        self.channel_proj_back = nn.Linear(d_model, out_channels)

        # FNet-Galerkin attention layers
        self.attention_layers = nn.ModuleList(
            [
                FNetGalerkinBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ffn=d_ffn,
                    fnet_ratio=fnet_ratio,
                    dropout=dropout,
                )
                for _ in range(n_attention_layers)
            ]
        )

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch c_out h//s w//s"]:
        """Forward pass through encoder block.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, C', H', W').

        """
        # Spatial downsampling
        x = self.downsample(x)
        batch, channels, height, width = x.shape

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x = rearrange(x, "b c h w -> b (h w) c")

        # Project to model dimension
        x = self.channel_proj(x)

        # Apply attention layers
        for layer in self.attention_layers:
            x = layer(x, height, width)

        # Project back to channels
        x = self.channel_proj_back(x)

        # Reshape to spatial: (B, H*W, C) -> (B, C, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)

        return x


class Encoder(nn.Module):
    """Analysis transform for video compression.

    Resolution-independent encoder using FNet mixing and Galerkin attention.
    Accepts any input size divisible by the downsample factor.

    Architecture:
        Input (B, 3, H, W)
        -> Initial Conv
        -> EncoderBlock (downsample 2x) x N
        -> Output Conv
        -> Latent (B, M, H/ds, W/ds)
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Initialize encoder.

        Args:
            config: Encoder configuration.

        """
        super().__init__()
        self.config = config

        # Calculate intermediate channels based on downsample factor
        n_downsamples = int(math.log2(config.downsample_factor))
        channel_mult = [2**i for i in range(n_downsamples + 1)]
        base_channels = config.latent_channels // channel_mult[-1]

        # Initial convolution (no downsampling)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                base_channels,
                kernel_size=5,
                padding=2,
            ),
            GDN(base_channels),
        )

        # Encoder blocks with progressive downsampling
        self.encoder_blocks = nn.ModuleList()
        in_ch = base_channels

        for i in range(n_downsamples):
            out_ch = base_channels * channel_mult[i + 1]

            # Distribute attention layers across blocks
            n_attn = max(1, config.n_layers // n_downsamples)

            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ffn=config.d_ffn,
                    n_attention_layers=n_attn,
                    fnet_ratio=config.fnet_ratio if config.use_fnet_mixing else 0.0,
                    dropout=config.dropout,
                    downsample_stride=2,
                )
            )
            in_ch = out_ch

        # Final projection to latent channels
        self.output_conv = nn.Conv2d(
            in_ch,
            config.latent_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: Float[Tensor, "batch 3 height width"],
    ) -> Float[Tensor, "batch latent h_out w_out"]:
        """Encode input to latent representation.

        Args:
            x: Input tensor (B, 3, H, W). H and W must be divisible
               by the downsample factor.

        Returns:
            Latent tensor (B, M, H/ds, W/ds).

        Raises:
            ValueError: If input dimensions not divisible by downsample factor.

        """
        _, _, h, w = x.shape
        ds = self.config.downsample_factor

        if h % ds != 0 or w % ds != 0:
            raise ValueError(
                f"Input dimensions ({h}, {w}) must be divisible by downsample factor ({ds})"
            )

        # Initial conv
        x = self.initial_conv(x)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Output projection
        x = self.output_conv(x)

        return x
