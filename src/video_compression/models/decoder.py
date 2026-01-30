"""Synthesis transform (decoder) for video compression.

Mirrors the encoder architecture with upsampling instead of downsampling.
Uses FNet mixing and Galerkin attention for resolution-independent reconstruction.

Architecture:
    Latent (B, M, H', W') -> Decoder Blocks -> Upsample -> Output (B, C, H, W)
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.config import DecoderConfig
from src.video_compression.models.encoder import (
    GDN,
    FNetGalerkinBlock,
)


class UpsampleBlock(nn.Module):
    """Spatial upsampling with channel reduction.

    Uses transposed convolution for upsampling with inverse GDN (IGDN).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        """Initialize upsample block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size.
            stride: Upsampling stride.
        """
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride - 1

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.igdn = GDN(out_channels, inverse=True)

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch c_out h*s w*s"]:
        """Apply upsampling.

        Args:
            x: Input tensor.

        Returns:
            Upsampled tensor.
        """
        return self.igdn(self.conv(x))


class DecoderBlock(nn.Module):
    """Complete decoder block with attention and upsampling.

    Structure:
        Patch to Sequence -> FNet-Galerkin Blocks -> Sequence to Patch -> Upsample
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
        upsample_stride: int = 2,
    ) -> None:
        """Initialize decoder block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            d_model: Attention model dimension.
            n_heads: Number of attention heads.
            d_ffn: FFN dimension.
            n_attention_layers: Number of attention layers.
            fnet_ratio: FNet to Galerkin ratio.
            dropout: Dropout rate.
            upsample_stride: Spatial upsampling factor.
        """
        super().__init__()

        # Channel to model dimension projection
        self.channel_proj = nn.Linear(in_channels, d_model)
        self.channel_proj_back = nn.Linear(d_model, in_channels)

        # FNet-Galerkin attention layers
        self.attention_layers = nn.ModuleList([
            FNetGalerkinBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ffn=d_ffn,
                fnet_ratio=fnet_ratio,
                dropout=dropout,
            )
            for _ in range(n_attention_layers)
        ])

        # Spatial upsampling
        self.upsample = UpsampleBlock(
            in_channels, out_channels, stride=upsample_stride
        )

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch c_out h*s w*s"]:
        """Forward pass through decoder block.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, C', H'*s, W'*s).
        """
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

        # Spatial upsampling
        x = self.upsample(x)

        return x


class Decoder(nn.Module):
    """Synthesis transform for video compression.

    Resolution-independent decoder using FNet mixing and Galerkin attention.
    Mirrors the encoder architecture for reconstruction.

    Architecture:
        Latent (B, M, H', W')
        -> Input Conv
        -> DecoderBlock (upsample 2x) x N
        -> Output Conv
        -> Output (B, 3, H, W)
    """

    def __init__(self, config: DecoderConfig) -> None:
        """Initialize decoder.

        Args:
            config: Decoder configuration.
        """
        super().__init__()
        self.config = config

        # Calculate intermediate channels based on upsample factor
        n_upsamples = int(math.log2(config.upsample_factor))
        channel_mult = [2 ** i for i in range(n_upsamples, -1, -1)]
        base_channels = config.latent_channels // channel_mult[0]

        # Input convolution (no upsampling)
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                config.latent_channels,
                base_channels * channel_mult[0],
                kernel_size=3,
                padding=1,
            ),
            GDN(base_channels * channel_mult[0], inverse=True),
        )

        # Decoder blocks with progressive upsampling
        self.decoder_blocks = nn.ModuleList()
        in_ch = base_channels * channel_mult[0]

        for i in range(n_upsamples):
            out_ch = base_channels * channel_mult[i + 1] if i + 1 < len(channel_mult) else base_channels

            # Distribute attention layers across blocks
            n_attn = max(1, config.n_layers // n_upsamples)

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ffn=config.d_ffn,
                    n_attention_layers=n_attn,
                    fnet_ratio=config.fnet_ratio if config.use_fnet_mixing else 0.0,
                    dropout=config.dropout,
                    upsample_stride=2,
                )
            )
            in_ch = out_ch

        # Final convolution to output channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                config.out_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        x: Float[Tensor, "batch latent h_lat w_lat"],
    ) -> Float[Tensor, "batch 3 height width"]:
        """Decode latent to reconstructed image.

        Args:
            x: Latent tensor (B, M, H', W').

        Returns:
            Reconstructed tensor (B, 3, H, W) in [0, 1] range.
        """
        # Input conv
        x = self.input_conv(x)

        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Output projection
        x = self.output_conv(x)

        return x


class TemporalDecoder(nn.Module):
    """Decoder with temporal context for video frames.

    Extends the base decoder with cross-attention to reference frames.
    Used for P-frames and B-frames in video compression.
    """

    def __init__(
        self,
        config: DecoderConfig,
        temporal_layers: int = 2,
    ) -> None:
        """Initialize temporal decoder.

        Args:
            config: Decoder configuration.
            temporal_layers: Number of temporal cross-attention layers.
        """
        super().__init__()
        self.base_decoder = Decoder(config)

        # Temporal cross-attention
        self.temporal_attention = nn.ModuleList([
            TemporalCrossAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
            for _ in range(temporal_layers)
        ])

        self.norm = nn.LayerNorm(config.latent_channels)

    def forward(
        self,
        x: Float[Tensor, "batch latent h w"],
        reference: Float[Tensor, "batch latent h w"] | None = None,
    ) -> Float[Tensor, "batch 3 height width"]:
        """Decode with optional temporal reference.

        Args:
            x: Current frame latent (B, M, H', W').
            reference: Reference frame latent for temporal prediction.

        Returns:
            Reconstructed tensor (B, 3, H, W).
        """
        if reference is not None:
            batch, channels, h, w = x.shape

            # Reshape for attention
            x_seq = rearrange(x, "b c h w -> b (h w) c")
            ref_seq = rearrange(reference, "b c h w -> b (h w) c")

            # Apply temporal attention
            for attn in self.temporal_attention:
                x_seq = attn(x_seq, ref_seq, h, w)

            # Reshape back
            x = rearrange(x_seq, "b (h w) c -> b c h w", h=h, w=w)
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return self.base_decoder(x)


class TemporalCrossAttention(nn.Module):
    """Cross-attention for temporal reference frames.

    Uses Galerkin-style linear attention for O(N) complexity
    between current frame and reference frame.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize temporal cross-attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads

        # Query from current frame
        self.to_q = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        # Key and value from reference frame
        self.to_k = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_v = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_out = nn.Linear(n_heads * self.d_key, d_model)

        self.norm_k = nn.LayerNorm(self.d_key)
        self.norm_v = nn.LayerNorm(self.d_key)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        reference: Float[Tensor, "batch n d"],
        height: int,
        width: int,
    ) -> Float[Tensor, "batch n d"]:
        """Apply temporal cross-attention.

        Args:
            x: Current frame features (B, N, D).
            reference: Reference frame features (B, N, D).
            height: Spatial height.
            width: Spatial width.

        Returns:
            Updated features (B, N, D).
        """
        batch, n, _ = x.shape

        # Query from current, K/V from reference (Galerkin-style)
        q = self.to_q(x)
        k = self.to_k(reference)
        v = self.to_v(reference)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        # Layer norm on K and V
        k = self.norm_k(k)
        v = self.norm_v(v)

        # Galerkin-style: Q @ (K^T @ V) / n
        kv = torch.matmul(k.transpose(-2, -1), v) / n
        out = torch.matmul(q, kv)

        # Reshape and project
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.dropout(out)

        # Residual connection
        return self.norm(x + out)
