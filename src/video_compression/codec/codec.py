"""Complete video codec implementation.

Combines encoder, decoder, entropy model, and rate control
into a unified encoding/decoding pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.config import CodecConfig
from src.video_compression.models.encoder import Encoder
from src.video_compression.models.decoder import Decoder, TemporalDecoder
from src.video_compression.models.hyperprior import (
    HyperpriorEntropyModel,
    create_entropy_model,
)
from src.video_compression.models.quantizer import create_quantizer
from src.video_compression.codec.entropy_coder import EntropyCoder, EncodedBitstream
from src.video_compression.codec.gop_manager import GOPManager, FrameInfo, FrameType


class CodecOutput(NamedTuple):
    """Output from encoding a frame."""

    bitstream: EncodedBitstream
    latent: Tensor
    reconstructed: Tensor
    rate: float  # Bits
    distortion: float  # MSE


@dataclass
class VideoHeader:
    """Header information for encoded video."""

    width: int
    height: int
    num_frames: int
    gop_size: int
    frame_rate: float = 30.0


class VideoCodec(nn.Module):
    """Neural video codec with FNet mixing and Galerkin attention.

    Architecture:
        Input Frame -> Encoder -> Quantize -> Entropy Code -> Bitstream
        Bitstream -> Entropy Decode -> Decoder -> Reconstructed Frame

    Features:
    - Resolution-independent via Galerkin attention
    - O(N log N) mixing via FNet
    - Learned hyperprior entropy model
    - MCTS-based rate control (optional)
    """

    def __init__(self, config: CodecConfig) -> None:
        """Initialize video codec.

        Args:
            config: Complete codec configuration.
        """
        super().__init__()
        self.config = config

        # Build encoder and decoder
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)

        # Temporal decoder for P/B frames
        self.temporal_decoder = TemporalDecoder(config.decoder)

        # Quantizer
        self.quantizer = create_quantizer(config.quantizer)

        # Entropy model
        self.entropy_model = create_entropy_model(config.entropy)

        # Entropy coder (for inference)
        self.entropy_coder = EntropyCoder(precision=config.entropy.precision)

        # GOP manager
        self.gop_manager = GOPManager(
            gop_size=config.mcts.gop_size,
            i_frame_interval=config.mcts.i_frame_interval,
            use_b_frames=config.mcts.use_b_frames,
            b_frame_count=config.mcts.b_frame_count,
        )

    def forward(
        self,
        x: Float[Tensor, "batch 3 height width"],
        reference: Float[Tensor, "batch 3 height width"] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for training.

        Args:
            x: Input frame (B, 3, H, W) in [0, 1].
            reference: Optional reference frame for temporal coding.

        Returns:
            Tuple of (reconstructed, rate, distortion).
        """
        # Encode
        y = self.encoder(x)

        # Entropy model (quantization + rate estimation)
        entropy_output = self.entropy_model(y, training=self.training)

        # Decode
        if reference is not None:
            ref_y = self.encoder(reference)
            ref_y_hat = self.quantizer(ref_y, training=self.training)
            x_hat = self.temporal_decoder(entropy_output.y_hat, ref_y_hat)
        else:
            x_hat = self.decoder(entropy_output.y_hat)

        # Compute distortion
        distortion = torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])

        return x_hat, entropy_output.rate, distortion

    def encode_frame(
        self,
        x: Float[Tensor, "1 3 height width"],
        frame_info: FrameInfo,
    ) -> CodecOutput:
        """Encode a single frame.

        Args:
            x: Input frame (1, 3, H, W) in [0, 1].
            frame_info: Frame metadata.

        Returns:
            CodecOutput with bitstream and metrics.
        """
        with torch.no_grad():
            # Encode to latent
            y = self.encoder(x)

            # Quantize
            y_hat = torch.round(y)

            # Get entropy model output for rate estimation
            entropy_output = self.entropy_model(y, training=False)

            # Compress with entropy coder
            compressed = self.entropy_model.compress(y)
            bitstream = self.entropy_coder.encode(
                compressed["y_symbols"],
                compressed["scales"],
            )

            # Decode for reconstruction
            if frame_info.frame_type == FrameType.I:
                x_hat = self.decoder(y_hat)
            else:
                # Get reference latent
                ref_latent = self.gop_manager.reference_buffer.get_latent(
                    frame_info.forward_ref
                )
                x_hat = self.temporal_decoder(y_hat, ref_latent)

            # Store reference if needed
            if frame_info.is_reference:
                self.gop_manager.reference_buffer.add(
                    frame_info.index, x_hat, y_hat
                )

            # Compute metrics
            rate = len(bitstream.data) * 8  # Bits
            distortion = torch.mean((x - x_hat) ** 2).item()

            return CodecOutput(
                bitstream=bitstream,
                latent=y_hat,
                reconstructed=x_hat,
                rate=rate,
                distortion=distortion,
            )

    def decode_frame(
        self,
        bitstream: EncodedBitstream,
        frame_info: FrameInfo,
        scales: Tensor,
    ) -> Float[Tensor, "1 3 height width"]:
        """Decode a single frame.

        Args:
            bitstream: Encoded bitstream.
            frame_info: Frame metadata.
            scales: Scale parameters for entropy decoding.

        Returns:
            Decoded frame (1, 3, H, W) in [0, 1].
        """
        with torch.no_grad():
            # Decode symbols
            symbols = self.entropy_coder.decode(bitstream, scales)
            y_hat = symbols.float().reshape(1, self.config.encoder.latent_channels, -1, -1)

            # Reconstruct
            if frame_info.frame_type == FrameType.I:
                x_hat = self.decoder(y_hat)
            else:
                ref_latent = self.gop_manager.reference_buffer.get_latent(
                    frame_info.forward_ref
                )
                x_hat = self.temporal_decoder(y_hat, ref_latent)

            # Store reference if needed
            if frame_info.is_reference:
                self.gop_manager.reference_buffer.add(
                    frame_info.index, x_hat, y_hat
                )

            return x_hat

    def encode_video(
        self,
        frames: Iterator[Float[Tensor, "1 3 height width"]],
        num_frames: int,
    ) -> Iterator[CodecOutput]:
        """Encode a video sequence.

        Args:
            frames: Iterator of input frames.
            num_frames: Total number of frames.

        Yields:
            CodecOutput for each frame.
        """
        self.eval()
        self.gop_manager.reset()

        for frame_idx, frame in enumerate(frames):
            if frame_idx >= num_frames:
                break

            frame_info = self.gop_manager.get_frame_info(frame_idx)
            output = self.encode_frame(frame, frame_info)
            yield output

    def compute_rd_loss(
        self,
        x: Float[Tensor, "batch 3 height width"],
        lambda_rd: float,
        reference: Float[Tensor, "batch 3 height width"] | None = None,
    ) -> dict[str, Tensor]:
        """Compute rate-distortion loss for training.

        Args:
            x: Input frames.
            lambda_rd: Rate-distortion tradeoff parameter.
            reference: Optional reference frames.

        Returns:
            Dictionary of loss components.
        """
        x_hat, rate, distortion = self(x, reference)

        # R-D loss
        rd_loss = distortion.mean() + lambda_rd * rate.mean()

        return {
            "rd_loss": rd_loss,
            "rate": rate.mean(),
            "distortion": distortion.mean(),
            "mse": distortion.mean(),
            "psnr": 10 * torch.log10(1.0 / (distortion.mean() + 1e-10)),
        }

    @torch.no_grad()
    def get_rate_distortion_point(
        self,
        x: Float[Tensor, "batch 3 height width"],
    ) -> tuple[float, float]:
        """Get rate-distortion point for input.

        Args:
            x: Input frames.

        Returns:
            Tuple of (rate_bpp, psnr_db).
        """
        self.eval()

        x_hat, rate, distortion = self(x, None)

        # Bits per pixel
        _, _, h, w = x.shape
        bpp = rate.mean().item() / (h * w)

        # PSNR
        psnr = 10 * torch.log10(1.0 / (distortion.mean() + 1e-10)).item()

        return bpp, psnr


def create_codec(config: CodecConfig | None = None) -> VideoCodec:
    """Factory function to create video codec.

    Args:
        config: Codec configuration. Uses defaults if None.

    Returns:
        Configured VideoCodec instance.
    """
    if config is None:
        config = CodecConfig(name="default")
    return VideoCodec(config)
