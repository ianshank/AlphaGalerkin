"""Complete video codec implementation.

Combines encoder, decoder, entropy model, and rate control
into a unified encoding/decoding pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.config import CodecConfig, RateControlMode
from src.video_compression.models.encoder import Encoder
from src.video_compression.models.decoder import Decoder, TemporalDecoder
from src.video_compression.models.hyperprior import (
    HyperpriorEntropyModel,
    create_entropy_model,
)
from src.video_compression.models.quantizer import create_quantizer
from src.video_compression.codec.entropy_coder import EntropyCoder, EncodedBitstream
from src.video_compression.codec.gop_manager import GOPManager, FrameInfo, FrameType
from src.video_compression.mcts.rate_control import (
    MCTSRateController,
    GOPPlanner,
    RateControlDecision,
)
from src.video_compression.mcts.networks import (
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)

logger = logging.getLogger(__name__)


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


class ReferenceFrameError(Exception):
    """Raised when reference frame is unavailable or invalid."""

    pass


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

    def __init__(
        self,
        config: CodecConfig,
        use_mcts_rate_control: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize video codec.

        Args:
            config: Complete codec configuration.
            use_mcts_rate_control: Whether to use MCTS-based rate control.
            device: Device for computation.
        """
        super().__init__()
        self.config = config
        self.device = device
        self._use_mcts = use_mcts_rate_control

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

        # MCTS rate controller (optional)
        self.rate_controller: MCTSRateController | None = None
        self.gop_planner: GOPPlanner | None = None

        if use_mcts_rate_control:
            self._init_rate_controller()

        # Statistics tracking
        self._encoding_stats: dict[str, list[float]] = {
            "bits": [],
            "distortion": [],
            "qp": [],
        }

    def _init_rate_controller(self) -> None:
        """Initialize MCTS rate controller networks."""
        latent_channels = self.config.encoder.latent_channels
        state_dim = 256

        # Create MCTS networks
        representation_net = RepresentationNetwork(
            latent_channels=latent_channels,
            state_dim=state_dim,
            n_layers=3,
        )
        dynamics_net = DynamicsNetwork(
            state_dim=state_dim,
            num_actions=self.config.mcts.qp_max - self.config.mcts.qp_min + 1,
            n_layers=2,
        )
        prediction_net = PredictionNetwork(
            state_dim=state_dim,
            num_actions=self.config.mcts.qp_max - self.config.mcts.qp_min + 1,
            support_size=self.config.mcts.value_support_size,
            hidden_dim=state_dim,
        )

        # Initialize rate controller
        self.rate_controller = MCTSRateController(
            config=self.config.mcts,
            representation_net=representation_net,
            dynamics_net=dynamics_net,
            prediction_net=prediction_net,
            device=self.device,
        )

        # Initialize GOP planner
        self.gop_planner = GOPPlanner(
            config=self.config.mcts,
            rate_controller=self.rate_controller,
        )

        logger.info(
            "Initialized MCTS rate controller with %d simulations",
            self.config.mcts.num_simulations,
        )

    def _validate_reference(
        self,
        frame_info: FrameInfo,
        strict: bool = True,
    ) -> None:
        """Validate that required references are available.

        Args:
            frame_info: Frame metadata with reference requirements.
            strict: If True, raise exception on missing reference.

        Raises:
            ReferenceFrameError: If required reference is missing.
        """
        if frame_info.frame_type == FrameType.I:
            return  # I-frames don't need references

        # Check forward reference
        if frame_info.forward_ref is not None:
            ref_frame = self.gop_manager.reference_buffer.get(frame_info.forward_ref)
            ref_latent = self.gop_manager.reference_buffer.get_latent(
                frame_info.forward_ref
            )

            if ref_frame is None or ref_latent is None:
                msg = (
                    f"Missing forward reference frame {frame_info.forward_ref} "
                    f"for frame {frame_info.index}"
                )
                logger.error(msg)
                if strict:
                    raise ReferenceFrameError(msg)

        # Check backward reference (for B-frames)
        if frame_info.backward_ref is not None:
            ref_frame = self.gop_manager.reference_buffer.get(frame_info.backward_ref)
            ref_latent = self.gop_manager.reference_buffer.get_latent(
                frame_info.backward_ref
            )

            if ref_frame is None or ref_latent is None:
                msg = (
                    f"Missing backward reference frame {frame_info.backward_ref} "
                    f"for frame {frame_info.index}"
                )
                logger.error(msg)
                if strict:
                    raise ReferenceFrameError(msg)

    def _get_reference_latent(
        self,
        frame_info: FrameInfo,
    ) -> Tensor | None:
        """Get reference latent for temporal coding.

        Args:
            frame_info: Frame metadata.

        Returns:
            Reference latent tensor or None for I-frames.
        """
        if frame_info.frame_type == FrameType.I:
            return None

        # For P-frames, use forward reference
        if frame_info.frame_type == FrameType.P:
            return self.gop_manager.reference_buffer.get_latent(frame_info.forward_ref)

        # For B-frames, use both references (combine)
        fwd_latent = self.gop_manager.reference_buffer.get_latent(frame_info.forward_ref)
        bwd_latent = self.gop_manager.reference_buffer.get_latent(frame_info.backward_ref)

        if fwd_latent is None:
            return bwd_latent
        if bwd_latent is None:
            return fwd_latent

        # Average of forward and backward references
        return (fwd_latent + bwd_latent) / 2

    def _select_qp(
        self,
        frame_latent: Tensor,
        frame_info: FrameInfo,
    ) -> int:
        """Select QP for frame using rate control.

        Args:
            frame_latent: Encoded frame latent.
            frame_info: Frame metadata.

        Returns:
            Selected QP value.
        """
        if self.rate_controller is not None:
            decision = self.rate_controller.select_qp(
                frame_latent,
                frame_type=frame_info.frame_type.value,
            )
            logger.debug(
                "MCTS selected QP=%d (confidence=%.3f) for frame %d",
                decision.qp,
                decision.confidence,
                frame_info.index,
            )
            return decision.qp

        # Fallback: use frame-type-based QP
        base_qp = self.config.mcts.crf_value
        qp_offset = {"I": -2, "P": 0, "B": 2}
        qp = base_qp + qp_offset.get(frame_info.frame_type.value, 0)

        return max(self.config.mcts.qp_min, min(self.config.mcts.qp_max, qp))

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
        validate_refs: bool = True,
    ) -> CodecOutput:
        """Encode a single frame.

        Args:
            x: Input frame (1, 3, H, W) in [0, 1].
            frame_info: Frame metadata.
            validate_refs: Whether to validate reference availability.

        Returns:
            CodecOutput with bitstream and metrics.

        Raises:
            ReferenceFrameError: If required reference is unavailable.
        """
        # Validate references before encoding
        if validate_refs:
            self._validate_reference(frame_info, strict=True)

        with torch.no_grad():
            # Encode to latent
            y = self.encoder(x)

            # Select QP using MCTS rate control (if enabled)
            qp = self._select_qp(y, frame_info)

            # Apply QP-based scaling before quantization
            qp_scale = 2.0 ** ((qp - self.config.mcts.crf_value) / 6.0)
            y_scaled = y / qp_scale

            # Quantize
            y_hat = torch.round(y_scaled) * qp_scale

            # Get entropy model output for rate estimation
            entropy_output = self.entropy_model(y_scaled, training=False)

            # Compress with entropy coder
            compressed = self.entropy_model.compress(y_scaled)
            bitstream = self.entropy_coder.encode(
                compressed["y_symbols"],
                compressed["scales"],
            )

            # Decode for reconstruction
            if frame_info.frame_type == FrameType.I:
                x_hat = self.decoder(y_hat)
            else:
                # Get reference latent with validation
                ref_latent = self._get_reference_latent(frame_info)
                if ref_latent is None:
                    logger.warning(
                        "No reference latent for frame %d, falling back to I-frame decode",
                        frame_info.index,
                    )
                    x_hat = self.decoder(y_hat)
                else:
                    x_hat = self.temporal_decoder(y_hat, ref_latent)

            # Store reference if needed
            if frame_info.is_reference:
                self.gop_manager.reference_buffer.add(
                    frame_info.index, x_hat, y_hat
                )
                logger.debug(
                    "Stored reference frame %d (type=%s)",
                    frame_info.index,
                    frame_info.frame_type.value,
                )

            # Compute metrics
            rate = len(bitstream.data) * 8  # Bits
            distortion = torch.mean((x - x_hat) ** 2).item()

            # Track statistics
            self._encoding_stats["bits"].append(rate)
            self._encoding_stats["distortion"].append(distortion)
            self._encoding_stats["qp"].append(qp)

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
        latent_shape: tuple[int, int] | None = None,
        qp: int | None = None,
        validate_refs: bool = True,
    ) -> Float[Tensor, "1 3 height width"]:
        """Decode a single frame.

        Args:
            bitstream: Encoded bitstream.
            frame_info: Frame metadata.
            scales: Scale parameters for entropy decoding.
            latent_shape: Optional (H, W) for latent reshape.
            qp: QP used during encoding (for inverse scaling).
            validate_refs: Whether to validate reference availability.

        Returns:
            Decoded frame (1, 3, H, W) in [0, 1].

        Raises:
            ReferenceFrameError: If required reference is unavailable.
        """
        # Validate references
        if validate_refs and frame_info.frame_type != FrameType.I:
            self._validate_reference(frame_info, strict=True)

        with torch.no_grad():
            # Decode symbols
            symbols = self.entropy_coder.decode(bitstream, scales)

            # Reshape to latent dimensions
            if latent_shape is not None:
                h, w = latent_shape
            else:
                # Estimate from scales
                h = w = int((symbols.numel() // self.config.encoder.latent_channels) ** 0.5)

            y_hat = symbols.float().reshape(
                1, self.config.encoder.latent_channels, h, w
            )

            # Apply inverse QP scaling
            if qp is not None:
                qp_scale = 2.0 ** ((qp - self.config.mcts.crf_value) / 6.0)
                y_hat = y_hat * qp_scale

            # Reconstruct
            if frame_info.frame_type == FrameType.I:
                x_hat = self.decoder(y_hat)
            else:
                ref_latent = self._get_reference_latent(frame_info)
                if ref_latent is None:
                    logger.warning(
                        "No reference for frame %d during decode, using I-frame path",
                        frame_info.index,
                    )
                    x_hat = self.decoder(y_hat)
                else:
                    x_hat = self.temporal_decoder(y_hat, ref_latent)

            # Store reference if needed
            if frame_info.is_reference:
                self.gop_manager.reference_buffer.add(frame_info.index, x_hat, y_hat)
                logger.debug(
                    "Stored decoded reference frame %d (type=%s)",
                    frame_info.index,
                    frame_info.frame_type.value,
                )

            return x_hat

    def decode_video(
        self,
        bitstreams: Iterator[tuple[EncodedBitstream, FrameInfo, Tensor]],
        num_frames: int,
        callback: Callable[[int, Tensor], None] | None = None,
    ) -> Iterator[Float[Tensor, "1 3 height width"]]:
        """Decode a video sequence.

        Args:
            bitstreams: Iterator of (bitstream, frame_info, scales) tuples.
            num_frames: Total number of frames.
            callback: Optional callback for progress reporting.

        Yields:
            Decoded frames.
        """
        self.eval()
        self.gop_manager.reset()

        logger.info("Starting video decoding: %d frames", num_frames)

        for frame_idx, (bitstream, frame_info, scales) in enumerate(bitstreams):
            if frame_idx >= num_frames:
                break

            try:
                x_hat = self.decode_frame(
                    bitstream,
                    frame_info,
                    scales,
                    validate_refs=True,
                )
            except ReferenceFrameError as e:
                logger.error("Decode failed at frame %d: %s", frame_idx, str(e))
                raise

            if callback is not None:
                callback(frame_idx, x_hat)

            if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                logger.info("Decoded frame %d/%d", frame_idx + 1, num_frames)

            yield x_hat

        logger.info("Video decoding complete: %d frames", num_frames)

    def encode_video(
        self,
        frames: Iterator[Float[Tensor, "1 3 height width"]],
        num_frames: int,
        callback: Callable[[int, CodecOutput], None] | None = None,
    ) -> Iterator[CodecOutput]:
        """Encode a video sequence.

        Args:
            frames: Iterator of input frames.
            num_frames: Total number of frames.
            callback: Optional callback for progress reporting.

        Yields:
            CodecOutput for each frame.
        """
        self.eval()
        self.gop_manager.reset()
        self._reset_stats()

        logger.info("Starting video encoding: %d frames", num_frames)

        # If MCTS GOP planning is enabled, pre-plan GOPs
        gop_plans: dict[int, list[RateControlDecision]] = {}

        for frame_idx, frame in enumerate(frames):
            if frame_idx >= num_frames:
                break

            # Check if we need to plan a new GOP
            if self.gop_planner is not None and frame_idx % self.config.mcts.gop_size == 0:
                gop_start = frame_idx
                # Collect GOP frames for planning (would need buffering in practice)
                logger.debug("Planning GOP starting at frame %d", gop_start)

            frame_info = self.gop_manager.get_frame_info(frame_idx)

            try:
                output = self.encode_frame(frame, frame_info, validate_refs=True)
            except ReferenceFrameError as e:
                logger.warning(
                    "Reference error at frame %d, encoding as I-frame: %s",
                    frame_idx,
                    str(e),
                )
                # Force I-frame on reference error
                frame_info = FrameInfo(
                    index=frame_idx,
                    gop_index=frame_info.gop_index,
                    frame_type=FrameType.I,
                    display_order=frame_info.display_order,
                    encode_order=frame_info.encode_order,
                    forward_ref=None,
                    backward_ref=None,
                )
                output = self.encode_frame(frame, frame_info, validate_refs=False)

            if callback is not None:
                callback(frame_idx, output)

            # Log progress
            if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                avg_bits = sum(self._encoding_stats["bits"]) / len(
                    self._encoding_stats["bits"]
                )
                avg_psnr = (
                    -10 * torch.log10(
                        torch.tensor(
                            sum(self._encoding_stats["distortion"])
                            / len(self._encoding_stats["distortion"])
                            + 1e-10
                        )
                    ).item()
                )
                logger.info(
                    "Encoded frame %d/%d - Avg bits: %.1f, Avg PSNR: %.2f dB",
                    frame_idx + 1,
                    num_frames,
                    avg_bits,
                    avg_psnr,
                )

            yield output

        logger.info("Video encoding complete: %d frames", num_frames)

    def _reset_stats(self) -> None:
        """Reset encoding statistics."""
        self._encoding_stats = {"bits": [], "distortion": [], "qp": []}

    def get_encoding_stats(self) -> dict[str, float]:
        """Get summary encoding statistics.

        Returns:
            Dictionary with average statistics.
        """
        if not self._encoding_stats["bits"]:
            return {}

        n = len(self._encoding_stats["bits"])
        total_bits = sum(self._encoding_stats["bits"])
        avg_distortion = sum(self._encoding_stats["distortion"]) / n
        avg_qp = sum(self._encoding_stats["qp"]) / n

        return {
            "total_bits": total_bits,
            "avg_bits_per_frame": total_bits / n,
            "avg_psnr": -10 * torch.log10(torch.tensor(avg_distortion + 1e-10)).item(),
            "avg_qp": avg_qp,
            "num_frames": n,
        }

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


def create_codec(
    config: CodecConfig | None = None,
    use_mcts_rate_control: bool = False,
    device: str = "cpu",
) -> VideoCodec:
    """Factory function to create video codec.

    Args:
        config: Codec configuration. Uses defaults if None.
        use_mcts_rate_control: Whether to enable MCTS-based rate control.
        device: Device for computation.

    Returns:
        Configured VideoCodec instance.
    """
    if config is None:
        config = CodecConfig(name="default")

    codec = VideoCodec(
        config=config,
        use_mcts_rate_control=use_mcts_rate_control,
        device=device,
    )

    logger.info(
        "Created video codec (MCTS rate control: %s, device: %s)",
        use_mcts_rate_control,
        device,
    )

    return codec


def load_codec(
    checkpoint_path: Path | str,
    config: CodecConfig | None = None,
    device: str = "cpu",
) -> VideoCodec:
    """Load codec from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Optional config override.
        device: Device for computation.

    Returns:
        Loaded VideoCodec instance.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading codec from %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use provided
    if config is None:
        if "config" in checkpoint:
            config = CodecConfig(**checkpoint["config"])
        else:
            config = CodecConfig(name="loaded")

    # Determine if MCTS was used
    use_mcts = "rate_controller" in checkpoint.get("model_state_dict", {})

    codec = create_codec(config, use_mcts_rate_control=use_mcts, device=device)

    # Load weights
    if "model_state_dict" in checkpoint:
        codec.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        codec.load_state_dict(checkpoint, strict=False)

    codec.to(device)
    codec.eval()

    logger.info("Loaded codec with %d parameters", sum(p.numel() for p in codec.parameters()))

    return codec
