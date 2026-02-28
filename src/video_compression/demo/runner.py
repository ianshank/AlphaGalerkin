"""Compression demo runner for end-to-end MVP demonstration.

Provides a reusable runner that demonstrates:
1. Synthetic video generation
2. Full encode/decode pipeline
3. Rate-distortion curve evaluation at multiple lambda values
4. Resolution independence verification
5. Bitstream I/O roundtrip

Usage:
    from src.video_compression.demo import CompressionDemoRunner, DemoConfig

    config = DemoConfig(num_frames=8, height=64, width=64)
    runner = CompressionDemoRunner(config)
    result = runner.run_full_demo()
    print(result.to_json())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.video_compression.codec.codec import VideoCodec, create_codec
from src.video_compression.codec.gop_manager import FrameInfo, FrameType
from src.video_compression.config import (
    CodecConfig,
    DecoderConfig,
    EncoderConfig,
    EntropyConfig,
    MCTSRateControlConfig,
    QuantizationMode,
    QuantizerConfig,
    TrainingConfig,
)
from src.video_compression.data.synthetic import (
    SyntheticPattern,
    SyntheticVideoConfig,
    SyntheticVideoGenerator,
)
from src.video_compression.demo.config import DemoConfig
from src.video_compression.metrics.quality import compute_psnr, compute_ssim
from src.video_compression.utils.bitstream import (
    BitstreamHeader,
    EncodedFrame,
    FrameHeader,
    load_bitstream,
    save_bitstream,
)
from src.video_compression.utils.padding import pad_to_multiple

logger = logging.getLogger(__name__)

# Sentinel for QP when frame_info.qp is None
_DEFAULT_QP = 32


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """Per-frame encoding result."""

    frame_idx: int
    frame_type: str
    rate_bits: float
    distortion_mse: float
    psnr_db: float
    ssim: float
    encode_time_ms: float


@dataclass
class LambdaResult:
    """Result for a single lambda value."""

    lambda_rd: float
    pattern: str
    height: int
    width: int
    num_frames: int
    frame_results: list[FrameResult] = field(default_factory=list)

    # Aggregated metrics
    avg_bpp: float = 0.0
    avg_psnr_db: float = 0.0
    avg_ssim: float = 0.0
    total_bits: int = 0
    total_encode_time_ms: float = 0.0
    total_decode_time_ms: float = 0.0

    # Bitstream info
    bitstream_path: str | None = None
    bitstream_size_bytes: int = 0


@dataclass
class ResolutionResult:
    """Result for a single resolution test."""

    pattern: str
    lambda_rd: float
    height: int
    width: int
    avg_psnr_db: float
    avg_ssim: float
    avg_bpp: float
    num_frames: int


@dataclass
class DemoResult:
    """Complete demo output."""

    lambda_results: list[LambdaResult] = field(default_factory=list)
    resolution_results: list[ResolutionResult] = field(default_factory=list)
    total_time_s: float = 0.0
    device: str = "cpu"
    config_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "lambda_results": [asdict(r) for r in self.lambda_results],
            "resolution_results": [asdict(r) for r in self.resolution_results],
            "total_time_s": round(self.total_time_s, 3),
            "device": self.device,
            "config": self.config_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------


class CompressionDemoRunner:
    """Orchestrates the full video compression MVP demo.

    Reusable from CLI scripts and tests.
    """

    def __init__(self, config: DemoConfig) -> None:
        """Initialize demo runner.

        Args:
            config: Demo configuration.

        """
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Set seed
        torch.manual_seed(config.seed)

        logger.info(
            "CompressionDemoRunner initialized: device=%s, patterns=%s, "
            "lambdas=%s, resolutions=%s",
            self.device,
            [p.value for p in config.patterns],
            config.lambda_values,
            config.resolution_sizes,
        )

    def _create_codec_config(self) -> CodecConfig:
        """Build CodecConfig from DemoConfig parameters.

        Maps demo-level settings to the full codec configuration
        hierarchy without hardcoding.

        Returns:
            Configured CodecConfig.

        """
        c = self.config

        encoder_config = EncoderConfig(
            name="demo_encoder",
            latent_channels=c.latent_channels,
            n_layers=c.n_layers,
            d_model=c.d_model,
            n_heads=c.n_heads,
            d_ffn=c.d_ffn,
            downsample_factor=c.downsample_factor,
        )

        decoder_config = DecoderConfig(
            name="demo_decoder",
            latent_channels=c.latent_channels,
            n_layers=c.n_layers,
            d_model=c.d_model,
            n_heads=c.n_heads,
            d_ffn=c.d_ffn,
            upsample_factor=c.downsample_factor,
        )

        quantizer_config = QuantizerConfig(
            name="demo_quantizer",
            mode=QuantizationMode.STE,
        )

        entropy_config = EntropyConfig(
            name="demo_entropy",
            hyper_channels=max(c.latent_channels // 2, 32),
            num_filters=c.latent_channels,
        )

        mcts_config = MCTSRateControlConfig(
            name="demo_mcts",
            gop_size=min(c.num_frames, 8),
            use_b_frames=False,
        )

        training_config = TrainingConfig(
            name="demo_training",
            lambda_rd=c.lambda_values[0],
        )

        return CodecConfig(
            name="demo_codec",
            encoder=encoder_config,
            decoder=decoder_config,
            quantizer=quantizer_config,
            entropy=entropy_config,
            mcts=mcts_config,
            training=training_config,
        )

    def _create_codec(self) -> VideoCodec:
        """Create a fresh codec instance.

        Returns:
            Newly initialized VideoCodec.

        """
        codec_config = self._create_codec_config()
        codec = create_codec(codec_config, device=str(self.device))
        codec.to(self.device)
        codec.eval()
        return codec

    def _generate_video(
        self,
        pattern: SyntheticPattern,
        height: int | None = None,
        width: int | None = None,
    ) -> Tensor:
        """Generate synthetic video for a given pattern.

        Args:
            pattern: Pattern type.
            height: Override height (uses config default if None).
            width: Override width (uses config default if None).

        Returns:
            Tensor of shape (T, 3, H, W) in [0, 1].

        """
        video_config = SyntheticVideoConfig(
            pattern=pattern,
            num_frames=self.config.num_frames,
            height=height or self.config.height,
            width=width or self.config.width,
            seed=self.config.seed,
        )
        generator = SyntheticVideoGenerator(video_config)
        return generator.generate()

    def run_single_lambda(
        self,
        frames: Tensor,
        lambda_rd: float,
        pattern_name: str = "unknown",
        write_bitstream: bool = False,
    ) -> LambdaResult:
        """Encode and decode frames at a single lambda value.

        Args:
            frames: Video tensor (T, 3, H, W) in [0, 1].
            lambda_rd: Rate-distortion tradeoff.
            pattern_name: Pattern name for logging.
            write_bitstream: Whether to write .agk file.

        Returns:
            LambdaResult with per-frame and aggregate metrics.

        """
        num_frames, _, height, width = frames.shape
        codec = self._create_codec()

        result = LambdaResult(
            lambda_rd=lambda_rd,
            pattern=pattern_name,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        total_bits = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        encoded_frames: list[EncodedFrame] = []

        logger.info(
            "Encoding %d frames (%s, %dx%d) at lambda=%.4f",
            num_frames,
            pattern_name,
            width,
            height,
            lambda_rd,
        )

        for i in range(num_frames):
            frame = frames[i].unsqueeze(0).to(self.device)

            # Pad to multiple of downsample_factor
            frame_padded, pad_info = pad_to_multiple(
                frame,
                align_to=self.config.downsample_factor,
            )

            # Get frame info from GOP manager
            frame_info = codec.gop_manager.get_frame_info(i)

            # Encode
            encode_start = time.perf_counter()
            try:
                output = codec.encode_frame(frame_padded, frame_info)
            except Exception as e:
                logger.warning(
                    "Encode failed for frame %d (type=%s): %s. "
                    "Forcing I-frame.",
                    i,
                    frame_info.frame_type.value,
                    e,
                )
                i_frame_info = FrameInfo(
                    index=i,
                    gop_index=0,
                    frame_type=FrameType.I,
                    display_order=i,
                    encode_order=i,
                    forward_ref=None,
                    backward_ref=None,
                    qp=_DEFAULT_QP,
                )
                output = codec.encode_frame(frame_padded, i_frame_info)

            encode_time_ms = (time.perf_counter() - encode_start) * 1000

            # Compute quality metrics against original (unpadded)
            reconstructed = output.reconstructed
            if reconstructed.shape[-2:] != frame.shape[-2:]:
                reconstructed = reconstructed[:, :, :height, :width]

            psnr = compute_psnr(
                reconstructed.cpu(),
                frame.cpu(),
            ).item()

            ssim_val = compute_ssim(
                reconstructed.cpu(),
                frame.cpu(),
            ).item()

            frame_result = FrameResult(
                frame_idx=i,
                frame_type=frame_info.frame_type.value,
                rate_bits=output.rate,
                distortion_mse=output.distortion,
                psnr_db=psnr,
                ssim=ssim_val,
                encode_time_ms=encode_time_ms,
            )
            result.frame_results.append(frame_result)
            total_bits += output.rate
            total_psnr += psnr
            total_ssim += ssim_val

            # Build encoded frame for bitstream
            if write_bitstream:
                latent_bytes = output.bitstream.data
                z_bytes = b""
                if output.z_bitstream is not None:
                    z_bytes = output.z_bitstream.data

                fwd_ref = (
                    frame_info.forward_ref
                    if frame_info.forward_ref is not None
                    else -1
                )
                bwd_ref = (
                    frame_info.backward_ref
                    if frame_info.backward_ref is not None
                    else -1
                )
                frame_header = FrameHeader(
                    frame_idx=i,
                    frame_type=frame_info.frame_type,
                    data_length=len(latent_bytes),
                    qp=frame_info.qp if frame_info.qp is not None else _DEFAULT_QP,
                    forward_ref_idx=fwd_ref,
                    backward_ref_idx=bwd_ref,
                )
                encoded_frames.append(
                    EncodedFrame(
                        header=frame_header,
                        data=latent_bytes,
                        z_data=z_bytes,
                    )
                )

            logger.debug(
                "Frame %d: type=%s, bits=%d, PSNR=%.2f dB, SSIM=%.4f, time=%.1f ms",
                i,
                frame_info.frame_type.value,
                int(output.rate),
                psnr,
                ssim_val,
                encode_time_ms,
            )

        # Aggregate metrics
        pixels_per_frame = height * width
        result.total_bits = int(total_bits)
        result.avg_bpp = total_bits / (num_frames * pixels_per_frame)
        result.avg_psnr_db = total_psnr / num_frames
        result.avg_ssim = total_ssim / num_frames
        result.total_encode_time_ms = sum(
            fr.encode_time_ms for fr in result.frame_results
        )

        # Write bitstream if requested
        if write_bitstream and encoded_frames:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            bitstream_path = output_dir / f"{pattern_name}_lambda{lambda_rd:.4f}.agk"

            ds_factor = self.config.downsample_factor
            padded_h = ((height + ds_factor - 1) // ds_factor) * ds_factor
            padded_w = ((width + ds_factor - 1) // ds_factor) * ds_factor

            header = BitstreamHeader(
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=self.config.frame_rate,
                gop_size=min(num_frames, 8),
                downsample_factor=ds_factor,
                latent_channels=self.config.latent_channels,
                padded_width=padded_w,
                padded_height=padded_h,
                lambda_rd=lambda_rd,
            )

            bytes_written = save_bitstream(bitstream_path, header, encoded_frames)
            result.bitstream_path = str(bitstream_path)
            result.bitstream_size_bytes = bytes_written
            logger.info("Wrote bitstream: %s (%d bytes)", bitstream_path, bytes_written)

            # Verify bitstream roundtrip: read back
            try:
                loaded_header, loaded_frames = load_bitstream(bitstream_path)
                assert loaded_header.num_frames == num_frames, (
                    f"Frame count mismatch: wrote {num_frames}, "
                    f"read {loaded_header.num_frames}"
                )
                assert len(loaded_frames) == len(encoded_frames), (
                    f"Encoded frame count mismatch: wrote {len(encoded_frames)}, "
                    f"read {len(loaded_frames)}"
                )
                logger.info(
                    "Bitstream roundtrip verified: %d frames, %dx%d",
                    loaded_header.num_frames,
                    loaded_header.width,
                    loaded_header.height,
                )
            except Exception as e:
                logger.warning("Bitstream roundtrip verification failed: %s", e)

        logger.info(
            "Lambda %.4f (%s): avg_bpp=%.4f, avg_PSNR=%.2f dB, avg_SSIM=%.4f",
            lambda_rd,
            pattern_name,
            result.avg_bpp,
            result.avg_psnr_db,
            result.avg_ssim,
        )

        return result

    def run_rd_curve(
        self,
        frames: Tensor,
        lambda_values: list[float],
        pattern_name: str = "unknown",
    ) -> list[LambdaResult]:
        """Evaluate R-D curve at multiple lambda values.

        Creates a fresh codec per lambda to avoid state leakage.

        Args:
            frames: Video tensor (T, 3, H, W).
            lambda_values: Lambda values to evaluate.
            pattern_name: Pattern name for logging.

        Returns:
            List of LambdaResult, one per lambda.

        """
        results = []
        for i, lam in enumerate(lambda_values):
            logger.info(
                "R-D curve point %d/%d: lambda=%.4f",
                i + 1,
                len(lambda_values),
                lam,
            )
            # Only write bitstream for first lambda to save disk
            write_bs = self.config.write_bitstream and (i == 0)
            result = self.run_single_lambda(frames, lam, pattern_name, write_bitstream=write_bs)
            results.append(result)

        return results

    def run_resolution_test(
        self,
        pattern: SyntheticPattern,
        lambda_rd: float,
    ) -> list[ResolutionResult]:
        """Test codec at multiple resolutions.

        Demonstrates resolution independence: same model architecture
        processes different spatial sizes.

        Args:
            pattern: Pattern to generate at each resolution.
            lambda_rd: Lambda value for encoding.

        Returns:
            List of ResolutionResult per resolution size.

        """
        results = []
        for h, w in self.config.resolution_sizes:
            logger.info(
                "Resolution test: %dx%d, pattern=%s, lambda=%.4f",
                w,
                h,
                pattern.value,
                lambda_rd,
            )

            frames = self._generate_video(pattern, height=h, width=w)
            lambda_result = self.run_single_lambda(
                frames,
                lambda_rd,
                pattern_name=f"{pattern.value}_{h}x{w}",
                write_bitstream=False,
            )

            results.append(
                ResolutionResult(
                    pattern=pattern.value,
                    lambda_rd=lambda_rd,
                    height=h,
                    width=w,
                    avg_psnr_db=lambda_result.avg_psnr_db,
                    avg_ssim=lambda_result.avg_ssim,
                    avg_bpp=lambda_result.avg_bpp,
                    num_frames=lambda_result.num_frames,
                )
            )

        return results

    def run_full_demo(self) -> DemoResult:
        """Run the complete MVP demo pipeline.

        Executes:
        1. R-D curve for each configured pattern
        2. Resolution independence test

        Returns:
            DemoResult with all metrics and metadata.

        """
        demo_start = time.perf_counter()

        result = DemoResult(
            device=str(self.device),
            config_summary=self.config.to_summary_dict(),
        )

        # ── R-D Curve per pattern ─────────────────────────────────────
        for pattern in self.config.patterns:
            logger.info(
                "=== Pattern: %s ===",
                pattern.value,
            )
            frames = self._generate_video(pattern)
            rd_results = self.run_rd_curve(
                frames,
                self.config.lambda_values,
                pattern_name=pattern.value,
            )
            result.lambda_results.extend(rd_results)

        # ── Resolution independence test ──────────────────────────────
        logger.info("=== Resolution Independence Test ===")
        # Use first pattern for resolution test
        test_pattern = self.config.patterns[0]
        res_results = self.run_resolution_test(
            test_pattern,
            self.config.resolution_lambda,
        )
        result.resolution_results.extend(res_results)

        result.total_time_s = time.perf_counter() - demo_start

        # Save JSON results
        if self.config.output_dir:
            self._save_results(result)

        logger.info(
            "Demo complete: %.1f s, %d R-D points, %d resolution tests",
            result.total_time_s,
            len(result.lambda_results),
            len(result.resolution_results),
        )

        return result

    def _save_results(self, result: DemoResult) -> None:
        """Save demo results as JSON.

        Args:
            result: Demo results to save.

        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "demo_results.json"

        json_path.write_text(result.to_json())
        logger.info("Saved demo results to %s", json_path)
