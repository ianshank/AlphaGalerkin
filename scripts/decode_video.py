"""CLI script for decoding compressed video files.

Usage:
    python -m scripts.decode_video \
        --input compressed.agk \
        --output decoded.mp4 \
        --checkpoint path/to/model.pt

Features:
    - Bitstream format validation
    - Automatic device selection
    - Progress reporting
    - Quality metrics computation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from src.templates.logging import (
    configure_module_logging,
    create_logger_class,
    DebugContext,
)

# Configure logging
configure_module_logging(level="INFO")
Logger = create_logger_class("decode_video")
logger = Logger("cli")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode compressed video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to compressed bitstream file (.agk)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output video file (.mp4, .avi, etc.)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for computation",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="Path to write quality metrics report (JSON)",
    )
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=None,
        help="Original video for quality comparison",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--prores",
        action="store_true",
        help="Use Apple ProRes 422 codec for .mov output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate bitstream without decoding",
    )

    return parser.parse_args()


def get_device(device_str: str) -> str:
    """Get computation device.

    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "mps").

    Returns:
        Resolved device string.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


def write_video_frames(
    frames: Iterator[Tensor],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    use_prores: bool = False,
) -> int:
    """Write decoded frames to video file.

    Args:
        frames: Iterator of decoded frame tensors (1, 3, H, W).
        output_path: Output video path.
        fps: Frame rate.
        width: Frame width.
        height: Frame height.
        use_prores: Whether to use ProRes codec for .mov.

    Returns:
        Number of frames written.
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Install with: pip install opencv-python")
        sys.exit(1)

    # Determine output format from extension
    ext = output_path.suffix.lower()
    fourcc_map = {
        ".mp4": "mp4v",
        ".avi": "XVID",
        ".mov": "mp4v",  # Standard QuickTime
        ".mkv": "XVID",
        ".webm": "VP80",
    }

    # Handle ProRes option
    if use_prores and ext == ".mov":
        fourcc_code = "apcn"  # ProRes 422
    else:
        # Default to mp4v if unknown, but log warning
        if ext not in fourcc_map:
            logger.warning(f"Unknown extension {ext}, defaulting to mp4v", extension=ext)
        fourcc_code = fourcc_map.get(ext, "mp4v")

    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        logger.error("video_writer_failed", path=str(output_path))
        sys.exit(1)

    frame_count = 0
    try:
        for frame in frames:
            # Convert tensor to numpy (B, C, H, W) -> (H, W, C)
            frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).clip(0, 255).astype("uint8")

            # RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            writer.write(frame_bgr)
            frame_count += 1

            if frame_count % 10 == 0:
                 logger.debug("frames_written", count=frame_count)

    finally:
        writer.release()

    return frame_count


def compute_quality_metrics(
    decoded_path: Path,
    reference_path: Path,
) -> dict[str, float]:
    """Compute quality metrics between decoded and reference videos.

    Args:
        decoded_path: Path to decoded video.
        reference_path: Path to reference video.

    Returns:
        Dictionary of quality metrics.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("deps_missing", message="OpenCV/NumPy not installed")
        return {}

    # Import quality metrics
    try:
        from src.video_compression.metrics.quality import compute_psnr, compute_ssim
    except ImportError:
        logger.warning("metrics_missing", message="Quality metrics not available")
        return {}

    decoded_cap = cv2.VideoCapture(str(decoded_path))
    reference_cap = cv2.VideoCapture(str(reference_path))

    if not decoded_cap.isOpened() or not reference_cap.isOpened():
        logger.error("comparison_failed", message="Failed to open video files")
        return {}

    psnr_values = []
    ssim_values = []
    frame_idx = 0

    while True:
        ret_dec, frame_dec = decoded_cap.read()
        ret_ref, frame_ref = reference_cap.read()

        if not ret_dec or not ret_ref:
            break

        # Convert to tensors
        dec_tensor = torch.from_numpy(frame_dec).float() / 255.0
        ref_tensor = torch.from_numpy(frame_ref).float() / 255.0

        # Compute metrics
        psnr = compute_psnr(dec_tensor, ref_tensor).item()
        ssim = compute_ssim(
            dec_tensor.permute(2, 0, 1).unsqueeze(0),
            ref_tensor.permute(2, 0, 1).unsqueeze(0),
        ).item()

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        frame_idx += 1

    decoded_cap.release()
    reference_cap.release()

    if not psnr_values:
        return {}

    return {
        "avg_psnr": float(np.mean(psnr_values)),
        "min_psnr": float(np.min(psnr_values)),
        "max_psnr": float(np.max(psnr_values)),
        "avg_ssim": float(np.mean(ssim_values)),
        "min_ssim": float(np.min(ssim_values)),
        "max_ssim": float(np.max(ssim_values)),
        "num_frames": frame_idx,
    }


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()

    if args.verbose:
        configure_module_logging(level="DEBUG")

    logger.info("decoding_start", input=str(args.input), output=str(args.output))

    # Validate input file
    if not args.input.exists():
        logger.error("input_not_found", path=str(args.input))
        return 1

    # Validate checkpoint
    if not args.checkpoint.exists():
        logger.error("checkpoint_not_found", path=str(args.checkpoint))
        return 1

    # Get device
    device = get_device(args.device)
    logger.info("device_selected", device=str(device))

    # Import codec modules
    try:
        from src.video_compression.codec import load_codec
        from src.video_compression.utils.bitstream import load_bitstream
    except ImportError as e:
        logger.error("import_failed", error=str(e))
        return 1

    with DebugContext("video_decoding", capture_memory=True) as ctx:
        # Load bitstream
        logger.info("loading_bitstream", path=str(args.input))
        try:
            header, encoded_frames = load_bitstream(args.input)
        except Exception as e:
            logger.error("bitstream_load_failed", error=str(e))
            return 1

        logger.info(
            "bitstream_info",
            resolution=f"{header.width}x{header.height}",
            frames=header.num_frames,
            fps=header.fps,
        )

        if args.dry_run:
            logger.info("dry_run_complete")
            return 0

        # Load codec
        logger.info("loading_codec", path=str(args.checkpoint))
        try:
            # Try using the load_codec utility first
            codec = load_codec(args.checkpoint, device=device)
        except Exception as e:
            logger.warning("load_codec_failed", error=str(e), message="Retrying with manual load")
            # Fallback: manual loading for robustness
            try:
                from src.video_compression.config import CodecConfig
                from src.video_compression.codec.codec import create_codec

                checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)

                # Try to reconstruct config from checkpoint, fallback to default
                if "config" in checkpoint_data and isinstance(checkpoint_data["config"], dict):
                    config = CodecConfig(**checkpoint_data["config"])
                else:
                    config = CodecConfig(name="decoder")

                codec = create_codec(config)
                codec.to(device)
                codec.eval()

                # Load state dict with flexible key detection
                if "model_state_dict" in checkpoint_data:
                    codec.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
                elif "model_state" in checkpoint_data:
                    codec.load_state_dict(checkpoint_data["model_state"], strict=False)
                else:
                    codec.load_state_dict(checkpoint_data, strict=False)

                logger.info("manual_load_success")
            except Exception as fallback_error:
                logger.error("codec_load_failed", error=str(fallback_error))
                return 1


        ctx.checkpoint("codec_loaded")

        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Decode frames
        logger.info("decoding_frames", count=header.num_frames)
        start_time = time.time()

        def frame_generator() -> Iterator[Tensor]:
            """Generate decoded frames."""
            from src.video_compression.codec.gop_manager import FrameInfo
            from src.video_compression.codec.entropy_coder import EncodedBitstream

            for frame in encoded_frames:
                # Reconstruct FrameInfo from stored metadata in frame header
                frame_info = FrameInfo(
                    index=frame.header.frame_idx,
                    gop_index=frame.header.frame_idx % codec.config.mcts.gop_size,
                    frame_type=frame.header.frame_type,
                    display_order=frame.header.frame_idx % codec.config.mcts.gop_size,
                    encode_order=frame.header.frame_idx % codec.config.mcts.gop_size,
                    forward_ref=frame.header.forward_ref_idx if frame.header.forward_ref_idx >= 0 else None,
                    backward_ref=frame.header.backward_ref_idx if frame.header.backward_ref_idx >= 0 else None,
                )

                # Compute latent shape from header dimensions (needed for bitstream)
                latent_h = header.padded_height // header.downsample_factor
                latent_w = header.padded_width // header.downsample_factor

                # Reconstruct bitstream object from raw bytes
                # EncodedBitstream requires: data, shape, min_val, max_val, num_symbols
                bitstream = EncodedBitstream(
                    data=frame.data,
                    shape=(1, header.latent_channels, latent_h, latent_w),
                    min_val=-128,  # Typical quantized symbol range
                    max_val=127,
                    num_symbols=header.latent_channels * latent_h * latent_w,
                )

                # Reconstruct hyperprior bitstream from z_data if available
                # This enables proper scale reconstruction during decoding
                z_bitstream = None
                if frame.z_data and len(frame.z_data) > 0:
                    # Compute z shape based on hyperprior architecture
                    # HyperAnalysis typically downsamples by 4x (2 conv layers with stride 2)
                    hyper_layers = getattr(codec.config.entropy, 'hyper_layers', 3)
                    z_downsample = 2 ** (hyper_layers - 1)
                    z_h = max(1, latent_h // z_downsample)
                    z_w = max(1, latent_w // z_downsample)
                    hyper_channels = getattr(codec.config.entropy, 'hyper_channels', 128)

                    z_bitstream = EncodedBitstream(
                        data=frame.z_data,
                        shape=(1, hyper_channels, z_h, z_w),
                        min_val=-128,
                        max_val=127,
                        num_symbols=hyper_channels * z_h * z_w,
                    )
                    logger.debug(
                        "hyperprior_available",
                        frame_idx=frame.header.frame_idx,
                        z_bytes_len=len(frame.z_data),
                        z_shape=(z_h, z_w),
                    )
                else:
                    # Log warning only once per video if no hyperprior data
                    if frame.header.frame_idx == 0:
                        logger.warning(
                            "no_hyperprior_data",
                            message="No hyperprior data in bitstream - using fallback uniform scales",
                        )

                # Decode frame using new API with z_bitstream for proper scale reconstruction
                decoded = codec.decode_frame(
                    bitstream=bitstream,
                    frame_info=frame_info,
                    scales=None,  # Let decoder reconstruct from z_bitstream
                    latent_shape=(latent_h, latent_w),
                    qp=frame.header.qp,
                    z_bitstream=z_bitstream,
                )

                # Crop to original dimensions if padding was applied
                if frame.header.padding_info is not None:
                    from src.video_compression.utils.padding import crop_to_original
                    decoded = crop_to_original(decoded, frame.header.padding_info)

                yield decoded

        # Write to output video
        num_written = write_video_frames(
            frames=frame_generator(),
            output_path=args.output,
            fps=header.fps,
            width=header.width,
            height=header.height,
            use_prores=args.prores,
        )

        ctx.checkpoint("frames_written", count=num_written)

        elapsed = time.time() - start_time
        fps = num_written / elapsed if elapsed > 0 else 0

        logger.info(
            "decoding_complete",
            frames=num_written,
            duration=round(elapsed, 2),
            fps=round(fps, 2),
            output=str(args.output),
        )

        # Compute quality metrics if reference provided
        if args.reference_video is not None:
            if not args.reference_video.exists():
                logger.warning("reference_missing", path=str(args.reference_video))
            else:
                logger.info("computing_metrics", reference=str(args.reference_video))
                metrics = compute_quality_metrics(args.output, args.reference_video)

                if metrics:
                    logger.info(
                        "quality_metrics",
                        avg_psnr=round(metrics["avg_psnr"], 2),
                        avg_ssim=round(metrics["avg_ssim"], 4),
                    )

                    # Write quality report if requested
                    if args.quality_report is not None:
                        import json

                        args.quality_report.parent.mkdir(parents=True, exist_ok=True)
                        with open(args.quality_report, "w") as f:
                            json.dump(metrics, f, indent=2)
                        logger.info("report_saved", path=str(args.quality_report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
