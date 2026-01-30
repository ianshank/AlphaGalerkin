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
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
) -> int:
    """Write decoded frames to video file.

    Args:
        frames: Iterator of decoded frame tensors (1, 3, H, W).
        output_path: Output video path.
        fps: Frame rate.
        width: Frame width.
        height: Frame height.

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
        ".mov": "mp4v",
        ".mkv": "XVID",
        ".webm": "VP80",
    }

    fourcc_code = fourcc_map.get(ext, "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        logger.error("Failed to open video writer for %s", output_path)
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
        logger.error("OpenCV/NumPy not installed")
        return {}

    # Import quality metrics
    try:
        from src.video_compression.metrics.quality import compute_psnr, compute_ssim
    except ImportError:
        logger.warning("Quality metrics not available")
        return {}

    decoded_cap = cv2.VideoCapture(str(decoded_path))
    reference_cap = cv2.VideoCapture(str(reference_path))

    if not decoded_cap.isOpened() or not reference_cap.isOpened():
        logger.error("Failed to open video files for comparison")
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
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    # Validate checkpoint
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    # Get device
    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # Import codec modules
    try:
        from src.video_compression.codec import load_codec
        from src.video_compression.utils.bitstream import (
            load_bitstream,
            BitstreamHeader,
        )
    except ImportError as e:
        logger.error("Failed to import codec modules: %s", e)
        return 1

    # Load bitstream
    logger.info("Loading bitstream from %s", args.input)
    try:
        header, encoded_frames = load_bitstream(args.input)
    except Exception as e:
        logger.error("Failed to load bitstream: %s", e)
        return 1

    logger.info(
        "Bitstream info: %dx%d, %d frames, %.2f fps",
        header.width,
        header.height,
        header.num_frames,
        header.fps,
    )

    if args.dry_run:
        logger.info("Dry run complete - bitstream is valid")
        return 0

    # Load codec
    logger.info("Loading codec from %s", args.checkpoint)
    try:
        codec = load_codec(args.checkpoint, device=device)
    except Exception as e:
        logger.error("Failed to load codec: %s", e)
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Decode frames
    logger.info("Decoding %d frames to %s", header.num_frames, args.output)
    start_time = time.time()

    def frame_generator():
        """Generate decoded frames."""
        from src.video_compression.codec.gop_manager import FrameInfo, FrameType
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

            # Reconstruct bitstream object from raw bytes
            bitstream = EncodedBitstream(data=frame.data, symbols=None)

            # Compute latent shape from header dimensions
            latent_h = header.padded_height // header.downsample_factor
            latent_w = header.padded_width // header.downsample_factor

            # Create scales tensor (uniform scales as placeholder)
            # In a full implementation, scales would be stored in z_data
            scales = torch.ones(1, header.latent_channels, latent_h, latent_w)

            # Decode frame
            decoded = codec.decode_frame(
                bitstream=bitstream,
                frame_info=frame_info,
                scales=scales,
                latent_shape=(latent_h, latent_w),
                qp=frame.header.qp,
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
    )

    elapsed = time.time() - start_time
    fps = num_written / elapsed if elapsed > 0 else 0

    logger.info(
        "Decoded %d frames in %.2fs (%.2f fps)",
        num_written,
        elapsed,
        fps,
    )

    # Compute quality metrics if reference provided
    if args.reference_video is not None:
        if not args.reference_video.exists():
            logger.warning("Reference video not found: %s", args.reference_video)
        else:
            logger.info("Computing quality metrics against %s", args.reference_video)
            metrics = compute_quality_metrics(args.output, args.reference_video)

            if metrics:
                logger.info("Quality metrics:")
                logger.info("  Avg PSNR: %.2f dB", metrics["avg_psnr"])
                logger.info("  Avg SSIM: %.4f", metrics["avg_ssim"])

                # Write quality report if requested
                if args.quality_report is not None:
                    import json

                    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
                    with open(args.quality_report, "w") as f:
                        json.dump(metrics, f, indent=2)
                    logger.info("Quality report written to %s", args.quality_report)

    logger.info("Decoding complete: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
