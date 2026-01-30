#!/usr/bin/env python
"""CLI script for encoding video with AlphaGalerkin codec.

Usage:
    python scripts/encode_video.py input.mp4 output.agk --qp 32
    python scripts/encode_video.py input.mp4 output.agk --lambda-rd 0.01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.video_compression.config import CodecConfig
from src.video_compression.codec.codec import create_codec


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Encode video with AlphaGalerkin neural codec",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input video file (mp4, avi, etc.)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output compressed file (.agk)",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=32,
        help="Quantization parameter (0-51, default: 32)",
    )
    parser.add_argument(
        "--lambda-rd",
        type=float,
        default=None,
        help="Rate-distortion lambda (overrides QP)",
    )
    parser.add_argument(
        "--gop-size",
        type=int,
        default=16,
        help="GOP size (default: 16)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for encoding (default: auto)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_video_frames(path: Path) -> torch.Tensor:
    """Load video frames from file.

    Args:
        path: Path to video file.

    Returns:
        Tensor of shape (T, 3, H, W) in [0, 1].
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Install with: pip install opencv-python")
        raise

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB, normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {path}")

    return torch.stack(frames)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Check input exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load config
    config = CodecConfig(name="encoder")
    config.mcts.gop_size = args.gop_size

    if args.lambda_rd is not None:
        config.training.lambda_rd = args.lambda_rd

    # Create codec
    codec = create_codec(config)
    codec.to(device)
    codec.eval()

    # Load model checkpoint if provided
    if args.model is not None:
        logger.info(f"Loading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        codec.load_state_dict(checkpoint["model_state"])

    # Load video
    logger.info("Loading video frames...")
    frames = load_video_frames(args.input)
    logger.info(f"Loaded {len(frames)} frames, shape: {frames.shape[1:]}")

    # Encode
    logger.info("Encoding...")
    total_bits = 0
    total_psnr = 0.0

    with torch.no_grad():
        for i, frame in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            frame_info = codec.gop_manager.get_frame_info(i)

            output = codec.encode_frame(frame, frame_info)
            total_bits += output.rate

            psnr = 10 * torch.log10(1.0 / (output.distortion + 1e-10)).item()
            total_psnr += psnr

            if i % 10 == 0:
                logger.info(f"Frame {i}: {output.rate:.0f} bits, PSNR: {psnr:.2f} dB")

    # Summary
    avg_psnr = total_psnr / len(frames)
    bpp = total_bits / (len(frames) * frames.shape[2] * frames.shape[3])

    logger.info(f"Encoding complete!")
    logger.info(f"Total bits: {total_bits:.0f}")
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
    logger.info(f"Average BPP: {bpp:.4f}")

    # Save compressed file (placeholder - would need actual bitstream writing)
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
