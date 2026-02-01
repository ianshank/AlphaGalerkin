#!/usr/bin/env python
"""CLI script for encoding video with AlphaGalerkin codec.

Usage:
    python scripts/encode_video.py input.mp4 output.agk --qp 32
    python scripts/encode_video.py input.mp4 output.agk --lambda-rd 0.01
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch

from src.video_compression.config import CodecConfig
from src.video_compression.codec.codec import create_codec
from src.video_compression.utils.bitstream import (
    BitstreamHeader,
    FrameHeader,
    EncodedFrame,
    save_bitstream,
)
from src.templates.logging import (
    configure_module_logging,
    create_logger_class,
    DebugContext,
)

# Configure logging
configure_module_logging(level="INFO")
Logger = create_logger_class("encode_video")
logger = Logger("cli")


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

    logger.info("loading_video", path=str(path))
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


def get_video_fps(path: Path) -> float:
    """Get video frame rate.

    Args:
        path: Path to video file.

    Returns:
        Frame rate (fps), defaults to 30.0 if unavailable.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30.0
    except ImportError:
        return 30.0


def serialize_latent(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes for bitstream storage.

    Args:
        tensor: Input tensor to serialize.

    Returns:
        Bytes representation of tensor.
    """
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return buffer.getvalue()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        configure_module_logging(level="DEBUG")

    logger.info("encoding_start", input=str(args.input), output=str(args.output))

    # Check input exists
    if not args.input.exists():
        logger.error("input_not_found", path=str(args.input))
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("device_selected", device=str(device))

    with DebugContext("video_encoding", capture_memory=True) as ctx:
        # Load config
        config = CodecConfig(name="encoder")
        config.mcts.gop_size = args.gop_size

        if args.lambda_rd is not None:
            config.training.lambda_rd = args.lambda_rd

        # Create codec on target device
        codec = create_codec(config, device=str(device))
        codec.to(device)
        codec.eval()

        # Load model checkpoint if provided
        if args.model is not None:
            logger.info("loading_model", path=str(args.model))
            try:
                checkpoint = torch.load(args.model, map_location=device, weights_only=False)
                logger.info("checkpoint_debug", type=str(type(checkpoint)), keys=str(list(checkpoint.keys())) if isinstance(checkpoint, dict) else "N/A")
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint

                missing, unexpected = codec.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning("model_load_partial", missing_keys_count=len(missing))
                    logger.debug("missing_keys", keys=missing)
                if unexpected:
                    logger.warning("model_load_unexpected", unexpected_keys_count=len(unexpected))
            except Exception as e:
                logger.exception("model_load_failed", error=str(e))
                sys.exit(1)

        # Load video
        frames = load_video_frames(args.input)
        num_frames = len(frames)
        _, height, width = frames.shape[1:]
        ctx.checkpoint("video_loaded", frames=num_frames, height=height, width=width)

        # Calculate padded dimensions (codec uses 16x downsampling)
        downsample_factor = config.encoder.downsample_factor
        padded_height = ((height + downsample_factor - 1) // downsample_factor) * downsample_factor
        padded_width = ((width + downsample_factor - 1) // downsample_factor) * downsample_factor

        # Create bitstream header
        header = BitstreamHeader(
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=get_video_fps(args.input),
            gop_size=args.gop_size,
            downsample_factor=downsample_factor,
            latent_channels=config.encoder.latent_channels,
            padded_width=padded_width,
            padded_height=padded_height,
            lambda_rd=args.lambda_rd if args.lambda_rd else config.training.lambda_rd,
        )

        # Encode and write to bitstream
        logger.info("encoding_frames", count=num_frames)
        total_bits = 0.0
        total_psnr = 0.0
        encoded_frames: list[EncodedFrame] = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                frame_tensor = frame.unsqueeze(0).to(device)
                frame_info = codec.gop_manager.get_frame_info(i)

                output = codec.encode_frame(frame_tensor, frame_info)
                total_bits += output.rate

                distortion_tensor = torch.tensor(output.distortion) if not isinstance(output.distortion, torch.Tensor) else output.distortion
                psnr = 10 * torch.log10(1.0 / (distortion_tensor + 1e-10)).item()
                total_psnr += psnr

                # Serialize latent to bytes for bitstream
                latent_bytes = serialize_latent(output.latent)
                # Note: CodecOutput doesn't expose hyperprior z tensor separately
                z_bytes = b""

                # Create frame header
                frame_header = FrameHeader(
                    frame_idx=i,
                    frame_type=frame_info.frame_type,
                    data_length=len(latent_bytes),
                    qp=args.qp,
                    forward_ref_idx=frame_info.forward_ref if frame_info.forward_ref else -1,
                    backward_ref_idx=frame_info.backward_ref if frame_info.backward_ref else -1,
                )

                encoded_frame = EncodedFrame(
                    header=frame_header,
                    data=latent_bytes,
                    z_data=z_bytes,
                )
                encoded_frames.append(encoded_frame)

                if i % 10 == 0 or i == num_frames - 1:
                    logger.info(
                        "frame_encoded",
                        index=i,
                        type=frame_info.frame_type,
                        bits=int(output.rate),
                        psnr=round(psnr, 2),
                    )

        # Write bitstream to file
        bytes_written = save_bitstream(args.output, header, encoded_frames)
        ctx.checkpoint("bitstream_saved", bytes=bytes_written)

        # Summary
        avg_psnr = total_psnr / num_frames
        bpp = total_bits / (num_frames * height * width)

        logger.info(
            "encoding_complete",
            total_bits=int(total_bits),
            avg_psnr=round(avg_psnr, 2),
            avg_bpp=round(bpp, 4),
            file_size=bytes_written,
            output_path=str(args.output),
        )


if __name__ == "__main__":
    main()
