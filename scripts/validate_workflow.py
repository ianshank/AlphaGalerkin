#!/usr/bin/env python
"""Validation script for AlphaGalerkin video compression workflow.

This script executes the full pipeline:
1. Detects available model checkpoint.
2. Encodes a test video.
3. Decodes the bitstream.
4. Validates output quality and format.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.templates.logging import (
    configure_module_logging,
    create_logger_class,
)

# Configure logging
configure_module_logging(level="INFO")
Logger = create_logger_class("validate_workflow")
logger = Logger("main")


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command and log it."""
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info("running_command", command=cmd_str)

    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(
            "command_failed",
            command=cmd_str,
            returncode=e.returncode,
            stderr=e.stderr.strip(),
        )
        raise


def validate_video_file(path: Path) -> dict[str, Any]:
    """Get video properties using FFmpeg/ffprobe if available, or basic checks.

    For now, we rely on file existence and size.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    return {
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
    }


def main() -> int:
    """Main validation workflow."""
    parser = argparse.ArgumentParser(description="Validate video compression workflow")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_outputs"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=32,
        help="Quantization parameter",
    )
    args = parser.parse_args()

    # 1. Locate Checkpoint
    checkpoint_file = None
    if args.checkpoint_dir.exists():
        # Prefer 'best.pt' or 'final.pt'
        candidates = list(args.checkpoint_dir.rglob("*.pt"))
        for cand in candidates:
            if "best" in cand.name or "final" in cand.name:
                checkpoint_file = cand
                break

        # Fallback to any .pt
        if not checkpoint_file and candidates:
            checkpoint_file = candidates[0]

    if not checkpoint_file:
        logger.error("no_checkpoint_found", search_dir=str(args.checkpoint_dir))
        # For validation purposes without a trained model, we might fail here
        # But if the user wants to test just the plumbing, we might need a dummy model
        # For now, let's fail.
        return 1

    logger.info("checkpoint_found", path=str(checkpoint_file))

    # 2. Setup Paths
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output has same extension as input
    input_ext = args.input.suffix
    output_stem = args.input.stem

    bitstream_file = args.output_dir / f"{output_stem}_qp{args.qp}.agk"
    decoded_file = args.output_dir / f"{output_stem}_qp{args.qp}_decoded{input_ext}"
    report_file = args.output_dir / f"{output_stem}_qp{args.qp}_metrics.json"

    # 3. Encode
    logger.info("step_1_encoding")
    try:
        run_command(
            [
                sys.executable,
                "scripts/encode_video.py",
                str(args.input),
                str(bitstream_file),
                "--qp",
                str(args.qp),
                "--model",
                str(checkpoint_file),
                "--device",
                "cuda",
            ]
        )
    except subprocess.CalledProcessError:
        logger.error("encoding_failed")
        return 1

    if not bitstream_file.exists():
        logger.error("bitstream_missing", path=str(bitstream_file))
        return 1

    logger.info("encoding_success", size=bitstream_file.stat().st_size)

    # 4. Decode
    logger.info("step_2_decoding")
    try:
        run_command(
            [
                sys.executable,
                "scripts/decode_video.py",
                "--input",
                str(bitstream_file),
                "--output",
                str(decoded_file),
                "--checkpoint",
                str(checkpoint_file),
                "--device",
                "cuda",
                "--quality-report",
                str(report_file),
                "--reference-video",
                str(args.input),  # enable metric computation
            ]
        )
    except subprocess.CalledProcessError:
        logger.error("decoding_failed")
        return 1

    if not decoded_file.exists():
        logger.error("decoded_file_missing", path=str(decoded_file))
        return 1

    # Verify extension matches
    if decoded_file.suffix.lower() != input_ext.lower():
        logger.error(
            "extension_mismatch",
            expected=input_ext,
            got=decoded_file.suffix,
        )
        return 1

    logger.info("decoding_success", output=str(decoded_file))

    # 5. Check Metrics (if available)
    if report_file.exists():
        import json

        with open(report_file) as f:
            metrics = json.load(f)

        avg_psnr = metrics.get("avg_psnr", 0.0)
        logger.info("validation_metrics", psnr=avg_psnr)

        if avg_psnr < 25.0:
            logger.warning("low_quality_warning", psnr=avg_psnr)
        else:
            logger.info("quality_check_passed")
    else:
        logger.warning("metrics_file_missing")

    logger.info("workflow_validation_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
