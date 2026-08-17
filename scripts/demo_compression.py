#!/usr/bin/env python
"""MVP demo for AlphaGalerkin neural video compression.

Self-contained demonstration of the full compression pipeline:
1. Generates synthetic video (no external data required)
2. Encodes frames through the neural codec
3. Writes/reads .agk bitstream format
4. Decodes frames and computes quality metrics
5. Evaluates R-D curve at multiple lambda values
6. Tests resolution independence

Usage:
    # Quick demo with defaults
    python scripts/demo_compression.py

    # Verbose demo with custom settings
    python scripts/demo_compression.py -v --num-frames 4 --height 64 --width 64

    # Custom R-D sweep
    python scripts/demo_compression.py --lambda-values 0.005,0.01,0.02

    # Test specific patterns
    python scripts/demo_compression.py --patterns gradient,waves,motion

    # Custom resolution test
    python scripts/demo_compression.py --resolution-sizes 32x32,64x64,128x128
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.video_compression.data.synthetic import SyntheticPattern
from src.video_compression.demo.config import DemoConfig
from src.video_compression.demo.runner import CompressionDemoRunner, DemoResult

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin Neural Video Compression MVP Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_compression.py                        # Quick demo
  python scripts/demo_compression.py -v                     # Verbose
  python scripts/demo_compression.py --patterns gradient    # Single pattern
  python scripts/demo_compression.py --lambda-values 0.01   # Single lambda
        """,
    )

    # Video generation
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames per pattern (default: 8)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Frame height in pixels (default: 64)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Frame width in pixels (default: 64)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="gradient,waves",
        help=(
            "Comma-separated patterns: gradient,motion,"
            "checkerboard,waves,noise (default: gradient,waves)"
        ),
    )

    # R-D sweep
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.005,0.01,0.02,0.05",
        help="Comma-separated lambda values for R-D curve (default: 0.005,0.01,0.02,0.05)",
    )

    # Resolution test
    parser.add_argument(
        "--resolution-sizes",
        type=str,
        default="64x64,128x128",
        help="Comma-separated HxW pairs for resolution test (default: 64x64,128x128)",
    )
    parser.add_argument(
        "--resolution-lambda",
        type=float,
        default=0.01,
        help="Lambda for resolution test (default: 0.01)",
    )

    # Codec architecture
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=64,
        help="Latent channels (default: 64)",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=8,
        help="Downsample factor, must be power of 2 (default: 8)",
    )

    # Runtime
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo_compression"),
        help="Output directory (default: outputs/demo_compression)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-bitstream",
        action="store_true",
        help="Skip writing .agk bitstream files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print JSON results to stdout",
    )

    return parser.parse_args()


def parse_patterns(patterns_str: str) -> list[SyntheticPattern]:
    """Parse comma-separated pattern names.

    Args:
        patterns_str: Comma-separated pattern names.

    Returns:
        List of SyntheticPattern enum values.

    Raises:
        ValueError: If an invalid pattern name is provided.

    """
    valid_names = {p.value for p in SyntheticPattern}
    patterns = []
    for name in patterns_str.split(","):
        name = name.strip().lower()
        if name not in valid_names:
            raise ValueError(
                f"Invalid pattern '{name}'. Valid patterns: {', '.join(sorted(valid_names))}"
            )
        patterns.append(SyntheticPattern(name))
    return patterns


def parse_lambda_values(lambda_str: str) -> list[float]:
    """Parse comma-separated lambda values.

    Args:
        lambda_str: Comma-separated float values.

    Returns:
        List of lambda values.

    """
    return [float(v.strip()) for v in lambda_str.split(",")]


def parse_resolution_sizes(res_str: str) -> list[tuple[int, int]]:
    """Parse comma-separated HxW resolution pairs.

    Args:
        res_str: Comma-separated "HxW" pairs.

    Returns:
        List of (height, width) tuples.

    """
    sizes = []
    for pair in res_str.split(","):
        pair = pair.strip()
        parts = pair.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution format '{pair}'. Use HxW (e.g., 64x64)")
        sizes.append((int(parts[0]), int(parts[1])))
    return sizes


def print_summary(result: DemoResult) -> None:
    """Print a human-readable summary of demo results.

    Args:
        result: Demo results to summarize.

    """
    print("\n" + "=" * 60)
    print("  AlphaGalerkin Video Compression MVP Demo Results")
    print("=" * 60)

    # R-D Results
    if result.lambda_results:
        print("\n--- Rate-Distortion Curve ---")
        print(f"{'Pattern':<15} {'Lambda':>8} {'BPP':>10} {'PSNR(dB)':>10} {'SSIM':>8}")
        print("-" * 55)
        for lr in result.lambda_results:
            print(
                f"{lr.pattern:<15} {lr.lambda_rd:>8.4f} "
                f"{lr.avg_bpp:>10.4f} {lr.avg_psnr_db:>10.2f} {lr.avg_ssim:>8.4f}"
            )

    # Resolution Independence
    if result.resolution_results:
        print("\n--- Resolution Independence ---")
        print(f"{'Pattern':<12} {'Resolution':>12} {'PSNR(dB)':>10} {'SSIM':>8} {'BPP':>10}")
        print("-" * 55)
        for rr in result.resolution_results:
            print(
                f"{rr.pattern:<12} {rr.width}x{rr.height:>5} "
                f"{rr.avg_psnr_db:>10.2f} {rr.avg_ssim:>8.4f} {rr.avg_bpp:>10.4f}"
            )

    # Summary
    print(f"\nTotal time: {result.total_time_s:.1f}s")
    print(f"Device: {result.device}")

    # Bitstream files
    bs_files = [lr.bitstream_path for lr in result.lambda_results if lr.bitstream_path]
    if bs_files:
        print("\nBitstream files written:")
        for f in bs_files:
            print(f"  {f}")

    print("=" * 60)


def main() -> None:
    """Main entry point for demo CLI."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        patterns = parse_patterns(args.patterns)
        lambda_values = parse_lambda_values(args.lambda_values)
        resolution_sizes = parse_resolution_sizes(args.resolution_sizes)
    except ValueError as e:
        logger.error("Invalid argument: %s", e)
        sys.exit(1)

    # Build config
    try:
        config = DemoConfig(
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            patterns=patterns,
            latent_channels=args.latent_channels,
            downsample_factor=args.downsample_factor,
            lambda_values=lambda_values,
            resolution_sizes=resolution_sizes,
            resolution_lambda=args.resolution_lambda,
            device=args.device,
            seed=args.seed,
            output_dir=str(args.output_dir),
            write_bitstream=not args.no_bitstream,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Config validation failed: %s", e)
        sys.exit(1)

    # Run demo
    runner = CompressionDemoRunner(config)
    result = runner.run_full_demo()

    # Output
    if args.json_output:
        print(result.to_json())
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
