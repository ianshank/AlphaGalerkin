"""Command-line interface for AlphaGalerkin."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin - Resolution-independent Go AI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # GTP command
    gtp_parser = subparsers.add_parser("gtp", help="Start GTP engine")
    gtp_parser.add_argument("--model", type=str, help="Path to model checkpoint")
    gtp_parser.add_argument("--board-size", type=int, default=19, help="Board size")
    gtp_parser.add_argument("--device", type=str, default="cpu", help="Device")

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify resolution invariance"
    )
    verify_parser.add_argument("--train-size", type=int, default=9)
    verify_parser.add_argument("--infer-size", type=int, default=19)
    verify_parser.add_argument("--device", type=str, default="cpu")

    # Colab generation command
    subparsers.add_parser(
        "generate-colab", help="Generate Colab-compatible notebook"
    )

    args = parser.parse_args()

    if args.command == "gtp":
        from src.tools.gtp import main as gtp_main

        sys.argv = ["gtp"]
        if args.model:
            sys.argv.extend(["--model", args.model])
        if args.board_size:
            sys.argv.extend(["--board-size", str(args.board_size)])
        if args.device:
            sys.argv.extend(["--device", args.device])
        gtp_main()

    elif args.command == "verify":
        from src.tools.verify_invariance import run_verification

        passed = run_verification(
            train_size=args.train_size,
            infer_size=args.infer_size,
            device=args.device,
        )
        sys.exit(0 if passed else 1)

    elif args.command == "generate-colab":
        from src.tools.colab import generate_colab_notebook

        generate_colab_notebook()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
