#!/usr/bin/env python
"""CLI for training a single video-compression zoo entry.

This script is intentionally thin: manifest parsing, codec-config
loading, logging, and exit codes live here; training logic stays in
``ZooTrainer``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.templates.logging import configure_module_logging, create_logger_class
from src.video_compression.training import ZooTrainer, build_training_config
from src.video_compression.zoo import VideoCodecZoo, load_manifest
from src.video_compression.zoo.cli_helpers import (
    load_codec_config as _load_codec_config,
)
from src.video_compression.zoo.cli_helpers import (
    load_dict as _load_dict,
)
from src.video_compression.zoo.cli_helpers import (
    override_entry as _override_entry,
)
from src.video_compression.zoo.cli_helpers import (
    resolve_codec_config_for_entry as _resolve_codec_config_for_entry,
)
from src.video_compression.zoo.cli_helpers import (
    resolve_device as _resolve_device,
)
from src.video_compression.zoo.cli_helpers import (
    resolve_entry as _resolve_entry,
)
from src.video_compression.zoo.cli_helpers import (
    resolve_path as _resolve_path,
)

_Logger = create_logger_class("train_compression_zoo_entry")

# Public re-exports for back-compat with existing imports.
__all__ = [
    "_load_dict",
    "_resolve_path",
    "_load_codec_config",
    "_resolve_entry",
    "_resolve_codec_config_for_entry",
    "_override_entry",
    "_resolve_device",
    "build_parser",
    "main",
]


def _cmd_dry_run(args: argparse.Namespace) -> int:
    logger = _Logger("cli", subcommand="dry-run")
    manifest = load_manifest(args.manifest)
    entry = _override_entry(
        _resolve_entry(manifest, args.entry_id),
        max_steps=args.max_steps,
        device=args.device,
    )
    device = _resolve_device(manifest, entry, device_override=args.device)
    codec_config = _resolve_codec_config_for_entry(
        manifest,
        entry,
        manifest_path=args.manifest,
    )
    training_config = build_training_config(entry, device=device)

    logger.info(
        "cli.dry_run.completed",
        manifest=str(args.manifest),
        entry_id=entry.entry_id,
        device=device,
        train_steps=training_config.total_steps,
        codec_name=codec_config.name,
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    logger = _Logger("cli", subcommand="train")
    manifest = load_manifest(args.manifest)
    entry = _override_entry(
        _resolve_entry(manifest, args.entry_id),
        max_steps=args.max_steps,
        device=args.device,
    )
    device = _resolve_device(manifest, entry, device_override=args.device)
    codec_config = _resolve_codec_config_for_entry(
        manifest,
        entry,
        manifest_path=args.manifest,
    )
    zoo = VideoCodecZoo(manifest.storage_root, backend=manifest.storage_backend)
    trainer = ZooTrainer(
        entry,
        zoo,
        codec_config=codec_config,
        device=device,
        output_root=args.output_root,
    )
    report = trainer.run()

    logger.info(
        "cli.train.completed",
        entry_id=report.entry_id,
        checkpoint_path=str(report.checkpoint_path),
        tolerance_passed=report.tolerance_passed,
        realized_bpp=report.realized_bpp,
        realized_psnr_db=report.realized_psnr_db,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse surface for the single-entry zoo trainer CLI."""
    parser = argparse.ArgumentParser(
        prog="train_compression_zoo_entry",
        description="Train or validate a single AlphaGalerkin codec zoo entry",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger level (default: INFO).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shorthand for --log-level DEBUG.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    dry_run_p = sub.add_parser(
        "dry-run",
        help="Validate one manifest entry and show resolved config.",
    )
    dry_run_p.add_argument("--manifest", type=Path, required=True)
    dry_run_p.add_argument("--entry-id", required=True)
    dry_run_p.add_argument("--device", default=None)
    dry_run_p.add_argument("--max-steps", type=int, default=None)
    dry_run_p.set_defaults(func=_cmd_dry_run)

    train_p = sub.add_parser("train", help="Train one zoo entry end-to-end.")
    train_p.add_argument("--manifest", type=Path, required=True)
    train_p.add_argument("--entry-id", required=True)
    train_p.add_argument("--device", default=None)
    train_p.add_argument("--max-steps", type=int, default=None)
    train_p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/video_compression/zoo"),
    )
    train_p.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for training or validating one zoo entry."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = "DEBUG" if args.debug else args.log_level
    configure_module_logging(level=level)

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised via tests via main()
    sys.exit(main())
