#!/usr/bin/env python
"""CLI for sweeping every entry of a video-compression zoo manifest.

This script wires :class:`ZooSweep` (Phase 2-D) into a user-facing CLI.
It is the multi-entry counterpart to
:mod:`scripts.train_compression_zoo_entry`. Manifest parsing,
codec-config resolution, logging, and exit codes live here; sweep
orchestration stays in :class:`ZooSweep`.

Subcommands:

* ``dry-run`` — load the manifest, build a :class:`ZooSweep`, and
  print the resolved device plan plus each entry's resume verdict
  (``should_skip``). No training is executed.
* ``train`` — run the sweep end-to-end. Exits 0 iff every requested
  entry either trained successfully or was skipped on a hash match.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from src.templates.logging import configure_module_logging, create_logger_class
from src.video_compression.config import CodecConfig
from src.video_compression.zoo import VideoCodecZoo, load_manifest
from src.video_compression.zoo.cli_helpers import (
    resolve_codec_config_for_entry,
)
from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.sweep import (
    SweepReport,
    ZooSweep,
    should_skip,
)

_Logger = create_logger_class("train_compression_zoo")


def _build_codec_resolver(
    manifest: ModelZooManifestConfig,
    *,
    manifest_path: Path,
) -> Callable[[ModelZooEntryConfig], CodecConfig]:
    """Return a closure that resolves a codec config for any entry."""

    def _resolver(entry: ModelZooEntryConfig) -> CodecConfig:
        return resolve_codec_config_for_entry(
            manifest,
            entry,
            manifest_path=manifest_path,
        )

    return _resolver


def _selected_entries(
    manifest: ModelZooManifestConfig,
    only_entry_ids: list[str] | None,
) -> list[ModelZooEntryConfig]:
    """Return manifest entries filtered by the optional allow-list."""
    if not only_entry_ids:
        return list(manifest.entries)
    allow = set(only_entry_ids)
    return [e for e in manifest.entries if e.entry_id in allow]


def _cmd_dry_run(args: argparse.Namespace) -> int:
    logger = _Logger("cli", subcommand="dry-run")
    manifest = load_manifest(args.manifest)
    zoo = VideoCodecZoo(manifest.storage_root, backend=manifest.storage_backend)

    only = args.only_entry_id or None
    sweep = ZooSweep(
        manifest,
        zoo,
        codec_config_for=_build_codec_resolver(
            manifest,
            manifest_path=args.manifest,
        ),
        output_root=args.output_root,
        only_entry_ids=only,
    )

    plan = sweep.plan
    for entry in _selected_entries(manifest, only):
        device = plan.device_for(entry.entry_id)
        skip, reason = should_skip(zoo, entry)
        logger.info(
            "cli.dry_run.entry",
            entry_id=entry.entry_id,
            device=device,
            would_skip=skip,
            skip_reason=reason,
            lambda_rd=entry.lambda_rd,
        )

    logger.info(
        "cli.dry_run.completed",
        manifest=str(args.manifest),
        manifest_name=manifest.name,
        strategy=plan.strategy.value,
        total_entries=len(_selected_entries(manifest, only)),
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    logger = _Logger("cli", subcommand="train")
    manifest = load_manifest(args.manifest)
    zoo = VideoCodecZoo(manifest.storage_root, backend=manifest.storage_backend)

    only = args.only_entry_id or None
    sweep = ZooSweep(
        manifest,
        zoo,
        codec_config_for=_build_codec_resolver(
            manifest,
            manifest_path=args.manifest,
        ),
        output_root=args.output_root,
        only_entry_ids=only,
    )

    report: SweepReport = sweep.run()

    logger.info(
        "cli.train.completed",
        manifest=str(args.manifest),
        manifest_name=report.manifest_name,
        total=report.total,
        trained=report.trained,
        skipped=report.skipped,
        failed=report.failed,
    )
    return 0 if report.failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse surface for the multi-entry zoo trainer CLI."""
    parser = argparse.ArgumentParser(
        prog="train_compression_zoo",
        description="Sweep every entry of an AlphaGalerkin codec zoo manifest.",
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
        help="Resolve manifest + plan and report skip verdicts. No training.",
    )
    dry_run_p.add_argument("--manifest", type=Path, required=True)
    dry_run_p.add_argument(
        "--only-entry-id",
        action="append",
        default=None,
        help="Restrict the sweep to these entry_ids (repeatable).",
    )
    dry_run_p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/video_compression/zoo"),
    )
    dry_run_p.set_defaults(func=_cmd_dry_run)

    train_p = sub.add_parser(
        "train",
        help="Run every (selected) entry of the manifest end-to-end.",
    )
    train_p.add_argument("--manifest", type=Path, required=True)
    train_p.add_argument(
        "--only-entry-id",
        action="append",
        default=None,
        help="Restrict the sweep to these entry_ids (repeatable).",
    )
    train_p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/video_compression/zoo"),
    )
    train_p.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for sweeping a manifest."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = "DEBUG" if args.debug else args.log_level
    configure_module_logging(level=level)

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    sys.exit(main())
