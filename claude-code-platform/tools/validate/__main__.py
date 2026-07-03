"""CLI entrypoint: ``python -m tools.validate [--root PATH] [--format ...]``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import structlog

from .config import ValidatorConfig
from .gates import run_all_gates

EXIT_CLEAN = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2


def find_repo_root(start: Path, marketplace_relpath: str) -> Path | None:
    """Walk upward from ``start`` to the directory holding the catalog."""
    for candidate in (start, *start.parents):
        if (candidate / marketplace_relpath).is_file():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools.validate",
        description="Run the marketplace static validation gates.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Marketplace repo root (default: auto-detect from cwd)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Violation output format on stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )
    log = structlog.get_logger("tools.validate")

    defaults = ValidatorConfig.model_fields["marketplace_relpath"].default
    root = args.root or find_repo_root(Path.cwd(), defaults)
    if root is None or not root.is_dir():
        log.error("root_not_found", cwd=str(Path.cwd()))
        return EXIT_USAGE

    config = ValidatorConfig(root=root.resolve())
    log.info("validation_started", root=str(config.root))
    violations = run_all_gates(config)

    if args.format == "json":
        print(json.dumps([asdict(v) for v in violations], indent=2))
    else:
        for violation in violations:
            print(f"[{violation.gate}] {violation.path}: {violation.message}")

    log.info(
        "validation_finished",
        violations=len(violations),
        clean=not violations,
    )
    return EXIT_VIOLATIONS if violations else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
