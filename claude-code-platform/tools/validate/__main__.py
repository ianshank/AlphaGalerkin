"""CLI entrypoint: ``python -m tools.validate [--root PATH] [--format ...]``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..logging_config import configure_tool_logging, get_tool_logger
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
    configure_tool_logging()
    log = get_tool_logger("tools.validate")

    defaults = ValidatorConfig.model_fields["marketplace_relpath"].default
    root = args.root or find_repo_root(Path.cwd(), defaults)
    if root is None or not root.is_dir():
        log.error("root_not_found", cwd=str(Path.cwd()))
        return EXIT_USAGE

    config = ValidatorConfig(root=root.resolve())
    log.info("validation_started", root=str(config.root))
    started = time.perf_counter()
    violations = run_all_gates(config)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    if args.format == "json":
        print(json.dumps([asdict(v) for v in violations], indent=2))
    else:
        for violation in violations:
            print(f"[{violation.gate}] {violation.path}: {violation.message}")

    per_gate = Counter(violation.gate for violation in violations)
    if per_gate:
        # Single 'counts' field: gate names are hyphenated, and stuffing
        # them through **kwargs relies on a CPython quirk.
        log.info("gate_summary", counts=dict(sorted(per_gate.items())))
    log.info(
        "validation_finished",
        violations=len(violations),
        clean=not violations,
        elapsed_ms=elapsed_ms,
    )
    return EXIT_VIOLATIONS if violations else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
