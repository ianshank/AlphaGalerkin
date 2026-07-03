"""Vendor the canonical hook runtime into every plugin (ADR-0002).

Installed plugins are cached per-directory and cannot import anything
outside their own root, so ``tools/hook_runtime`` is copied verbatim into
``<plugin>/hooks/scripts/_runtime/`` for every plugin that ships hook
scripts. Copies are byte-identical (no header rewriting) so the CI parity
gate is a plain content compare.

Usage:
    python -m tools.sync_runtime --write   # vendor / refresh copies
    python -m tools.sync_runtime --check   # CI mode: exit 1 on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from .validate.config import ValidatorConfig
from .validate.gates import discover_plugin_dirs, gate_vendored_runtime

EXIT_CLEAN = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2

log = structlog.get_logger("tools.sync_runtime")


def sync_plugin(config: ValidatorConfig, plugin_dir: Path) -> list[str]:
    """Copy canonical runtime files into one plugin; returns changed names."""
    canonical_dir = config.root / config.runtime_src_relpath
    vendored_dir = plugin_dir / config.vendored_runtime_relpath
    vendored_dir.mkdir(parents=True, exist_ok=True)
    changed: list[str] = []
    canonical_files = {
        p.name: p.read_bytes() for p in sorted(canonical_dir.glob("*.py"))
    }
    for name, content in canonical_files.items():
        target = vendored_dir / name
        if not target.is_file() or target.read_bytes() != content:
            target.write_bytes(content)
            changed.append(name)
    for stray in sorted(vendored_dir.glob("*.py")):
        if stray.name not in canonical_files:
            stray.unlink()
            changed.append(f"removed:{stray.name}")
    return changed


def run(config: ValidatorConfig, *, check_only: bool) -> int:
    if check_only:
        violations = gate_vendored_runtime(config)
        for violation in violations:
            log.warning(
                "vendored_runtime_drift", path=violation.path, detail=violation.message
            )
        log.info("sync_runtime_check_finished", drift_count=len(violations))
        return EXIT_DRIFT if violations else EXIT_CLEAN

    for plugin_dir in discover_plugin_dirs(config):
        if not (plugin_dir / config.hook_scripts_relpath).is_dir():
            log.info("plugin_skipped_no_hooks", plugin=plugin_dir.name)
            continue
        changed = sync_plugin(config, plugin_dir)
        log.info("plugin_synced", plugin=plugin_dir.name, changed=changed)
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.sync_runtime", description=__doc__.splitlines()[0]
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="vendor/refresh copies")
    mode.add_argument("--check", action="store_true", help="exit 1 on drift (CI)")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    if not root.is_dir():
        log.error("root_not_found", root=str(root))
        return EXIT_USAGE
    return run(ValidatorConfig(root=root), check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
