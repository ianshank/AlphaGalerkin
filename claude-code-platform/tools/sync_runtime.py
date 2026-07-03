"""Vendor the canonical hook runtime into every plugin (ADR-0002).

Installed plugins are cached per-directory and cannot import anything
outside their own root, so ``tools/hook_runtime`` is copied verbatim into
``<plugin>/hooks/scripts/_runtime/`` for every plugin that ships hook
scripts. The copy is recursive and content-complete (any suffix, any
depth; bytecode caches excluded), strays at any depth are removed, and a
symlinked ``_runtime`` is replaced with a real directory — mirroring
exactly what the ``vendored-runtime-parity`` gate enforces.

Usage:
    python -m tools.sync_runtime --write   # vendor / refresh copies
    python -m tools.sync_runtime --check   # CI mode: exit 1 on drift
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .logging_config import configure_tool_logging, get_tool_logger
from .validate.config import ValidatorConfig
from .validate.gates import (
    discover_plugin_dirs,
    gate_vendored_runtime,
    plugin_hook_scripts,
    relative_file_map,
)

EXIT_CLEAN = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2

log = get_tool_logger("tools.sync_runtime")


def sync_plugin(config: ValidatorConfig, plugin_dir: Path) -> list[str]:
    """Vendor the canonical runtime into one plugin; returns changed relpaths.

    Raises:
        ValueError: when the canonical runtime is missing or empty — the
            check runs BEFORE any mutation, so a broken source tree can
            never wipe a plugin's existing vendored copy.
    """
    canonical_dir = config.root / config.runtime_src_relpath
    canonical = {
        rel: path.read_bytes() for rel, path in relative_file_map(canonical_dir).items()
    }
    if not canonical:
        raise ValueError(
            f"canonical hook runtime at {canonical_dir} is missing or empty; "
            "refusing to sync (would delete vendored copies)"
        )
    vendored_dir = plugin_dir / config.vendored_runtime_relpath
    changed: list[str] = []
    if vendored_dir.is_symlink():
        vendored_dir.unlink()
        changed.append("removed:symlink:_runtime")
    vendored_dir.mkdir(parents=True, exist_ok=True)
    # Symlinks anywhere in the vendored tree are gate violations, and
    # relative_file_map deliberately skips them — remove them up front so
    # write mode always repairs what the parity gate flags.
    for stray_link in sorted(vendored_dir.rglob("*")):
        if stray_link.is_symlink():
            rel = stray_link.relative_to(vendored_dir).as_posix()
            stray_link.unlink()
            changed.append(f"removed:symlink:{rel}")
    for rel, content in canonical.items():
        target = vendored_dir / rel
        if not target.is_file() or target.read_bytes() != content:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            changed.append(rel)
    for rel in sorted(set(relative_file_map(vendored_dir)) - set(canonical)):
        (vendored_dir / rel).unlink()
        changed.append(f"removed:{rel}")
    _remove_cache_and_empty_dirs(vendored_dir)
    return changed


def _remove_cache_and_empty_dirs(vendored_dir: Path) -> None:
    for path in sorted(vendored_dir.rglob("*"), reverse=True):
        if path.is_dir() and (path.name == "__pycache__" or not any(path.iterdir())):
            shutil.rmtree(path, ignore_errors=True)


def has_hook_scripts(config: ValidatorConfig, plugin_dir: Path) -> bool:
    vendored_dir = plugin_dir / config.vendored_runtime_relpath
    return any(
        not path.is_relative_to(vendored_dir)
        for path in plugin_hook_scripts(config, plugin_dir)
    )


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
        if not has_hook_scripts(config, plugin_dir):
            log.info("plugin_skipped_no_hooks", plugin=plugin_dir.name)
            continue
        try:
            changed = sync_plugin(config, plugin_dir)
        except ValueError as exc:
            log.error("sync_runtime_refused", plugin=plugin_dir.name, error=str(exc))
            return EXIT_USAGE
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

    configure_tool_logging()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    if not root.is_dir():
        log.error("root_not_found", root=str(root))
        return EXIT_USAGE
    return run(ValidatorConfig(root=root), check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
