"""Regenerate the marketplace catalog from plugin manifests + release pins.

The catalog's ``plugins`` array is DOWNSTREAM of two sources of truth
(ADR-0003):

- each plugin's ``.claude-plugin/plugin.json`` (name, version, description);
- ``release/pins.json`` (released version → git sha).

Marketplace identity fields (``name``, ``owner``, ``metadata``) are
preserved from the existing catalog — they are authored, not generated.
A plugin whose manifest version matches its pin gets a self-referential
sha-pinned github source; otherwise it gets a repo-relative dev source.

Usage:
    python -m tools.sync_catalog --write   # regenerate plugins entries
    python -m tools.sync_catalog --check   # CI mode: exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import structlog

from .validate.config import ValidatorConfig
from .validate.gates import load_manifests
from .validate.schemas import PinsDocument

EXIT_CLEAN = 0
EXIT_STALE = 1
EXIT_ERROR = 2

JSON_INDENT = 2

log = structlog.get_logger("tools.sync_catalog")


def load_pins(config: ValidatorConfig) -> PinsDocument:
    path = config.root / config.pins_relpath
    if not path.is_file():
        return PinsDocument(schema_version=1)
    return PinsDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def build_plugin_entries(config: ValidatorConfig) -> list[dict[str, Any]]:
    """Deterministic (name-sorted) catalog entries from manifests + pins."""
    manifests, violations = load_manifests(config)
    if violations:
        details = "; ".join(f"{v.path}: {v.message}" for v in violations)
        raise ValueError(f"cannot regenerate catalog from invalid manifests: {details}")
    pins = load_pins(config)
    entries: list[dict[str, Any]] = []
    for name in sorted(manifests):
        manifest = manifests[name]
        pin = pins.pins.get(name)
        source: str | dict[str, Any]
        if pin is not None and pin.version == manifest.version:
            if not pins.repo:
                raise ValueError(
                    f"pin for {name!r} requires 'repo' in {config.pins_relpath}"
                )
            source = {"source": "github", "repo": pins.repo, "sha": pin.sha}
            if pin.ref:
                source["ref"] = pin.ref
        else:
            source = f"./{config.plugins_dirname}/{name}"
        entries.append(
            {
                "name": name,
                "source": source,
                "description": manifest.description,
                "version": manifest.version,
            }
        )
    return entries


def render_catalog(config: ValidatorConfig) -> str:
    path = config.root / config.marketplace_relpath
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    document["plugins"] = build_plugin_entries(config)
    return json.dumps(document, indent=JSON_INDENT, ensure_ascii=False) + "\n"


def run(config: ValidatorConfig, *, check_only: bool) -> int:
    path = config.root / config.marketplace_relpath
    try:
        rendered = render_catalog(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.error("catalog_generation_failed", error=str(exc))
        return EXIT_ERROR
    current = path.read_text(encoding="utf-8")
    if check_only:
        stale = rendered != current
        log.info("sync_catalog_check_finished", stale=stale, path=str(path))
        return EXIT_STALE if stale else EXIT_CLEAN
    if rendered != current:
        path.write_text(rendered, encoding="utf-8")
        log.info("catalog_regenerated", path=str(path))
    else:
        log.info("catalog_already_current", path=str(path))
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.sync_catalog", description=__doc__.splitlines()[0]
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate entries")
    mode.add_argument("--check", action="store_true", help="exit 1 if stale (CI)")
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
    return run(ValidatorConfig(root=root), check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
