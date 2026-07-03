#!/usr/bin/env python3
"""PostToolUse quality scan: warn-only hardcoded-literal and secret detection.

Reads the hook event from stdin, scans content written by file-editing
tools against configurable regex categories (``config/defaults.json``,
overridable via ``CCP_*`` env vars — overrides work even when the file is
absent, via in-code fallbacks), and logs structured findings to stderr.
Matched text is NEVER logged — only category, line number, and pattern —
so a detected secret cannot leak into logs or transcripts.

Gating contract: exits 0 (warn-only) by default. When the ``gating``
tunable is true, findings exit 2 AND unexpected crashes fail CLOSED
(exit 2) — the gating flag is re-resolved at crash time by the fail-safe
wrapper. Stdlib-only; imports the vendored ``_runtime`` package
(ADR-0002).
"""

from __future__ import annotations

import logging
import re
import sys
from os import environ
from pathlib import Path
from typing import Any

from _runtime import constants
from _runtime.failsafe import main_entry
from _runtime.models import HookInput, parse_hook_input
from _runtime.tunables import load_tunables

COMPONENT = "eng-standards.quality_scan"

#: tool_input keys that may carry newly written content, in priority order.
CONTENT_KEYS: tuple[str, ...] = ("content", "new_string", "new_source")
FILE_PATH_KEY = "file_path"
GATING_KEY = "gating"

#: In-code fallbacks; the shipped config/defaults.json overrides these and
#: CCP_* env vars override both (types are coerced against these values).
FALLBACK_TUNABLES: dict[str, Any] = {
    "scan_tools": ["Write", "Edit", "MultiEdit", "NotebookEdit"],
    "max_file_bytes": 1_048_576,
    GATING_KEY: False,
    "patterns": {},
    "exclude_path_substrings": ["/.git/", "/_runtime/", "/node_modules/"],
}


def effective_tunables() -> dict[str, Any]:
    return load_tunables(fallbacks=FALLBACK_TUNABLES)


def resolve_gating() -> bool:
    """Crash-time gating resolution for the fail-safe wrapper.

    Never raises: falls back to the raw ``CCP_GATING`` env flag when the
    tunables file is unreadable, and to False when nothing is set.
    """
    try:
        return bool(effective_tunables().get(GATING_KEY, False))
    except Exception:
        raw = environ.get(constants.ENV_PREFIX + GATING_KEY.upper(), "")
        return raw.strip().lower() in constants.TRUTHY_VALUES


def gather_text(event: HookInput, max_file_bytes: int) -> tuple[str, str]:
    """Return ``(text, origin)`` to scan; prefers in-payload content over disk."""
    chunks = [
        value
        for key in CONTENT_KEYS
        if isinstance((value := event.tool_input.get(key)), str) and value
    ]
    if chunks:
        return "\n".join(chunks), "tool_input"
    path_value = event.tool_input.get(FILE_PATH_KEY)
    if isinstance(path_value, str) and path_value:
        try:
            path = Path(path_value)
            if path.is_file() and path.stat().st_size <= max_file_bytes:
                return path.read_text(encoding="utf-8", errors="replace"), "file"
        except (OSError, ValueError):
            return "", "unreadable"
    return "", "none"


def scan_text(text: str, patterns: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Scan line-by-line; findings carry category/line/pattern, never content."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for category, expressions in patterns.items():
        for expression in expressions:
            compiled.append((category, re.compile(expression)))
    findings: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for category, pattern in compiled:
            if pattern.search(line):
                findings.append(
                    {"category": category, "line": lineno, "pattern": pattern.pattern}
                )
    return findings


def is_excluded(path_value: str, exclusions: list[str]) -> bool:
    return any(substring in path_value for substring in exclusions)


def run(logger: logging.Logger) -> int:
    event = parse_hook_input(sys.stdin)
    tunables = effective_tunables()
    scan_tools = tunables["scan_tools"]  # always present via FALLBACK_TUNABLES
    if event.tool_name not in scan_tools:
        logger.debug("quality_scan_skipped", extra={"tool_name": event.tool_name})
        return constants.EXIT_OK

    file_path = str(event.tool_input.get(FILE_PATH_KEY, ""))
    exclusions = tunables["exclude_path_substrings"]
    if file_path and is_excluded(file_path, exclusions):
        logger.debug("quality_scan_excluded", extra={"file_path": file_path})
        return constants.EXIT_OK

    max_file_bytes = int(tunables["max_file_bytes"])
    text, origin = gather_text(event, max_file_bytes)
    if not text:
        logger.debug("quality_scan_no_content", extra={"origin": origin})
        return constants.EXIT_OK

    findings = scan_text(text, tunables["patterns"])
    for finding in findings:
        logger.warning(
            "quality_finding",
            extra={
                "file_path": file_path,
                "tool_name": event.tool_name,
                "origin": origin,
                **finding,
            },
        )
    gating = bool(tunables[GATING_KEY])
    logger.info(
        "quality_scan_finished",
        extra={
            "file_path": file_path,
            "finding_count": len(findings),
            "gating": gating,
        },
    )
    if findings and gating:
        return constants.EXIT_BLOCK
    return constants.EXIT_OK


if __name__ == "__main__":
    main_entry(run, component=COMPONENT, gating=resolve_gating)
