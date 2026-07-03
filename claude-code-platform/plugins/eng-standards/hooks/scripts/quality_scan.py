#!/usr/bin/env python3
"""PostToolUse quality scan: warn-only hardcoded-literal and secret detection.

Reads the hook event from stdin, scans content written by file-editing
tools against configurable regex categories (``config/defaults.json``,
overridable via ``CCP_*`` env vars), and logs structured findings to
stderr. Matched text is NEVER logged — only category, line number, and
pattern — so a detected secret cannot leak into logs or transcripts.

Fail-safe contract: always exits 0 (warn-only) unless the ``gating``
tunable is true, in which case findings exit 2. Stdlib-only; imports the
vendored ``_runtime`` package (ADR-0002).
"""

from __future__ import annotations

import logging
import re
import sys
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

#: In-code fallbacks used when the tunables file is absent.
FALLBACK_SCAN_TOOLS: tuple[str, ...] = ("Write", "Edit", "MultiEdit", "NotebookEdit")
FALLBACK_MAX_FILE_BYTES = 1_048_576


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
        path = Path(path_value)
        try:
            if path.is_file() and path.stat().st_size <= max_file_bytes:
                return path.read_text(encoding="utf-8", errors="replace"), "file"
        except OSError:
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
    tunables = load_tunables()
    scan_tools = tunables.get("scan_tools", list(FALLBACK_SCAN_TOOLS))
    if event.tool_name not in scan_tools:
        logger.debug("quality_scan_skipped", extra={"tool_name": event.tool_name})
        return constants.EXIT_OK

    file_path = str(event.tool_input.get(FILE_PATH_KEY, ""))
    exclusions = tunables.get("exclude_path_substrings", [])
    if file_path and is_excluded(file_path, exclusions):
        logger.debug("quality_scan_excluded", extra={"file_path": file_path})
        return constants.EXIT_OK

    max_file_bytes = int(tunables.get("max_file_bytes", FALLBACK_MAX_FILE_BYTES))
    text, origin = gather_text(event, max_file_bytes)
    if not text:
        logger.debug("quality_scan_no_content", extra={"origin": origin})
        return constants.EXIT_OK

    patterns = tunables.get("patterns", {})
    findings = scan_text(text, patterns)
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
    gating = bool(tunables.get("gating", False))
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
    main_entry(run, component=COMPONENT)
