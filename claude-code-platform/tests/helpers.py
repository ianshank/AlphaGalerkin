"""Shared test helpers (single source of truth — do not duplicate in files).

Covers the three recurring needs across the suite:
- driving the real quality_scan hook as a subprocess (the exact contract
  Claude Code uses: stdin JSON, ``CLAUDE_PLUGIN_ROOT`` env, exit code);
- importing the hook script as a module for white-box unit coverage;
- small JSON read/write and violation-inspection utilities.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from tools.validate.gates import Violation

SUBTREE_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_RELPATH = Path("plugins/eng-standards")
SCRIPT_RELPATH = PLUGIN_RELPATH / "hooks" / "scripts" / "quality_scan.py"
HOOK_TIMEOUT_SECONDS = 30

#: A canned AWS-style secret line that the shipped patterns must flag.
SECRET_LINE = 'aws = "AKIA' + "A" * 16 + '"\n'


def run_hook_script(
    repo_root: Path,
    payload: Mapping[str, object] | str,
    *,
    env_overrides: dict[str, str] | None = None,
    plugin_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute the real hook script exactly as Claude Code does."""
    root = plugin_root if plugin_root is not None else repo_root / PLUGIN_RELPATH
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(repo_root / SCRIPT_RELPATH)],
        input=text,
        env={**os.environ, "CLAUDE_PLUGIN_ROOT": str(root), **(env_overrides or {})},
        capture_output=True,
        text=True,
        timeout=HOOK_TIMEOUT_SECONDS,
        check=False,
    )


def stderr_events(result: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
    """Parse the hook's JSON-lines stderr log into documents."""
    return [json.loads(line) for line in result.stderr.splitlines() if line.strip()]


def write_event(file_path: str, content: str) -> dict[str, object]:
    """A PostToolUse Write event payload."""
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": file_path, "content": content},
    }


def load_quality_scan() -> ModuleType:
    """Import the vendored hook script as a module (white-box coverage).

    Puts the scripts directory on ``sys.path`` so both ``quality_scan``
    and its plugin-local ``_runtime`` package resolve, mirroring how the
    interpreter resolves them when the script runs as a process.
    """
    scripts_dir = str(SUBTREE_ROOT / SCRIPT_RELPATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("quality_scan")


def gate_names(violations: list[Violation]) -> set[str]:
    return {violation.gate for violation in violations}


def read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
