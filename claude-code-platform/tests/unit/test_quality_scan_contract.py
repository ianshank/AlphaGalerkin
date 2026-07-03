"""Subprocess contract tests for the eng-standards quality_scan hook.

The script is executed exactly as Claude Code executes it: a fresh
``python3`` process, the hook event on stdin, ``CLAUDE_PLUGIN_ROOT`` in
the environment, JSON logs on stderr, contract expressed via exit code.
Runs against the REAL vendored script inside the plugin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

PLUGIN_RELPATH = Path("plugins/eng-standards")
SCRIPT_RELPATH = PLUGIN_RELPATH / "hooks" / "scripts" / "quality_scan.py"


def run_hook(
    repo_root: Path,
    payload: Mapping[str, object] | str,
    *,
    env_overrides: dict[str, str] | None = None,
    plugin_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    root = plugin_root if plugin_root is not None else repo_root / PLUGIN_RELPATH
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": str(root),
        **(env_overrides or {}),
    }
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(repo_root / SCRIPT_RELPATH)],
        input=text,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def stderr_events(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in result.stderr.splitlines() if line.strip()]


def write_event(file_path: str, content: str) -> dict[str, object]:
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": file_path, "content": content},
    }


def test_clean_content_exits_ok_no_findings(repo_root: Path) -> None:
    result = run_hook(repo_root, write_event("/workspace/app.py", "x = 1\n"))
    assert result.returncode == 0
    events = stderr_events(result)
    assert any(e["event"] == "quality_scan_finished" for e in events)
    assert not any(e["event"] == "quality_finding" for e in events)


def test_secret_pattern_warns_but_exits_ok(repo_root: Path) -> None:
    content = 'aws = "AKIA' + "A" * 16 + '"\n'
    result = run_hook(repo_root, write_event("/workspace/app.py", content))
    assert result.returncode == 0
    findings = [e for e in stderr_events(result) if e["event"] == "quality_finding"]
    assert findings and findings[0]["category"] == "secret"
    # Matched text must never appear in logs (secret-leak prevention).
    assert "AKIA" + "A" * 16 not in result.stderr


def test_hardcoded_path_detected(repo_root: Path) -> None:
    content = 'DATA = "/Users/someone/data.csv"\n'
    result = run_hook(repo_root, write_event("/workspace/app.py", content))
    findings = [e for e in stderr_events(result) if e["event"] == "quality_finding"]
    assert findings and findings[0]["category"] == "hardcoded_path"
    assert result.returncode == 0


def test_unscanned_tool_skipped(repo_root: Path) -> None:
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Read",
        "tool_input": {"file_path": "/Users/someone/x"},
    }
    result = run_hook(repo_root, payload, env_overrides={"CCP_DEBUG": "1"})
    assert result.returncode == 0
    assert any(e["event"] == "quality_scan_skipped" for e in stderr_events(result))


def test_excluded_path_skipped(repo_root: Path) -> None:
    event = write_event("/workspace/.git/config", 'p = "/Users/someone/x/"\n')
    result = run_hook(repo_root, event, env_overrides={"CCP_DEBUG": "1"})
    assert result.returncode == 0
    assert any(e["event"] == "quality_scan_excluded" for e in stderr_events(result))


def test_malformed_stdin_failsafe_exits_ok(repo_root: Path) -> None:
    result = run_hook(repo_root, "this is not json")
    assert result.returncode == 0
    events = stderr_events(result)
    assert any(e["event"] == "hook_failsafe_triggered" for e in events)


def test_gating_env_override_blocks_on_finding(repo_root: Path) -> None:
    content = 'aws = "AKIA' + "A" * 16 + '"\n'
    result = run_hook(
        repo_root,
        write_event("/workspace/app.py", content),
        env_overrides={"CCP_GATING": "1"},
    )
    assert result.returncode == 2


def test_gating_env_override_clean_content_still_ok(repo_root: Path) -> None:
    result = run_hook(
        repo_root,
        write_event("/workspace/app.py", "x = 1\n"),
        env_overrides={"CCP_GATING": "1"},
    )
    assert result.returncode == 0


def test_debug_env_enables_debug_logs(repo_root: Path) -> None:
    result = run_hook(
        repo_root,
        {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {}},
        env_overrides={"CCP_DEBUG": "1"},
    )
    assert any(e["level"] == "DEBUG" for e in stderr_events(result))


def test_missing_tunables_file_uses_fallbacks(repo_root: Path, tmp_path: Path) -> None:
    """A plugin root without config/defaults.json still scans nothing but runs."""
    result = run_hook(
        repo_root,
        write_event("/workspace/app.py", "x = 1\n"),
        plugin_root=tmp_path,
    )
    assert result.returncode == 0
    assert any(e["event"] == "quality_scan_finished" for e in stderr_events(result))


def test_edit_tool_new_string_scanned(repo_root: Path) -> None:
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "/workspace/app.py",
            "old_string": "x",
            "new_string": 'token = "AKIA' + "B" * 16 + '"',
        },
    }
    result = run_hook(repo_root, payload)
    findings = [e for e in stderr_events(result) if e["event"] == "quality_finding"]
    assert findings and findings[0]["origin"] == "tool_input"


def test_file_fallback_when_no_inline_content(repo_root: Path, tmp_path: Path) -> None:
    target = tmp_path / "written.py"
    target.write_text('key = "/home/someone/secrets/"\n', encoding="utf-8")
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "NotebookEdit",
        "tool_input": {"file_path": str(target)},
    }
    result = run_hook(repo_root, payload)
    findings = [e for e in stderr_events(result) if e["event"] == "quality_finding"]
    assert findings and findings[0]["origin"] == "file"
