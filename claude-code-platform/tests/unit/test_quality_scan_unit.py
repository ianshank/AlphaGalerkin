"""White-box unit tests for the quality_scan hook (direct module import).

The subprocess contract tests prove the process-level behavior; these
import the script as a module so its logic is measurable by coverage and
each branch is testable in isolation.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import SECRET_LINE, load_quality_scan
from tools.hook_runtime.constants import EXIT_BLOCK, EXIT_OK
from tools.hook_runtime.models import parse_hook_input

quality_scan = load_quality_scan()

SECRET_PATTERNS = {"secret": ["AKIA[0-9A-Z]{16}"]}


@pytest.fixture()
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test.quality_scan_unit")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture()
def real_plugin_env(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    plugin_root = repo_root / "plugins" / "eng-standards"
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(plugin_root))
    return plugin_root


def event(payload: dict[str, Any]) -> Any:
    return parse_hook_input(json.dumps(payload))


def feed_stdin(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


class TestGatherText:
    def test_prefers_inline_content_over_disk(self, tmp_path: Path) -> None:
        on_disk = tmp_path / "f.py"
        on_disk.write_text("disk\n", encoding="utf-8")
        hook_event = event(
            {
                "hook_event_name": "PostToolUse",
                "tool_input": {"file_path": str(on_disk), "content": "inline"},
            }
        )
        assert quality_scan.gather_text(hook_event, 1024) == ("inline", "tool_input")

    def test_concatenates_multiple_content_keys(self) -> None:
        hook_event = event(
            {
                "hook_event_name": "PostToolUse",
                "tool_input": {"content": "a", "new_string": "b"},
            }
        )
        text, origin = quality_scan.gather_text(hook_event, 1024)
        assert text == "a\nb" and origin == "tool_input"

    def test_file_fallback(self, tmp_path: Path) -> None:
        on_disk = tmp_path / "f.py"
        on_disk.write_text("disk\n", encoding="utf-8")
        hook_event = event(
            {
                "hook_event_name": "PostToolUse",
                "tool_input": {"file_path": str(on_disk)},
            }
        )
        assert quality_scan.gather_text(hook_event, 1024) == ("disk\n", "file")

    def test_oversized_file_not_read(self, tmp_path: Path) -> None:
        on_disk = tmp_path / "f.py"
        on_disk.write_text("x" * 100, encoding="utf-8")
        hook_event = event(
            {
                "hook_event_name": "PostToolUse",
                "tool_input": {"file_path": str(on_disk)},
            }
        )
        assert quality_scan.gather_text(hook_event, 10) == ("", "none")

    def test_nul_byte_path_does_not_crash(self) -> None:
        # pathlib treats the invalid path as a non-file (no exception);
        # the guard exists for platforms/paths where stat raises instead.
        hook_event = event(
            {"hook_event_name": "PostToolUse", "tool_input": {"file_path": "a\x00b"}}
        )
        text, origin = quality_scan.gather_text(hook_event, 1024)
        assert text == "" and origin in ("none", "unreadable")

    def test_no_path_no_content(self) -> None:
        hook_event = event({"hook_event_name": "PostToolUse", "tool_input": {}})
        assert quality_scan.gather_text(hook_event, 1024) == ("", "none")


class TestScanText:
    def test_findings_carry_category_line_pattern_only(self) -> None:
        findings = quality_scan.scan_text("ok\n" + SECRET_LINE, SECRET_PATTERNS)
        assert findings == [
            {"category": "secret", "line": 2, "pattern": "AKIA[0-9A-Z]{16}"}
        ]

    def test_empty_patterns_yield_nothing(self) -> None:
        assert quality_scan.scan_text(SECRET_LINE, {}) == []

    def test_multiple_categories_and_lines(self) -> None:
        patterns = {"a": ["foo"], "b": ["bar"]}
        findings = quality_scan.scan_text("foo\nbar\nfoo bar\n", patterns)
        assert len(findings) == 4


class TestIsExcluded:
    def test_substring_match(self) -> None:
        assert quality_scan.is_excluded("/x/.git/config", ["/.git/"])

    def test_no_match(self) -> None:
        assert not quality_scan.is_excluded("/x/src/app.py", ["/.git/"])


class TestResolveGating:
    def test_default_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.delenv("CCP_GATING", raising=False)
        assert quality_scan.resolve_gating() is False

    def test_env_flag_enables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("CCP_GATING", "1")
        assert quality_scan.resolve_gating() is True

    def test_broken_tunables_file_falls_back_to_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "defaults.json").write_text("{", encoding="utf-8")
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.setenv("CCP_GATING", "true")
        assert quality_scan.resolve_gating() is True

    def test_config_file_enables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "defaults.json").write_text(
            json.dumps({"gating": True}), encoding="utf-8"
        )
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
        monkeypatch.delenv("CCP_GATING", raising=False)
        assert quality_scan.resolve_gating() is True


class TestRun:
    def test_unscanned_tool_skips(
        self,
        real_plugin_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        silent_logger: logging.Logger,
    ) -> None:
        feed_stdin(
            monkeypatch,
            {"hook_event_name": "PostToolUse", "tool_name": "Read", "tool_input": {}},
        )
        assert quality_scan.run(silent_logger) == EXIT_OK

    def test_excluded_path_skips(
        self,
        real_plugin_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        silent_logger: logging.Logger,
    ) -> None:
        feed_stdin(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/x/.git/config", "content": SECRET_LINE},
            },
        )
        assert quality_scan.run(silent_logger) == EXIT_OK

    def test_no_content_skips(
        self,
        real_plugin_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        silent_logger: logging.Logger,
    ) -> None:
        feed_stdin(
            monkeypatch,
            {"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {}},
        )
        assert quality_scan.run(silent_logger) == EXIT_OK

    def test_finding_warn_only_exits_ok(
        self,
        real_plugin_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        silent_logger: logging.Logger,
    ) -> None:
        feed_stdin(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/w/app.py", "content": SECRET_LINE},
            },
        )
        monkeypatch.delenv("CCP_GATING", raising=False)
        assert quality_scan.run(silent_logger) == EXIT_OK

    def test_finding_with_gating_blocks(
        self,
        real_plugin_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        silent_logger: logging.Logger,
    ) -> None:
        feed_stdin(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/w/app.py", "content": SECRET_LINE},
            },
        )
        monkeypatch.setenv("CCP_GATING", "1")
        assert quality_scan.run(silent_logger) == EXIT_BLOCK

    def test_effective_tunables_merge_real_plugin_defaults(
        self, real_plugin_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CCP_GATING", raising=False)
        tunables = quality_scan.effective_tunables()
        assert tunables["gating"] is False
        assert "secret" in tunables["patterns"]  # shipped defaults.json loaded
