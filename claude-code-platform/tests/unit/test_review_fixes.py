"""Regression tests for the adversarial-review findings (F1-F7).

Each test reproduces a bypass/failure the original green build missed and
pins the fix. Naming: F<n> matches the review report ordering.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from tests.helpers import (
    SECRET_LINE,
    gate_names,
)
from tests.helpers import (
    run_hook_script as run_hook,
)
from tools.hook_runtime import constants
from tools.hook_runtime.failsafe import run_failsafe
from tools.hook_runtime.tunables import load_tunables
from tools.sync_runtime import sync_plugin
from tools.validate.config import ValidatorConfig
from tools.validate.gates import run_all_gates


def write_event(content: str) -> dict[str, object]:
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": "/workspace/app.py", "content": content},
    }


class TestF1GatingFailsClosed:
    """A gating hook must fail CLOSED (exit 2) on unexpected crashes."""

    def test_malformed_stdin_with_gating_blocks(self, repo_root: Path) -> None:
        result = run_hook(repo_root, "not json", env_overrides={"CCP_GATING": "1"})
        assert result.returncode == constants.EXIT_BLOCK

    def test_broken_env_override_with_gating_blocks(self, repo_root: Path) -> None:
        result = run_hook(
            repo_root,
            write_event("x = 1\n"),
            env_overrides={"CCP_GATING": "1", "CCP_MAX_FILE_BYTES": "abc"},
        )
        assert result.returncode == constants.EXIT_BLOCK

    def test_bad_pattern_with_gating_blocks(self, repo_root: Path) -> None:
        result = run_hook(
            repo_root,
            write_event("x = 1\n"),
            env_overrides={"CCP_GATING": "1", "CCP_PATTERNS": '{"secret": ["("]}'},
        )
        assert result.returncode == constants.EXIT_BLOCK

    def test_crash_without_gating_still_warns_only(self, repo_root: Path) -> None:
        result = run_hook(repo_root, "not json", env_overrides={})
        assert result.returncode == constants.EXIT_OK

    def test_failsafe_log_reports_effective_gating(self, repo_root: Path) -> None:
        result = run_hook(repo_root, "not json", env_overrides={"CCP_GATING": "1"})
        events = [json.loads(line) for line in result.stderr.splitlines()]
        triggered = [e for e in events if e["event"] == "hook_failsafe_triggered"]
        assert triggered and triggered[0]["gating"] is True

    def test_callable_gating_resolver_used_on_crash(self) -> None:
        def boom(_log: logging.Logger) -> int:
            raise RuntimeError("crash")

        assert (
            run_failsafe(boom, component="t.g", gating=lambda: True)
            == constants.EXIT_BLOCK
        )

    def test_raising_resolver_falls_back_to_warn_only(self) -> None:
        def boom(_log: logging.Logger) -> int:
            raise RuntimeError("crash")

        def bad_resolver() -> bool:
            raise ValueError("broken config")

        assert (
            run_failsafe(boom, component="t.g2", gating=bad_resolver)
            == constants.EXIT_OK
        )


class TestF2HookScriptsOutsideScriptsDir:
    """Hook entry points anywhere in the plugin are gated."""

    def test_nonstdlib_import_outside_scripts_dir_flagged(
        self, synthetic_marketplace: Path
    ) -> None:
        plugin = synthetic_marketplace / "plugins" / "demo-plugin"
        (plugin / "hooks" / "evil.py").write_text("import requests\n", encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert "stdlib-imports" in gate_names(violations)

    def test_command_referencing_missing_script_flagged(
        self, synthetic_marketplace: Path
    ) -> None:
        plugin = synthetic_marketplace / "plugins" / "demo-plugin"
        hooks_json = plugin / "hooks" / "hooks.json"
        document = json.loads(hooks_json.read_text(encoding="utf-8"))
        document["hooks"]["PostToolUse"][0]["hooks"][0]["command"] = (
            'python3 "${CLAUDE_PLUGIN_ROOT}/hooks/ghost.py"'
        )
        hooks_json.write_text(json.dumps(document, indent=2), encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert any(
            "ghost.py" in v.message for v in violations if v.gate == "path-literals"
        )

    def test_command_referenced_script_outside_hooks_is_scanned(
        self, synthetic_marketplace: Path
    ) -> None:
        plugin = synthetic_marketplace / "plugins" / "demo-plugin"
        (plugin / "bin").mkdir()
        (plugin / "bin" / "extra.py").write_text("import numpy\n", encoding="utf-8")
        hooks_json = plugin / "hooks" / "hooks.json"
        document = json.loads(hooks_json.read_text(encoding="utf-8"))
        document["hooks"]["PostToolUse"][0]["hooks"].append(
            {
                "type": "command",
                "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/bin/extra.py"',
            }
        )
        hooks_json.write_text(json.dumps(document, indent=2), encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert any(
            "numpy" in v.message for v in violations if v.gate == "stdlib-imports"
        )


class TestF3RecursiveVendoredParity:
    """Nested / non-.py content inside _runtime is drift."""

    def test_nested_stray_file_flagged(self, synthetic_marketplace: Path) -> None:
        vendored = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "hooks"
            / "scripts"
            / "_runtime"
        )
        (vendored / "vendor").mkdir()
        (vendored / "vendor" / "payload.py").write_text("x = 1\n", encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert any(
            "payload.py" in v.path
            for v in violations
            if v.gate == "vendored-runtime-parity"
        )

    def test_non_py_stray_flagged(self, synthetic_marketplace: Path) -> None:
        vendored = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "hooks"
            / "scripts"
            / "_runtime"
        )
        (vendored / "extra.txt").write_text("data\n", encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert any(
            "extra.txt" in v.path
            for v in violations
            if v.gate == "vendored-runtime-parity"
        )

    def test_sync_removes_nested_stray(self, synthetic_marketplace: Path) -> None:
        config = ValidatorConfig(root=synthetic_marketplace)
        plugin = synthetic_marketplace / "plugins" / "demo-plugin"
        vendored = plugin / "hooks" / "scripts" / "_runtime"
        (vendored / "vendor").mkdir()
        (vendored / "vendor" / "payload.py").write_text("x = 1\n", encoding="utf-8")
        changed = sync_plugin(config, plugin)
        assert "removed:vendor/payload.py" in changed
        assert not (vendored / "vendor").exists()


class TestF4SymlinkedRuntime:
    """A symlinked _runtime must be rejected and repaired."""

    def test_symlinked_runtime_dir_flagged(self, synthetic_marketplace: Path) -> None:
        plugin = synthetic_marketplace / "plugins" / "demo-plugin"
        vendored = plugin / "hooks" / "scripts" / "_runtime"
        shutil.rmtree(vendored)
        vendored.symlink_to(synthetic_marketplace / "tools" / "hook_runtime")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert any(
            "symlink" in v.message
            for v in violations
            if v.gate == "vendored-runtime-parity"
        )

    def test_sync_replaces_symlinked_runtime(self, synthetic_marketplace: Path) -> None:
        config = ValidatorConfig(root=synthetic_marketplace)
        plugin = synthetic_marketplace / "plugins" / "demo-plugin"
        vendored = plugin / "hooks" / "scripts" / "_runtime"
        shutil.rmtree(vendored)
        vendored.symlink_to(synthetic_marketplace / "tools" / "hook_runtime")
        sync_plugin(config, plugin)
        assert not vendored.is_symlink() and vendored.is_dir()
        assert run_all_gates(config) == []


class TestF5DynamicImports:
    """importlib / __import__ are banned in hook scripts."""

    @pytest.mark.parametrize(
        "code",
        [
            "import importlib\nimportlib.import_module('numpy')\n",
            "mod = __import__('requests')\n",
            "from importlib import import_module\n",
        ],
    )
    def test_dynamic_import_flagged(
        self, synthetic_marketplace: Path, code: str
    ) -> None:
        script = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "hooks"
            / "scripts"
            / "demo_hook.py"
        )
        script.write_text(code, encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert "stdlib-imports" in gate_names(violations)


class TestF6PathLiteralCoverage:
    """Trailing-slash-less paths and extensionless files are caught."""

    def test_home_path_without_trailing_slash_flagged(
        self, synthetic_marketplace: Path
    ) -> None:
        skill = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "skills"
            / "demo-skill"
            / "SKILL.md"
        )
        skill.write_text(
            skill.read_text(encoding="utf-8") + '\npath = "/home/someuser"\n',
            encoding="utf-8",
        )
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert "path-literals" in gate_names(violations)

    def test_tilde_file_path_flagged(self, synthetic_marketplace: Path) -> None:
        skill = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "skills"
            / "demo-skill"
            / "SKILL.md"
        )
        skill.write_text(
            skill.read_text(encoding="utf-8") + "\nsource ~/secrets.env\n",
            encoding="utf-8",
        )
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert "path-literals" in gate_names(violations)

    def test_extensionless_file_scanned(self, synthetic_marketplace: Path) -> None:
        runner = synthetic_marketplace / "plugins" / "demo-plugin" / "hooks" / "runner"
        runner.write_text('#!/bin/sh\nDATA="/Users/someone/data"\n', encoding="utf-8")
        violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
        assert "path-literals" in gate_names(violations)


class TestF7EnvOverridesWithoutDefaultsFile:
    """CCP_* overrides must work when config/defaults.json is absent."""

    def test_fallbacks_enable_env_overrides(self, tmp_path: Path) -> None:
        tunables = load_tunables(
            tmp_path,
            env={"CCP_GATING": "1"},
            fallbacks={"gating": False, "max_file_bytes": 10},
        )
        assert tunables["gating"] is True
        assert tunables["max_file_bytes"] == 10

    def test_file_overrides_fallbacks_env_overrides_both(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "defaults.json").write_text(
            json.dumps({"max_file_bytes": 20}), encoding="utf-8"
        )
        tunables = load_tunables(
            tmp_path,
            env={"CCP_MAX_FILE_BYTES": "30"},
            fallbacks={"max_file_bytes": 10},
        )
        assert tunables["max_file_bytes"] == 30

    def test_hook_scans_via_env_patterns_without_defaults_file(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        patterns = json.dumps({"secret": ["AKIA[0-9A-Z]{16}"]})
        result = run_hook(
            repo_root,
            write_event(SECRET_LINE),
            env_overrides={"CCP_PATTERNS": patterns, "CCP_GATING": "1"},
            plugin_root=tmp_path,  # no config/defaults.json here
        )
        assert result.returncode == constants.EXIT_BLOCK
        events = [json.loads(line) for line in result.stderr.splitlines()]
        assert any(e["event"] == "quality_finding" for e in events)
