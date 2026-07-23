"""End-to-end smoke: the official `claude plugin validate` accepts the repo.

This exercises component correctness through the real CLI (review
Finding 5: the marketplace *install* path is a manual release-checklist
item — it cannot be scripted headlessly). Skips when the CLI is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.validate.config import ValidatorConfig
from tools.validate.gates import discover_plugin_dirs

pytestmark = pytest.mark.e2e

CLI_TIMEOUT_SECONDS = 120

claude_cli = shutil.which("claude")
requires_cli = pytest.mark.skipif(claude_cli is None, reason="claude CLI not installed")


def validate(path: Path) -> subprocess.CompletedProcess[str]:
    assert claude_cli is not None
    return subprocess.run(
        [claude_cli, "plugin", "validate", str(path)],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )


@requires_cli
def test_marketplace_passes_official_validation(repo_root: Path) -> None:
    result = validate(repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


@requires_cli
def test_every_plugin_passes_official_validation(repo_root: Path) -> None:
    plugin_dirs = discover_plugin_dirs(ValidatorConfig(root=repo_root))
    assert plugin_dirs, "no plugins discovered"
    for plugin_dir in plugin_dirs:
        result = validate(plugin_dir)
        assert result.returncode == 0, (
            f"{plugin_dir.name}: " + result.stdout + result.stderr
        )
