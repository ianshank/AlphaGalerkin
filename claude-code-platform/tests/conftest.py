"""Shared fixtures: import path setup and a synthetic marketplace factory."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

SUBTREE_ROOT = Path(__file__).resolve().parents[1]
if str(SUBTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBTREE_ROOT))

from tests.helpers import write_json  # noqa: E402


@pytest.fixture()
def repo_root() -> Path:
    """The real marketplace repo root (this subtree)."""
    return SUBTREE_ROOT


@pytest.fixture()
def synthetic_marketplace(tmp_path: Path) -> Path:
    """A minimal, valid marketplace tree for gate tests to mutate.

    Copies the real canonical runtime so vendored-parity semantics match
    production, and ships one plugin with one hook script.
    """
    root = tmp_path / "marketplace"
    plugin = root / "plugins" / "demo-plugin"
    (root / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / "hooks" / "scripts").mkdir(parents=True)
    (plugin / "skills" / "demo-skill").mkdir(parents=True)
    (plugin / "agents").mkdir(parents=True)
    (root / "release").mkdir()

    shutil.copytree(
        SUBTREE_ROOT / "tools" / "hook_runtime",
        root / "tools" / "hook_runtime",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        SUBTREE_ROOT / "tools" / "hook_runtime",
        plugin / "hooks" / "scripts" / "_runtime",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    description = "Demo plugin for gate tests."
    write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "demo-marketplace",
            "owner": {"name": "Test Owner"},
            "plugins": [
                {
                    "name": "demo-plugin",
                    "source": "./plugins/demo-plugin",
                    "description": description,
                    "version": "0.1.0",
                }
            ],
        },
    )
    write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "demo-plugin", "version": "0.1.0", "description": description},
    )
    write_json(
        plugin / "hooks" / "hooks.json",
        {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    'python3 "${CLAUDE_PLUGIN_ROOT}'
                                    '/hooks/scripts/demo_hook.py"'
                                ),
                            }
                        ],
                    }
                ]
            }
        },
    )
    write_json(root / "release" / "pins.json", {"schema_version": 1, "pins": {}})
    (plugin / "hooks" / "scripts" / "demo_hook.py").write_text(
        "import json\nimport sys\n\nfrom _runtime import constants\n\n"
        "print(json.dumps({}), file=sys.stderr)\n"
        "sys.exit(constants.EXIT_OK)\n",
        encoding="utf-8",
    )
    (plugin / "skills" / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo skill for tests.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    (plugin / "agents" / "demo-agent.md").write_text(
        "---\nname: demo-agent\ndescription: A demo agent for tests.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    return root
