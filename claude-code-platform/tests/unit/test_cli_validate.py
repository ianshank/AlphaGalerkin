"""Unit tests for the tools.validate CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate.__main__ import (
    EXIT_CLEAN,
    EXIT_USAGE,
    EXIT_VIOLATIONS,
    find_repo_root,
    main,
)

MARKETPLACE_RELPATH = ".claude-plugin/marketplace.json"


class TestMain:
    def test_clean_marketplace_exits_zero(self, synthetic_marketplace: Path) -> None:
        assert main(["--root", str(synthetic_marketplace)]) == EXIT_CLEAN

    def test_violation_exits_one(self, synthetic_marketplace: Path) -> None:
        (synthetic_marketplace / "release" / "pins.json").unlink()
        assert main(["--root", str(synthetic_marketplace)]) == EXIT_VIOLATIONS

    def test_json_format_emits_parseable_violation_list(
        self,
        synthetic_marketplace: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (synthetic_marketplace / "release" / "pins.json").unlink()
        exit_code = main(["--root", str(synthetic_marketplace), "--format", "json"])
        assert exit_code == EXIT_VIOLATIONS
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert payload
        for violation in payload:
            assert set(violation) == {"gate", "path", "message"}

    def test_json_format_clean_emits_empty_list(
        self,
        synthetic_marketplace: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--root", str(synthetic_marketplace), "--format", "json"])
        assert exit_code == EXIT_CLEAN
        assert json.loads(capsys.readouterr().out) == []

    def test_nonexistent_root_exits_usage(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        assert main(["--root", str(missing)]) == EXIT_USAGE


class TestFindRepoRoot:
    def test_finds_root_from_nested_directory(
        self, synthetic_marketplace: Path
    ) -> None:
        nested = synthetic_marketplace / "plugins" / "demo-plugin" / "hooks" / "scripts"
        assert find_repo_root(nested, MARKETPLACE_RELPATH) == synthetic_marketplace

    def test_finds_root_from_root_itself(self, synthetic_marketplace: Path) -> None:
        assert (
            find_repo_root(synthetic_marketplace, MARKETPLACE_RELPATH)
            == synthetic_marketplace
        )

    def test_returns_none_when_no_catalog_upward(self, tmp_path: Path) -> None:
        orphan = tmp_path / "no" / "marketplace" / "here"
        orphan.mkdir(parents=True)
        assert find_repo_root(orphan, MARKETPLACE_RELPATH) is None


def test_gate_summary_logged_with_hyphenated_gate_names(
    synthetic_marketplace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Copilot review: gate_summary must not crash on hyphenated gate names."""
    manifest = (
        synthetic_marketplace
        / "plugins"
        / "demo-plugin"
        / ".claude-plugin"
        / "plugin.json"
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["description"] = "Drifted."
    manifest.write_text(json.dumps(document, indent=2), encoding="utf-8")

    assert main(["--root", str(synthetic_marketplace)]) == 1
    stderr = capsys.readouterr().err
    assert "gate_summary" in stderr and "catalog-parity" in stderr
