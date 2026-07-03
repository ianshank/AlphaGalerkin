"""Unit tests for tools.sync_runtime (vendoring the canonical hook runtime)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.sync_runtime import EXIT_CLEAN, EXIT_DRIFT, run, sync_plugin
from tools.validate.config import ValidatorConfig
from tools.validate.gates import run_all_gates


@pytest.fixture()
def config(synthetic_marketplace: Path) -> ValidatorConfig:
    return ValidatorConfig(root=synthetic_marketplace)


@pytest.fixture()
def plugin(synthetic_marketplace: Path) -> Path:
    return synthetic_marketplace / "plugins" / "demo-plugin"


def canonical_names(config: ValidatorConfig) -> set[str]:
    canonical_dir = config.root / config.runtime_src_relpath
    return {p.name for p in canonical_dir.glob("*.py")}


def vendored_dir(config: ValidatorConfig, plugin: Path) -> Path:
    return plugin / config.vendored_runtime_relpath


class TestSyncPlugin:
    def test_vendors_all_canonical_files_from_scratch(
        self, config: ValidatorConfig, plugin: Path
    ) -> None:
        shutil.rmtree(vendored_dir(config, plugin))
        changed = sync_plugin(config, plugin)
        expected = canonical_names(config)
        assert set(changed) == expected
        assert {p.name for p in vendored_dir(config, plugin).glob("*.py")} == expected

    def test_second_run_is_idempotent(
        self, config: ValidatorConfig, plugin: Path
    ) -> None:
        sync_plugin(config, plugin)
        assert sync_plugin(config, plugin) == []

    def test_drift_is_repaired(self, config: ValidatorConfig, plugin: Path) -> None:
        target = vendored_dir(config, plugin) / "constants.py"
        canonical = (
            config.root / config.runtime_src_relpath / "constants.py"
        ).read_bytes()
        target.write_bytes(canonical + b"\n# drift\n")
        changed = sync_plugin(config, plugin)
        assert changed == ["constants.py"]
        assert target.read_bytes() == canonical

    def test_stray_file_is_removed(self, config: ValidatorConfig, plugin: Path) -> None:
        stray = vendored_dir(config, plugin) / "stray.py"
        stray.write_text("# not canonical\n", encoding="utf-8")
        changed = sync_plugin(config, plugin)
        assert changed == ["removed:stray.py"]
        assert not stray.exists()


class TestRunCheckOnly:
    def test_clean_tree_returns_zero(self, config: ValidatorConfig) -> None:
        assert run(config, check_only=True) == EXIT_CLEAN

    def test_drift_returns_one(self, config: ValidatorConfig, plugin: Path) -> None:
        target = vendored_dir(config, plugin) / "constants.py"
        target.write_bytes(target.read_bytes() + b"\n# drift\n")
        assert run(config, check_only=True) == EXIT_DRIFT

    def test_write_mode_repairs_then_check_is_clean(
        self, config: ValidatorConfig, plugin: Path
    ) -> None:
        shutil.rmtree(vendored_dir(config, plugin))
        assert run(config, check_only=True) == EXIT_DRIFT
        assert run(config, check_only=False) == EXIT_CLEAN
        assert run(config, check_only=True) == EXIT_CLEAN


def test_write_removes_nested_stray_symlink(
    synthetic_marketplace: Path,
) -> None:
    """Copilot review: a stray symlink under _runtime must be repaired."""
    config = ValidatorConfig(root=synthetic_marketplace)
    plugin = synthetic_marketplace / "plugins" / "demo-plugin"
    vendored = plugin / "hooks" / "scripts" / "_runtime"
    (vendored / "sub").mkdir()
    stray = vendored / "sub" / "link.py"
    stray.symlink_to(synthetic_marketplace / "tools" / "hook_runtime" / "constants.py")
    assert run_all_gates(config) != []  # gate flags the symlink

    changed = sync_plugin(config, plugin)

    assert "removed:symlink:sub/link.py" in changed
    assert not stray.exists() and not stray.is_symlink()
    assert run_all_gates(config) == []  # write mode fully repairs drift
