"""Unit tests for tools.sync_catalog (catalog regeneration from manifests + pins)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import read_json, write_json
from tools.sync_catalog import (
    EXIT_CLEAN,
    EXIT_ERROR,
    EXIT_STALE,
    build_plugin_entries,
    run,
)
from tools.validate.config import ValidatorConfig

VALID_SHA = "c" * 40


@pytest.fixture()
def config(synthetic_marketplace: Path) -> ValidatorConfig:
    return ValidatorConfig(root=synthetic_marketplace)


def manifest_path(root: Path, name: str = "demo-plugin") -> Path:
    return root / "plugins" / name / ".claude-plugin" / "plugin.json"


def add_plugin(root: Path, name: str, description: str) -> None:
    write_json(
        manifest_path(root, name),
        {"name": name, "version": "0.2.0", "description": description},
    )


class TestBuildPluginEntries:
    def test_entries_are_name_sorted_and_deterministic(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        add_plugin(synthetic_marketplace, "alpha-plugin", "First alphabetically.")
        add_plugin(synthetic_marketplace, "zeta-plugin", "Last alphabetically.")
        first = build_plugin_entries(config)
        second = build_plugin_entries(config)
        assert [e["name"] for e in first] == [
            "alpha-plugin",
            "demo-plugin",
            "zeta-plugin",
        ]
        assert first == second

    def test_manifest_description_change_regenerates_entry(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        path = manifest_path(synthetic_marketplace)
        document = read_json(path)
        document["description"] = "A brand new description."
        write_json(path, document)
        entries = build_plugin_entries(config)
        assert entries[0]["description"] == "A brand new description."

    def test_matching_pin_renders_github_source(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "repo": "owner/marketplace-repo",
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": VALID_SHA}},
            },
        )
        entries = build_plugin_entries(config)
        assert entries[0]["source"] == {
            "source": "github",
            "repo": "owner/marketplace-repo",
            "sha": VALID_SHA,
        }

    def test_non_matching_pin_keeps_relative_source(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "repo": "owner/marketplace-repo",
                "pins": {"demo-plugin": {"version": "9.9.9", "sha": VALID_SHA}},
            },
        )
        entries = build_plugin_entries(config)
        assert entries[0]["source"] == "./plugins/demo-plugin"

    def test_matching_pin_without_repo_raises(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": VALID_SHA}},
            },
        )
        with pytest.raises(ValueError, match="requires 'repo'"):
            build_plugin_entries(config)


class TestRun:
    def test_check_only_clean_on_fresh_synthetic_tree(
        self, config: ValidatorConfig
    ) -> None:
        assert run(config, check_only=True) == EXIT_CLEAN

    def test_check_only_stale_after_description_change_then_clean_after_write(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        path = manifest_path(synthetic_marketplace)
        document = read_json(path)
        document["description"] = "Changed by test."
        write_json(path, document)
        assert run(config, check_only=True) == EXIT_STALE
        assert run(config, check_only=False) == EXIT_CLEAN
        assert run(config, check_only=True) == EXIT_CLEAN
        catalog = read_json(
            synthetic_marketplace / ".claude-plugin" / "marketplace.json"
        )
        assert catalog["plugins"][0]["description"] == "Changed by test."

    def test_matching_pin_without_repo_returns_exit_error(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": VALID_SHA}},
            },
        )
        assert run(config, check_only=True) == EXIT_ERROR
