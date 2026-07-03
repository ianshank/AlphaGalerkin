"""CLI-path and error-branch tests for the sync tools' ``main()``."""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import read_json, write_json
from tools import sync_catalog, sync_runtime
from tools.validate.config import ValidatorConfig


def add_hookless_plugin(root: Path, name: str = "docs-only") -> Path:
    """A plugin with a manifest but no hooks (exercises the skip branch)."""
    plugin = root / "plugins" / name
    (plugin / ".claude-plugin").mkdir(parents=True)
    description = "Docs-only plugin."
    write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": name, "version": "0.1.0", "description": description},
    )
    catalog_path = root / ".claude-plugin" / "marketplace.json"
    document = read_json(catalog_path)
    document["plugins"].append(
        {
            "name": name,
            "source": f"./plugins/{name}",
            "description": description,
            "version": "0.1.0",
        }
    )
    write_json(catalog_path, document)
    return plugin


class TestSyncRuntimeMain:
    def test_check_clean(self, synthetic_marketplace: Path) -> None:
        argv = ["--check", "--root", str(synthetic_marketplace)]
        assert sync_runtime.main(argv) == sync_runtime.EXIT_CLEAN

    def test_check_reports_drift(self, synthetic_marketplace: Path) -> None:
        vendored = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "hooks"
            / "scripts"
            / "_runtime"
        )
        (vendored / "constants.py").write_text("drift\n", encoding="utf-8")
        argv = ["--check", "--root", str(synthetic_marketplace)]
        assert sync_runtime.main(argv) == sync_runtime.EXIT_DRIFT

    def test_write_repairs_and_skips_hookless(
        self, synthetic_marketplace: Path
    ) -> None:
        add_hookless_plugin(synthetic_marketplace)
        vendored = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / "hooks"
            / "scripts"
            / "_runtime"
        )
        (vendored / "constants.py").write_text("drift\n", encoding="utf-8")
        assert (
            sync_runtime.main(["--write", "--root", str(synthetic_marketplace)])
            == sync_runtime.EXIT_CLEAN
        )
        assert (
            sync_runtime.main(["--check", "--root", str(synthetic_marketplace)])
            == sync_runtime.EXIT_CLEAN
        )

    def test_nonexistent_root_is_usage_error(self, tmp_path: Path) -> None:
        argv = ["--check", "--root", str(tmp_path / "missing")]
        assert sync_runtime.main(argv) == sync_runtime.EXIT_USAGE

    def test_has_hook_scripts_false_for_hookless_plugin(
        self, synthetic_marketplace: Path
    ) -> None:
        plugin = add_hookless_plugin(synthetic_marketplace)
        config = ValidatorConfig(root=synthetic_marketplace)
        assert sync_runtime.has_hook_scripts(config, plugin) is False


class TestSyncCatalogMain:
    def test_check_clean(self, synthetic_marketplace: Path) -> None:
        argv = ["--check", "--root", str(synthetic_marketplace)]
        assert sync_catalog.main(argv) == sync_catalog.EXIT_CLEAN

    def test_check_stale_then_write_then_clean(
        self, synthetic_marketplace: Path
    ) -> None:
        manifest = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / ".claude-plugin"
            / "plugin.json"
        )
        document = read_json(manifest)
        document["description"] = "Changed description."
        write_json(manifest, document)
        root_argv = ["--root", str(synthetic_marketplace)]
        assert sync_catalog.main(["--check", *root_argv]) == sync_catalog.EXIT_STALE
        assert sync_catalog.main(["--write", *root_argv]) == sync_catalog.EXIT_CLEAN
        assert sync_catalog.main(["--check", *root_argv]) == sync_catalog.EXIT_CLEAN

    def test_invalid_manifest_is_generation_error(
        self, synthetic_marketplace: Path
    ) -> None:
        manifest = (
            synthetic_marketplace
            / "plugins"
            / "demo-plugin"
            / ".claude-plugin"
            / "plugin.json"
        )
        manifest.write_text("{", encoding="utf-8")
        argv = ["--check", "--root", str(synthetic_marketplace)]
        assert sync_catalog.main(argv) == sync_catalog.EXIT_ERROR

    def test_non_object_catalog_is_generation_error(
        self, synthetic_marketplace: Path
    ) -> None:
        catalog = synthetic_marketplace / ".claude-plugin" / "marketplace.json"
        catalog.write_text("[]", encoding="utf-8")
        argv = ["--check", "--root", str(synthetic_marketplace)]
        assert sync_catalog.main(argv) == sync_catalog.EXIT_ERROR

    def test_missing_pins_file_defaults_to_relative_sources(
        self, synthetic_marketplace: Path
    ) -> None:
        (synthetic_marketplace / "release" / "pins.json").unlink()
        config = ValidatorConfig(root=synthetic_marketplace)
        entries = sync_catalog.build_plugin_entries(config)
        assert entries[0]["source"] == "./plugins/demo-plugin"

    def test_write_mode_noop_when_current(self, synthetic_marketplace: Path) -> None:
        catalog = synthetic_marketplace / ".claude-plugin" / "marketplace.json"
        before = catalog.read_text(encoding="utf-8")
        argv = ["--write", "--root", str(synthetic_marketplace)]
        assert sync_catalog.main(argv) == sync_catalog.EXIT_CLEAN
        assert catalog.read_text(encoding="utf-8") == before

    def test_pin_ref_included_in_github_source(
        self, synthetic_marketplace: Path
    ) -> None:
        sha = "a" * 40
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "repo": "owner/repo",
                "pins": {
                    "demo-plugin": {"version": "0.1.0", "ref": "v0.1.0", "sha": sha}
                },
            },
        )
        config = ValidatorConfig(root=synthetic_marketplace)
        (entry,) = sync_catalog.build_plugin_entries(config)
        assert entry["source"] == {
            "source": "github",
            "repo": "owner/repo",
            "sha": sha,
            "ref": "v0.1.0",
        }

    def test_render_catalog_is_stable_json(self, synthetic_marketplace: Path) -> None:
        config = ValidatorConfig(root=synthetic_marketplace)
        rendered = sync_catalog.render_catalog(config)
        assert json.loads(rendered)["plugins"][0]["name"] == "demo-plugin"
        assert rendered.endswith("\n")
