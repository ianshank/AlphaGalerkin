"""Gate-by-gate mutation tests: start clean, break one thing, assert the gate fires."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.helpers import gate_names, read_json, write_json
from tools.validate.config import ValidatorConfig
from tools.validate.gates import run_all_gates

VALID_SHA = "b" * 40


@pytest.fixture()
def config(synthetic_marketplace: Path) -> ValidatorConfig:
    return ValidatorConfig(root=synthetic_marketplace)


def marketplace_path(root: Path) -> Path:
    return root / ".claude-plugin" / "marketplace.json"


def plugin_dir(root: Path) -> Path:
    return root / "plugins" / "demo-plugin"


class TestCleanBaseline:
    def test_synthetic_marketplace_passes_all_gates(
        self, config: ValidatorConfig
    ) -> None:
        assert run_all_gates(config) == []


class TestMarketplaceSchemaGate:
    def test_missing_marketplace_json(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        marketplace_path(synthetic_marketplace).unlink()
        assert "marketplace-schema" in gate_names(run_all_gates(config))

    def test_corrupt_marketplace_json(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        marketplace_path(synthetic_marketplace).write_text(
            "{ not valid json", encoding="utf-8"
        )
        assert "marketplace-schema" in gate_names(run_all_gates(config))


class TestCatalogParityGate:
    def test_catalog_description_drift(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        path = marketplace_path(synthetic_marketplace)
        document = read_json(path)
        document["plugins"][0]["description"] = "Drifted description."
        write_json(path, document)
        violations = run_all_gates(config)
        assert "catalog-parity" in gate_names(violations)

    def test_plugin_missing_from_catalog(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        path = marketplace_path(synthetic_marketplace)
        document = read_json(path)
        # Replace the demo-plugin entry so the plugin directory has no
        # catalog entry (plugins must stay non-empty for schema validity).
        document["plugins"] = [
            {"name": "other-plugin", "source": "./plugins/other-plugin"}
        ]
        write_json(path, document)
        violations = run_all_gates(config)
        parity_messages = [v.message for v in violations if v.gate == "catalog-parity"]
        assert any("missing from catalog" in m for m in parity_messages)


class TestPluginManifestSchemaGate:
    def test_manifest_name_directory_mismatch(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        path = plugin_dir(synthetic_marketplace) / ".claude-plugin" / "plugin.json"
        document = read_json(path)
        document["name"] = "renamed-plugin"
        write_json(path, document)
        violations = run_all_gates(config)
        manifest_messages = [
            v.message for v in violations if v.gate == "plugin-manifest-schema"
        ]
        assert any("!= directory" in m for m in manifest_messages)


class TestVendoredRuntimeParityGate:
    def test_modified_vendored_file(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        vendored = (
            plugin_dir(synthetic_marketplace)
            / "hooks"
            / "scripts"
            / "_runtime"
            / "constants.py"
        )
        vendored.write_text(
            vendored.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
        )
        assert "vendored-runtime-parity" in gate_names(run_all_gates(config))

    def test_deleted_vendored_dir(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        shutil.rmtree(
            plugin_dir(synthetic_marketplace) / "hooks" / "scripts" / "_runtime"
        )
        assert "vendored-runtime-parity" in gate_names(run_all_gates(config))

    def test_stray_file_in_vendored_dir(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        stray = (
            plugin_dir(synthetic_marketplace)
            / "hooks"
            / "scripts"
            / "_runtime"
            / "stray.py"
        )
        stray.write_text("# not canonical\n", encoding="utf-8")
        violations = run_all_gates(config)
        parity_messages = [
            v.message for v in violations if v.gate == "vendored-runtime-parity"
        ]
        assert any("stray file" in m for m in parity_messages)


class TestPathLiteralsGate:
    def test_home_path_literal_in_skill(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        skill = plugin_dir(synthetic_marketplace) / "skills" / "demo-skill" / "SKILL.md"
        skill.write_text(
            skill.read_text(encoding="utf-8") + "\nSee /home/someuser/x for data.\n",
            encoding="utf-8",
        )
        violations = run_all_gates(config)
        literal_violations = [v for v in violations if v.gate == "path-literals"]
        assert literal_violations
        assert any(str(skill) == v.path for v in literal_violations)

    def test_hooks_command_without_plugin_root_token(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        path = plugin_dir(synthetic_marketplace) / "hooks" / "hooks.json"
        document = read_json(path)
        document["hooks"]["PostToolUse"][0]["hooks"][0]["command"] = (
            "python3 hooks/scripts/demo_hook.py"
        )
        write_json(path, document)
        violations = run_all_gates(config)
        literal_messages = [v.message for v in violations if v.gate == "path-literals"]
        assert any("${CLAUDE_PLUGIN_ROOT}" in m for m in literal_messages)


class TestStdlibImportsGate:
    def test_third_party_import_flagged(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        script = plugin_dir(synthetic_marketplace) / "hooks" / "scripts" / "bad_hook.py"
        script.write_text("import requests\n", encoding="utf-8")
        violations = run_all_gates(config)
        import_messages = [v.message for v in violations if v.gate == "stdlib-imports"]
        assert any("'requests'" in m for m in import_messages)

    def test_runtime_and_stdlib_imports_allowed(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        script = plugin_dir(synthetic_marketplace) / "hooks" / "scripts" / "ok_hook.py"
        script.write_text("import _runtime\nimport json\n", encoding="utf-8")
        violations = run_all_gates(config)
        assert "stdlib-imports" not in gate_names(violations)

    def test_relative_import_in_file_invoked_script_flagged(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        """Copilot review: `from . import x` in a top-level hook script
        raises ImportError at runtime (no parent package) — the gate must
        not bless it. Inside the vendored _runtime package it stays legal
        (the clean-baseline test covers that: _runtime uses them)."""
        script = plugin_dir(synthetic_marketplace) / "hooks" / "scripts" / "bad_rel.py"
        script.write_text("from . import helper\n", encoding="utf-8")
        violations = run_all_gates(config)
        assert any(
            "relative import" in v.message
            for v in violations
            if v.gate == "stdlib-imports"
        )


class TestFrontmatterGate:
    def _skill_path(self, root: Path) -> Path:
        return plugin_dir(root) / "skills" / "demo-skill" / "SKILL.md"

    def test_skill_without_frontmatter(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        self._skill_path(synthetic_marketplace).write_text(
            "# Demo\n\nNo frontmatter here.\n", encoding="utf-8"
        )
        violations = run_all_gates(config)
        fm_messages = [v.message for v in violations if v.gate == "frontmatter"]
        assert any("missing or malformed" in m for m in fm_messages)

    def test_frontmatter_name_mismatches_dirname(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        self._skill_path(synthetic_marketplace).write_text(
            "---\nname: wrong-name\ndescription: A demo skill.\n---\n\n# Demo\n",
            encoding="utf-8",
        )
        violations = run_all_gates(config)
        fm_messages = [v.message for v in violations if v.gate == "frontmatter"]
        assert any("!= expected" in m for m in fm_messages)

    def test_skill_over_line_limit(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        body = "\n".join(f"Line {i}" for i in range(160))
        self._skill_path(synthetic_marketplace).write_text(
            "---\nname: demo-skill\ndescription: A demo skill.\n---\n" + body + "\n",
            encoding="utf-8",
        )
        violations = run_all_gates(config)
        fm_messages = [v.message for v in violations if v.gate == "frontmatter"]
        assert any("progressive-disclosure" in m for m in fm_messages)


class TestReleasePinsGate:
    def _pins_path(self, root: Path) -> Path:
        return root / "release" / "pins.json"

    def test_pin_for_unknown_plugin(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            self._pins_path(synthetic_marketplace),
            {
                "schema_version": 1,
                "repo": "owner/repo",
                "pins": {"ghost-plugin": {"version": "1.0.0", "sha": VALID_SHA}},
            },
        )
        violations = run_all_gates(config)
        pin_messages = [v.message for v in violations if v.gate == "release-pins"]
        assert any("unknown plugin" in m for m in pin_messages)

    def test_pinned_version_with_relative_catalog_source(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            self._pins_path(synthetic_marketplace),
            {
                "schema_version": 1,
                "repo": "owner/repo",
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": VALID_SHA}},
            },
        )
        violations = run_all_gates(config)
        pin_messages = [v.message for v in violations if v.gate == "release-pins"]
        assert any("relative source" in m for m in pin_messages)

    def test_pin_satisfied_by_matching_github_source_is_clean(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        write_json(
            self._pins_path(synthetic_marketplace),
            {
                "schema_version": 1,
                "repo": "owner/repo",
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": VALID_SHA}},
            },
        )
        catalog_path = marketplace_path(synthetic_marketplace)
        document = read_json(catalog_path)
        document["plugins"][0]["source"] = {
            "source": "github",
            "repo": "owner/repo",
            "sha": VALID_SHA,
        }
        write_json(catalog_path, document)
        assert run_all_gates(config) == []

    def test_missing_pins_file(
        self, synthetic_marketplace: Path, config: ValidatorConfig
    ) -> None:
        self._pins_path(synthetic_marketplace).unlink()
        violations = run_all_gates(config)
        pin_messages = [v.message for v in violations if v.gate == "release-pins"]
        assert any("missing" in m for m in pin_messages)


def test_missing_catalog_description_is_parity_violation(
    synthetic_marketplace: Path,
) -> None:
    """Copilot review: a deleted catalog description must not pass parity."""
    catalog = synthetic_marketplace / ".claude-plugin" / "marketplace.json"
    document = read_json(catalog)
    del document["plugins"][0]["description"]
    write_json(catalog, document)
    violations = run_all_gates(ValidatorConfig(root=synthetic_marketplace))
    assert any(
        v.gate == "catalog-parity" and "missing description" in v.message
        for v in violations
    )
