"""Error-branch coverage for the validation gates.

Complements test_validate_gates.py (happy-path + primary mutations) with
the malformed-input and edge branches: corrupt/schema-invalid documents,
empty canonical runtime, symlinked files, syntax errors, frontmatter
edge cases, and non-command hook invocations.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tests.helpers import gate_names, read_json, write_json
from tools.validate.config import ValidatorConfig
from tools.validate.gates import (
    discover_plugin_dirs,
    gate_frontmatter,
    gate_pins,
    gate_stdlib_imports,
    gate_vendored_runtime,
    load_manifests,
    load_marketplace,
    run_all_gates,
)


def config_for(root: Path, **overrides: object) -> ValidatorConfig:
    return ValidatorConfig(root=root, **overrides)  # type: ignore[arg-type]


def plugin_dir(root: Path) -> Path:
    return root / "plugins" / "demo-plugin"


class TestDocumentErrorBranches:
    def test_no_plugins_dir_discovers_nothing(self, tmp_path: Path) -> None:
        assert discover_plugin_dirs(config_for(tmp_path)) == []

    def test_marketplace_schema_invalid_name(self, synthetic_marketplace: Path) -> None:
        catalog = synthetic_marketplace / ".claude-plugin" / "marketplace.json"
        document = read_json(catalog)
        document["name"] = "Not Kebab Case"
        write_json(catalog, document)
        marketplace, violations = load_marketplace(config_for(synthetic_marketplace))
        assert marketplace is None
        assert violations and violations[0].gate == "marketplace-schema"

    def test_manifest_corrupt_json(self, synthetic_marketplace: Path) -> None:
        manifest = plugin_dir(synthetic_marketplace) / ".claude-plugin" / "plugin.json"
        manifest.write_text("{", encoding="utf-8")
        manifests, violations = load_manifests(config_for(synthetic_marketplace))
        assert manifests == {}
        assert any("unreadable JSON" in v.message for v in violations)

    def test_manifest_schema_invalid_version(self, synthetic_marketplace: Path) -> None:
        manifest = plugin_dir(synthetic_marketplace) / ".claude-plugin" / "plugin.json"
        document = read_json(manifest)
        document["version"] = "one.two"
        write_json(manifest, document)
        _, violations = load_manifests(config_for(synthetic_marketplace))
        assert any(v.gate == "plugin-manifest-schema" for v in violations)

    def test_pins_corrupt_json(self, synthetic_marketplace: Path) -> None:
        (synthetic_marketplace / "release" / "pins.json").write_text(
            "{", encoding="utf-8"
        )
        violations = run_all_gates(config_for(synthetic_marketplace))
        assert any(
            v.gate == "release-pins" and "unreadable JSON" in v.message
            for v in violations
        )

    def test_pins_schema_invalid_sha(self, synthetic_marketplace: Path) -> None:
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": "short"}},
            },
        )
        violations = run_all_gates(config_for(synthetic_marketplace))
        assert any(v.gate == "release-pins" for v in violations)

    def test_pin_sha_mismatch_with_github_source(
        self, synthetic_marketplace: Path
    ) -> None:
        catalog = synthetic_marketplace / ".claude-plugin" / "marketplace.json"
        document = read_json(catalog)
        document["plugins"][0]["source"] = {
            "source": "github",
            "repo": "owner/repo",
            "sha": "b" * 40,
        }
        write_json(catalog, document)
        write_json(
            synthetic_marketplace / "release" / "pins.json",
            {
                "schema_version": 1,
                "repo": "owner/repo",
                "pins": {"demo-plugin": {"version": "0.1.0", "sha": "a" * 40}},
            },
        )
        config = config_for(synthetic_marketplace)
        marketplace, _ = load_marketplace(config)
        manifests, _ = load_manifests(config)
        assert marketplace is not None
        violations = gate_pins(config, marketplace, manifests)
        assert any("catalog sha" in v.message for v in violations)


class TestRuntimeAndImportEdges:
    def test_empty_canonical_runtime_is_violation(
        self, synthetic_marketplace: Path
    ) -> None:
        shutil.rmtree(synthetic_marketplace / "tools" / "hook_runtime")
        violations = gate_vendored_runtime(config_for(synthetic_marketplace))
        assert violations and "canonical hook runtime is empty" in violations[0].message

    def test_symlinked_file_inside_runtime_flagged(
        self, synthetic_marketplace: Path
    ) -> None:
        vendored = plugin_dir(synthetic_marketplace) / "hooks" / "scripts" / "_runtime"
        target = vendored / "constants.py"
        target.unlink()
        target.symlink_to(
            synthetic_marketplace / "tools" / "hook_runtime" / "constants.py"
        )
        violations = gate_vendored_runtime(config_for(synthetic_marketplace))
        assert any("symlink" in v.message for v in violations)

    def test_syntax_error_in_hook_script_flagged(
        self, synthetic_marketplace: Path
    ) -> None:
        script = (
            plugin_dir(synthetic_marketplace) / "hooks" / "scripts" / "demo_hook.py"
        )
        script.write_text("def broken(:\n", encoding="utf-8")
        violations = gate_stdlib_imports(config_for(synthetic_marketplace))
        assert any("syntax error" in v.message for v in violations)


class TestHooksDocumentEdges:
    def test_corrupt_hooks_json_is_schema_violation(
        self, synthetic_marketplace: Path
    ) -> None:
        hooks = plugin_dir(synthetic_marketplace) / "hooks" / "hooks.json"
        hooks.write_text("{", encoding="utf-8")
        violations = run_all_gates(config_for(synthetic_marketplace))
        assert "hooks-schema" in gate_names(violations)

    def test_empty_hooks_mapping_is_schema_violation(
        self, synthetic_marketplace: Path
    ) -> None:
        write_json(
            plugin_dir(synthetic_marketplace) / "hooks" / "hooks.json", {"hooks": {}}
        )
        violations = run_all_gates(config_for(synthetic_marketplace))
        assert "hooks-schema" in gate_names(violations)

    def test_non_command_invocation_skips_token_check(
        self, synthetic_marketplace: Path
    ) -> None:
        write_json(
            plugin_dir(synthetic_marketplace) / "hooks" / "hooks.json",
            {
                "hooks": {
                    "PostToolUse": [
                        {"hooks": [{"type": "prompt", "prompt": "review this"}]}
                    ]
                }
            },
        )
        violations = run_all_gates(config_for(synthetic_marketplace))
        assert "path-literals" not in gate_names(violations)


class TestFrontmatterEdges:
    def skill_path(self, root: Path) -> Path:
        return plugin_dir(root) / "skills" / "demo-skill" / "SKILL.md"

    def test_unterminated_frontmatter(self, synthetic_marketplace: Path) -> None:
        self.skill_path(synthetic_marketplace).write_text(
            "---\nname: demo-skill\n# never closed\n", encoding="utf-8"
        )
        violations = gate_frontmatter(config_for(synthetic_marketplace))
        assert any("malformed" in v.message for v in violations)

    def test_non_mapping_frontmatter(self, synthetic_marketplace: Path) -> None:
        self.skill_path(synthetic_marketplace).write_text(
            "---\n- just\n- a list\n---\nbody\n", encoding="utf-8"
        )
        violations = gate_frontmatter(config_for(synthetic_marketplace))
        assert any("malformed" in v.message for v in violations)

    def test_non_kebab_name(self, synthetic_marketplace: Path) -> None:
        self.skill_path(synthetic_marketplace).write_text(
            "---\nname: Demo_Skill\ndescription: d\n---\nbody\n", encoding="utf-8"
        )
        violations = gate_frontmatter(config_for(synthetic_marketplace))
        assert any("kebab-case" in v.message for v in violations)

    def test_description_over_limit(self, synthetic_marketplace: Path) -> None:
        self.skill_path(synthetic_marketplace).write_text(
            "---\nname: demo-skill\ndescription: " + "d" * 50 + "\n---\nbody\n",
            encoding="utf-8",
        )
        config = config_for(synthetic_marketplace, max_frontmatter_description_chars=10)
        violations = gate_frontmatter(config)
        assert any("exceeds" in v.message for v in violations)

    def test_missing_description_key(self, synthetic_marketplace: Path) -> None:
        self.skill_path(synthetic_marketplace).write_text(
            "---\nname: demo-skill\n---\nbody\n", encoding="utf-8"
        )
        violations = gate_frontmatter(config_for(synthetic_marketplace))
        assert any("'description' is required" in v.message for v in violations)
