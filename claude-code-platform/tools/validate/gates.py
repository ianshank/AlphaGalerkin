"""Static validation gates for the marketplace repo.

Each gate is an independently testable function returning a list of
:class:`Violation`. ``run_all_gates`` orchestrates them. Version parity
between catalog entries and plugin manifests is intentionally NOT gated
here — ``claude plugin validate`` already enforces it (review Finding 7);
this module covers only what the official validator does not.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config import KEBAB_CASE_PATTERN, ValidatorConfig
from .schemas import (
    HooksDocument,
    MarketplaceDocument,
    PinsDocument,
    PluginManifest,
)

FRONTMATTER_DELIMITER = "---"
REQUIRED_FRONTMATTER_KEYS = ("name", "description")
SKILL_FILENAME = "SKILL.md"
IGNORED_CACHE_DIRNAME = "__pycache__"
IGNORED_CACHE_SUFFIXES = (".pyc", ".pyo")
#: Dynamic-import machinery banned from hook scripts: it defeats the
#: static stdlib-import analysis (ADR-0002).
DYNAMIC_IMPORT_MODULES = ("importlib",)
DYNAMIC_IMPORT_BUILTIN = "__import__"


@dataclass(frozen=True)
class Violation:
    """One gate failure, addressed to a file."""

    gate: str
    path: str
    message: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_violations(
    gate: str, path: Path, error: ValidationError
) -> list[Violation]:
    return [
        Violation(
            gate=gate,
            path=str(path),
            message=f"{'.'.join(str(loc) for loc in issue['loc'])}: {issue['msg']}",
        )
        for issue in error.errors()
    ]


def discover_plugin_dirs(config: ValidatorConfig) -> list[Path]:
    plugins_root = config.root / config.plugins_dirname
    if not plugins_root.is_dir():
        return []
    return sorted(
        child
        for child in plugins_root.iterdir()
        if child.is_dir() and (child / config.manifest_relpath).is_file()
    )


def load_marketplace(
    config: ValidatorConfig,
) -> tuple[MarketplaceDocument | None, list[Violation]]:
    gate = "marketplace-schema"
    path = config.root / config.marketplace_relpath
    if not path.is_file():
        return None, [Violation(gate, str(path), "marketplace.json is missing")]
    try:
        document = MarketplaceDocument.model_validate(_read_json(path))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [Violation(gate, str(path), f"unreadable JSON: {exc}")]
    except ValidationError as exc:
        return None, _schema_violations(gate, path, exc)
    return document, []


def load_manifests(
    config: ValidatorConfig,
) -> tuple[dict[str, PluginManifest], list[Violation]]:
    gate = "plugin-manifest-schema"
    manifests: dict[str, PluginManifest] = {}
    violations: list[Violation] = []
    for plugin_dir in discover_plugin_dirs(config):
        path = plugin_dir / config.manifest_relpath
        try:
            manifest = PluginManifest.model_validate(_read_json(path))
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(Violation(gate, str(path), f"unreadable JSON: {exc}"))
            continue
        except ValidationError as exc:
            violations.extend(_schema_violations(gate, path, exc))
            continue
        if manifest.name != plugin_dir.name:
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"manifest name {manifest.name!r} != directory "
                    f"name {plugin_dir.name!r}",
                )
            )
            continue
        manifests[manifest.name] = manifest
    return manifests, violations


def gate_catalog_parity(
    config: ValidatorConfig,
    marketplace: MarketplaceDocument,
    manifests: dict[str, PluginManifest],
) -> list[Violation]:
    """Catalog entries must mirror manifest name/description exactly."""
    gate = "catalog-parity"
    path = config.root / config.marketplace_relpath
    violations: list[Violation] = []
    catalog_names = {entry.name for entry in marketplace.plugins}
    for missing in sorted(set(manifests) - catalog_names):
        violations.append(
            Violation(gate, str(path), f"plugin {missing!r} missing from catalog")
        )
    for entry in marketplace.plugins:
        manifest = manifests.get(entry.name)
        if manifest is None:
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"catalog entry {entry.name!r} has no plugin directory",
                )
            )
            continue
        # None counts as drift too: sync_catalog always writes a
        # description, so absence means the entry was hand-edited.
        if entry.description != manifest.description:
            detail = (
                "missing description"
                if entry.description is None
                else "catalog description differs from manifest"
            )
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"{entry.name}: {detail} "
                    "(regenerate with `python -m tools.sync_catalog --write`)",
                )
            )
    return violations


def gate_pins(
    config: ValidatorConfig,
    marketplace: MarketplaceDocument,
    manifests: dict[str, PluginManifest],
) -> list[Violation]:
    """Released versions must ship as self-referential sha-pinned sources."""
    gate = "release-pins"
    path = config.root / config.pins_relpath
    if not path.is_file():
        return [Violation(gate, str(path), "release pin manifest is missing")]
    try:
        pins = PinsDocument.model_validate(_read_json(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [Violation(gate, str(path), f"unreadable JSON: {exc}")]
    except ValidationError as exc:
        return _schema_violations(gate, path, exc)

    violations: list[Violation] = []
    entries = {entry.name: entry for entry in marketplace.plugins}
    for plugin_name, pin in sorted(pins.pins.items()):
        if plugin_name not in manifests:
            violations.append(
                Violation(gate, str(path), f"pin for unknown plugin {plugin_name!r}")
            )
            continue
        entry = entries.get(plugin_name)
        if entry is None or manifests[plugin_name].version != pin.version:
            continue  # pin applies to a released version, not the working tree
        source = entry.source
        if isinstance(source, str):
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"{plugin_name}@{pin.version} is pinned but the catalog "
                    "uses a relative source (run sync_catalog --write)",
                )
            )
        elif source.sha != pin.sha:
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"{plugin_name}@{pin.version}: catalog sha "
                    f"{source.sha!r} != pinned sha {pin.sha!r}",
                )
            )
    return violations


def relative_file_map(directory: Path) -> dict[str, Path]:
    """All regular files under ``directory`` keyed by posix relpath.

    Bytecode caches are ignored; symlinks are NOT followed here — they are
    reported separately (a symlinked runtime breaks after plugin install).
    """
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if IGNORED_CACHE_DIRNAME in path.parts:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix in IGNORED_CACHE_SUFFIXES:
            continue
        files[path.relative_to(directory).as_posix()] = path
    return files


#: Backwards-compat alias for the pre-0.1.0 private name.
_relative_file_map = relative_file_map


def _symlink_violations(gate: str, directory: Path) -> list[Violation]:
    violations: list[Violation] = []
    if directory.is_symlink():
        violations.append(
            Violation(
                gate,
                str(directory),
                "must be a real directory, not a symlink (symlinks break "
                "after plugin install; ADR-0002)",
            )
        )
        return violations
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            violations.append(
                Violation(
                    gate,
                    str(path),
                    "symlink not allowed inside the vendored runtime "
                    "(breaks after plugin install; ADR-0002)",
                )
            )
    return violations


def plugin_hook_scripts(config: ValidatorConfig, plugin_dir: Path) -> list[Path]:
    """Every Python file under ``hooks/`` (vendored runtime included) plus
    any file referenced by a hooks.json command — the actual entry points,
    wherever they live in the plugin."""
    hooks_dir = plugin_dir / config.hooks_dirname
    scripts: set[Path] = set()
    if hooks_dir.is_dir():
        scripts.update(
            p
            for p in hooks_dir.rglob("*.py")
            if IGNORED_CACHE_DIRNAME not in p.parts and p.is_file()
        )
    for relative in _hooks_command_paths(config, plugin_dir):
        candidate = plugin_dir / relative
        if candidate.suffix == ".py" and candidate.is_file():
            scripts.add(candidate)
    return sorted(scripts)


def _hooks_command_paths(config: ValidatorConfig, plugin_dir: Path) -> list[str]:
    """Plugin-relative paths referenced via the root token in hook commands."""
    hooks_path = plugin_dir / config.hooks_relpath
    if not hooks_path.is_file():
        return []
    try:
        document = HooksDocument.model_validate(_read_json(hooks_path))
    except (OSError, json.JSONDecodeError, ValidationError):
        return []  # reported by the hooks-schema checks elsewhere
    token_path = re.compile(re.escape(config.plugin_root_token) + r"/([^\"'\s]+)")
    paths: list[str] = []
    for groups in document.hooks.values():
        for group in groups:
            for invocation in group.hooks:
                paths.extend(token_path.findall(invocation.command or ""))
    return paths


def gate_vendored_runtime(config: ValidatorConfig) -> list[Violation]:
    """Vendored ``_runtime`` copies must be byte-identical to the canonical lib.

    Recursive and content-complete: every file (any suffix, any depth,
    bytecode caches excluded) is compared, and symlinks are rejected. A
    plugin triggers the requirement as soon as it ships any hook script.
    """
    gate = "vendored-runtime-parity"
    canonical_dir = config.root / config.runtime_src_relpath
    canonical = relative_file_map(canonical_dir) if canonical_dir.is_dir() else {}
    violations: list[Violation] = []
    if not canonical:
        return [Violation(gate, str(canonical_dir), "canonical hook runtime is empty")]
    canonical_bytes = {rel: path.read_bytes() for rel, path in canonical.items()}
    for plugin_dir in discover_plugin_dirs(config):
        vendored_dir = plugin_dir / config.vendored_runtime_relpath
        has_scripts = any(
            not path.is_relative_to(vendored_dir)
            for path in plugin_hook_scripts(config, plugin_dir)
        )
        if not has_scripts:
            continue  # plugin ships no hook scripts; nothing to vendor
        hint = "run `python -m tools.sync_runtime --write`"
        if vendored_dir.is_symlink() or not vendored_dir.is_dir():
            violations.extend(_symlink_violations(gate, vendored_dir))
            if not vendored_dir.is_symlink():
                violations.append(
                    Violation(
                        gate, str(vendored_dir), f"vendored runtime missing; {hint}"
                    )
                )
            continue
        violations.extend(_symlink_violations(gate, vendored_dir))
        vendored = {
            rel: path.read_bytes()
            for rel, path in relative_file_map(vendored_dir).items()
        }
        for rel in sorted(set(canonical_bytes) - set(vendored)):
            violations.append(
                Violation(gate, str(vendored_dir / rel), f"missing file; {hint}")
            )
        for rel in sorted(set(vendored) - set(canonical_bytes)):
            violations.append(
                Violation(gate, str(vendored_dir / rel), f"stray file; {hint}")
            )
        for rel in sorted(set(canonical_bytes) & set(vendored)):
            if canonical_bytes[rel] != vendored[rel]:
                violations.append(
                    Violation(
                        gate,
                        str(vendored_dir / rel),
                        f"content drift from canonical runtime; {hint}",
                    )
                )
    return violations


def gate_path_literals(config: ValidatorConfig) -> list[Violation]:
    """No machine-specific literal paths; hook commands use the root token."""
    gate = "path-literals"
    patterns = [re.compile(p) for p in config.path_literal_patterns]
    violations: list[Violation] = []
    for plugin_dir in discover_plugin_dirs(config):
        for file_path in sorted(plugin_dir.rglob("*")):
            if IGNORED_CACHE_DIRNAME in file_path.parts:
                continue
            if not file_path.is_file():
                continue
            # Extensionless files (shebang scripts, dotfiles) are scanned
            # too — skipping them was a gate bypass (review F6).
            if file_path.suffix and file_path.suffix not in config.scannable_suffixes:
                continue
            relative = file_path.relative_to(plugin_dir).as_posix()
            if relative in config.path_literal_exclude_relpaths:
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for pattern in patterns:
                    if pattern.search(line):
                        violations.append(
                            Violation(
                                gate,
                                str(file_path),
                                f"line {lineno}: machine-specific path matches "
                                f"{pattern.pattern!r}",
                            )
                        )
        violations.extend(_hooks_command_token_violations(config, plugin_dir))
    return violations


def _hooks_command_token_violations(
    config: ValidatorConfig, plugin_dir: Path
) -> list[Violation]:
    gate = "path-literals"
    hooks_path = plugin_dir / config.hooks_relpath
    if not hooks_path.is_file():
        return []
    try:
        document = HooksDocument.model_validate(_read_json(hooks_path))
    except (OSError, json.JSONDecodeError) as exc:
        return [Violation("hooks-schema", str(hooks_path), f"unreadable JSON: {exc}")]
    except ValidationError as exc:
        return _schema_violations("hooks-schema", hooks_path, exc)
    violations: list[Violation] = []
    for event, groups in document.hooks.items():
        for group in groups:
            for invocation in group.hooks:
                command = invocation.command or ""
                if invocation.type == "command" and (
                    config.plugin_root_token not in command
                ):
                    violations.append(
                        Violation(
                            gate,
                            str(hooks_path),
                            f"{event}: command must reference scripts via "
                            f"{config.plugin_root_token}",
                        )
                    )
    for relative in _hooks_command_paths(config, plugin_dir):
        if not (plugin_dir / relative).is_file():
            violations.append(
                Violation(
                    gate,
                    str(hooks_path),
                    f"command references {relative!r} which does not exist "
                    "in the plugin",
                )
            )
    return violations


def gate_stdlib_imports(config: ValidatorConfig) -> list[Violation]:
    """Hook scripts import stdlib + plugin-local only; no dynamic imports.

    Scans every Python file under ``hooks/`` (vendored runtime included)
    plus any file referenced by a hooks.json command wherever it lives —
    not just ``hooks/scripts/`` (review F2). Dynamic-import machinery
    (``importlib``, ``__import__``) is banned outright because it defeats
    static analysis (review F5).
    """
    gate = "stdlib-imports"
    allowed = (set(sys.stdlib_module_names) | {config.vendored_import_name}) - set(
        DYNAMIC_IMPORT_MODULES
    )
    violations: list[Violation] = []
    for plugin_dir in discover_plugin_dirs(config):
        for script in plugin_hook_scripts(config, plugin_dir):
            try:
                tree = ast.parse(
                    script.read_text(encoding="utf-8"), filename=str(script)
                )
            except SyntaxError as exc:
                violations.append(
                    Violation(gate, str(script), f"syntax error: {exc.msg}")
                )
                continue
            violations.extend(_script_import_violations(gate, script, tree, allowed))
    return violations


def _script_import_violations(
    gate: str, script: Path, tree: ast.AST, allowed: set[str]
) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == DYNAMIC_IMPORT_BUILTIN:
            violations.append(
                Violation(
                    gate,
                    str(script),
                    f"line {node.lineno}: {DYNAMIC_IMPORT_BUILTIN} is banned "
                    "in hook scripts (dynamic imports defeat static "
                    "analysis; ADR-0002)",
                )
            )
            continue
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import: plugin-local by construction
            names = [node.module.split(".")[0]] if node.module else []
        else:
            continue
        for name in names:
            if name in DYNAMIC_IMPORT_MODULES:
                violations.append(
                    Violation(
                        gate,
                        str(script),
                        f"line {node.lineno}: {name!r} is banned in hook "
                        "scripts (dynamic imports defeat static analysis; "
                        "ADR-0002)",
                    )
                )
            elif name not in allowed:
                violations.append(
                    Violation(
                        gate,
                        str(script),
                        f"line {node.lineno}: non-stdlib import "
                        f"{name!r} (hook scripts are stdlib-only, "
                        "ADR-0002)",
                    )
                )
    return violations


def _parse_frontmatter(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_DELIMITER:
            block = "\n".join(lines[1:index])
            loaded = yaml.safe_load(block)
            return loaded if isinstance(loaded, dict) else None
    return None


def gate_frontmatter(config: ValidatorConfig) -> list[Violation]:
    """SKILL.md / agent markdown carry valid frontmatter and size limits."""
    gate = "frontmatter"
    violations: list[Violation] = []
    kebab = re.compile(KEBAB_CASE_PATTERN)

    def check(path: Path, expected_name: str, enforce_line_limit: bool) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        frontmatter = _parse_frontmatter(text)
        if frontmatter is None:
            violations.append(
                Violation(gate, str(path), "missing or malformed YAML frontmatter")
            )
            return
        for key in REQUIRED_FRONTMATTER_KEYS:
            value = frontmatter.get(key)
            if not isinstance(value, str) or not value.strip():
                violations.append(
                    Violation(gate, str(path), f"frontmatter key {key!r} is required")
                )
                return
        name = str(frontmatter["name"])
        if not kebab.fullmatch(name):
            violations.append(
                Violation(gate, str(path), f"frontmatter name {name!r} not kebab-case")
            )
        if name != expected_name:
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"frontmatter name {name!r} != expected {expected_name!r}",
                )
            )
        description = str(frontmatter["description"])
        if len(description) > config.max_frontmatter_description_chars:
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"description exceeds "
                    f"{config.max_frontmatter_description_chars} characters",
                )
            )
        line_count = len(text.splitlines())
        if enforce_line_limit and line_count > config.max_skill_lines:
            violations.append(
                Violation(
                    gate,
                    str(path),
                    f"{line_count} lines exceeds the {config.max_skill_lines}-line "
                    "progressive-disclosure limit; move detail to references/",
                )
            )

    for plugin_dir in discover_plugin_dirs(config):
        skills_dir = plugin_dir / config.skills_dirname
        if skills_dir.is_dir():
            for skill_md in sorted(skills_dir.glob(f"*/{SKILL_FILENAME}")):
                check(skill_md, skill_md.parent.name, enforce_line_limit=True)
        agents_dir = plugin_dir / config.agents_dirname
        if agents_dir.is_dir():
            for agent_md in sorted(agents_dir.glob("*.md")):
                check(agent_md, agent_md.stem, enforce_line_limit=False)
    return violations


#: Gates that need only the config (schema-independent).
STANDALONE_GATES: tuple[Callable[[ValidatorConfig], list[Violation]], ...] = (
    gate_vendored_runtime,
    gate_path_literals,
    gate_stdlib_imports,
    gate_frontmatter,
)


def run_all_gates(config: ValidatorConfig) -> list[Violation]:
    """Run every gate; returns all violations (empty list = clean)."""
    violations: list[Violation] = []
    marketplace, marketplace_violations = load_marketplace(config)
    violations.extend(marketplace_violations)
    manifests, manifest_violations = load_manifests(config)
    violations.extend(manifest_violations)
    if marketplace is not None:
        violations.extend(gate_catalog_parity(config, marketplace, manifests))
        violations.extend(gate_pins(config, marketplace, manifests))
    for gate in STANDALONE_GATES:
        violations.extend(gate(config))
    return violations
