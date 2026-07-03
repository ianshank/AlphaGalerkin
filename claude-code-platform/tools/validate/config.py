"""Validator configuration — every gate limit and pattern is a typed field.

Dev-side only (pydantic is permitted here, never in hook scripts —
ADR-0002). Defaults encode the repo conventions; nothing in the gates
module carries inline literals.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

#: Kebab-case identifier required for marketplace/plugin names.
KEBAB_CASE_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

#: Semantic version (MAJOR.MINOR.PATCH with optional pre-release/build).
SEMVER_PATTERN = (
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?(?:\+[0-9A-Za-z][0-9A-Za-z.-]*)?$"
)

#: Full-length lowercase git commit sha.
GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"


class ValidatorConfig(BaseModel):
    """Paths, limits, and patterns driving the static validation gates."""

    model_config = ConfigDict(frozen=True)

    root: Path = Field(description="Marketplace repository root")
    plugins_dirname: str = Field(default="plugins")
    marketplace_relpath: str = Field(default=".claude-plugin/marketplace.json")
    manifest_relpath: str = Field(default=".claude-plugin/plugin.json")
    hooks_dirname: str = Field(default="hooks")
    hooks_relpath: str = Field(default="hooks/hooks.json")
    pins_relpath: str = Field(default="release/pins.json")
    runtime_src_relpath: str = Field(
        default="tools/hook_runtime",
        description="Canonical hook runtime vendored into plugins",
    )
    vendored_runtime_relpath: str = Field(
        default="hooks/scripts/_runtime",
        description="Vendored runtime location inside each plugin",
    )
    hook_scripts_relpath: str = Field(default="hooks/scripts")
    skills_dirname: str = Field(default="skills")
    agents_dirname: str = Field(default="agents")

    max_skill_lines: int = Field(
        default=150,
        gt=0,
        description="Progressive-disclosure ceiling for SKILL.md files",
    )
    max_frontmatter_description_chars: int = Field(default=1024, gt=0)

    plugin_root_token: str = Field(
        default="${CLAUDE_PLUGIN_ROOT}",
        description="Required token in every hooks.json command path",
    )
    path_literal_patterns: tuple[str, ...] = Field(
        default=(
            # No trailing-slash requirement: "/home/user" without a
            # trailing segment is just as machine-specific (review F6).
            r"(?<![\[\w/])/Users/[A-Za-z][A-Za-z0-9._-]*",
            r"(?<![\[\w/])/home/[A-Za-z][A-Za-z0-9._-]*",
            r"[A-Za-z]:\\Users\\[A-Za-z][A-Za-z0-9._-]*",
            r"(?<![\w/])~/[A-Za-z0-9._-]+",
        ),
        description="Regexes flagging machine-specific literal paths",
    )
    path_literal_exclude_relpaths: tuple[str, ...] = Field(
        default=("config/defaults.json",),
        description=(
            "Plugin-relative files exempt from the path-literal gate "
            "(e.g. files that legitimately contain such regexes as data)"
        ),
    )
    scannable_suffixes: tuple[str, ...] = Field(
        default=(".md", ".json", ".py", ".sh", ".yaml", ".yml", ".toml", ".txt"),
    )
    vendored_import_name: str = Field(
        default="_runtime",
        description="Import name of the vendored runtime inside hook scripts",
    )
