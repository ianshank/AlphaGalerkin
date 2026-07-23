"""Pydantic schemas for marketplace, manifest, hooks, and pin documents.

Best-effort derivations from the official plugins-reference /
plugin-marketplaces docs (fetched 2026-07-03 — see CLAUDE.md doc-facts
log). All models use ``extra="ignore"`` so newer official fields never
fail validation (forward compatibility); required fields and formats are
enforced strictly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import GIT_SHA_PATTERN, KEBAB_CASE_PATTERN, SEMVER_PATTERN


class MarketplaceOwner(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    email: str | None = None


class GitHubSource(BaseModel):
    """Remote plugin source; ``sha`` (when set) takes precedence over ``ref``."""

    model_config = ConfigDict(extra="ignore")

    source: Literal["github"]
    repo: str = Field(min_length=1)
    ref: str | None = None
    sha: str | None = Field(default=None, pattern=GIT_SHA_PATTERN)


class MarketplacePluginEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(pattern=KEBAB_CASE_PATTERN)
    source: str | GitHubSource
    description: str | None = None
    version: str | None = Field(default=None, pattern=SEMVER_PATTERN)

    @field_validator("source")
    @classmethod
    def _relative_string_source(cls, value: str | GitHubSource) -> str | GitHubSource:
        if isinstance(value, str) and not value.startswith("./"):
            raise ValueError(
                "string sources must be repo-relative paths starting with './'"
            )
        return value


class MarketplaceDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(pattern=KEBAB_CASE_PATTERN)
    owner: MarketplaceOwner
    plugins: list[MarketplacePluginEntry] = Field(min_length=1)
    metadata: dict[str, Any] | None = None

    @field_validator("plugins")
    @classmethod
    def _unique_plugin_names(
        cls, value: list[MarketplacePluginEntry]
    ) -> list[MarketplacePluginEntry]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for entry in value:
            if entry.name in seen:
                duplicates.add(entry.name)
            seen.add(entry.name)
        if duplicates:
            raise ValueError(f"duplicate plugin names in catalog: {sorted(duplicates)}")
        return value


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(pattern=KEBAB_CASE_PATTERN)
    version: str = Field(pattern=SEMVER_PATTERN)
    description: str = Field(min_length=1)
    author: dict[str, Any] | None = None


class HookInvocation(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1)
    command: str | None = None
    timeout: int | None = Field(default=None, gt=0)


class HookMatcherGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    matcher: str | None = None
    hooks: list[HookInvocation] = Field(min_length=1)


class HooksDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    hooks: dict[str, list[HookMatcherGroup]] = Field(min_length=1)


class PinEntry(BaseModel):
    """One released plugin version pinned to an exact commit (ADR-0003)."""

    model_config = ConfigDict(extra="ignore")

    version: str = Field(pattern=SEMVER_PATTERN)
    ref: str | None = None
    sha: str = Field(pattern=GIT_SHA_PATTERN)


class PinsDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(ge=1)
    repo: str | None = Field(
        default=None,
        description="owner/repo used for self-referential github sources",
    )
    pins: dict[str, PinEntry] = Field(default_factory=dict)
