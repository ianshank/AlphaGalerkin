"""Unit tests for the pydantic document schemas in tools.validate.schemas."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from tools.validate.config import KEBAB_CASE_PATTERN
from tools.validate.schemas import (
    GitHubSource,
    HooksDocument,
    MarketplaceDocument,
    MarketplaceOwner,
    MarketplacePluginEntry,
    PinEntry,
    PinsDocument,
    PluginManifest,
)

VALID_SHA = "a" * 40


class TestKebabCaseNames:
    @pytest.mark.parametrize(
        "bad_name",
        ["Bad_Name", "UPPER", "has space", "trailing-", "-leading", "double--dash", ""],
    )
    def test_non_kebab_plugin_entry_name_rejected(self, bad_name: str) -> None:
        with pytest.raises(ValidationError):
            MarketplacePluginEntry(name=bad_name, source="./plugins/x")

    def test_kebab_case_name_accepted(self) -> None:
        entry = MarketplacePluginEntry(name="demo-plugin-2", source="./plugins/x")
        assert entry.name == "demo-plugin-2"

    @settings(max_examples=20, deadline=None)
    @given(name=st.from_regex(KEBAB_CASE_PATTERN, fullmatch=True))
    def test_generated_kebab_names_always_validate(self, name: str) -> None:
        entry = MarketplacePluginEntry(name=name, source="./plugins/x")
        assert entry.name == name


class TestSemver:
    @pytest.mark.parametrize("bad_version", ["1.0", "v1.0.0", "1.0.0.0", "abc", ""])
    def test_non_semver_manifest_version_rejected(self, bad_version: str) -> None:
        with pytest.raises(ValidationError):
            PluginManifest(name="demo", version=bad_version, description="d")

    @pytest.mark.parametrize(
        "good_version", ["0.1.0", "1.2.3", "1.0.0-rc.1", "1.0.0+build.5"]
    )
    def test_semver_manifest_version_accepted(self, good_version: str) -> None:
        manifest = PluginManifest(name="demo", version=good_version, description="d")
        assert manifest.version == good_version


class TestGitShaPattern:
    def test_forty_hex_sha_accepted(self) -> None:
        source = GitHubSource(source="github", repo="owner/repo", sha=VALID_SHA)
        assert source.sha == VALID_SHA

    @pytest.mark.parametrize(
        "bad_sha",
        ["a" * 39, "a" * 41, "A" * 40, "g" * 40, "not-a-sha"],
    )
    def test_non_forty_hex_sha_rejected(self, bad_sha: str) -> None:
        with pytest.raises(ValidationError):
            GitHubSource(source="github", repo="owner/repo", sha=bad_sha)

    def test_pin_entry_sha_pattern_enforced(self) -> None:
        with pytest.raises(ValidationError):
            PinEntry(version="1.0.0", sha="deadbeef")


class TestRelativeStringSource:
    def test_string_source_without_dot_slash_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"starting with './'"):
            MarketplacePluginEntry(name="demo", source="plugins/demo")

    def test_string_source_with_dot_slash_accepted(self) -> None:
        entry = MarketplacePluginEntry(name="demo", source="./plugins/demo")
        assert entry.source == "./plugins/demo"

    def test_github_source_dict_accepted(self) -> None:
        entry = MarketplacePluginEntry(
            name="demo",
            source=GitHubSource(source="github", repo="owner/repo", sha=VALID_SHA),
        )
        assert isinstance(entry.source, GitHubSource)


class TestMarketplaceDocument:
    def test_duplicate_plugin_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate plugin names"):
            MarketplaceDocument(
                name="mkt",
                owner=MarketplaceOwner(name="o"),
                plugins=[
                    MarketplacePluginEntry(name="dup", source="./plugins/dup"),
                    MarketplacePluginEntry(name="dup", source="./plugins/dup"),
                ],
            )

    def test_empty_plugins_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MarketplaceDocument(
                name="mkt", owner=MarketplaceOwner(name="o"), plugins=[]
            )

    def test_extra_unknown_fields_ignored_forward_compat(self) -> None:
        document = MarketplaceDocument.model_validate(
            {
                "name": "mkt",
                "owner": {"name": "o", "future_owner_field": True},
                "plugins": [
                    {
                        "name": "demo",
                        "source": "./plugins/demo",
                        "future_entry_field": "x",
                    }
                ],
                "future_top_level_field": {"nested": 1},
            }
        )
        assert document.plugins[0].name == "demo"
        assert not hasattr(document, "future_top_level_field")


class TestHooksDocument:
    def test_minimal_valid_document(self) -> None:
        document = HooksDocument.model_validate(
            {
                "hooks": {
                    "PostToolUse": [
                        {"hooks": [{"type": "command", "command": "echo hi"}]}
                    ]
                }
            }
        )
        assert "PostToolUse" in document.hooks

    def test_empty_hooks_dict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HooksDocument.model_validate({"hooks": {}})


class TestPinsDocument:
    def test_defaults(self) -> None:
        document = PinsDocument(schema_version=1)
        assert document.repo is None
        assert document.pins == {}

    def test_schema_version_below_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PinsDocument(schema_version=0)
