"""Unit tests for tunables loading and env overrides (tools.hook_runtime.tunables)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tools.hook_runtime import constants
from tools.hook_runtime.tunables import (
    TunablesError,
    apply_env_overrides,
    load_tunables,
    plugin_root,
)


@pytest.fixture()
def plugin_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "defaults.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gating": False,
                "max_file_bytes": 1024,
                "ratio": 0.5,
                "scan_tools": ["Write"],
                "patterns": {"secret": ["x"]},
                "label": "default",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestLoadTunables:
    def test_loads_defaults_without_schema_version(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={})
        assert tunables["max_file_bytes"] == 1024
        assert "schema_version" not in tunables

    def test_missing_file_yields_empty(self, tmp_path: Path) -> None:
        assert load_tunables(tmp_path, env={}) == {}

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "defaults.json").write_text("{", encoding="utf-8")
        with pytest.raises(TunablesError, match="unreadable"):
            load_tunables(tmp_path, env={})

    def test_non_object_document_raises(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "defaults.json").write_text("[1]", encoding="utf-8")
        with pytest.raises(TunablesError, match="JSON object"):
            load_tunables(tmp_path, env={})

    def test_forward_compat_newer_schema_keeps_keys(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "defaults.json").write_text(
            json.dumps({"schema_version": 99, "known": 1, "future_key": {"a": 2}}),
            encoding="utf-8",
        )
        tunables = load_tunables(tmp_path, env={})
        assert tunables == {"known": 1, "future_key": {"a": 2}}


class TestEnvOverrides:
    def test_bool_coercion(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_GATING": "true"})
        assert tunables["gating"] is True

    def test_int_coercion(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_MAX_FILE_BYTES": "99"})
        assert tunables["max_file_bytes"] == 99

    def test_float_coercion(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_RATIO": "0.75"})
        assert tunables["ratio"] == 0.75

    def test_list_coercion_from_json(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_SCAN_TOOLS": '["Edit"]'})
        assert tunables["scan_tools"] == ["Edit"]

    def test_dict_coercion_from_json(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_PATTERNS": '{"a": []}'})
        assert tunables["patterns"] == {"a": []}

    def test_string_passthrough(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_LABEL": "override"})
        assert tunables["label"] == "override"

    def test_bad_int_raises(self, plugin_dir: Path) -> None:
        with pytest.raises(TunablesError, match="CCP_MAX_FILE_BYTES"):
            load_tunables(plugin_dir, env={"CCP_MAX_FILE_BYTES": "many"})

    def test_bad_json_raises(self, plugin_dir: Path) -> None:
        with pytest.raises(TunablesError, match="CCP_SCAN_TOOLS"):
            load_tunables(plugin_dir, env={"CCP_SCAN_TOOLS": "not-json"})

    def test_wrong_json_type_raises(self, plugin_dir: Path) -> None:
        with pytest.raises(TunablesError, match="list"):
            load_tunables(plugin_dir, env={"CCP_SCAN_TOOLS": '{"a": 1}'})

    def test_unknown_env_keys_ignored(self, plugin_dir: Path) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_NOT_A_KEY": "1"})
        assert "not_a_key" not in tunables

    @given(
        defaults=st.dictionaries(
            st.from_regex(r"[a-z][a-z_]{0,10}", fullmatch=True),
            st.one_of(st.booleans(), st.integers(), st.text(max_size=20)),
            max_size=5,
        )
    )
    def test_no_env_is_identity(self, defaults: dict[str, object]) -> None:
        assert apply_env_overrides(defaults, {}) == defaults


class TestPluginRoot:
    def test_env_wins(self, tmp_path: Path) -> None:
        env = {constants.ENV_PLUGIN_ROOT: str(tmp_path)}
        assert plugin_root(env) == tmp_path

    def test_fallback_is_vendored_layout(self) -> None:
        # canonical file lives at tools/hook_runtime/tunables.py; parents[3]
        # mirrors <plugin>/hooks/scripts/_runtime → plugin root depth.
        assert plugin_root({}).is_dir()


class TestStrictBoolCoercion:
    """Copilot review: a boolean typo must fail loudly, not read as False."""

    def test_bool_typo_raises(self, plugin_dir: Path) -> None:
        with pytest.raises(TunablesError, match="CCP_GATING must be a boolean"):
            load_tunables(plugin_dir, env={"CCP_GATING": "tru"})

    @pytest.mark.parametrize("value", ["0", "false", "NO", "off", ""])
    def test_explicit_falsy_values_accepted(self, plugin_dir: Path, value: str) -> None:
        tunables = load_tunables(plugin_dir, env={"CCP_GATING": value})
        assert tunables["gating"] is False
