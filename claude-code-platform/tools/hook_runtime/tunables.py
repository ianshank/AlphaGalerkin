"""Tunables: plugin-owned defaults file + ``CCP_*`` env overrides.

The plugin ships its defaults in ``config/defaults.json`` (read via the
plugin root — never a hardcoded path). Any top-level key ``foo`` can be
overridden by the environment variable ``CCP_FOO``; the override is
coerced to the type of the shipped default, so misconfiguration fails
loudly instead of silently changing semantics.

Note: plugin-root ``settings.json`` is NOT used for tunables — Claude Code
reserves it for ``agent``/``subagentStatusLine`` (review Finding 3).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import environ
from pathlib import Path
from typing import Any

from . import constants


class TunablesError(ValueError):
    """Raised when the defaults file or an env override is invalid."""


def plugin_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the plugin root from ``CLAUDE_PLUGIN_ROOT``.

    Falls back to the vendored layout (``<plugin>/hooks/scripts/_runtime``)
    when the variable is unset, e.g. when a script is exercised directly
    in tests.
    """
    source = environ if env is None else env
    configured = source.get(constants.ENV_PLUGIN_ROOT, "")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3]


def load_tunables(
    root: Path | None = None,
    *,
    relpath: str = constants.DEFAULT_TUNABLES_RELPATH,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load defaults from the plugin's tunables file and apply env overrides.

    A missing file yields ``{}`` plus overrides (hooks must run with
    sensible in-code fallbacks); an unreadable or malformed file raises
    :class:`TunablesError` so the fail-safe wrapper can log it.
    """
    source = environ if env is None else env
    base = plugin_root(source) if root is None else root
    path = base / relpath
    defaults: dict[str, Any] = {}
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TunablesError(f"unreadable tunables file {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise TunablesError(f"tunables file {path} must contain a JSON object")
        # Forward compatibility: a newer schema_version may carry unknown
        # keys; known keys keep working, unknown keys pass through untouched.
        defaults = {k: v for k, v in document.items() if k != "schema_version"}
    return apply_env_overrides(defaults, source)


def apply_env_overrides(
    defaults: Mapping[str, Any], env: Mapping[str, str]
) -> dict[str, Any]:
    """Return ``defaults`` with ``CCP_<KEY>`` env values applied (env wins)."""
    merged = dict(defaults)
    for key, default in defaults.items():
        env_key = constants.ENV_PREFIX + key.upper()
        if env_key in env:
            merged[key] = _coerce(env[env_key], default, env_key)
    return merged


def _coerce(raw: str, default: Any, env_key: str) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in constants.TRUTHY_VALUES
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise TunablesError(f"{env_key} must be an integer, got {raw!r}") from exc
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise TunablesError(f"{env_key} must be a number, got {raw!r}") from exc
    if isinstance(default, (list, dict)):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TunablesError(f"{env_key} must be JSON, got {raw!r}") from exc
        if not isinstance(value, type(default)):
            raise TunablesError(
                f"{env_key} must decode to {type(default).__name__}, "
                f"got {type(value).__name__}"
            )
        return value
    return raw
