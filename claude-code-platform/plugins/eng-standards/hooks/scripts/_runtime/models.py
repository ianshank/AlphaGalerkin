"""Typed parsing of Claude Code hook stdin payloads (stdlib-only).

Hook scripts receive a single JSON object on stdin. :func:`parse_hook_input`
validates the fields the runtime relies on and preserves the complete
payload in :attr:`HookInput.raw`, so fields added by future Claude Code
versions remain accessible without a runtime change (forward
compatibility).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import IO, Any


class HookInputError(ValueError):
    """Raised when the stdin payload is not a valid hook event."""


@dataclass(frozen=True)
class HookInput:
    """Validated hook event.

    ``tool_input`` is strictly required to be an object when present
    (tool parameters are always objects); ``tool_response`` is leniently
    coerced because its shape varies by tool.
    """

    hook_event_name: str
    session_id: str = ""
    cwd: str = ""
    transcript_path: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_response: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _string_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HookInputError(
            f"field {key!r} must be a string, got {type(value).__name__}"
        )
    return value


def _strict_object_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HookInputError(
            f"field {key!r} must be an object, got {type(value).__name__}"
        )
    return value


def _lenient_object_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"value": value}


def parse_hook_input(source: str | IO[str]) -> HookInput:
    """Parse and validate a hook event from raw JSON text or a stream.

    Raises:
        HookInputError: on malformed JSON, a non-object payload, a missing
            ``hook_event_name``, or wrongly typed known fields.
    """
    text = source if isinstance(source, str) else source.read()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HookInputError(f"stdin is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HookInputError(
            f"hook payload must be a JSON object, got {type(payload).__name__}"
        )
    event_name = _string_field(payload, "hook_event_name")
    if not event_name:
        raise HookInputError("hook payload is missing 'hook_event_name'")
    return HookInput(
        hook_event_name=event_name,
        session_id=_string_field(payload, "session_id"),
        cwd=_string_field(payload, "cwd"),
        transcript_path=_string_field(payload, "transcript_path"),
        tool_name=_string_field(payload, "tool_name"),
        tool_input=_strict_object_field(payload, "tool_input"),
        tool_response=_lenient_object_field(payload, "tool_response"),
        raw=payload,
    )
