"""Unit tests for hook stdin parsing (tools.hook_runtime.models)."""

from __future__ import annotations

import io
import json

import pytest

from tools.hook_runtime.models import HookInput, HookInputError, parse_hook_input

FULL_PAYLOAD = {
    "hook_event_name": "PostToolUse",
    "session_id": "sess-1",
    "cwd": "/workspace/project",
    "transcript_path": "/workspace/transcript.jsonl",
    "tool_name": "Write",
    "tool_input": {"file_path": "a.py", "content": "x = 1\n"},
    "tool_response": {"success": True},
}


def test_parse_full_payload() -> None:
    event = parse_hook_input(json.dumps(FULL_PAYLOAD))
    assert event.hook_event_name == "PostToolUse"
    assert event.tool_name == "Write"
    assert event.tool_input["file_path"] == "a.py"
    assert event.tool_response == {"success": True}


def test_parse_accepts_stream() -> None:
    event = parse_hook_input(io.StringIO(json.dumps(FULL_PAYLOAD)))
    assert event.session_id == "sess-1"


def test_minimal_payload_defaults() -> None:
    event = parse_hook_input(json.dumps({"hook_event_name": "SessionStart"}))
    assert event.hook_event_name == "SessionStart"
    assert event.tool_name == ""
    assert event.tool_input == {}
    assert event.tool_response == {}


def test_unknown_fields_preserved_in_raw() -> None:
    payload = dict(FULL_PAYLOAD, future_field={"nested": 1})
    event = parse_hook_input(json.dumps(payload))
    assert event.raw["future_field"] == {"nested": 1}


def test_null_known_field_treated_as_absent() -> None:
    event = parse_hook_input(
        json.dumps({"hook_event_name": "Stop", "tool_name": None, "tool_input": None})
    )
    assert event.tool_name == ""
    assert event.tool_input == {}


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[1, 2]", '"string"', "42"],
)
def test_malformed_payloads_raise(text: str) -> None:
    with pytest.raises(HookInputError):
        parse_hook_input(text)


def test_missing_event_name_raises() -> None:
    with pytest.raises(HookInputError, match="hook_event_name"):
        parse_hook_input(json.dumps({"tool_name": "Write"}))


def test_non_string_known_field_raises() -> None:
    with pytest.raises(HookInputError, match="tool_name"):
        parse_hook_input(json.dumps({"hook_event_name": "X", "tool_name": 5}))


def test_non_object_tool_input_raises() -> None:
    with pytest.raises(HookInputError, match="tool_input"):
        parse_hook_input(json.dumps({"hook_event_name": "X", "tool_input": [1]}))


def test_non_object_tool_response_coerced_leniently() -> None:
    event = parse_hook_input(
        json.dumps({"hook_event_name": "X", "tool_response": "ok"})
    )
    assert event.tool_response == {"value": "ok"}


def test_hook_input_is_frozen() -> None:
    event = HookInput(hook_event_name="X")
    with pytest.raises(AttributeError):
        event.tool_name = "Write"  # type: ignore[misc]
