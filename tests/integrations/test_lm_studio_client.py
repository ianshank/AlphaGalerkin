"""LMStudioClient tests — mocked openai SDK, retry/fallback/error paths."""

from __future__ import annotations

import inspect

import pytest

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.schema import (
    LMStudioActionSpaceMismatchError,
    LMStudioConnectionError,
    LMStudioParseError,
)
from tests.integrations.conftest import (
    FakeOpenAIModule,
    fake_completion,
    policy_json,
    policy_json_with_logits,
)


def _config(**overrides: object) -> LMStudioConfig:
    return LMStudioConfig(
        preflight_on_construct=False,
        backoff_base_s=0.0001,
        **overrides,  # type: ignore[arg-type]
    )


def test_happy_path_emits_log_event(
    fake_openai: FakeOpenAIModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)  # warm last_client
    client = LMStudioClient(_config())
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json(n_actions=4, value=0.25))
    ]
    response = client.complete_policy(
        "prompt-body",
        expected_action_size=4,
        seed=7,
    )
    assert response.value == pytest.approx(0.25)
    assert len(response.logits) == 4
    call = fake_openai.last_client.chat.completions.calls[0]
    assert call["seed"] == 7
    assert call["response_format"] == {"type": "json_object"}
    assert call["max_tokens"] == LMStudioConfig().max_tokens


def test_parse_error_retries_then_succeeds(fake_openai: FakeOpenAIModule) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config())
    fake_openai.last_client.chat.completions.responses = [
        fake_completion("not-json {{{"),
        fake_completion(policy_json(n_actions=3)),
    ]
    response = client.complete_policy("p", expected_action_size=3, seed=1)
    assert len(response.logits) == 3


def test_action_size_mismatch_retries_with_corrective_user_turn(
    fake_openai: FakeOpenAIModule,
) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config())
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json_with_logits([0.1, 0.2])),
        fake_completion(policy_json(n_actions=4)),
    ]
    response = client.complete_policy("p", expected_action_size=4, seed=1)
    assert len(response.logits) == 4
    second_messages = fake_openai.last_client.chat.completions.calls[1]["messages"]
    corrective = second_messages[-1]
    assert corrective["role"] == "user"
    assert "logits of length 2" in corrective["content"]
    assert "action space size is 4" in corrective["content"]


def test_action_size_mismatch_then_success_does_not_raise(
    fake_openai: FakeOpenAIModule,
) -> None:
    """Branch-coverage: mismatch on attempt 1, success on attempt 2."""
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=1))
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json_with_logits([0.1])),
        fake_completion(policy_json(n_actions=4)),
    ]
    response = client.complete_policy("p", expected_action_size=4, seed=2)
    assert len(response.logits) == 4


def test_exhausted_retries_raises_by_default(fake_openai: FakeOpenAIModule) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=1))
    fake_openai.last_client.chat.completions.responses = [
        fake_completion("oops"),
        fake_completion("still bad"),
    ]
    with pytest.raises(LMStudioParseError):
        client.complete_policy("p", expected_action_size=3, seed=1)


def test_connection_error_retries_then_raises(fake_openai: FakeOpenAIModule) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=2))
    err = fake_openai.APIConnectionError("network down")
    fake_openai.last_client.chat.completions.responses = [err, err, err]
    with pytest.raises(LMStudioConnectionError):
        client.complete_policy("p", expected_action_size=3, seed=1)


def test_timeout_error_distinct_from_connection(fake_openai: FakeOpenAIModule) -> None:
    """Branch-coverage: APITimeoutError must also coerce to LMStudioConnectionError."""
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=0))
    fake_openai.last_client.chat.completions.responses = [
        fake_openai.APITimeoutError("timed out"),
    ]
    with pytest.raises(LMStudioConnectionError) as exc_info:
        client.complete_policy("p", expected_action_size=3, seed=1)
    assert "APITimeoutError" in str(exc_info.value)


def test_action_size_mismatch_exhausts_then_raises(
    fake_openai: FakeOpenAIModule,
) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=1))
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json_with_logits([0.1])),
        fake_completion(policy_json_with_logits([0.1, 0.2])),
    ]
    with pytest.raises(LMStudioActionSpaceMismatchError):
        client.complete_policy("p", expected_action_size=4, seed=1)


def test_enabled_false_refuses_construction(fake_openai: FakeOpenAIModule) -> None:
    with pytest.raises(Exception):  # noqa: PT011 - typed LMStudioError covered elsewhere
        LMStudioClient(_config(enabled=False))


def test_preflight_on_construct_runs_and_passes(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the `preflight_on_construct=True` happy path (client.py:117-126)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    client = LMStudioClient(LMStudioConfig(preflight_on_construct=True, backoff_base_s=0.0001))
    assert client.sdk_client is not None


def test_preflight_on_construct_failure_raises(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight failure inside `__init__` must raise `LMStudioPreflightError`."""
    import torch

    from src.integrations.lm_studio.schema import LMStudioPreflightError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = LMStudioConfig(
        preflight_on_construct=True,
        model="not-in-server-response",
        backoff_base_s=0.0001,
    )
    with pytest.raises(LMStudioPreflightError):
        LMStudioClient(config)


def test_context_manager_closes_sdk_client(fake_openai: FakeOpenAIModule) -> None:
    """`with LMStudioClient(...) as client` must close the SDK client on exit."""
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    with LMStudioClient(_config()) as client:
        assert client.sdk_client is not None
    # The fake records `closed=True` when `close()` is called.
    assert fake_openai.last_client.closed is True


def test_unknown_exception_coerces_to_connection_error(
    fake_openai: FakeOpenAIModule,
) -> None:
    """Unknown exception types fall through to a generic `LMStudioConnectionError`."""
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=0))
    fake_openai.last_client.chat.completions.responses = [ValueError("weird")]
    with pytest.raises(LMStudioConnectionError) as exc_info:
        client.complete_policy("p", expected_action_size=3, seed=1)
    assert "ValueError" in str(exc_info.value)


def test_extract_content_missing_choices(fake_openai: FakeOpenAIModule) -> None:
    """Defensive branches when SDK returns malformed completion shapes."""
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config(max_retries=0))
    bad_completion = type(
        "BadCompletion",
        (),
        {"choices": [], "usage": None},
    )()
    fake_openai.last_client.chat.completions.responses = [bad_completion]
    with pytest.raises(LMStudioParseError):
        client.complete_policy("p", expected_action_size=3, seed=1)


def test_action_size_mismatch_corrective_messages_have_three_entries(
    fake_openai: FakeOpenAIModule,
) -> None:
    """Tighten the corrective-user-turn assertion: must contain system + 2 user turns."""
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config())
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json_with_logits([0.1, 0.2])),
        fake_completion(policy_json(n_actions=4)),
    ]
    client.complete_policy("p", expected_action_size=4, seed=1)
    second_messages = fake_openai.last_client.chat.completions.calls[1]["messages"]
    assert len(second_messages) == 3
    assert second_messages[0]["role"] == "system"
    assert second_messages[1]["role"] == "user"
    assert second_messages[2]["role"] == "user"


def test_openai_sdk_signature_compat_sentinel() -> None:
    """If openai is installed, both `seed` and `response_format` must be accepted.

    Falls through silently if the openai SDK is absent (CPU CI without the
    `[lm-studio]` extra installed); the sentinel is a guard against
    incompatible upgrades, not a hard import.
    """
    try:
        import openai  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("openai SDK not installed; signature sentinel skipped")
    from openai import OpenAI

    sig = inspect.signature(OpenAI.__init__)
    assert "base_url" in sig.parameters
    assert "api_key" in sig.parameters
