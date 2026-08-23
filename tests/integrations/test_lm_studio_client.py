"""LMStudioClient tests — mocked openai SDK, retry/fallback/error paths."""

from __future__ import annotations

import inspect

import pytest

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.prompt import prompt_hash as compute_prompt_hash
from src.integrations.lm_studio.schema import (
    LMStudioActionSpaceMismatchError,
    LMStudioConnectionError,
    LMStudioError,
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
    from src.integrations.lm_studio.schema import LMStudioError

    with pytest.raises(LMStudioError, match="enabled=False"):
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


def test_expected_action_size_must_be_positive(fake_openai: FakeOpenAIModule) -> None:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(_config())
    with pytest.raises(ValueError, match="expected_action_size must be > 0"):
        client.complete_policy("p", expected_action_size=0, seed=1)


# ---------------------------------------------------------------------------
# Retry/backoff correctness (exponential growth, config-bound retry count,
# permanent-vs-transient treatment). See src/integrations/AGENT.md and
# CLAUDE.md's LLM-prior Regression Surface row.
# ---------------------------------------------------------------------------


def test_backoff_is_exponential_and_bounded_by_config(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backoff must double every attempt and be bounded by config.

    `backoff_base_s * 2 ** attempt` (exponential), and the retry count must
    be bounded by `config.max_retries` -- not a hardcoded count or a
    fixed/linear sleep duration.
    """
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "src.integrations.lm_studio.client.time.sleep",
        lambda d: sleep_calls.append(d),
    )
    base = 1.0
    client = LMStudioClient(
        LMStudioConfig(preflight_on_construct=False, backoff_base_s=base, max_retries=3)
    )
    err = fake_openai.APIConnectionError("network down")
    fake_openai.last_client.chat.completions.responses = [err, err, err, err]
    with pytest.raises(LMStudioConnectionError):
        client.complete_policy("p", expected_action_size=3, seed=1)

    # 3 retries -> 3 sleeps, each doubling: base*2^0, base*2^1, base*2^2.
    assert sleep_calls == [base * 1, base * 2, base * 4]
    # Not linear (base, 2*base, 3*base) and not constant (base, base, base).
    assert sleep_calls != [base, base * 2, base * 3]
    assert sleep_calls != [base, base, base]
    # Exactly max_retries + 1 SDK calls: the initial attempt plus 3 retries.
    assert len(fake_openai.last_client.chat.completions.calls) == 4


def test_permanent_error_fails_immediately_no_retry(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-retryable SDK error (auth failure) must NOT be retried.

    Exactly one SDK call, zero sleeps, and the typed exception is the plain
    `LMStudioError` parent -- not `LMStudioConnectionError`, which is
    reserved for genuinely transient failures.
    """
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "src.integrations.lm_studio.client.time.sleep",
        lambda d: sleep_calls.append(d),
    )
    # Generous retry budget: if the permanent-vs-transient distinction were
    # broken, this budget would be enough to mask the bug by retrying all
    # the way to the queued success response below.
    client = LMStudioClient(
        LMStudioConfig(preflight_on_construct=False, backoff_base_s=1.0, max_retries=5)
    )
    auth_err = fake_openai.AuthenticationError("invalid API key")
    success = fake_completion(policy_json(n_actions=3))
    fake_openai.last_client.chat.completions.responses = [auth_err, success]

    with pytest.raises(LMStudioError) as exc_info:
        client.complete_policy("p", expected_action_size=3, seed=1)

    # Coerced to the plain parent type, NOT the retryable subclass.
    assert type(exc_info.value) is LMStudioError
    assert not isinstance(exc_info.value, LMStudioConnectionError)
    assert "AuthenticationError" in str(exc_info.value)

    # Zero backoff sleeps: the permanent failure short-circuits the retry
    # loop entirely rather than paying `max_retries` rounds of backoff.
    assert sleep_calls == []
    # Exactly one SDK call: the queued `success` response was never touched.
    calls = fake_openai.last_client.chat.completions.calls
    assert len(calls) == 1
    assert fake_openai.last_client.chat.completions.responses == [success]


def test_transient_vs_permanent_treatment_under_multi_failure_sequence(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end proof that transient and permanent failures are treated differently.

    A realistic sequence of `complete_policy` calls (as MCTS would issue one
    per leaf evaluation):

        call 1: transient (APIConnectionError) x2, then success -> retries, recovers
        call 2: permanent (NotFoundError) -> fails on the FIRST attempt, no retry
        call 3: transient (RateLimitError) exhausts all retries -> raises

    A regression that treated permanent failures as retryable would show up
    here as call 2 burning more than one SDK call / a non-zero sleep.
    """
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "src.integrations.lm_studio.client.time.sleep",
        lambda d: sleep_calls.append(d),
    )
    client = LMStudioClient(
        LMStudioConfig(preflight_on_construct=False, backoff_base_s=0.5, max_retries=2)
    )
    completions = fake_openai.last_client.chat.completions

    # --- call 1: transient, recovers after 2 retries ---
    conn_err = fake_openai.APIConnectionError("down")
    completions.responses = [conn_err, conn_err, fake_completion(policy_json(n_actions=3))]
    response = client.complete_policy("p1", expected_action_size=3, seed=1)
    assert len(response.logits) == 3
    assert len(completions.calls) == 3  # initial + 2 retries
    assert sleep_calls == [0.5 * 1, 0.5 * 2]

    sleep_calls.clear()
    completions.calls.clear()

    # --- call 2: permanent, fails on the very first attempt ---
    not_found = fake_openai.NotFoundError("model not loaded")
    success_never_reached = fake_completion(policy_json(n_actions=3))
    completions.responses = [not_found, success_never_reached]
    with pytest.raises(LMStudioError) as exc_info:
        client.complete_policy("p2", expected_action_size=3, seed=2)
    assert not isinstance(exc_info.value, LMStudioConnectionError)
    assert len(completions.calls) == 1  # no retry burned on a permanent failure
    assert sleep_calls == []  # zero backoff -- no wasted latency
    assert completions.responses == [success_never_reached]  # untouched

    sleep_calls.clear()
    completions.calls.clear()

    # --- call 3: transient, exhausts the retry budget and raises ---
    rate_err = fake_openai.RateLimitError("slow down")
    completions.responses = [rate_err, rate_err, rate_err]
    with pytest.raises(LMStudioConnectionError):
        client.complete_policy("p3", expected_action_size=3, seed=3)
    assert len(completions.calls) == 3  # initial + 2 retries, bounded by max_retries=2
    assert sleep_calls == [0.5 * 1, 0.5 * 2]


def test_retry_and_call_events_emitted_with_correct_fields_per_attempt(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry and call events must fire per-attempt with correct fields.

    `lm_studio_retry` must fire once per retry attempt (not only alongside
    the final outcome) with the right `attempt`/`reason`/`prompt_hash`, and
    `lm_studio_call` must fire exactly once per `complete_policy` call with
    `retries_used`/`parse_ok`/`tokens_in`/`tokens_out`/`prompt_hash` populated.
    """
    import src.integrations.lm_studio.client as client_mod

    recorded: list[tuple[str, str, dict[str, object]]] = []

    class _RecordingLogger:
        def info(self, event: str, **kw: object) -> None:
            recorded.append(("info", event, kw))

        def warning(self, event: str, **kw: object) -> None:
            recorded.append(("warning", event, kw))

    monkeypatch.setattr(client_mod, "logger", _RecordingLogger())
    monkeypatch.setattr("src.integrations.lm_studio.client.time.sleep", lambda d: None)

    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(
        LMStudioConfig(preflight_on_construct=False, backoff_base_s=0.001, max_retries=3)
    )
    fake_openai.last_client.chat.completions.responses = [
        fake_completion("not-json {{{"),  # -> LMStudioParseError
        fake_completion(policy_json_with_logits([0.1])),  # -> mismatch (expects 4)
        fake_openai.APIConnectionError("down"),  # -> LMStudioConnectionError
        fake_completion(policy_json(n_actions=4, value=0.5)),  # -> success
    ]
    prompt = "prompt-body-for-log-test"
    response = client.complete_policy(prompt, expected_action_size=4, seed=9)
    assert response.value == pytest.approx(0.5)

    expected_hash = compute_prompt_hash(prompt)

    retry_events = [kw for _level, event, kw in recorded if event == "lm_studio_retry"]
    assert [(e["attempt"], e["reason"]) for e in retry_events] == [
        (0, "LMStudioParseError"),
        (1, "LMStudioActionSpaceMismatchError"),
        (2, "LMStudioConnectionError"),
    ]
    assert all(e["prompt_hash"] == expected_hash for e in retry_events)
    # All three retry events are warnings, not silent/absent.
    assert all(level == "warning" for level, event, _kw in recorded if event == "lm_studio_retry")

    call_events = [kw for _level, event, kw in recorded if event == "lm_studio_call"]
    assert len(call_events) == 1  # exactly one terminal event per complete_policy call
    call = call_events[0]
    assert call["prompt_hash"] == expected_hash
    assert call["retries_used"] == 3
    assert call["parse_ok"] is True
    assert call["tokens_in"] == 11
    assert call["tokens_out"] == 22
    assert call["error"] is None
    assert call["latency_ms"] >= 0.0


def test_call_event_emitted_on_terminal_failure_too(
    fake_openai: FakeOpenAIModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lm_studio_call` must still fire when retries are exhausted.

    With `parse_ok=False` and the error type populated, not only on success.
    """
    import src.integrations.lm_studio.client as client_mod

    recorded: list[tuple[str, dict[str, object]]] = []

    class _RecordingLogger:
        def info(self, event: str, **kw: object) -> None:
            recorded.append((event, kw))

        def warning(self, event: str, **kw: object) -> None:
            pass

    monkeypatch.setattr(client_mod, "logger", _RecordingLogger())
    monkeypatch.setattr("src.integrations.lm_studio.client.time.sleep", lambda d: None)

    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    client = LMStudioClient(
        LMStudioConfig(preflight_on_construct=False, backoff_base_s=0.001, max_retries=1)
    )
    fake_openai.last_client.chat.completions.responses = [
        fake_completion("bad"),
        fake_completion("still bad"),
    ]
    with pytest.raises(LMStudioParseError):
        client.complete_policy("p", expected_action_size=3, seed=1)

    call_events = [kw for event, kw in recorded if event == "lm_studio_call"]
    assert len(call_events) == 1
    assert call_events[0]["parse_ok"] is False
    assert call_events[0]["retries_used"] == 1
    assert call_events[0]["error"] == "LMStudioParseError"


def test_openai_sdk_signature_compat_sentinel() -> None:
    """If openai is installed, the SDK surface our client depends on must hold.

    Specifically:
      - ``openai.OpenAI(base_url=..., api_key=...)`` must remain constructable.
      - ``chat.completions.create`` must accept both ``seed=`` and
        ``response_format=`` simultaneously (we pass both on every call).

    Falls through silently if the openai SDK is absent (CPU CI without the
    ``[lm-studio]`` extra installed); the sentinel is a guard against
    incompatible upgrades, not a hard import.
    """
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai SDK not installed; signature sentinel skipped")
    from openai import OpenAI

    # Constructor surface
    constructor_sig = inspect.signature(OpenAI.__init__)
    assert "base_url" in constructor_sig.parameters
    assert "api_key" in constructor_sig.parameters

    # Completions surface — `seed` and `response_format` are the two kwargs
    # we pass on every call. If either disappears, retries and JSON-mode
    # both break and this sentinel surfaces the breakage at test time.
    # `OpenAI.chat` is a cached_property so we need an instance; the SDK
    # defers API-key validation until the first request, so a dummy key
    # and URL are fine for inspecting the bound method's signature.
    sdk_client = OpenAI(api_key="sentinel", base_url="http://127.0.0.1:0/v1")
    completions_sig = inspect.signature(sdk_client.chat.completions.create)
    assert "seed" in completions_sig.parameters
    assert "response_format" in completions_sig.parameters
