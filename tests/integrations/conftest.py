"""Shared fixtures for `tests/integrations/`.

The fixtures here let the LM Studio integration be exercised on CPU CI
without importing the real ``openai`` SDK. Each fixture installs a fake
``openai`` module into ``src.integrations.lm_studio.client._import_openai``
via monkeypatch so the production import path is exercised end-to-end.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeChoice:
    """One choice from a fake `chat.completions.create` response."""

    def __init__(self, content: str) -> None:
        self.message = MagicMock()
        self.message.content = content


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 11, completion_tokens: int = 22) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    """A fake `ChatCompletion` matching the attributes our client reads."""

    def __init__(
        self,
        content: str,
        *,
        prompt_tokens: int = 11,
        completion_tokens: int = 22,
    ) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeChatCompletions:
    """Programmable chat.completions endpoint.

    Set ``responses`` to a list whose elements are either
    ``_FakeCompletion`` objects or ``Exception`` instances. Each
    ``.create`` call pops one element and either returns it or raises it.
    """

    def __init__(self) -> None:
        self.responses: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeChatCompletions exhausted; configure more responses")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeModelsList:
    def __init__(self, model_ids: Iterable[str]) -> None:
        self.data = [MagicMock(id=mid) for mid in model_ids]


class _FakeModels:
    def __init__(self, model_ids: Iterable[str]) -> None:
        self._ids = list(model_ids)

    def list(self) -> _FakeModelsList:
        return _FakeModelsList(self._ids)


class FakeOpenAIClient:
    """Stand-in for ``openai.OpenAI`` covering only the surface we touch."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        available_models: Iterable[str] = ("qwen2.5-14b-instruct",),
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.chat = MagicMock()
        self.chat.completions = _FakeChatCompletions()
        self.models = _FakeModels(available_models)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeAPIConnectionError(Exception):
    """Exception type matching ``openai.APIConnectionError``."""


class _FakeAPITimeoutError(Exception):
    """Exception type matching ``openai.APITimeoutError``."""


class FakeOpenAIModule:
    """Module-shaped fake exposing the exact symbols ``client.py`` imports."""

    def __init__(self, *, available_models: Iterable[str] = ("qwen2.5-14b-instruct",)) -> None:
        self._available_models = list(available_models)
        self.last_client: FakeOpenAIClient | None = None
        self.APIConnectionError = _FakeAPIConnectionError
        self.APITimeoutError = _FakeAPITimeoutError

    def OpenAI(  # noqa: N802 - matches openai SDK casing
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
    ) -> FakeOpenAIClient:
        self.last_client = FakeOpenAIClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            available_models=self._available_models,
        )
        return self.last_client


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> Generator[FakeOpenAIModule, None, None]:
    """Patch the lazy openai import to return a fake module."""
    fake = FakeOpenAIModule()

    def _fake_importer() -> FakeOpenAIModule:
        return fake

    monkeypatch.setattr(
        "src.integrations.lm_studio.client._import_openai",
        _fake_importer,
    )
    monkeypatch.setattr(
        "src.integrations.lm_studio.preflight._BYTES_PER_GIB",
        1024**3,
    )
    yield fake


def policy_json(
    *,
    n_actions: int,
    value: float = 0.0,
    reasoning: str = "stub",
    fill: float = 0.0,
) -> str:
    """Build a valid LMStudioPolicyResponse JSON body."""
    return json.dumps(
        {
            "logits": [fill] * n_actions,
            "value": value,
            "reasoning": reasoning,
        }
    )


def policy_json_with_logits(logits: list[float], *, value: float = 0.0) -> str:
    return json.dumps({"logits": logits, "value": value, "reasoning": ""})


def fake_completion(content: str) -> _FakeCompletion:
    return _FakeCompletion(content)
