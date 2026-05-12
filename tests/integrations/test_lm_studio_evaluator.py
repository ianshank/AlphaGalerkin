"""LMStudioEvaluator tests — protocol compliance, masking, fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.evaluator import LMStudioEvaluator
from src.integrations.lm_studio.schema import LMStudioParseError
from src.mcts.evaluator import EvaluationResult
from tests.integrations.conftest import (
    FakeOpenAIModule,
    fake_completion,
    policy_json,
)


def _state(channels: int = 7, side: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((channels, side, side)).astype(np.float32)


def _build_client(
    fake_openai: FakeOpenAIModule,
    *,
    fallback_to_uniform_on_parse_error: bool = False,
) -> LMStudioClient:
    fake_openai.OpenAI(base_url="x", api_key="x", timeout=1.0)
    return LMStudioClient(
        LMStudioConfig(
            preflight_on_construct=False,
            backoff_base_s=0.0001,
            fallback_to_uniform_on_parse_error=fallback_to_uniform_on_parse_error,
        )
    )


def _build_evaluator(client: LMStudioClient, n_actions: int = 4) -> LMStudioEvaluator:
    return LMStudioEvaluator(
        client,
        action_space_size=n_actions,
        pde_family="poisson",
        basis_descriptions=[f"basis_{i}" for i in range(n_actions)],
        seed=1,
    )


def test_protocol_compliance(fake_openai: FakeOpenAIModule) -> None:
    """Hasattr + callable check mirroring tests/mcts/test_evaluator.py."""
    evaluator: Any = _build_evaluator(_build_client(fake_openai))
    assert hasattr(evaluator, "evaluate")
    assert callable(evaluator.evaluate)
    assert hasattr(evaluator, "evaluate_batch")
    assert callable(evaluator.evaluate_batch)


def test_illegal_actions_masked_to_zero(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai)
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json(n_actions=4, fill=1.0))
    ]
    evaluator = _build_evaluator(client, n_actions=4)
    result = evaluator.evaluate(_state(), legal_actions=[0, 2])
    assert isinstance(result, EvaluationResult)
    assert result.policy[1] == pytest.approx(0.0)
    assert result.policy[3] == pytest.approx(0.0)


def test_policy_sums_to_one(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai)
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json(n_actions=4, fill=0.5))
    ]
    evaluator = _build_evaluator(client, n_actions=4)
    result = evaluator.evaluate(_state(), legal_actions=[0, 1, 2, 3])
    assert result.policy.sum() == pytest.approx(1.0, abs=1e-5)


def test_batch_matches_loop(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai)
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json(n_actions=4, fill=0.1)),
        fake_completion(policy_json(n_actions=4, fill=0.2)),
        fake_completion(policy_json(n_actions=4, fill=0.1)),
        fake_completion(policy_json(n_actions=4, fill=0.2)),
    ]
    evaluator = _build_evaluator(client, n_actions=4)
    states = [_state(), _state()]
    legal = [[0, 1], [2, 3]]
    batched = evaluator.evaluate_batch(states, legal)
    looped = [evaluator.evaluate(s, la) for s, la in zip(states, legal, strict=True)]
    for b, l in zip(batched, looped, strict=True):
        np.testing.assert_allclose(b.policy, l.policy, atol=1e-6)
        assert b.value == pytest.approx(l.value)


def test_uniform_fallback_when_configured(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai, fallback_to_uniform_on_parse_error=True)
    # Two failing responses (max_retries default = 3 so we need to exhaust)
    fake_openai.last_client.chat.completions.responses = [
        fake_completion("garbage"),
        fake_completion("still"),
        fake_completion("bad"),
        fake_completion("done"),
    ]
    evaluator = _build_evaluator(client, n_actions=4)
    result = evaluator.evaluate(_state(), legal_actions=[1, 3])
    assert result.value == 0.0
    assert result.policy[0] == pytest.approx(0.0)
    assert result.policy[1] == pytest.approx(0.5)
    assert result.policy[2] == pytest.approx(0.0)
    assert result.policy[3] == pytest.approx(0.5)


def test_no_legal_actions_returns_zero_policy(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai)
    evaluator = _build_evaluator(client, n_actions=4)
    result = evaluator.evaluate(_state(), legal_actions=[])
    assert result.policy.sum() == 0.0
    assert result.value == 0.0


def test_latencies_recorded(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai)
    fake_openai.last_client.chat.completions.responses = [
        fake_completion(policy_json(n_actions=4, fill=0.0))
    ]
    evaluator = _build_evaluator(client, n_actions=4)
    evaluator.evaluate(_state(), legal_actions=[0, 1, 2, 3])
    assert len(evaluator.latencies_ms) == 1
    evaluator.reset_latencies()
    assert evaluator.latencies_ms == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"action_space_size": 0}, "action_space_size must be > 0"),
        ({"basis_descriptions": ["only_one"]}, "basis_descriptions length"),
        ({"temperature": 0.0}, "temperature must be > 0"),
    ],
)
def test_evaluator_constructor_validators(
    fake_openai: FakeOpenAIModule, kwargs: dict[str, Any], match: str
) -> None:
    """Cover the three constructor guards at evaluator.py:88-97."""
    client = _build_client(fake_openai)
    base: dict[str, Any] = {
        "action_space_size": 4,
        "pde_family": "poisson",
        "basis_descriptions": [f"b_{i}" for i in range(4)],
        "seed": 1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LMStudioEvaluator(client, **base)


def test_evaluate_batch_length_mismatch_raises(fake_openai: FakeOpenAIModule) -> None:
    """`evaluate_batch` rejects mis-aligned state / legal-actions lists."""
    evaluator = _build_evaluator(_build_client(fake_openai), n_actions=4)
    with pytest.raises(ValueError, match="legal_actions_batch length"):
        evaluator.evaluate_batch([_state()], [[0, 1], [2, 3]])


def test_raises_when_fallback_disabled(fake_openai: FakeOpenAIModule) -> None:
    client = _build_client(fake_openai, fallback_to_uniform_on_parse_error=False)
    fake_openai.last_client.chat.completions.responses = [
        fake_completion("garbage"),
        fake_completion("still"),
        fake_completion("bad"),
        fake_completion("done"),
    ]
    evaluator = _build_evaluator(client, n_actions=4)
    with pytest.raises(LMStudioParseError):
        evaluator.evaluate(_state(), legal_actions=[0, 1])
