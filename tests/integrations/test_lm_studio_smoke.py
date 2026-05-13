"""GPU-required real-server smoke tests for the LM Studio integration.

These tests run only on machines with CUDA available AND with the
``LM_STUDIO_URL`` environment variable pointing at a live LM Studio
endpoint. Both gates are required:

    * ``@pytest.mark.gpu_required`` is auto-skipped by the root
      ``conftest.py`` when CUDA is absent.
    * Missing ``LM_STUDIO_URL`` short-circuits the suite via
      ``pytest.skip`` so a CUDA-only box without LM Studio is also fine.
"""

from __future__ import annotations

import os
import time

import pytest

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.preflight import check_lm_studio_server

pytestmark = [pytest.mark.gpu_required, pytest.mark.integration]


def _config_from_env() -> LMStudioConfig:
    base_url = os.environ.get("LM_STUDIO_URL")
    model = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")
    if not base_url:
        pytest.skip("LM_STUDIO_URL not set; skipping real-server smoke test")
    return LMStudioConfig(base_url=base_url, model=model, preflight_on_construct=False)


def _build_prompt(n_actions: int) -> str:
    return (
        "Return JSON only with keys: logits (list of "
        f"{n_actions} real numbers), value (real in [-1,1]), reasoning "
        "(string). No commentary."
    )


def test_real_server_complete_policy_roundtrip() -> None:
    config = _config_from_env()
    client = LMStudioClient(config)
    start = time.perf_counter()
    response = client.complete_policy(
        _build_prompt(4),
        expected_action_size=4,
        seed=42,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    assert len(response.logits) == 4
    assert -1.0 <= response.value <= 1.0
    # Loose ceiling: the headline 3000 ms is asserted by the scenario;
    # this is a sanity bound to catch >10s pathologies.
    assert duration_ms < 10_000.0


def test_real_server_seed_reproducibility() -> None:
    config = _config_from_env()
    client = LMStudioClient(config)
    prompt = _build_prompt(4)
    response_a = client.complete_policy(prompt, expected_action_size=4, seed=42)
    response_b = client.complete_policy(prompt, expected_action_size=4, seed=42)
    # llama.cpp seed determinism is best-effort; accept either logit-
    # closeness OR identical argmax as evidence of seed honour.
    import numpy as np

    logits_a = np.asarray(response_a.logits)
    logits_b = np.asarray(response_b.logits)
    close = bool(np.allclose(logits_a, logits_b, atol=1e-2))
    same_argmax = int(np.argmax(logits_a)) == int(np.argmax(logits_b))
    assert close or same_argmax


def test_real_server_preflight_passes() -> None:
    config = _config_from_env()
    report = check_lm_studio_server(config)
    assert report.passed, report.failure_reason
