"""GPU-required real-server smoke test for the scaling-law scenario.

Runs only with CUDA available (``@pytest.mark.gpu_required``, auto-skipped by
the root ``conftest.py`` on CPU CI) AND with ``LM_STUDIO_URL`` set when the
LLM arm is exercised. A bare random-arm sweep runs on any CUDA box; the LLM
arm is added only when a live LM Studio endpoint is configured.
"""

from __future__ import annotations

import os

import pytest

from src.integrations.lm_studio.config import LMStudioConfig
from src.poc.config import ScenarioStatus
from src.poc.scenarios.scaling_law import ScalingLawScenario
from src.poc.scenarios.scaling_law_config import ScalingLawConfig

pytestmark = [pytest.mark.gpu_required, pytest.mark.integration]


def test_random_arm_scaling_sweep_on_gpu() -> None:
    """A real random-arm sweep on CUDA records a scaling fit."""
    config = ScalingLawConfig(
        arms=["random"],
        simulation_budgets=[8, 16, 32],
        n_seeds=3,
        device="cuda",
        max_basis_functions=8,
        n_candidate_bases=16,
        min_residual_decay=0.0,
        min_fit_r2=0.0,
    )
    result = ScalingLawScenario(config).run()
    assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
    assert "residual_scaling_exponent" in result.metrics
    assert "residual_fit_r2" in result.metrics


def test_llm_arm_scaling_sweep_on_gpu() -> None:
    """A real LLM-arm sweep against a live LM Studio endpoint on CUDA."""
    base_url = os.environ.get("LM_STUDIO_URL")
    if not base_url:
        pytest.skip("LM_STUDIO_URL not set; skipping LLM-arm GPU smoke test")
    model = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")
    config = ScalingLawConfig(
        arms=["random", "llm"],
        simulation_budgets=[8, 16],
        n_seeds=2,
        device="cuda",
        max_basis_functions=6,
        n_candidate_bases=12,
        min_residual_decay=0.0,
        min_fit_r2=0.0,
        lm_studio=LMStudioConfig(
            base_url=base_url,
            model=model,
            preflight_on_construct=False,
        ),
    )
    result = ScalingLawScenario(config).run()
    assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
    # The LLM arm should have produced its own scaling fit (preflight passed).
    assert "llm_residual_scaling_exponent" in result.metrics
