"""GPU-required real-server smoke test for the research-loop harness.

Runs only with CUDA available (``@pytest.mark.gpu_required``, auto-skipped by
the root ``conftest.py`` on CPU CI). A bare random-arm manifest sweep runs on
any CUDA box; the LLM arm is added only when ``LM_STUDIO_URL`` is configured.
"""

from __future__ import annotations

import os

import pytest

from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
from src.agents.research_loop import ResearchLoopOrchestrator
from src.integrations.lm_studio.config import LMStudioConfig
from src.templates.base import ExecutionStatus

pytestmark = [pytest.mark.gpu_required, pytest.mark.integration]


def test_random_arm_manifest_sweep_on_gpu() -> None:
    """A real random-arm sweep across an OOD manifest on CUDA builds a ledger."""
    config = ResearchLoopConfig(
        name="smoke",
        problems=[
            ResearchProblemSpec(name="poisson_id", pde="poisson"),
            ResearchProblemSpec(name="helmholtz_ood", pde="helmholtz"),
            ResearchProblemSpec(name="biharmonic_ood", pde="biharmonic"),
        ],
        default_arms=["random"],
        n_seeds=3,
        device="cuda",
        n_mcts_simulations=16,
        max_rollouts=256,
        max_basis_functions=8,
        n_candidate_bases=16,
        min_solved_fraction=0.0,
    )
    result = ResearchLoopOrchestrator(config).run()
    assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
    ledger = result.metadata["discovery_ledger"]
    assert set(ledger) == {"poisson_id", "helmholtz_ood", "biharmonic_ood"}


def test_llm_arm_manifest_sweep_on_gpu() -> None:
    """A real LLM-arm manifest sweep against a live LM Studio endpoint on CUDA."""
    base_url = os.environ.get("LM_STUDIO_URL")
    if not base_url:
        pytest.skip("LM_STUDIO_URL not set; skipping LLM-arm GPU smoke test")
    model = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")
    config = ResearchLoopConfig(
        name="smoke_llm",
        problems=[
            ResearchProblemSpec(name="poisson_id", pde="poisson"),
            ResearchProblemSpec(name="helmholtz_ood", pde="helmholtz"),
        ],
        default_arms=["random", "llm"],
        n_seeds=2,
        device="cuda",
        n_mcts_simulations=16,
        max_rollouts=256,
        max_basis_functions=6,
        n_candidate_bases=12,
        min_solved_fraction=0.0,
        lm_studio=LMStudioConfig(
            base_url=base_url,
            model=model,
            preflight_on_construct=False,
        ),
    )
    loop = ResearchLoopOrchestrator(config)
    result = loop.run()
    assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
    assert "llm" in loop._available_arms
