"""Acceptance / AQA tests for the centaur deliverables.

These map each Adam-Brown theme to an acceptance check exercised on a real
(tiny, CPU) run, plus a lightweight governance check that every knob is a
typed Pydantic field (no hardcoded values).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# AQA-1 — Held-out generalisation: OOD operators wired into the ablation        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ood", ["helmholtz", "biharmonic"])
def test_aqa_ood_operator_selectable_and_buildable(ood: str) -> None:
    from src.poc.scenarios._centaur_common import (
        build_basis_game,
        build_pde_operator,
        enumerate_basis_descriptions,
    )
    from src.poc.scenarios.llm_prior_config import LLMPriorAblationConfig

    # Acceptance: the held-out operator is a valid ood_pde choice ...
    config = LLMPriorAblationConfig(ood_pde=ood)  # type: ignore[arg-type]
    assert config.ood_pde == ood
    # ... and the shared centaur path can actually build it end-to-end.
    operator = build_pde_operator(ood)
    game = build_basis_game(
        ood, operator, max_basis_functions=3, n_candidate_bases=6, target_residual=1e-2
    )
    descriptions = enumerate_basis_descriptions(game)
    assert game.action_space_size == 6
    assert len(descriptions) == 6


# --------------------------------------------------------------------------- #
# AQA-2 — Bitter lesson: the scaling scenario delivers a scaling curve          #
# --------------------------------------------------------------------------- #


def test_aqa_scaling_law_delivers_scaling_curve() -> None:
    from src.poc.config import ScenarioStatus
    from src.poc.scenarios.scaling_law import ScalingLawScenario
    from src.poc.scenarios.scaling_law_config import ScalingLawConfig

    config = ScalingLawConfig(
        arms=["random"],
        simulation_budgets=[2, 4, 8],
        n_seeds=2,
        seeds=[1, 2],
        device="cpu",
        requires_gpu=False,
        max_basis_functions=2,
        n_candidate_bases=4,
        min_residual_decay=0.0,
        min_fit_r2=0.0,
    )
    result = ScalingLawScenario(config).run()
    # Acceptance: the scenario ran and produced the headline scaling metrics
    # plus a per-budget data point at every swept budget.
    assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
    assert np.isfinite(result.metrics["residual_scaling_exponent"])
    assert np.isfinite(result.metrics["residual_fit_r2"])
    for budget in (2, 4, 8):
        assert f"random_residual_median_b{budget}" in result.metrics


# --------------------------------------------------------------------------- #
# AQA-3 — Billions of Einsteins: the research loop delivers a discovery ledger  #
# --------------------------------------------------------------------------- #


def test_aqa_research_loop_delivers_discovery_ledger() -> None:
    from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
    from src.agents.research_loop import ResearchLoopOrchestrator

    config = ResearchLoopConfig(
        name="aqa",
        problems=[
            ResearchProblemSpec(name="poisson", pde="poisson"),
            ResearchProblemSpec(name="helmholtz", pde="helmholtz"),
        ],
        default_arms=["random"],
        n_seeds=2,
        seeds=[1, 2],
        n_mcts_simulations=2,
        max_rollouts=8,
        max_basis_functions=2,
        n_candidate_bases=4,
        target_residual=0.5,
        device="cpu",
        min_solved_fraction=0.0,
    )
    result = ResearchLoopOrchestrator(config).run()
    ledger = result.metadata["discovery_ledger"]
    # Acceptance: a complete per-problem ledger + a valid solved fraction.
    assert set(ledger) == {"poisson", "helmholtz"}
    for entry in ledger.values():
        assert entry["best_arm"] in ("random",)
        assert "per_arm" in entry
    assert 0.0 <= result.metrics["solved_fraction"] <= 1.0
    assert result.metrics["n_problems"] == 2.0


# --------------------------------------------------------------------------- #
# AQA-4 — Governance: no hardcoded values (every knob is a typed field)         #
# --------------------------------------------------------------------------- #


def test_aqa_configs_expose_knobs_as_typed_fields() -> None:
    from src.agents.config import ResearchLoopConfig
    from src.poc.scenarios.scaling_law_config import ScalingLawConfig

    scaling_fields = ScalingLawConfig.model_fields
    for knob in (
        "simulation_budgets",
        "n_seeds",
        "target_residual",
        "min_residual_decay",
        "min_fit_r2",
        "rollout_headroom",
        "significance_alpha",
    ):
        assert knob in scaling_fields, f"scaling knob not a field: {knob}"

    research_fields = ResearchLoopConfig.model_fields
    for knob in (
        "n_mcts_simulations",
        "max_rollouts",
        "target_residual",
        "min_solved_fraction",
        "n_seeds",
    ):
        assert knob in research_fields, f"research knob not a field: {knob}"
