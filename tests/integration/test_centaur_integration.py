"""Integration tests for the centaur deliverables — real component wiring.

These exercise the actual operator → game → adapter → MCTS stack, the
scenario through the real ScenarioRunner, and the research-loop orchestrator
across a real multi-PDE manifest. No mocks: only the budgets are tiny so the
suite stays CPU-fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS
from src.pde.config import BasisSelectionConfig, PDEConfig, PDEGameConfig, PDEType
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.registry import get_pde_operator

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# OOD operators through the full MCTS stack                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "pde_type"),
    [("helmholtz", PDEType.HELMHOLTZ), ("biharmonic", PDEType.BIHARMONIC)],
)
def test_ood_operator_drives_real_mcts(name: str, pde_type: PDEType) -> None:
    """A held-out operator solves through BasisSelectionGame + PDEGameAdapter + MCTS."""
    np.random.seed(0)
    operator = get_pde_operator(name)(PDEConfig(name=name, pde_type=pde_type))
    game = BasisSelectionGame(
        operator,
        PDEGameConfig(
            name=f"{name}_game",
            pde_config=operator.config,
            game_mode="basis_selection",
            basis_config=BasisSelectionConfig(
                name=f"{name}_basis", max_basis_functions=3, n_candidate_bases=6
            ),
            error_tolerance=1e-9,
        ),
    )
    adapter = PDEGameAdapter(game)
    initial_error = float(adapter.current_error)
    assert np.isfinite(initial_error)

    mcts = MCTS(evaluator=RandomEvaluator(n_actions=game.action_space_size), n_simulations=4)
    steps = 0
    while not adapter.is_terminal() and steps < 3:
        action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
        if action < 0:
            break
        adapter.apply_action(action)
        mcts.advance(action)
        steps += 1

    assert steps >= 1
    assert np.isfinite(float(adapter.current_error))
    # The error history is tracked and finite throughout.
    assert all(np.isfinite(e) for e in adapter.error_history)


# --------------------------------------------------------------------------- #
# Scaling-law scenario through the real ScenarioRunner                         #
# --------------------------------------------------------------------------- #


def test_scaling_law_via_runner(tmp_path) -> None:
    from src.poc.config import ScenarioStatus
    from src.poc.registry import ScenarioRegistry
    from src.poc.runner import ScenarioRunner
    from src.poc.scenarios.scaling_law import ScalingLawScenario
    from src.poc.scenarios.scaling_law_config import ScalingLawConfig

    registry = ScenarioRegistry()
    if registry.get("scaling_law") is None:
        registry.register("scaling_law", ScalingLawScenario)
    config = ScalingLawConfig(
        arms=["random"],
        simulation_budgets=[2, 4],
        n_seeds=2,
        seeds=[1, 2],
        device="cpu",
        requires_gpu=False,
        max_basis_functions=2,
        n_candidate_bases=4,
        min_residual_decay=0.0,
        min_fit_r2=0.0,
    )
    runner = ScenarioRunner(output_dir=tmp_path)
    result = runner.run("scaling_law", config=config)
    assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
    assert "residual_scaling_exponent" in result.metrics
    assert "random_residual_median_b2" in result.metrics


# --------------------------------------------------------------------------- #
# Research-loop orchestrator across a real multi-PDE manifest                   #
# --------------------------------------------------------------------------- #


def test_research_loop_real_multi_pde_manifest() -> None:
    from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
    from src.agents.research_loop import ResearchLoopOrchestrator
    from src.templates.base import ExecutionStatus

    config = ResearchLoopConfig(
        name="integration",
        problems=[
            ResearchProblemSpec(name="poisson", pde="poisson"),
            ResearchProblemSpec(name="helmholtz", pde="helmholtz"),
            ResearchProblemSpec(name="biharmonic", pde="biharmonic"),
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
    assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
    ledger = result.metadata["discovery_ledger"]
    assert set(ledger) == {"poisson", "helmholtz", "biharmonic"}
    for entry in ledger.values():
        assert entry["best_arm"] == "random"
        assert isinstance(entry["solved"], bool)
    assert 0.0 <= result.metrics["solved_fraction"] <= 1.0


def test_research_loop_parallel_produces_consistent_ledger() -> None:
    """Real (CPU) parallel run produces a complete, uncorrupted ledger.

    The lock-guarded seeded solve guarantees no RNG race corrupts the per-cell
    work; the resulting ledger structure (problem keys + best arm) matches the
    sequential run. Exact residuals are not asserted because the collocation
    sampler draws from an independent (unseeded) generator.
    """
    from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
    from src.agents.research_loop import ResearchLoopOrchestrator

    def _make(parallel: bool) -> ResearchLoopConfig:
        return ResearchLoopConfig(
            name="par" if parallel else "seq",
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
            parallel=parallel,
        )

    seq = ResearchLoopOrchestrator(_make(False)).run()
    par = ResearchLoopOrchestrator(_make(True)).run()
    seq_ledger = seq.metadata["discovery_ledger"]
    par_ledger = par.metadata["discovery_ledger"]
    assert set(seq_ledger) == set(par_ledger) == {"poisson", "helmholtz"}
    for name in ("poisson", "helmholtz"):
        assert par_ledger[name]["best_arm"] == seq_ledger[name]["best_arm"] == "random"
        assert np.isfinite(par_ledger[name]["best_residual"])


# --------------------------------------------------------------------------- #
# Shared gating helpers integrated with the real LMStudioConfig                #
# --------------------------------------------------------------------------- #


def test_gate_llm_client_integrates_with_real_config() -> None:
    from src.integrations.lm_studio.config import LMStudioConfig
    from src.integrations.lm_studio.preflight import PreflightReport
    from src.poc.scenarios._centaur_common import gate_llm_client

    cfg = LMStudioConfig(preflight_on_construct=False)
    constructed: dict[str, object] = {}

    def _factory(passed_cfg: LMStudioConfig) -> str:
        constructed["cfg"] = passed_cfg
        return "client"

    report = PreflightReport(
        server_reachable=True,
        model_available=True,
        available_models=["m"],
        free_vram_gib=16.0,
        vram_sufficient=True,
        failure_reason="",
    )
    client = gate_llm_client(
        cfg,
        cell_logger=__import__("structlog").get_logger("test"),
        preflight_fn=lambda _c: report,
        client_factory=_factory,
    )
    assert client == "client"
    assert isinstance(constructed["cfg"], LMStudioConfig)
    assert constructed["cfg"].preflight_on_construct is False
