"""Tests for the centaur research-loop harness.

Config validation plus orchestration are tested via a synthetic subclass that
overrides ``_solve_cell`` with canned outcomes, so the manifest sweep,
discovery-ledger aggregation, arm gating, parallel/sequential equivalence, and
status logic are validated deterministically without a live LM Studio server.
A single tiny real random-arm run exercises the shared MCTS cell end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# research_loop is imported as a module so fixtures can patch the gating
# helpers (check_lm_studio_server / LMStudioClient) it references by name.
import src.agents.research_loop as research_module
from src.agents.config import (
    AgentType,
    ResearchLoopConfig,
    ResearchProblemSpec,
)
from src.agents.research_loop import ResearchLoopOrchestrator
from src.integrations.lm_studio.preflight import PreflightReport
from src.poc.scenarios._centaur_common import CellOutcome
from src.templates.base import ExecutionStatus

# --------------------------------------------------------------------------- #
# Config validation                                                           #
# --------------------------------------------------------------------------- #


def test_agent_type_has_research() -> None:
    assert AgentType.RESEARCH.value == "research"


def _problem(name: str, pde: str = "poisson", arms: list[str] | None = None) -> ResearchProblemSpec:
    return ResearchProblemSpec(name=name, pde=pde, arms=arms)  # type: ignore[arg-type]


def test_config_requires_at_least_one_problem() -> None:
    with pytest.raises(ValueError, match="at least 1 item|min_length|too_short|Problems"):
        ResearchLoopConfig(name="c", problems=[])


def test_config_problem_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="problem names must be unique"):
        ResearchLoopConfig(
            name="c",
            problems=[_problem("dup"), _problem("dup", pde="heat")],
        )


def test_config_default_arms_non_empty() -> None:
    with pytest.raises(ValueError, match="default_arms must be non-empty"):
        ResearchLoopConfig(name="c", problems=[_problem("p")], default_arms=[])


def test_config_default_arms_dedup() -> None:
    cfg = ResearchLoopConfig(
        name="c", problems=[_problem("p")], default_arms=["random", "llm", "random"]
    )
    assert cfg.default_arms == ["random", "llm"]


def test_config_resolved_seeds_derived() -> None:
    cfg = ResearchLoopConfig(name="c", problems=[_problem("p")], seed=3, n_seeds=2)
    assert cfg.resolved_seeds() == [3, 3 + 1009]


def test_config_arms_for_uses_override() -> None:
    cfg = ResearchLoopConfig(
        name="c",
        problems=[_problem("p", arms=["llm"])],
        default_arms=["random"],
    )
    assert cfg.arms_for(cfg.problems[0]) == ["llm"]


def test_config_arms_for_falls_back_to_default() -> None:
    cfg = ResearchLoopConfig(name="c", problems=[_problem("p")], default_arms=["random", "trained"])
    assert cfg.arms_for(cfg.problems[0]) == ["random", "trained"]


# --------------------------------------------------------------------------- #
# Synthetic orchestrator harness                                              #
# --------------------------------------------------------------------------- #


class _SyntheticLoop(ResearchLoopOrchestrator):
    """Research loop returning canned (rollouts, residual) per (problem, arm)."""

    def __init__(self, config: ResearchLoopConfig, cells: dict[tuple[str, str], CellOutcome]):
        super().__init__(config)
        self._cells = cells

    def _solve_cell(  # type: ignore[override]
        self, problem, operator, descriptions, arm, seed
    ) -> CellOutcome:
        base = self._cells[(problem.name, arm)]
        # Per-seed jitter keeps medians well-defined without changing the winner.
        return CellOutcome(base.rollouts_used, base.final_residual * (1.0 + (seed % 2) * 1e-6))


@pytest.fixture
def passing_preflight(monkeypatch: pytest.MonkeyPatch) -> PreflightReport:
    report = PreflightReport(
        server_reachable=True,
        model_available=True,
        available_models=["qwen2.5-14b-instruct"],
        free_vram_gib=16.0,
        vram_sufficient=True,
        failure_reason="",
    )
    monkeypatch.setattr(research_module, "check_lm_studio_server", lambda config: report)
    monkeypatch.setattr(
        research_module, "LMStudioClient", lambda config: MagicMock(name="LMStudioClient")
    )
    return report


@pytest.fixture
def failing_preflight(monkeypatch: pytest.MonkeyPatch) -> PreflightReport:
    report = PreflightReport(
        server_reachable=False,
        model_available=False,
        available_models=[],
        free_vram_gib=None,
        vram_sufficient=True,
        failure_reason="server unreachable",
    )
    monkeypatch.setattr(research_module, "check_lm_studio_server", lambda config: report)
    return report


def _cpu_config(**overrides: object) -> ResearchLoopConfig:
    base: dict[str, object] = {
        "name": "loop",
        "problems": [_problem("p_a"), _problem("p_b")],
        "default_arms": ["random"],
        "n_seeds": 2,
        "seeds": [1, 2],
        "device": "cpu",
        "n_mcts_simulations": 2,
        "max_rollouts": 8,
        "max_basis_functions": 2,
        "n_candidate_bases": 4,
        "target_residual": 1e-2,
    }
    base.update(overrides)
    return ResearchLoopConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def test_ledger_picks_lowest_residual_arm(passing_preflight: PreflightReport) -> None:
    config = _cpu_config(default_arms=["random", "llm"], min_solved_fraction=0.0)
    cells = {
        ("p_a", "random"): CellOutcome(8, 0.2),
        ("p_a", "llm"): CellOutcome(4, 0.005),  # llm wins p_a (below target)
        ("p_b", "random"): CellOutcome(8, 0.05),  # random wins p_b
        ("p_b", "llm"): CellOutcome(4, 0.3),
    }
    result = _SyntheticLoop(config, cells).run()
    assert result.status == ExecutionStatus.COMPLETED
    ledger = result.metadata["discovery_ledger"]
    assert ledger["p_a"]["best_arm"] == "llm"
    assert ledger["p_a"]["solved"] is True
    assert ledger["p_b"]["best_arm"] == "random"
    assert result.metrics["arm_wins_llm"] == 1.0
    assert result.metrics["arm_wins_random"] == 1.0
    assert result.metrics["solved_fraction"] == pytest.approx(0.5)


def test_status_failed_when_below_min_solved_fraction() -> None:
    config = _cpu_config(min_solved_fraction=1.0)
    cells = {
        ("p_a", "random"): CellOutcome(8, 0.2),  # neither solved (target 1e-2)
        ("p_b", "random"): CellOutcome(8, 0.2),
    }
    result = _SyntheticLoop(config, cells).run()
    assert result.status == ExecutionStatus.FAILED
    assert result.metrics["solved_fraction"] == pytest.approx(0.0)


def test_parallel_matches_sequential(passing_preflight: PreflightReport) -> None:
    cells = {
        ("p_a", "random"): CellOutcome(8, 0.2),
        ("p_a", "llm"): CellOutcome(4, 0.005),
        ("p_b", "random"): CellOutcome(8, 0.05),
        ("p_b", "llm"): CellOutcome(4, 0.3),
    }
    seq = _SyntheticLoop(_cpu_config(default_arms=["random", "llm"]), cells).run()
    par = _SyntheticLoop(_cpu_config(default_arms=["random", "llm"], parallel=True), cells).run()
    # Discovery ledgers (best arm + solved) match regardless of dispatch.
    seq_led = {k: (v["best_arm"], v["solved"]) for k, v in seq.metadata["discovery_ledger"].items()}
    par_led = {k: (v["best_arm"], v["solved"]) for k, v in par.metadata["discovery_ledger"].items()}
    assert seq_led == par_led
    assert seq.metrics["solved_fraction"] == par.metrics["solved_fraction"]


def test_llm_arm_gated_off_on_preflight_failure(failing_preflight: PreflightReport) -> None:
    config = _cpu_config(default_arms=["random", "llm"])
    cells = {
        ("p_a", "random"): CellOutcome(8, 0.005),
        ("p_b", "random"): CellOutcome(8, 0.005),
    }
    loop = _SyntheticLoop(config, cells)
    result = loop.run()
    # llm preflight failed -> only random ran; no llm metrics recorded.
    assert "llm" not in loop._available_arms
    assert "p_a_llm_median_residual" not in result.metrics
    assert result.metrics["arm_wins_random"] == 2.0


def test_trained_arm_skipped_without_checkpoint() -> None:
    config = _cpu_config(default_arms=["random", "trained"])
    cells = {
        ("p_a", "random"): CellOutcome(8, 0.005),
        ("p_b", "random"): CellOutcome(8, 0.005),
    }
    loop = _SyntheticLoop(config, cells)
    loop.run()
    assert loop._available_arms == {"random"}


def test_no_available_arms_skips(failing_preflight: PreflightReport) -> None:
    config = _cpu_config(default_arms=["llm"])
    loop = _SyntheticLoop(config, cells={})
    result = loop.run()
    assert result.status == ExecutionStatus.SKIPPED


def test_per_problem_arm_override(passing_preflight: PreflightReport) -> None:
    config = _cpu_config(
        problems=[_problem("p_a", arms=["llm"]), _problem("p_b", arms=["random"])],
        default_arms=["random"],
    )
    cells = {
        ("p_a", "llm"): CellOutcome(4, 0.005),
        ("p_b", "random"): CellOutcome(8, 0.005),
    }
    result = _SyntheticLoop(config, cells).run()
    ledger = result.metadata["discovery_ledger"]
    assert set(ledger["p_a"]["per_arm"]) == {"llm"}
    assert set(ledger["p_b"]["per_arm"]) == {"random"}


# --------------------------------------------------------------------------- #
# Real MCTS micro-run (CPU, random arm)                                       #
# --------------------------------------------------------------------------- #


def test_real_random_arm_micro_run_completes() -> None:
    config = _cpu_config(
        problems=[_problem("p_poisson", pde="poisson")],
        default_arms=["random"],
        target_residual=0.5,
        min_solved_fraction=0.0,
    )
    result = ResearchLoopOrchestrator(config).run()
    assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
    assert "solved_fraction" in result.metrics
    assert "discovery_ledger" in result.metadata
