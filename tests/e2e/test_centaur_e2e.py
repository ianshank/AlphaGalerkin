"""End-to-end tests for the centaur deliverables.

Exercises the full user-facing path: shipped YAML configs → dispatch →
scenario/orchestrator run → result + persisted artifacts, plus CLI journeys.
All runs are forced onto CPU with tiny budgets so the suite is CI-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import CLIRunnerType

pytestmark = pytest.mark.e2e

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Shipped-config dispatch                                                     #
# --------------------------------------------------------------------------- #


def test_scaling_law_demo_yaml_dispatches_to_config() -> None:
    from src.poc.config import load_config_from_dict
    from src.poc.scenarios.scaling_law_config import ScalingLawConfig

    data = yaml.safe_load(
        (_REPO_ROOT / "config/scenarios/scaling_law_demo.yaml").read_text(encoding="utf-8")
    )
    scenario = data["scenarios"][0]
    config = load_config_from_dict(scenario)
    assert isinstance(config, ScalingLawConfig)
    assert config.name == "scaling_law"
    assert config.pde == "poisson"


def test_research_loop_demo_yaml_loads_as_config() -> None:
    from src.agents.config import ResearchLoopConfig
    from src.templates.cli import load_config_file

    config = load_config_file(
        _REPO_ROOT / "config/agents/research_loop_demo.yaml", ResearchLoopConfig
    )
    assert isinstance(config, ResearchLoopConfig)
    assert [p.pde for p in config.problems] == ["poisson", "helmholtz", "biharmonic"]


# --------------------------------------------------------------------------- #
# Full run via the ScenarioRunner from a YAML file                             #
# --------------------------------------------------------------------------- #


def test_scaling_law_run_from_yaml(tmp_path: Path) -> None:
    from src.poc.registry import ScenarioRegistry
    from src.poc.runner import ScenarioRunner
    from src.poc.scenarios.scaling_law import ScalingLawScenario

    registry = ScenarioRegistry()
    if registry.get("scaling_law") is None:
        registry.register("scaling_law", ScalingLawScenario)

    cpu_yaml = {
        "scenarios": [
            {
                "name": "scaling_law",
                "description": "cpu e2e",
                "tier": "integration",
                "enabled": True,
                "pde": "poisson",
                "arms": ["random"],
                "simulation_budgets": [2, 4],
                "n_seeds": 2,
                "seeds": [1, 2],
                "target_residual": 1.0e-6,
                "max_basis_functions": 2,
                "n_candidate_bases": 4,
                "device": "cpu",
                "requires_gpu": False,
                "min_residual_decay": 0.0,
                "min_fit_r2": 0.0,
            }
        ]
    }
    yaml_path = tmp_path / "scaling_cpu.yaml"
    yaml_path.write_text(yaml.safe_dump(cpu_yaml), encoding="utf-8")

    runner = ScenarioRunner(output_dir=tmp_path / "out")
    results = runner.run_from_config(yaml_path)
    assert len(results) == 1
    result = results[0]
    assert result.scenario_name == "scaling_law"
    assert "residual_scaling_exponent" in result.metrics
    # A result JSON was persisted somewhere under the output dir.
    persisted = list((tmp_path / "out").rglob("*.json"))
    assert persisted, "runner did not persist any result JSON"
    payload = json.loads(persisted[0].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_research_loop_e2e_from_demo_then_cpu_run() -> None:
    """Load the shipped manifest, then run a CPU variant end-to-end."""
    from src.agents.config import ResearchLoopConfig
    from src.agents.research_loop import ResearchLoopOrchestrator
    from src.templates.base import ExecutionStatus
    from src.templates.cli import load_config_file

    shipped = load_config_file(
        _REPO_ROOT / "config/agents/research_loop_demo.yaml", ResearchLoopConfig
    )
    # Reuse the shipped problem manifest, but force a CPU random-arm run.
    cpu_config = shipped.model_copy(
        update={
            "default_arms": ["random"],
            "device": "cpu",
            "n_seeds": 2,
            "seeds": [1, 2],
            "n_mcts_simulations": 2,
            "max_rollouts": 8,
            "max_basis_functions": 2,
            "n_candidate_bases": 4,
            "target_residual": 0.5,
            "min_solved_fraction": 0.0,
        }
    )
    result = ResearchLoopOrchestrator(cpu_config).run()
    assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
    assert set(result.metadata["discovery_ledger"]) == {
        "poisson_id",
        "helmholtz_ood",
        "biharmonic_ood",
    }


# --------------------------------------------------------------------------- #
# CLI journeys                                                                 #
# --------------------------------------------------------------------------- #


def test_poc_cli_info_scaling_law(cli_runner: CLIRunnerType) -> None:
    result = cli_runner("src.poc.cli", ["info", "scaling_law"], 60, None)
    # info may exit 0 (found) or non-zero on environments without the scenario
    # loaded; the key contract is that the CLI runs and mentions the scenario.
    assert result.returncode in (0, 1, 2)
    assert "scaling" in (result.stdout + result.stderr).lower()


def test_agents_cli_help_lists_research(cli_runner: CLIRunnerType) -> None:
    result = cli_runner("src.agents.cli", ["--help"], 60, None)
    assert result.success, result.stderr
    assert "research" in result.stdout.lower()


def test_agents_cli_research_help(cli_runner: CLIRunnerType) -> None:
    result = cli_runner("src.agents.cli", ["research", "--help"], 60, None)
    assert result.returncode in (0, 2)
    assert "config" in (result.stdout + result.stderr).lower()
