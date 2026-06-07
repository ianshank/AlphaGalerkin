"""Sanity (smoke) tests for the AI-for-physics centaur deliverables.

Fast, dependency-light checks that the new surfaces import, register, and
validate. These are the first line of defence: if any of these fail the
deeper integration/e2e suites cannot be trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_imports_succeed() -> None:
    import src.agents.research_loop  # noqa: F401
    import src.poc.scenarios._centaur_common  # noqa: F401
    import src.poc.scenarios.scaling_law  # noqa: F401
    from src.pde.operators import BiharmonicOperator, HelmholtzOperator  # noqa: F401


def test_ood_operators_registered() -> None:
    from src.pde.registry import list_pde_operators

    names = list_pde_operators()
    assert "helmholtz" in names
    assert "biharmonic" in names


def test_pde_type_map_has_ood_operators() -> None:
    from src.poc.scenarios._centaur_common import PDE_TYPE_MAP

    assert {"helmholtz", "biharmonic"} <= set(PDE_TYPE_MAP)


def test_scaling_law_scenario_registered() -> None:
    from src.poc.registry import ScenarioRegistry
    from src.poc.scenarios.scaling_law import ScalingLawScenario

    # The scenario auto-registers on import; re-register only if a sibling test
    # cleared the singleton (register() raises on duplicates).
    registry = ScenarioRegistry()
    if registry.get("scaling_law") is None:
        registry.register("scaling_law", ScalingLawScenario)
    assert registry.get("scaling_law") is ScalingLawScenario


def test_research_agent_type_present() -> None:
    from src.agents.config import AgentType

    assert AgentType.RESEARCH.value == "research"


def test_default_configs_validate() -> None:
    from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
    from src.poc.scenarios.scaling_law_config import ScalingLawConfig

    assert ScalingLawConfig().name == "scaling_law"
    cfg = ResearchLoopConfig(name="c", problems=[ResearchProblemSpec(name="p", pde="poisson")])
    assert cfg.problems[0].pde == "poisson"


@pytest.mark.parametrize(
    "rel_path",
    [
        "config/scenarios/scaling_law_demo.yaml",
        "config/agents/research_loop_demo.yaml",
    ],
)
def test_demo_yaml_parses(rel_path: str) -> None:
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"missing demo config: {rel_path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_ood_pde_literal_accepts_new_operators() -> None:
    from src.poc.scenarios.llm_prior_config import LLMPriorAblationConfig

    for pde in ("helmholtz", "biharmonic"):
        cfg = LLMPriorAblationConfig(ood_pde=pde)  # type: ignore[arg-type]
        assert cfg.ood_pde == pde
