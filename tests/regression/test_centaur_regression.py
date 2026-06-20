"""Regression tests pinning the centaur deliverables' behavioural contracts.

Each test guards an invariant that must not silently change: the OOD operator
math, the backwards-compatible llm_prior delegation, the scaling-fit sign, the
discovery-ledger winner selection, and the gating-helper skip semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# OOD operator math                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["helmholtz", "biharmonic"])
def test_ood_residual_vanishes_on_exact_solution(name: str) -> None:
    from src.pde.config import PDEConfig, PDEType
    from src.pde.registry import get_pde_operator

    pde_type = PDEType.HELMHOLTZ if name == "helmholtz" else PDEType.BIHARMONIC
    operator = get_pde_operator(name)(PDEConfig(name=name, pde_type=pde_type))
    coords = torch.rand(64, 2, dtype=torch.float32, requires_grad=True)
    u = operator.exact_solution(coords)
    assert isinstance(u, torch.Tensor)
    residual = operator.residual(u, coords)
    # Contract: manufactured residual stays at/below 1e-3 (float32 autodiff).
    assert residual.l2_norm < 1e-3


def test_pde_type_enum_has_ood_members() -> None:
    from src.pde.config import PDEType

    assert PDEType.HELMHOLTZ.value == "helmholtz"
    assert PDEType.BIHARMONIC.value == "biharmonic"


# --------------------------------------------------------------------------- #
# Backwards-compatible llm_prior delegation                                   #
# --------------------------------------------------------------------------- #


def test_llm_prior_pde_map_is_shared_object() -> None:
    from src.poc.scenarios import llm_prior_ablation
    from src.poc.scenarios._centaur_common import PDE_TYPE_MAP

    # The module-level alias must remain the *same* shared object so a new
    # operator is wired in exactly once.
    assert llm_prior_ablation._PDE_TYPE_MAP is PDE_TYPE_MAP


def test_llm_prior_scenario_private_api_intact() -> None:
    from src.poc.scenarios.llm_prior_ablation import LLMPriorAblationScenario

    for method in (
        "_run_cell",
        "_build_pde_operator",
        "_build_game",
        "_build_evaluator",
        "_enumerate_basis_descriptions",
        "_build_llm_evaluator",
        "_build_trained_evaluator",
    ):
        assert callable(getattr(LLMPriorAblationScenario, method)), method


# --------------------------------------------------------------------------- #
# Scaling-fit sign contract                                                   #
# --------------------------------------------------------------------------- #


def test_scaling_fit_is_negative_for_decaying_residual() -> None:
    from src.poc.scenarios.scaling_law import fit_log_log

    slope, r2 = fit_log_log([2, 4, 8, 16], [1.0, 0.5, 0.25, 0.125])
    assert slope == pytest.approx(-1.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Discovery-ledger winner selection                                           #
# --------------------------------------------------------------------------- #


def test_discovery_ledger_selects_lowest_residual_arm() -> None:
    from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
    from src.agents.research_loop import ResearchLoopOrchestrator
    from src.poc.scenarios._centaur_common import CellOutcome

    config = ResearchLoopConfig(
        name="reg",
        problems=[ResearchProblemSpec(name="p", pde="poisson")],
        default_arms=["random", "llm"],
        target_residual=1e-2,
        device="cpu",
    )
    orch = ResearchLoopOrchestrator(config)
    orch._available_arms = {"random", "llm"}
    results = {
        "p": {
            "random": [CellOutcome(8, 0.2), CellOutcome(8, 0.2)],
            "llm": [CellOutcome(4, 0.001), CellOutcome(4, 0.001)],
        }
    }
    metrics, metadata = orch._aggregate(results)
    ledger = metadata["discovery_ledger"]["p"]
    assert ledger["best_arm"] == "llm"  # lowest residual wins
    assert ledger["solved"] is True  # 0.001 <= target 1e-2
    assert metrics["arm_wins_llm"] == 1.0
    assert metrics["solved_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Gating-helper skip semantics                                                #
# --------------------------------------------------------------------------- #


def test_gate_llm_client_returns_none_on_failed_preflight() -> None:
    import structlog

    from src.integrations.lm_studio.config import LMStudioConfig
    from src.integrations.lm_studio.preflight import PreflightReport
    from src.poc.scenarios._centaur_common import gate_llm_client

    failing = PreflightReport(
        server_reachable=False,
        model_available=False,
        available_models=[],
        free_vram_gib=None,
        vram_sufficient=True,
        failure_reason="down",
    )
    client = gate_llm_client(
        LMStudioConfig(preflight_on_construct=False),
        cell_logger=structlog.get_logger("t"),
        preflight_fn=lambda _c: failing,
        client_factory=lambda _c: object(),
    )
    assert client is None


def test_gate_trained_model_returns_none_on_load_error() -> None:
    import structlog

    from src.poc.scenarios._centaur_common import gate_trained_model

    def _raise(*_a: object, **_k: object) -> tuple[object, object]:
        raise RuntimeError("bad ckpt")

    result = gate_trained_model(
        "ckpt.pt", "cpu", cell_logger=structlog.get_logger("t"), loader=_raise
    )
    assert result is None


def test_residual_is_array_finite() -> None:
    """Helmholtz residual values stay a finite 1-D array (broadcasting guard)."""
    from src.pde.config import PDEConfig, PDEType
    from src.pde.registry import get_pde_operator

    operator = get_pde_operator("helmholtz")(PDEConfig(name="h", pde_type=PDEType.HELMHOLTZ))
    coords = torch.rand(16, 2, dtype=torch.float32, requires_grad=True)
    u = operator.exact_solution(coords).unsqueeze(-1)  # (N, 1) column form
    residual = operator.residual(u, coords)
    assert residual.values.shape == (16,)
    assert bool(np.all(np.isfinite(residual.values.detach().numpy())))
