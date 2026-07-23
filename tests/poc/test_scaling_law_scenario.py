"""ScalingLawScenario tests — fit, gating, aggregation, threshold mutation.

The MCTS inner loop is exercised only by a single tiny real random-arm run;
the sweep/fit/threshold orchestration is tested through a synthetic harness
that returns canned residuals decaying with the budget (a clean power law),
so the log-log fit and threshold logic are validated deterministically
without a live LM Studio server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.integrations.lm_studio.preflight import PreflightReport
from src.poc.config import ScenarioStatus
from src.poc.registry import ScenarioRegistry
from src.poc.scenarios import scaling_law as scaling_module
from src.poc.scenarios.scaling_law import ScalingLawScenario, fit_log_log
from src.poc.scenarios.scaling_law_config import (
    SCALING_SCENARIO_NAME,
    ScalingLawConfig,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    ScenarioRegistry().clear()
    ScenarioRegistry().register(SCALING_SCENARIO_NAME, ScalingLawScenario)


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
    monkeypatch.setattr(scaling_module, "check_lm_studio_server", lambda config: report)
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
    monkeypatch.setattr(scaling_module, "check_lm_studio_server", lambda config: report)
    return report


@pytest.fixture
def stub_lm_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock(name="LMStudioClient")
    monkeypatch.setattr(
        scaling_module,
        "LMStudioClient",
        lambda config, scenario_logger=None: stub,
    )
    return stub


class _SyntheticScenario(ScalingLawScenario):
    """ScalingLawScenario returning a canned power-law residual per cell.

    residual(budget) = base[arm] / budget, so log(residual) is exactly
    linear in log(budget) with slope -1 and R² = 1.
    """

    def __init__(self, config: ScalingLawConfig, base: dict[str, float]) -> None:
        super().__init__(config)
        self._base = base

    def _run_cell(  # type: ignore[override]
        self,
        *,
        arm: str,
        operator: Any,
        basis_descriptions: list[str],
        budget: int,
        seed: int,
        cell_logger: Any,
    ) -> float:
        # Slight per-seed jitter keeps the significance test well-defined.
        jitter = 1.0 + (seed % 3) * 1e-3
        return (self._base[arm] / budget) * jitter


def _cpu_config(**overrides: Any) -> ScalingLawConfig:
    base = {
        "arms": ["random"],
        "simulation_budgets": [4, 8, 16, 32],
        "n_seeds": 3,
        "seeds": [1, 2, 3],
        "device": "cpu",
        "requires_gpu": False,
        "max_basis_functions": 3,
        "n_candidate_bases": 6,
    }
    base.update(overrides)
    return ScalingLawConfig(**base)


# --------------------------------------------------------------------------- #
# fit_log_log unit tests                                                      #
# --------------------------------------------------------------------------- #


def test_fit_log_log_perfect_power_law() -> None:
    slope, r2 = fit_log_log([1, 2, 4, 8], [1.0, 0.5, 0.25, 0.125])
    assert slope == pytest.approx(-1.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_log_log_too_few_points() -> None:
    assert fit_log_log([8], [0.5]) == (0.0, 0.0)


def test_fit_log_log_handles_nan() -> None:
    slope, r2 = fit_log_log([4, 8, 16], [float("nan"), 0.5, 0.25])
    # Two finite points remain -> slope still negative.
    assert slope < 0.0


def test_fit_log_log_floors_zero_residual() -> None:
    # A zero residual must not produce log(0) = -inf.
    slope, r2 = fit_log_log([4, 8], [0.0, 0.0])
    assert slope == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Scenario orchestration                                                      #
# --------------------------------------------------------------------------- #


def test_scenario_registered() -> None:
    assert ScenarioRegistry().get(SCALING_SCENARIO_NAME) is ScalingLawScenario


def test_synthetic_random_arm_passes_thresholds() -> None:
    config = _cpu_config(min_residual_decay=0.5, min_fit_r2=0.9)
    scenario = _SyntheticScenario(config, base={"random": 1.0})
    result = scenario.run()
    assert result.status == ScenarioStatus.PASSED
    assert result.passed is True
    # Primary aliases recorded, slope ~ -1, r2 ~ 1.
    assert result.metrics["residual_scaling_exponent"] == pytest.approx(-1.0, abs=1e-2)
    assert result.metrics["residual_fit_r2"] == pytest.approx(1.0, abs=1e-2)
    # Per-arm + per-budget metrics present.
    assert "random_residual_scaling_exponent" in result.metrics
    for budget in config.simulation_budgets:
        assert f"random_residual_median_b{budget}" in result.metrics


def test_synthetic_two_arms_records_comparison(
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    config = _cpu_config(arms=["random", "llm"], min_residual_decay=0.5, min_fit_r2=0.9)
    scenario = _SyntheticScenario(config, base={"random": 1.0, "llm": 0.5})
    result = scenario.run()
    assert "arm_comparison_p" in result.metrics
    assert "arm_comparison_significant" in result.metrics
    # Both arms produced their own scaling fits.
    assert "random_residual_scaling_exponent" in result.metrics
    assert "llm_residual_scaling_exponent" in result.metrics


def test_llm_only_arm_preflight_fail_skips(failing_preflight: PreflightReport) -> None:
    config = _cpu_config(arms=["llm"])
    scenario = _SyntheticScenario(config, base={"llm": 1.0})
    result = scenario.run()
    # No active arms -> scenario skips.
    assert result.status == ScenarioStatus.SKIPPED
    # Primary (llm) gated off -> headline thresholds dropped.
    names = {t.name for t in scenario.config.thresholds}
    assert "residual_scaling_exponent" not in names
    assert "residual_fit_r2" not in names


def test_trained_arm_without_checkpoint_skipped() -> None:
    config = _cpu_config(arms=["random", "trained"])
    scenario = _SyntheticScenario(config, base={"random": 1.0, "trained": 1.0})
    scenario.setup()
    assert scenario._active_arms == ["random"]


def test_html_artifact_emitted() -> None:
    pytest.importorskip("matplotlib")
    config = _cpu_config(min_residual_decay=0.0, min_fit_r2=0.0)
    scenario = _SyntheticScenario(config, base={"random": 1.0})
    result = scenario.run()
    assert "html_report" in result.artifacts


# --------------------------------------------------------------------------- #
# Real MCTS micro-run (CPU, random arm)                                       #
# --------------------------------------------------------------------------- #


def test_real_random_arm_micro_run_completes() -> None:
    """A tiny real MCTS sweep on CPU exercises the shared cell end-to-end."""
    config = _cpu_config(
        simulation_budgets=[2, 4],
        n_seeds=2,
        seeds=[1, 2],
        max_basis_functions=2,
        n_candidate_bases=4,
        rollout_headroom=1,
        min_residual_decay=0.0,
        min_fit_r2=0.0,
    )
    scenario = ScalingLawScenario(config)
    result = scenario.run()
    assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
    assert "residual_scaling_exponent" in result.metrics
    assert "random_residual_median_b2" in result.metrics


# --------------------------------------------------------------------------- #
# Gating-path coverage                                                        #
# --------------------------------------------------------------------------- #


def test_trained_arm_loads_when_checkpoint_present(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.training.checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module,
        "create_model_from_checkpoint",
        lambda ckpt, device, strict: (object(), {"cfg": 1}),
    )
    config = _cpu_config(
        arms=["random", "trained"],
        trained_checkpoint_path="dummy.pt",
        min_residual_decay=0.0,
        min_fit_r2=0.0,
    )
    scenario = _SyntheticScenario(config, base={"random": 1.0, "trained": 0.5})
    scenario.run()
    assert "trained" in scenario._active_arms


def test_trained_arm_load_failure_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.training.checkpoint as checkpoint_module

    def _raise(*_a: object, **_k: object) -> tuple[object, object]:
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(checkpoint_module, "create_model_from_checkpoint", _raise)
    config = _cpu_config(arms=["random", "trained"], trained_checkpoint_path="dummy.pt")
    scenario = _SyntheticScenario(config, base={"random": 1.0})
    scenario.run()
    assert scenario._active_arms == ["random"]


def test_llm_disabled_by_config_skips() -> None:
    from src.integrations.lm_studio.config import LMStudioConfig

    config = _cpu_config(
        arms=["random", "llm"],
        lm_studio=LMStudioConfig(enabled=False, preflight_on_construct=False),
    )
    scenario = _SyntheticScenario(config, base={"random": 1.0})
    scenario.run()
    assert scenario._active_arms == ["random"]
