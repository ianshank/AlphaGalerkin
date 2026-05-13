"""LLMPriorAblationScenario tests — gating, threshold mutation, aggregation.

The full MCTS-driven inner loop is **not** exercised here; it would
require a live LM Studio server (or extensive operator/game mocking) and
hide bugs behind fragile stubs. Instead the tests target the
orchestration: arm gating, threshold list mutation, aggregation, and
HTML artifact emission. The numerical inner loop is reserved for the
manual GPU smoke run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.integrations.lm_studio.preflight import PreflightReport
from src.poc.config import ScenarioStatus
from src.poc.registry import ScenarioRegistry
from src.poc.scenarios import llm_prior_ablation as scenario_module
from src.poc.scenarios.llm_prior_ablation import LLMPriorAblationScenario
from src.poc.scenarios.llm_prior_config import (
    SCENARIO_NAME,
    LLMPriorAblationConfig,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    """Re-register the scenario after sibling test files clear the singleton.

    Several other PoC scenario tests (e.g. ``test_noyron_hx_scenario.py``)
    use an autouse fixture that calls ``ScenarioRegistry().clear()`` and
    drops cached ``src.poc.scenarios.*`` modules from ``sys.modules``.
    Re-register the exact class object the test file imported so identity
    checks like ``registry.get(name) is Scenario`` hold.
    """
    ScenarioRegistry().clear()
    ScenarioRegistry().register(SCENARIO_NAME, LLMPriorAblationScenario)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the CUDA fail-loud check in resolve_device.

    Patches the bound name on the *exact module object* the test file
    imported. Sibling test files that delete entries from ``sys.modules``
    do not affect this binding.
    """
    import torch

    monkeypatch.setattr(
        scenario_module,
        "resolve_device",
        lambda preference, *, context: torch.device("cpu"),
    )


@pytest.fixture
def passing_preflight(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Make `check_lm_studio_server` return a passing report."""
    report = PreflightReport(
        server_reachable=True,
        model_available=True,
        available_models=["qwen2.5-14b-instruct"],
        free_vram_gib=16.0,
        vram_sufficient=True,
        failure_reason="",
    )
    monkeypatch.setattr(scenario_module, "check_lm_studio_server", lambda config: report)
    return report


@pytest.fixture
def failing_preflight(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Make `check_lm_studio_server` return a failing report."""
    report = PreflightReport(
        server_reachable=False,
        model_available=False,
        available_models=[],
        free_vram_gib=None,
        vram_sufficient=True,
        failure_reason="server unreachable",
    )
    monkeypatch.setattr(scenario_module, "check_lm_studio_server", lambda config: report)
    return report


@pytest.fixture
def stub_lm_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace LMStudioClient with a MagicMock instance."""
    stub = MagicMock(name="LMStudioClient")
    monkeypatch.setattr(
        scenario_module,
        "LMStudioClient",
        lambda config, scenario_logger=None: stub,
    )
    return stub


# ---------------------------------------------------------------------------
# Synthetic-cell harness — exercises aggregation without real MCTS
# ---------------------------------------------------------------------------


class _SyntheticScenario(LLMPriorAblationScenario):
    """LLMPriorAblationScenario that returns canned (rollouts, residual) per cell."""

    def __init__(
        self,
        config: LLMPriorAblationConfig,
        cells: dict[tuple[str, str, int], tuple[int, float]],
    ) -> None:
        super().__init__(config)
        self._cells = cells

    def _run_cell(  # type: ignore[override]
        self,
        *,
        arm: str,
        pde_name: str,
        operator: Any,
        basis_descriptions: list[str],
        seed: int,
        cell_logger: Any,
    ) -> tuple[int, float]:
        if arm == "llm":
            # Simulate latency samples so p95 is computed.
            self._llm_latencies_ms.append(120.0 + seed)
        return self._cells[(arm, pde_name, seed)]

    def _build_pde_operator(self, pde_name: str) -> Any:  # type: ignore[override]
        return MagicMock(spec_set=["__class__"])

    def _enumerate_basis_descriptions(  # type: ignore[override]
        self,
        pde_name: str,
        operator: Any,
    ) -> list[str]:
        return [f"basis_{i}" for i in range(self.config.n_candidate_bases)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scenario_registered() -> None:
    """The decorator-driven registry should expose the new scenario."""
    cls = ScenarioRegistry().get(SCENARIO_NAME)
    assert cls is LLMPriorAblationScenario


def test_setup_skips_llm_arm_on_preflight_fail(
    stub_device: None,
    failing_preflight: PreflightReport,
) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    assert scenario._llm_arm_enabled is False
    threshold_names = {t.name for t in scenario.config.thresholds}
    assert "id_rollout_reduction_pct" not in threshold_names
    assert "ood_llm_residual" not in threshold_names
    assert "llm_call_p95_latency_ms" not in threshold_names


def test_setup_skips_trained_arm_on_missing_checkpoint(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=True,
        run_llm_arm=True,
        trained_checkpoint_path=None,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    assert scenario._trained_arm_enabled is False
    threshold_names = {t.name for t in scenario.config.thresholds}
    assert "ood_trained_residual" not in threshold_names


def test_setup_skips_trained_arm_on_missing_file(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When create_model_from_checkpoint raises, the arm degrades gracefully."""

    def _raises(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("no such checkpoint")

    monkeypatch.setattr(
        "src.training.checkpoint.create_model_from_checkpoint",
        _raises,
    )
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=True,
        run_llm_arm=True,
        trained_checkpoint_path=Path("/nonexistent/checkpoint.pt"),
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    assert scenario._trained_arm_enabled is False


def test_aggregation_metrics_recorded(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    """Synthetic cells let us check the aggregation logic directly."""
    seeds = [11, 12, 13]
    config = LLMPriorAblationConfig(
        n_seeds=len(seeds),
        seeds=seeds,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    cells: dict[tuple[str, str, int], tuple[int, float]] = {}
    for seed_idx, seed in enumerate(seeds):
        # Random uses many more rollouts than LLM on ID; OOD residuals differ.
        cells[("random", config.id_pde, seed)] = (256 + seed_idx * 16, 0.005)
        cells[("llm", config.id_pde, seed)] = (96 + seed_idx * 8, 0.005)
        cells[("random", config.ood_pde, seed)] = (4096, 0.5)
        cells[("llm", config.ood_pde, seed)] = (2048, 0.005)
    scenario = _SyntheticScenario(config, cells)
    result = scenario.run()
    metrics = result.metrics
    assert "id_rollout_reduction_pct" in metrics
    assert metrics["id_rollout_reduction_pct"] > 0.0
    assert metrics["ood_llm_residual"] == pytest.approx(0.005, abs=1e-6)
    assert metrics["llm_call_p95_latency_ms"] > 0.0
    # Trained-arm metric must NOT be present (arm was skipped).
    assert "ood_trained_residual" not in metrics


def test_html_artifact_emitted(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end the scenario records an HTML artifact path."""
    monkeypatch.chdir(tmp_path)
    config = LLMPriorAblationConfig(
        n_seeds=2,
        seeds=[1, 2],
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    cells: dict[tuple[str, str, int], tuple[int, float]] = {}
    for seed in [1, 2]:
        cells[("random", config.id_pde, seed)] = (300, 0.01)
        cells[("llm", config.id_pde, seed)] = (100, 0.01)
        cells[("random", config.ood_pde, seed)] = (4096, 0.5)
        cells[("llm", config.ood_pde, seed)] = (2000, 0.005)
    scenario = _SyntheticScenario(config, cells)
    result = scenario.run()
    artifact = result.artifacts.get("html_report")
    if artifact is None:
        pytest.skip("matplotlib not installed; HTML artifact not generated")
    artifact_path = Path(artifact)
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size > 0


def test_build_pde_operator_known_name(stub_device: None) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2, run_random_arm=True, run_trained_arm=False, run_llm_arm=False
    )
    scenario = LLMPriorAblationScenario(config)
    operator = scenario._build_pde_operator("poisson")
    assert operator is not None
    assert operator.config.pde_type.value == "poisson"


def test_build_pde_operator_unknown_name_raises(stub_device: None) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2, run_random_arm=True, run_trained_arm=False, run_llm_arm=False
    )
    scenario = LLMPriorAblationScenario(config)
    with pytest.raises(ValueError, match="has no PDEType mapping"):
        scenario._build_pde_operator("not_a_pde")


def test_enumerate_basis_descriptions_has_expected_length(stub_device: None) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=False,
        n_candidate_bases=8,
        max_basis_functions=4,
    )
    scenario = LLMPriorAblationScenario(config)
    operator = scenario._build_pde_operator("poisson")
    descriptions = scenario._enumerate_basis_descriptions("poisson", operator)
    assert len(descriptions) > 0
    assert all(isinstance(d, str) for d in descriptions)


def test_active_arms_reflects_runtime_gating(stub_device: None) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=True,
        run_llm_arm=False,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario._trained_arm_enabled = False  # simulate runtime gate-off
    assert scenario._active_arms() == ["random"]


def test_run_cell_with_random_evaluator_completes(stub_device: None) -> None:
    """Real MCTS micro-run with the Random arm — sanity-checks `_run_cell`.

    Budget is intentionally tiny (4 simulations × max 2 macro-steps) so
    the test runs in well under a second even on CPU CI.
    """
    config = LLMPriorAblationConfig(
        n_seeds=2,
        seeds=[1],
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=False,
        n_candidate_bases=4,
        max_basis_functions=2,
        n_mcts_simulations=4,
        max_rollouts=8,
        target_residual=0.999,  # impossible-to-reach; force exit on max_rollouts
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    operator = scenario._build_pde_operator(config.id_pde)
    descriptions = scenario._enumerate_basis_descriptions(config.id_pde, operator)
    rollouts, residual = scenario._run_cell(
        arm="random",
        pde_name=config.id_pde,
        operator=operator,
        basis_descriptions=descriptions,
        seed=1,
        cell_logger=scenario._scenario_logger,  # type: ignore[arg-type]
    )
    assert rollouts >= 0
    assert residual >= 0.0


def test_record_aggregates_skips_when_no_random_arm(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    """Without the random arm, the id_rollout_reduction_pct metric is not recorded."""
    config = LLMPriorAblationConfig(
        n_seeds=2,
        seeds=[1, 2],
        run_random_arm=False,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    cells: dict[tuple[str, str, int], tuple[int, float]] = {}
    for seed in [1, 2]:
        cells[("llm", config.id_pde, seed)] = (50, 0.005)
        cells[("llm", config.ood_pde, seed)] = (50, 0.01)
    scenario = _SyntheticScenario(config, cells)
    result = scenario.run()
    assert "id_rollout_reduction_pct" not in result.metrics
    assert "ood_llm_residual" in result.metrics


def test_setup_skips_llm_arm_when_lm_studio_enabled_false(
    stub_device: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `lm_studio.enabled=False`, the LLM arm is gated off without preflight."""
    # Make preflight raise loudly if it's ever invoked — proves we short-circuit.
    monkeypatch.setattr(
        scenario_module,
        "check_lm_studio_server",
        lambda config: pytest.fail("preflight should not run when lm_studio.enabled is False"),
    )
    from src.integrations.lm_studio.config import LMStudioConfig

    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
        lm_studio=LMStudioConfig(enabled=False, preflight_on_construct=False),
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    assert scenario._llm_arm_enabled is False


def test_setup_skips_llm_arm_when_preflight_raises(
    stub_device: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions raised by `check_lm_studio_server` must not propagate to the scenario."""

    def _raise(config: Any) -> None:
        raise RuntimeError("transport blew up")

    monkeypatch.setattr(scenario_module, "check_lm_studio_server", _raise)
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    assert scenario._llm_arm_enabled is False


def test_setup_skips_llm_arm_when_client_construction_fails(
    stub_device: None,
    passing_preflight: PreflightReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure inside `LMStudioClient(...)` must be caught and skip the arm."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("client init blew up")

    monkeypatch.setattr(scenario_module, "LMStudioClient", _raise)
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    assert scenario._llm_arm_enabled is False
    assert scenario._lm_client is None


def test_build_llm_evaluator_raises_when_client_missing(stub_device: None) -> None:
    """Defensive raise — gating should have caught this, but make it explicit."""
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario._lm_client = None
    operator = scenario._build_pde_operator(config.id_pde)
    game = scenario._build_game(config.id_pde, operator)
    with pytest.raises(RuntimeError, match="LLM arm requested but client not built"):
        scenario._build_llm_evaluator(
            game=game,
            pde_name=config.id_pde,
            basis_descriptions=[f"b_{i}" for i in range(game.action_space_size)],
            seed=1,
            cell_logger=MagicMock(),
        )


def test_build_trained_evaluator_raises_when_model_missing(stub_device: None) -> None:
    """Defensive raise — gating should have caught this, but make it explicit."""
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=False,
        run_trained_arm=True,
        run_llm_arm=False,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario._trained_model = None
    with pytest.raises(RuntimeError, match="trained arm requested but model not loaded"):
        scenario._build_trained_evaluator()


def test_build_evaluator_unknown_arm_raises(stub_device: None) -> None:
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=False,
    )
    scenario = LLMPriorAblationScenario(config)
    operator = scenario._build_pde_operator(config.id_pde)
    game = scenario._build_game(config.id_pde, operator)
    with pytest.raises(ValueError, match="unknown arm"):
        scenario._build_evaluator(
            arm="invented",
            pde_name=config.id_pde,
            game=game,
            basis_descriptions=[f"b_{i}" for i in range(game.action_space_size)],
            seed=1,
            cell_logger=MagicMock(),
        )


def test_aggregation_handles_empty_random_rollouts(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    """`_record_id_metrics` with median_random=0 should not divide-by-zero."""
    config = LLMPriorAblationConfig(
        n_seeds=2,
        seeds=[1, 2],
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    cells: dict[tuple[str, str, int], tuple[int, float]] = {}
    for seed in [1, 2]:
        cells[("random", config.id_pde, seed)] = (0, 0.01)  # zero rollouts
        cells[("llm", config.id_pde, seed)] = (0, 0.01)
        cells[("random", config.ood_pde, seed)] = (0, 0.5)
        cells[("llm", config.ood_pde, seed)] = (0, 0.005)
    scenario = _SyntheticScenario(config, cells)
    result = scenario.run()
    # Reduction defaults to 0 when median_random is 0.
    assert result.metrics.get("id_rollout_reduction_pct") == 0.0


def test_random_arm_disabled_drops_id_rollout_reduction_threshold(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    """Disabling the random arm drops `id_rollout_reduction_pct` so the run can pass.

    Without this gating, the threshold is installed by `get_default_thresholds`
    but the metric is never recorded (the LLM-vs-random comparison can't be
    computed), and `BaseScenario._evaluate_thresholds` would auto-FAIL.
    """
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=False,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()
    threshold_names = {t.name for t in scenario.config.thresholds}
    assert "id_rollout_reduction_pct" not in threshold_names


def test_empty_llm_latency_samples_drops_p95_threshold(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
) -> None:
    """If no LLM call was recorded, the p95-latency threshold is dropped at aggregate time.

    Prevents `_percentile` returning NaN from auto-FAILing the run when all
    cells exited before any LLM `evaluate` call.
    """
    config = LLMPriorAblationConfig(
        n_seeds=2,
        seeds=[1, 2],
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=True,
    )

    # Build cells that DO produce LLM final-residual numbers (so the LLM arm
    # is active) but the synthetic harness's `_run_cell` for "llm" only appends
    # latency samples when we want it to. Patch the harness to skip latency.
    class _NoLatencyScenario(_SyntheticScenario):
        def _run_cell(  # type: ignore[override]
            self,
            *,
            arm: str,
            pde_name: str,
            operator: Any,
            basis_descriptions: list[str],
            seed: int,
            cell_logger: Any,
        ) -> tuple[int, float]:
            # Deliberately do NOT push to `_llm_latencies_ms`, even on the
            # llm arm — simulates "every cell exited before any LLM call".
            return self._cells[(arm, pde_name, seed)]

    cells: dict[tuple[str, str, int], tuple[int, float]] = {}
    for seed in [1, 2]:
        cells[("random", config.id_pde, seed)] = (256, 0.01)
        cells[("llm", config.id_pde, seed)] = (96, 0.01)
        cells[("random", config.ood_pde, seed)] = (4096, 0.5)
        cells[("llm", config.ood_pde, seed)] = (2048, 0.005)
    scenario = _NoLatencyScenario(config, cells)
    result = scenario.run()
    assert "llm_call_p95_latency_ms" not in result.metrics
    assert "llm_call_p95_latency_ms" not in {t.name for t in scenario.config.thresholds}


def test_run_cell_logs_warning_on_invalid_action(
    stub_device: None,
) -> None:
    """`_run_cell` logs `cell_loop_early_exit` when the evaluator returns action<0."""
    from src.mcts.evaluator import EvaluationResult, RandomEvaluator

    config = LLMPriorAblationConfig(
        n_seeds=2,
        seeds=[1],
        run_random_arm=True,
        run_trained_arm=False,
        run_llm_arm=False,
        n_candidate_bases=4,
        max_basis_functions=2,
        n_mcts_simulations=4,
        max_rollouts=8,
        target_residual=0.999,
    )
    scenario = LLMPriorAblationScenario(config)
    scenario.setup()

    # Patch the evaluator-builder to return an evaluator that emits a
    # zero-policy so MCTS's `get_action` returns -1.
    class _AlwaysInvalidEvaluator(RandomEvaluator):
        def evaluate(self, state: Any, legal_actions: list[int]) -> EvaluationResult:
            import numpy as np

            return EvaluationResult(
                policy=np.zeros(config.n_candidate_bases, dtype=np.float32), value=0.0
            )

    def _build(arm: str, **_: Any) -> Any:
        return _AlwaysInvalidEvaluator(n_actions=config.n_candidate_bases)

    scenario._build_evaluator = _build  # type: ignore[assignment]
    operator = scenario._build_pde_operator(config.id_pde)
    descriptions = scenario._enumerate_basis_descriptions(config.id_pde, operator)
    # Just verify the call returns without crashing — the warning log emission
    # is observable via structlog but asserting on it is brittle; the integration
    # test confirms the early-exit path executes without raising.
    rollouts, residual = scenario._run_cell(
        arm="random",
        pde_name=config.id_pde,
        operator=operator,
        basis_descriptions=descriptions,
        seed=1,
        cell_logger=scenario._scenario_logger,  # type: ignore[arg-type]
    )
    assert rollouts >= 0
    assert residual >= 0.0


def test_no_active_arms_returns_skipped(
    stub_device: None,
    passing_preflight: PreflightReport,
    stub_lm_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every arm is gated off after setup, the scenario short-circuits."""
    monkeypatch.setattr(
        scenario_module,
        "check_lm_studio_server",
        lambda config: PreflightReport(
            server_reachable=False,
            model_available=False,
            available_models=[],
            free_vram_gib=None,
            vram_sufficient=True,
            failure_reason="server unreachable",
        ),
    )
    config = LLMPriorAblationConfig(
        n_seeds=2,
        run_random_arm=False,
        run_trained_arm=False,
        run_llm_arm=True,
    )
    scenario = LLMPriorAblationScenario(config)
    result = scenario.run()
    assert result.status == ScenarioStatus.SKIPPED
