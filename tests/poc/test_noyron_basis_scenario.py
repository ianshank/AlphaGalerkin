"""Tests for NoyronBasisScenario (Leap 71 v2.2).

Covers the reusable manufactured-target helper, arm gating + threshold-drop
semantics, and a real CPU random-arm micro-run proving the first
MCTS-on-Noyron pipeline executes and produces a monotone (>=0) error reduction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.preflight import PreflightReport
from src.integrations.lm_studio.schema import LMStudioConnectionError
from src.pde.register_games import _create_helical_pde_config
from src.pde.registry import PDEOperatorRegistry
from src.poc.config import ScenarioStatus
from src.poc.scenarios.noyron_basis import (
    NoyronBasisScenario,
    make_manufactured_operator,
)
from src.poc.scenarios.noyron_basis_config import SCENARIO_NAME, NoyronBasisConfig


def _enabled_lm_studio() -> LMStudioConfig:
    return LMStudioConfig(enabled=True, preflight_on_construct=False)


def _passing_report() -> PreflightReport:
    # `passed` is a computed property (all three bools True).
    return PreflightReport(server_reachable=True, model_available=True, vram_sufficient=True)


def _failing_report() -> PreflightReport:
    return PreflightReport(
        server_reachable=False,
        model_available=False,
        vram_sufficient=False,
        failure_reason="server unreachable",
    )


def _config(**overrides: object) -> NoyronBasisConfig:
    params: dict[str, object] = {
        "name": SCENARIO_NAME,
        "description": "test",
        "arms": ["random"],
        "n_seeds": 1,
        "n_simulations": 2,
        "max_basis_functions": 3,
        "n_candidate_bases": 6,
        "device": "cpu",
    }
    params.update(overrides)
    return NoyronBasisConfig(**params)  # type: ignore[arg-type]


def _helical_operator(name: str = "helical_heat"):  # type: ignore[no-untyped-def]
    cfg = _create_helical_pde_config(name)
    return PDEOperatorRegistry().get_or_raise(name)(cfg)


class TestManufacturedOperator:
    def test_makes_homogeneous_operator_non_degenerate(self) -> None:
        """The raw helical operator has no steady exact solution; the wrapper adds one."""
        op = _helical_operator()
        pts = op.generate_collocation_points(32)
        assert op.exact_solution(pts) is None  # homogeneous baseline

        wrapped = make_manufactured_operator(op, wavenumber=1)
        field = wrapped.exact_solution(pts)
        assert field is not None
        arr = field.numpy() if isinstance(field, torch.Tensor) else field
        assert arr.shape == (32,)
        assert float(np.abs(arr).max()) > 0.0  # non-trivial target

    def test_tensor_and_numpy_inputs(self) -> None:
        op = _helical_operator()
        wrapped = make_manufactured_operator(op, wavenumber=2)
        pts_np = op.generate_collocation_points(8)
        out_np = wrapped.exact_solution(pts_np)
        out_t = wrapped.exact_solution(torch.as_tensor(pts_np))
        assert isinstance(out_t, torch.Tensor)
        np.testing.assert_allclose(
            out_np if isinstance(out_np, np.ndarray) else out_np.numpy(),
            out_t.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_source_and_residual_delegate_to_base(self) -> None:
        """The wrapper only overrides exact_solution; other behaviour is inherited."""
        op = _helical_operator()
        wrapped = make_manufactured_operator(op, wavenumber=1)
        assert type(wrapped).__mro__[1] is type(op)


class TestArmGating:
    def test_trained_arm_without_checkpoint_disabled(self) -> None:
        scenario = NoyronBasisScenario(_config(arms=["random", "trained"]))
        scenario.setup()
        assert scenario._active_arms() == ["random"]
        # random is primary → its thresholds remain
        names = {t.name for t in scenario.config.thresholds}
        assert names == {"error_reduction_pct", "final_residual"}

    def test_llm_arm_without_config_disabled(self) -> None:
        scenario = NoyronBasisScenario(_config(arms=["random", "llm"]))
        scenario.setup()
        assert scenario._active_arms() == ["random"]

    def test_primary_arm_disabled_drops_thresholds(self) -> None:
        # primary = trained (arms[0]); no checkpoint → disabled → thresholds dropped
        scenario = NoyronBasisScenario(_config(arms=["trained", "random"]))
        scenario.setup()
        assert scenario.config.primary_arm == "trained"
        assert scenario.config.thresholds == []

    def test_no_active_arms_skips(self) -> None:
        scenario = NoyronBasisScenario(_config(arms=["trained"]))
        result = scenario.run()
        assert result.status == ScenarioStatus.SKIPPED


class TestLLMGating:
    """Cover the LLM-enabled preflight + client-construction branches."""

    def test_llm_preflight_fails_disables_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.integrations.lm_studio.preflight.check_lm_studio_server",
            lambda cfg: _failing_report(),
        )
        scenario = NoyronBasisScenario(
            _config(arms=["random", "llm"], lm_studio=_enabled_lm_studio())
        )
        scenario.setup()
        assert scenario._active_arms() == ["random"]

    def test_llm_preflight_raises_disables_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(cfg: object) -> PreflightReport:
            raise LMStudioConnectionError("boom")

        monkeypatch.setattr("src.integrations.lm_studio.preflight.check_lm_studio_server", _raise)
        scenario = NoyronBasisScenario(
            _config(arms=["random", "llm"], lm_studio=_enabled_lm_studio())
        )
        scenario.setup()
        assert scenario._active_arms() == ["random"]

    def test_llm_client_construction_failure_disables_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.integrations.lm_studio.preflight.check_lm_studio_server",
            lambda cfg: _passing_report(),
        )

        def _raise(*_a: object, **_k: object) -> object:
            raise RuntimeError("client build failed")

        monkeypatch.setattr("src.integrations.lm_studio.client.LMStudioClient", _raise)
        scenario = NoyronBasisScenario(
            _config(arms=["random", "llm"], lm_studio=_enabled_lm_studio())
        )
        scenario.setup()
        assert scenario._active_arms() == ["random"]

    def test_llm_arm_enabled_when_preflight_and_client_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.integrations.lm_studio.preflight.check_lm_studio_server",
            lambda cfg: _passing_report(),
        )
        monkeypatch.setattr(
            "src.integrations.lm_studio.client.LMStudioClient",
            lambda *a, **k: MagicMock(name="LMStudioClient"),
        )
        scenario = NoyronBasisScenario(
            _config(arms=["random", "llm"], lm_studio=_enabled_lm_studio())
        )
        scenario.setup()
        assert set(scenario._active_arms()) == {"random", "llm"}
        scenario.teardown()  # exercises client.close()

    def test_llm_disabled_by_config_flag(self) -> None:
        scenario = NoyronBasisScenario(
            _config(
                arms=["random", "llm"],
                lm_studio=LMStudioConfig(enabled=False, preflight_on_construct=False),
            )
        )
        scenario.setup()
        assert scenario._active_arms() == ["random"]


class TestTrainedGating:
    def test_trained_arm_loads_when_checkpoint_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.training.checkpoint.create_model_from_checkpoint",
            lambda *a, **k: (MagicMock(name="model"), {}),
        )
        scenario = NoyronBasisScenario(
            _config(arms=["trained", "random"], trained_checkpoint_path="/fake/ckpt.pt")
        )
        scenario.setup()
        assert "trained" in scenario._active_arms()

    def test_trained_arm_load_failure_disables_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: object, **_k: object) -> object:
            raise FileNotFoundError("no such checkpoint")

        monkeypatch.setattr("src.training.checkpoint.create_model_from_checkpoint", _raise)
        scenario = NoyronBasisScenario(
            _config(arms=["random", "trained"], trained_checkpoint_path="/fake/ckpt.pt")
        )
        scenario.setup()
        assert scenario._active_arms() == ["random"]


class TestReductionPct:
    def test_zero_initial_error_floored(self) -> None:
        # denominator floor prevents div-by-zero; equal init/final -> 0% reduction
        assert NoyronBasisScenario._reduction_pct(0.0, 0.0) == 0.0

    def test_positive_reduction(self) -> None:
        assert NoyronBasisScenario._reduction_pct(1.0, 0.5) == pytest.approx(50.0)


class TestRealMicroRun:
    def test_cpu_random_arm_runs_and_is_monotone(self) -> None:
        """First MCTS-on-Noyron: pipeline runs, reduction is monotone (>=0)."""
        scenario = NoyronBasisScenario(_config(arms=["random"], n_seeds=1))
        result = scenario.run()
        assert result.status == ScenarioStatus.PASSED
        assert result.passed
        assert "error_reduction_pct" in result.metrics
        assert "final_residual" in result.metrics
        # Least-squares basis addition cannot increase the fit residual.
        assert result.metrics["error_reduction_pct"] >= 0.0
        assert result.metrics["final_residual"] >= 0.0
