"""Smoke tests for the TransferScenario lifecycle.

Closes Section 2.2 of docs/PLAN_2026-04-27.md.

The transfer scenario was historically lightly-tested (~19% coverage on
master at PR #62). It is *not* dead code — registered via
``@scenario("transfer")``, exposed in the CLI, listed in the dashboard,
and exercised indirectly by ``tests/poc/test_runner.py`` and the e2e
suite. This module adds focused lifecycle + orchestration tests that
lift coverage without committing to the full feature path.

The expensive ``_train_model`` and ``_evaluate_at_resolution`` methods
are stubbed via ``unittest.mock.patch`` so the tests run in a few
seconds rather than minutes; the goal here is plumbing-correctness, not
end-to-end training validation. The e2e suite covers the latter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.poc.config import ScenarioStatus, TransferScenarioConfig
from src.poc.registry import ScenarioRegistry


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    """Reset the singleton registry between tests.

    Mirrors the autouse fixtures in the other PoC test modules so that
    re-importing ``src.poc.scenarios`` in a fresh test does not warn about
    duplicate registrations.
    """
    ScenarioRegistry().clear()
    for mod_name in list(sys.modules):
        if mod_name.startswith("src.poc.scenarios"):
            del sys.modules[mod_name]


def _import_scenario_class() -> type:
    """Fresh-import so ``@scenario("transfer")`` registers in this test."""
    import importlib

    module = importlib.import_module("src.poc.scenarios.transfer")
    return module.TransferScenario


def _smoke_config(**overrides: object) -> TransferScenarioConfig:
    """Build the minimal valid TransferScenarioConfig for fast tests.

    Defaults are kept tiny so any test that *does* exercise the full
    pipeline (none here, all are mocked) would still complete quickly.
    """
    # All values pinned at the per-field minima documented in
    # ``TransferScenarioConfig`` (n_train_samples ge=100, n_eval_samples
    # ge=10, n_charges ge=1, n_epochs ge=1, d_model ge=16, n_heads ge=1,
    # n_layers ge=1, n_fourier_features ge=8). The execute-path tests
    # never actually train (heavy methods are mocked), so the values
    # only need to satisfy the validators, not be physically meaningful.
    base: dict[str, object] = {
        "name": "transfer",
        "description": "smoke test",
        "train_resolution": 5,
        "eval_resolutions": [5, 7],
        "primary_eval_resolution": 7,
        "n_train_samples": 100,
        "n_eval_samples": 10,
        "n_charges": 2,
        "n_epochs": 1,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "n_fourier_features": 8,
        "mse_threshold": 1.0,
    }
    base.update(overrides)
    return TransferScenarioConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestTransferScenarioRegistration:
    def test_scenario_is_registered(self) -> None:
        expected_cls = _import_scenario_class()
        cls = ScenarioRegistry().get("transfer")
        assert cls is expected_cls


# ---------------------------------------------------------------------------
# Lifecycle: __init__ / setup / teardown
# ---------------------------------------------------------------------------


class TestTransferScenarioLifecycle:
    def test_init_stores_config(self) -> None:
        cls = _import_scenario_class()
        cfg = _smoke_config()
        scenario = cls(config=cfg)
        assert scenario.config is cfg
        assert scenario._model is None
        assert scenario._device is None
        assert scenario._output_dir is None
        assert scenario._scenario_logger is None

    def test_setup_creates_output_dir(self, tmp_path: Path, monkeypatch) -> None:
        # Redirect cwd so setup()'s hardcoded ``outputs/poc/transfer`` lands
        # under the tmp dir rather than polluting the repo's outputs/.
        monkeypatch.chdir(tmp_path)

        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()

        assert scenario._output_dir is not None
        assert scenario._output_dir.exists()
        assert scenario._output_dir == Path("outputs/poc/transfer")

    def test_setup_resolves_device(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)

        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()

        assert scenario._device is not None
        # Device should be either cuda or cpu — the scenario's hardcoded
        # auto-selection (cuda if available else cpu).
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert scenario._device == expected

    def test_setup_creates_scenario_logger(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()
        assert scenario._scenario_logger is not None

    def test_teardown_clears_model_reference(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()
        # Plant a sentinel; teardown must clear it.
        scenario._model = MagicMock()
        scenario.teardown()
        assert scenario._model is None

    def test_teardown_runs_without_setup(self) -> None:
        """teardown() must be safe to call before setup() (idempotent cleanup)."""
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        # No setup() call — teardown must not raise.
        scenario.teardown()


# ---------------------------------------------------------------------------
# Execute: orchestration with mocked heavy operations.
# ---------------------------------------------------------------------------


def _stub_eval(mse: float) -> dict[str, float]:
    """Simulate the dict shape returned by _evaluate_at_resolution."""
    return {"mse": mse, "rmse": mse**0.5, "max_error": mse * 2.0}


class TestTransferScenarioExecute:
    """Exercise execute()'s orchestration logic.

    Both ``_train_model`` and ``_evaluate_at_resolution`` are stubbed
    via ``patch.object`` so the tests run in a few hundred ms; the goal
    is plumbing-correctness, not training-correctness.
    """

    def _build_scenario(self, tmp_path: Path, monkeypatch, **cfg_overrides):
        monkeypatch.chdir(tmp_path)
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config(**cfg_overrides))
        scenario.setup()
        return scenario

    def test_execute_calls_train_then_eval_per_resolution(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch)
        # Mock both heavy methods.
        with (
            patch.object(scenario, "_train_model", return_value=MagicMock()) as train_mock,
            patch.object(
                scenario,
                "_evaluate_at_resolution",
                side_effect=lambda res: _stub_eval(0.5),
            ) as eval_mock,
            patch.object(scenario, "_save_model"),
        ):
            from datetime import datetime

            scenario._start_time = datetime.now()
            result = scenario.execute()

        train_mock.assert_called_once()
        # eval called once per resolution in [5, 7].
        assert eval_mock.call_count == 2
        assert result.scenario_name == "transfer"

    def test_execute_records_per_resolution_metrics(self, tmp_path: Path, monkeypatch) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch)
        from datetime import datetime

        scenario._start_time = datetime.now()
        with (
            patch.object(scenario, "_train_model", return_value=MagicMock()),
            patch.object(
                scenario,
                "_evaluate_at_resolution",
                side_effect=lambda res: _stub_eval(0.5 if res == 5 else 0.8),
            ),
            patch.object(scenario, "_save_model"),
        ):
            result = scenario.execute()

        # Each resolution gets a per-metric record like ``mse_5x5``.
        assert "mse_5x5" in result.metrics
        assert "mse_7x7" in result.metrics
        assert result.metrics["mse_5x5"] == pytest.approx(0.5)
        assert result.metrics["mse_7x7"] == pytest.approx(0.8)

    def test_execute_passes_when_all_thresholds_met(self, tmp_path: Path, monkeypatch) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch, mse_threshold=1.0)
        from datetime import datetime

        scenario._start_time = datetime.now()
        with (
            patch.object(scenario, "_train_model", return_value=MagicMock()),
            patch.object(
                scenario,
                "_evaluate_at_resolution",
                side_effect=lambda res: _stub_eval(0.1),
            ),
            patch.object(scenario, "_save_model"),
        ):
            result = scenario.execute()

        assert result.status == ScenarioStatus.PASSED
        assert result.passed is True
        assert all(result.threshold_results.values())

    def test_execute_fails_when_any_threshold_missed(self, tmp_path: Path, monkeypatch) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch, mse_threshold=0.3)
        from datetime import datetime

        scenario._start_time = datetime.now()
        with (
            patch.object(scenario, "_train_model", return_value=MagicMock()),
            patch.object(
                scenario,
                "_evaluate_at_resolution",
                # 5x5 passes (0.1 < 0.3); 7x7 fails (0.5 >= 0.3).
                side_effect=lambda res: _stub_eval(0.1 if res == 5 else 0.5),
            ),
            patch.object(scenario, "_save_model"),
        ):
            result = scenario.execute()

        assert result.status == ScenarioStatus.FAILED
        assert result.passed is False
        assert result.threshold_results["mse_5x5"] is True
        assert result.threshold_results["mse_7x7"] is False

    def test_execute_saves_model_artifact(self, tmp_path: Path, monkeypatch) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch)
        from datetime import datetime

        scenario._start_time = datetime.now()
        with (
            patch.object(scenario, "_train_model", return_value=MagicMock()),
            patch.object(
                scenario,
                "_evaluate_at_resolution",
                side_effect=lambda res: _stub_eval(0.5),
            ),
            patch.object(scenario, "_save_model") as save_mock,
        ):
            result = scenario.execute()

        save_mock.assert_called_once()
        assert "model" in result.artifacts

    def test_execute_records_torch_and_python_versions(self, tmp_path: Path, monkeypatch) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch)
        from datetime import datetime

        scenario._start_time = datetime.now()
        with (
            patch.object(scenario, "_train_model", return_value=MagicMock()),
            patch.object(
                scenario,
                "_evaluate_at_resolution",
                side_effect=lambda res: _stub_eval(0.5),
            ),
            patch.object(scenario, "_save_model"),
        ):
            result = scenario.execute()

        assert result.torch_version == torch.__version__
        assert result.python_version != ""
        assert result.python_version == sys.version
