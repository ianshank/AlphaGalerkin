"""Tests for the zero-shot transfer scenario (``src/poc/scenarios/transfer.py``).

``TransferScenario`` previously had no dedicated test module — it was reached
only indirectly (registry listings, CLI journeys). This module covers the
lifecycle surface, and in particular the ``setup()`` device resolution that was
changed from an inline ``torch.device("cuda" if ... else "cpu")`` to the shared
``resolve_device`` policy helper.

Validates:
    - ``setup()`` resolves the device through ``src.poc.device.resolve_device``
      with the ``"auto"`` preference (silent CPU fallback), not a hardcoded
      ternary, and lands on CPU on a CPU-only box.
    - ``setup()`` creates the output directory and the scenario logger.
    - Sibling scenarios (``complexity``, ``stability``) use the same policy.
    - ``teardown()`` releases the model.
    - Registration and config defaults.

``execute()`` is deliberately not run: it trains a PhysicsOperator for
``n_epochs`` and is covered by the scenario-runner/e2e surfaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src.poc.config import ScenarioStatus, TransferScenarioConfig
from src.poc.registry import ScenarioRegistry
from src.poc.scenarios.transfer import TransferScenario
from unittest.mock import MagicMock, patch


@pytest.fixture
def scenario() -> TransferScenario:
    """Provides a default TransferScenario instance."""
    cfg = TransferScenarioConfig(
        name="transfer",
        description="test",
        train_resolution=5,
        eval_resolutions=[5],
        primary_eval_resolution=5,
        n_train_samples=100,
        n_eval_samples=10,
        n_charges=1,
        n_epochs=1,
        d_model=16,
        n_heads=1,
        n_layers=1,
        n_fourier_features=8,
        mse_threshold=1.0,
    )
    return TransferScenario(config=cfg)


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

    def _build_scenario(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        **cfg_overrides: object,
    ) -> object:
        """Build a setup() -ready scenario for execute() orchestration tests.

        Centralizes the repeated boilerplate: chdir into tmp_path so the
        scenario's hardcoded outputs/poc/transfer dir lands under it,
        instantiate via the registered class, run setup(), and stamp
        ``_start_time``. Every execute() test needs all four; pulling
        them here keeps the test bodies focused on the assertion.
        """
        monkeypatch.chdir(tmp_path)
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config(**cfg_overrides))
        scenario.setup()
        # ``execute`` reads ``self._start_time`` to compute duration; the
        # framework normally sets this from BaseScenario.run() but our
        # tests call execute() directly, so seed it here.
        scenario._start_time = datetime.now()
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
            result = scenario.execute()

        train_mock.assert_called_once()
        # eval called once per resolution in [5, 7].
        assert eval_mock.call_count == 2
        assert result.scenario_name == "transfer"

    def test_execute_records_per_resolution_metrics(self, tmp_path: Path, monkeypatch) -> None:
        scenario = self._build_scenario(tmp_path, monkeypatch)
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
=======
from src.poc.config import ScenarioTier, TransferScenarioConfig
from src.poc.registry import ScenarioRegistry
from src.poc.scenarios.transfer import TransferScenario


@pytest.fixture
def small_config() -> TransferScenarioConfig:
    """A minimal valid config (nothing here triggers training)."""
    return TransferScenarioConfig(
        name="transfer",
        description="test transfer scenario",
        train_resolution=9,
        eval_resolutions=[9, 13],
        primary_eval_resolution=13,
        n_train_samples=100,
        n_eval_samples=10,
        n_epochs=1,
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_fourier_features=8,
        seed=42,
    )


class TestTransferScenarioInit:
    """Construction and registration."""

    def test_registered_under_transfer(self) -> None:
        """The ``@scenario('transfer')`` decorator registered the class.

        Both sides are resolved from the *current* ``sys.modules`` rather than
        the module-level import. Sibling files in this package purge
        ``sys.modules['src.poc.scenarios*']`` to make the decorators re-fire, so
        after they run the registry holds a freshly-imported class object while
        the top-level ``TransferScenario`` still points at the original — two
        distinct classes with identical names, which makes an ``is`` comparison
        fail with the baffling ``assert <class 'X'> is X``.
        """
        import src.poc.scenarios.transfer as transfer_mod

        assert ScenarioRegistry().get("transfer") is transfer_mod.TransferScenario

    def test_config_class_is_transfer_config(self) -> None:
        """The scenario declares its Pydantic config class."""
        assert TransferScenario.config_class is TransferScenarioConfig

    def test_lazy_attributes_start_unset(self, small_config: TransferScenarioConfig) -> None:
        """Everything ``setup()`` owns is ``None`` before ``setup()`` runs."""
        s = TransferScenario(small_config)
        assert s._device is None
        assert s._model is None
        assert s._output_dir is None
        assert s._scenario_logger is None

    def test_default_tier_is_integration(self) -> None:
        """Transfer is an integration-tier scenario."""
        assert TransferScenarioConfig().tier is ScenarioTier.INTEGRATION


class TestTransferScenarioSetup:
    """``setup()`` device resolution and resource creation."""

    def test_setup_resolves_device_to_cpu_on_a_cpu_box(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With CUDA unavailable, ``"auto"`` must fall back to CPU silently."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        s = TransferScenario(small_config)
        s.setup()
        assert s._device == torch.device("cpu")

    def test_setup_delegates_to_resolve_device_with_auto(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The device comes from the shared policy helper, not a local ternary.

        This is the regression guard for the refactor: an inline
        ``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` would
        produce the same value on this box but never call ``resolve_device``,
        so asserting the value alone cannot catch a revert.

        The scenario class is taken off ``transfer_mod`` rather than the
        module-level import: sibling files in this package purge
        ``sys.modules['src.poc.scenarios*']``, so after they run this re-import
        yields a *different* module object than the one the top-level
        ``TransferScenario`` was bound from. Patching one and instantiating the
        other left the spy uninstalled on the class actually under test, which
        made this assertion pass alone and fail in a full-package run.
        """
        import src.poc.scenarios.transfer as transfer_mod

        calls: list[tuple[str, str]] = []

        def _spy(preference: str, *, context: str = "scenario") -> torch.device:
            calls.append((preference, context))
            return torch.device("cpu")

        monkeypatch.setattr(transfer_mod, "resolve_device", _spy)
        s = transfer_mod.TransferScenario(small_config)
        s.setup()

        assert calls == [("auto", "transfer")]
        assert s._device == torch.device("cpu")

    def test_setup_uses_auto_not_cuda_so_cpu_ci_never_raises(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``"auto"`` (not ``"cuda"``) — CPU CI must not hit the fail-loud path.

        ``resolve_device("cuda", ...)`` raises ``RuntimeError`` when CUDA is
        missing. Wiring the classic scenarios to ``"cuda"`` would break every
        CPU run, so the preference is pinned here.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        s = TransferScenario(small_config)
        s.setup()  # must not raise
        assert s._device is not None
        assert s._device.type == "cpu"

    def test_setup_creates_output_directory(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The artifact directory exists after setup."""
        monkeypatch.chdir(tmp_path)
        s = TransferScenario(small_config)
        s.setup()
        assert s._output_dir == Path("outputs/poc/transfer")
        assert (tmp_path / "outputs" / "poc" / "transfer").is_dir()

    def test_setup_creates_scenario_logger(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A ``ScenarioLogger`` bound to this scenario is created."""
        monkeypatch.chdir(tmp_path)
        s = TransferScenario(small_config)
        s.setup()
        assert s._scenario_logger is not None
        assert s._scenario_logger._context["scenario"] == "transfer"

    def test_setup_is_idempotent(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Calling ``setup()`` twice does not fail on the existing directory."""
        monkeypatch.chdir(tmp_path)
        s = TransferScenario(small_config)
        s.setup()
        s.setup()
        assert s._device is not None


class TestTransferScenarioTeardown:
    """``teardown()`` cleanup."""

    def test_teardown_releases_model(
        self, small_config: TransferScenarioConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The model reference is dropped so the GPU/CPU memory can be freed."""
        monkeypatch.chdir(tmp_path)
        s = TransferScenario(small_config)
        s.setup()
        s._model = torch.nn.Linear(2, 2)
        s.teardown()
        assert s._model is None

    def test_teardown_without_setup_is_safe(self, small_config: TransferScenarioConfig) -> None:
        """Teardown on a never-setup scenario must not raise."""
        TransferScenario(small_config).teardown()


class TestClassicScenariosShareTheDevicePolicy:
    """All three classic scenarios route through ``resolve_device('auto')``."""

    @pytest.mark.parametrize(
        ("module_path", "context"),
        [
            ("src.poc.scenarios.transfer", "transfer"),
            ("src.poc.scenarios.complexity", "complexity"),
            ("src.poc.scenarios.stability", "stability"),
        ],
    )
    def test_module_imports_the_shared_helper(self, module_path: str, context: str) -> None:
        """Each module holds a reference to the shared helper.

        A revert to the inline ternary would drop this import; the
        ``context`` argument each scenario passes is asserted by its own
        spy test (see ``test_setup_delegates_to_resolve_device_with_auto``).
        """
        import importlib

        from src.poc.device import resolve_device

        module = importlib.import_module(module_path)
        assert module.resolve_device is resolve_device
        assert context  # scenario name the helper is called with
>>>>>>> origin/claude/alphagalerkin-implementation-4zGEN
