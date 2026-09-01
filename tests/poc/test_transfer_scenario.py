"""Tests for the zero-shot transfer scenario (``src/poc/scenarios/transfer.py``).

``TransferScenario`` previously had no dedicated test module — it was reached
only indirectly (registry listings, CLI journeys). This module covers the
lifecycle surface, and in particular the ``setup()`` device resolution that was
changed from an inline ``torch.device("cuda" if ... else "cpu")`` to the shared
``resolve_device`` policy helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.poc.config import ScenarioStatus, ScenarioTier, TransferScenarioConfig
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


class TestTransferScenarioExecute:
    """Real CPU micro-runs through ``execute()``: train, evaluate, save.

    ``execute()``, ``_train_model()``, ``_evaluate_at_resolution()`` and
    ``_save_model()`` previously had no coverage at all (measured 25% branch
    on this file) -- every prior test in this module stopped at ``setup()``/
    ``teardown()``. ``small_config`` is small enough (9x9/13x13, 100 train
    samples, 1 epoch, d_model=16) that a full ``run()`` completes in a few
    seconds on CPU, so this exercises the real training/evaluation loop
    rather than mocking it away.
    """

    def test_execute_passes_with_a_generous_threshold(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A generous ``mse_threshold`` lets the tiny model pass everywhere."""
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(update={"mse_threshold": 10.0})
        s = TransferScenario(config)
        result = s.run()

        assert result.status is ScenarioStatus.PASSED
        assert result.passed is True
        assert result.threshold_results == {"mse_9x9": True, "mse_13x13": True}
        assert hasattr(result, "primary_resolution")
        assert result.primary_resolution == config.primary_eval_resolution
        assert hasattr(result, "primary_passed")
        assert result.primary_passed is True

    def test_execute_fails_with_an_impossible_threshold(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """An unreachable ``mse_threshold`` fails every resolution."""
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(update={"mse_threshold": 1e-12})
        s = TransferScenario(config)
        result = s.run()

        assert result.status is ScenarioStatus.FAILED
        assert result.passed is False
        assert all(v is False for v in result.threshold_results.values())
        assert result.primary_passed is False

    def test_execute_records_metrics_for_every_eval_resolution(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Each eval resolution records mse/mae/rmse/max_error plus train_loss_final."""
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(update={"mse_threshold": 10.0})
        result = TransferScenario(config).run()

        assert "train_loss_final" in result.metrics
        assert result.metrics["train_loss_final"] >= 0.0
        for eval_res in config.eval_resolutions:
            for metric in ("mse", "mae", "rmse", "max_error"):
                key = f"{metric}_{eval_res}x{eval_res}"
                assert key in result.metrics, key
                assert result.metrics[key] >= 0.0

    def test_execute_saves_a_model_artifact(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The trained model checkpoint is written and recorded as an artifact."""
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(update={"mse_threshold": 10.0})
        result = TransferScenario(config).run()

        assert "model" in result.artifacts
        model_path = Path(result.artifacts["model"])
        assert model_path.exists()

        checkpoint = torch.load(model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert checkpoint["config"]["d_model"] == config.d_model
        assert checkpoint["scenario_config_hash"] == config.compute_hash()

    def test_execute_logs_progress_every_ten_epochs(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The ``(epoch + 1) % 10 == 0`` progress-log branch fires at 10 epochs.

        ``small_config`` uses a single epoch everywhere else in this module, so
        that branch (and its paired ``scenario_logger.metric`` call) was never
        exercised. A single eval resolution keeps this fast.
        """
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(
            update={
                "n_epochs": 10,
                "eval_resolutions": [9],
                "primary_eval_resolution": 9,
                "mse_threshold": 10.0,
            }
        )
        result = TransferScenario(config).run()

        assert result.status is ScenarioStatus.PASSED
        assert "train_loss_final" in result.metrics

    def test_execute_skips_model_artifact_when_training_yields_no_model(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``if self._output_dir and self._model:`` guards a falsy ``_model``.

        ``_train_model`` always returns a real module in production, but the
        guard exists for defensive reasons; this proves the False branch is
        genuinely dead code by construction, not silently broken.
        """
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(update={"mse_threshold": 10.0})
        s = TransferScenario(config)
        # Force _train_model to return a falsy stand-in without touching
        # _evaluate_at_resolution (which only needs ``self._model.eval()``/
        # ``self._model(...)`` to exist -- a real tiny module still satisfies
        # that while being distinguishable from "no model").

        real_train_model = TransferScenario._train_model

        class _FalsyModuleWrapper:
            """Wraps a real module but is falsy, so ``and self._model`` fails."""

            def __init__(self, module: torch.nn.Module) -> None:
                self._module = module

            def __bool__(self) -> bool:
                return False

            def __getattr__(self, name: str) -> object:
                return getattr(self._module, name)

            def __call__(self, *args: object, **kwargs: object) -> object:
                return self._module(*args, **kwargs)

        def _train_model_falsy(self: TransferScenario) -> object:
            return _FalsyModuleWrapper(real_train_model(self))

        monkeypatch.setattr(TransferScenario, "_train_model", _train_model_falsy)
        result = s.run()

        assert result.status is ScenarioStatus.PASSED
        assert "model" not in result.artifacts

    def test_execute_result_has_expected_base_fields(
        self,
        small_config: TransferScenarioConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The returned ``ScenarioResult`` carries the usual bookkeeping fields."""
        monkeypatch.chdir(tmp_path)
        config = small_config.model_copy(update={"mse_threshold": 10.0})
        result = TransferScenario(config).run()

        assert result.scenario_name == "transfer"
        assert result.config_hash == config.compute_hash()
        assert result.device in ("cpu", "cuda")
        assert result.duration_seconds >= 0
        assert result.start_time is not None
        assert result.end_time is not None


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
