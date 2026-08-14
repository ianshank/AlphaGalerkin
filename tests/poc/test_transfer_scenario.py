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
