"""Unit tests for ``BaseScenario.execution_device_label``.

The method exists because ``ScenarioResult.device`` -- documented as
"Computation device used" -- was filled at both construction sites with
``"cuda" if torch.cuda.is_available() else "cpu"``, i.e. host *availability*.
A scenario configured ``device: cpu`` on a CUDA host therefore persisted
``"cuda"``: the inverse of the field's meaning, in the one artifact a reader
consults to find out where a run executed.

That fix was guarded **only** by two E2E assertions of the form
``payload["device"] == e2e_device``. Every workflow runs on ``ubuntu-latest``
and ``test-e2e`` pins ``E2E_DEVICE: cpu``, so on CI both the fixed and the
pre-fix expression yield ``"cpu"`` and the assertions are indistinguishable.
Verified: restoring the pre-fix body left both journey tests green.

The discriminating case needs a CUDA host, so it is produced here by
monkeypatching ``torch.cuda.is_available`` -- which makes the whole 2x2
(``_device`` set/unset x CUDA available/not) reachable on a CPU runner in
milliseconds, with no subprocess and no GPU.
"""

from __future__ import annotations

import pytest

from src.poc.config import BaseScenarioConfig
from src.poc.registry import BaseScenario


class _Scenario(BaseScenario):
    """Minimal concrete scenario; the ABC's other hooks are not exercised."""

    def execute(self) -> dict[str, float]:  # pragma: no cover - not called here
        return {}


def _scenario() -> _Scenario:
    return _Scenario(
        BaseScenarioConfig(name="device_label_probe", description="device label probe")
    )


@pytest.fixture
def _cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this is a CUDA host, whatever the runner actually is."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def _no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this is a CPU-only host, whatever the runner actually is."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class TestTheResolvedDeviceWins:
    """A scenario that resolved a device reports *that*, not host availability."""

    def test_a_cpu_pinned_scenario_on_a_cuda_host_records_cpu(self, _cuda_available: None) -> None:
        """The discriminating case, and the whole reason the method exists.

        This is the assertion no CI runner could ever make: the pre-fix
        expression returns ``"cuda"`` here.
        """
        scenario = _scenario()
        scenario._device = "cpu"
        assert scenario.execution_device_label() == "cpu"

    def test_an_indexed_device_is_preserved_verbatim(self, _cuda_available: None) -> None:
        """``cuda:1`` must not be flattened to ``cuda`` on a dual-card host."""
        scenario = _scenario()
        scenario._device = "cuda:1"
        assert scenario.execution_device_label() == "cuda:1"

    def test_a_torch_device_object_is_stringified(self, _no_cuda: None) -> None:
        """Scenarios store ``torch.device``, not always ``str``."""
        import torch

        scenario = _scenario()
        scenario._device = torch.device("cpu")
        assert scenario.execution_device_label() == "cpu"


class TestTheAvailabilityFallback:
    """Scenarios with no device concept keep the historical expression.

    The numpy-only comparison harnesses never set ``_device``; changing what
    they record would be a result-shape change for a run that is genuinely
    device-irrelevant.
    """

    def test_no_resolved_device_on_a_cuda_host_reports_cuda(self, _cuda_available: None) -> None:
        scenario = _scenario()
        assert not hasattr(scenario, "_device")
        assert scenario.execution_device_label() == "cuda"

    def test_no_resolved_device_on_a_cpu_host_reports_cpu(self, _no_cuda: None) -> None:
        assert _scenario().execution_device_label() == "cpu"

    def test_an_explicit_none_falls_back_rather_than_recording_none(self, _no_cuda: None) -> None:
        """``_device = None`` means "setup() has not resolved one yet".

        Guards the ``is not None`` check specifically: a truthiness test would
        behave the same here, but would also mistake a legitimately falsy
        device label for "unset".
        """
        scenario = _scenario()
        scenario._device = None
        assert scenario.execution_device_label() == "cpu"
