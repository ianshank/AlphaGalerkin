"""WS1 registry + dispatch + sentinel behaviour."""

from __future__ import annotations

import pytest

from src.pde.certificate.registry import (
    VerifierRegistry,
    capture_hardware_meta,
    get_verifier,
    register_verifier,
)
from src.pde.certificate.types import VerifierUnavailableError

# --- Registry keys --------------------------------------------------------


def test_builtin_sentinels_are_registered() -> None:
    """All four rigorous-tier backends land as sentinel entries in WS1."""
    reg = VerifierRegistry()
    for key in ("autolirpa", "delta_crown", "jax_verify", "dreal"):
        assert key in reg, f"missing sentinel {key!r}"


def test_heuristic_grid_is_registered() -> None:
    """WS1 ships the always-available verifier."""
    reg = VerifierRegistry()
    assert "heuristic_grid" in reg


# --- Sentinel dispatch: raises VerifierUnavailableError ------------------


@pytest.mark.parametrize("backend", ["autolirpa", "delta_crown", "jax_verify", "dreal"])
def test_sentinel_dispatch_raises(backend: str) -> None:
    """Spec §4 AC1 — unavailable backends fail closed with a typed error."""
    with pytest.raises(VerifierUnavailableError) as excinfo:
        get_verifier(backend)  # type: ignore[arg-type]
    assert excinfo.value.backend == backend


def test_jax_verify_error_names_jax_extra() -> None:
    """The `jax_verify` sentinel points users at the right extra."""
    with pytest.raises(VerifierUnavailableError) as excinfo:
        get_verifier("jax_verify")
    assert excinfo.value.extra == "jax"
    # Message body mentions ADR risk.
    assert "unmaintained" in str(excinfo.value) or "ADR" in str(excinfo.value)


def test_heuristic_grid_dispatch_returns_instance() -> None:
    """The always-available verifier constructs without extras."""
    from src.pde.certificate.verifiers import HeuristicGridResidualVerifier

    v = get_verifier("heuristic_grid")
    assert isinstance(v, HeuristicGridResidualVerifier)


# --- get_or_raise on unknown keys -----------------------------------------


def test_unknown_backend_raises_key_error() -> None:
    with pytest.raises(KeyError):
        VerifierRegistry().get_or_raise("no_such_backend")


# --- register_verifier(replace=True) --------------------------------------


def test_register_replace_overrides_existing() -> None:
    """WS2 uses ``replace=True`` to drop-in a real verifier for a sentinel."""

    @register_verifier("autolirpa", replace=True)
    class _Fake:
        backend_name = "autolirpa"

        def certify(self, **kwargs: object) -> None:  # pragma: no cover
            raise NotImplementedError

    cls = VerifierRegistry().get_or_raise("autolirpa")
    assert cls is _Fake


def test_register_without_replace_raises_on_duplicate() -> None:
    """Default ``replace=False`` preserves the underlying duplicate-raise."""
    with pytest.raises(ValueError, match="already registered"):

        @register_verifier("heuristic_grid")
        class _Dup:
            backend_name = "heuristic_grid"

            def certify(self, **kwargs: object) -> None:  # pragma: no cover
                raise NotImplementedError


# --- capture_hardware_meta ------------------------------------------------


def test_capture_hardware_meta_populates_device_dtype() -> None:
    hw = capture_hardware_meta(device="cpu", dtype="float64")
    assert hw.device == "cpu"
    assert hw.dtype == "float64"


def test_capture_hardware_meta_versions_are_strings_or_none() -> None:
    hw = capture_hardware_meta(device="cpu", dtype="float32")
    for attr in ("torch_version", "jax_version", "jax_verify_version"):
        v = getattr(hw, attr)
        assert v is None or isinstance(v, str)
