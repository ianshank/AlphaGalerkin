"""Spec §4 AC6 — backwards compatibility of :class:`CertificateConfig`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pde.certificate.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_HEURISTIC_GRID_RESOLUTION,
    DEFAULT_RECORD_COMPILE_TIME,
    DEFAULT_TRACK_A_OVERHEAD_FRACTION,
    DEFAULT_TRACK_B_BUDGET_S_CPU,
    DEFAULT_VERIFIER_BACKEND,
    CertificateConfig,
)


# The pre-WS1 minimal constructor call — a scenario that has never touched
# the verifier boundary must still parse.
def test_pre_ws1_construction_still_validates() -> None:
    """Byte-identical construction path from PR #1 keeps working."""
    cfg = CertificateConfig()
    assert cfg.enabled is True
    assert cfg.prefer_rigorous is True
    assert cfg.track_a_overhead_fraction == DEFAULT_TRACK_A_OVERHEAD_FRACTION
    assert cfg.track_b_budget_s == DEFAULT_TRACK_B_BUDGET_S_CPU
    assert cfg.heuristic_grid_resolution == DEFAULT_HEURISTIC_GRID_RESOLUTION
    assert cfg.thresholds == []


def test_ws1_defaults_are_backwards_compatible() -> None:
    """New WS1 fields default to values that preserve pre-WS1 behaviour."""
    cfg = CertificateConfig()
    assert cfg.verifier_backend == DEFAULT_VERIFIER_BACKEND == "heuristic_grid"
    assert cfg.device == DEFAULT_DEVICE == "auto"
    assert cfg.dtype == DEFAULT_DTYPE == "float64"
    assert cfg.record_compile_time is DEFAULT_RECORD_COMPILE_TIME is True
    # ``budget`` default: max_wall_s == DEFAULT_TRACK_B_BUDGET_S_CPU, no fallback.
    assert cfg.budget.max_wall_s == DEFAULT_TRACK_B_BUDGET_S_CPU
    assert cfg.budget.allow_heuristic_fallback is False


def test_pre_ws1_dump_is_subset_of_ws1_dump() -> None:
    """The pre-WS1 field set is a strict subset of the WS1 dump."""
    cfg = CertificateConfig()
    dump = cfg.model_dump()
    for key in (
        "enabled",
        "prefer_rigorous",
        "track_a_overhead_fraction",
        "track_b_budget_s",
        "heuristic_grid_resolution",
        "thresholds",
    ):
        assert key in dump, f"pre-WS1 field {key!r} missing from dump"
    for key in ("verifier_backend", "budget", "device", "dtype", "record_compile_time"):
        assert key in dump, f"WS1 field {key!r} missing from dump"


def test_verifier_backend_validated_against_literal() -> None:
    """Typos in ``verifier_backend`` fail at construction time."""
    with pytest.raises(ValidationError):
        CertificateConfig(verifier_backend="not_a_real_backend")  # type: ignore[arg-type]


def test_device_validated_against_literal() -> None:
    with pytest.raises(ValidationError):
        CertificateConfig(device="bogus")  # type: ignore[arg-type]


def test_dtype_validated_against_literal() -> None:
    with pytest.raises(ValidationError):
        CertificateConfig(dtype="float128")  # type: ignore[arg-type]


def test_explicit_rigorous_choice_is_accepted() -> None:
    """The config validates against the Literal even though the sentinel is unavailable.

    Availability is a run-time concern (dispatch time), not a validation-time
    concern. This lets a scenario ship a rigorous choice for a future
    environment that will have the extra installed.
    """
    cfg = CertificateConfig(verifier_backend="jax_verify")
    assert cfg.verifier_backend == "jax_verify"


def test_frozen_config_rejects_mutation() -> None:
    cfg = CertificateConfig()
    with pytest.raises((TypeError, ValidationError)):
        cfg.enabled = False  # type: ignore[misc]


def test_extra_fields_still_forbidden() -> None:
    """WS1 preserves PR #1's typo-catch on the runtime config."""
    with pytest.raises(ValidationError):
        CertificateConfig(unknown_knob=42)  # type: ignore[call-arg]
