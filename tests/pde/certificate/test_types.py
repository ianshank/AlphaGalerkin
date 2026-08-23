"""WS1 type contracts — frozenness, extra=ignore, and structural rigor guard."""

from __future__ import annotations

import pytest

from src.pde.certificate.types import (
    CertificationBudget,
    CertifiedModel,
    CertifiedResidualBound,
    DomainSpec,
    HardwareMeta,
    VerifierUnavailableError,
)


def _minimal_hw() -> HardwareMeta:
    return HardwareMeta(device="cpu", dtype="float64")


_BOUND_DEFAULTS: dict[str, object] = {
    "upper_bound": 1.0,
    "rigor": "heuristic",
    "backend": "heuristic_grid",
    "domain_coverage": "grid_sampled",
    "compile_wall_s": 0.0,
    "cert_wall_s": 0.1,
    "steady_state_wall_s": 0.1,
    "failure_reason": None,
    "notes": "",
}


def _valid_bound(**overrides: object) -> CertifiedResidualBound:
    kwargs: dict[str, object] = {**_BOUND_DEFAULTS, "hardware_meta": _minimal_hw()}
    kwargs.update(overrides)
    return CertifiedResidualBound(**kwargs)  # type: ignore[arg-type]


# --- CertifiedResidualBound: AC3 / AC5 structural guards ------------------


def test_rigorous_requires_full_coverage() -> None:
    """Spec §4 AC3 — a grid-sampled bound cannot claim rigor."""
    with pytest.raises(ValueError, match="rigor='rigorous' requires domain_coverage='full'"):
        _valid_bound(rigor="rigorous", domain_coverage="grid_sampled")


def test_rigorous_full_coverage_accepted() -> None:
    """The one path that admits rigor: full domain coverage."""
    bound = _valid_bound(
        rigor="rigorous",
        backend="autolirpa",
        domain_coverage="full",
    )
    assert bound.rigor == "rigorous"


def test_partial_coverage_cannot_be_rigorous() -> None:
    """A partial-coverage bound is still not admissible as ``rigor='rigorous'``."""
    with pytest.raises(ValueError, match="rigor='rigorous' requires"):
        _valid_bound(rigor="rigorous", domain_coverage="partial")


def test_cert_wall_must_include_compile_wall() -> None:
    """Spec §4 AC5 — cost accounting must be coherent."""
    with pytest.raises(ValueError, match="cert_wall_s .* must be >= compile_wall_s"):
        _valid_bound(compile_wall_s=1.0, cert_wall_s=0.5)


def test_failed_requires_reason() -> None:
    with pytest.raises(ValueError, match="rigor='failed' requires a non-empty failure_reason"):
        _valid_bound(rigor="failed", failure_reason=None)


def test_non_failed_cannot_carry_reason() -> None:
    with pytest.raises(ValueError, match="cannot carry a failure_reason"):
        _valid_bound(rigor="heuristic", failure_reason="oops")


def test_bound_is_frozen() -> None:
    bound = _valid_bound()
    with pytest.raises((TypeError, ValueError)):  # pydantic frozen raises ValidationError
        bound.upper_bound = 999.0  # type: ignore[misc]


def test_upper_bound_non_negative() -> None:
    with pytest.raises(ValueError):
        _valid_bound(upper_bound=-0.1)


# --- DomainSpec validators ------------------------------------------------


def test_rectangular_requires_bounds() -> None:
    with pytest.raises(ValueError, match="kind='rectangular' requires non-empty bounds"):
        DomainSpec(kind="rectangular", bounds=())


def test_rectangular_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="must satisfy hi > lo"):
        DomainSpec(kind="rectangular", bounds=((1.0, 1.0),))


def test_sdf_requires_reference() -> None:
    with pytest.raises(ValueError, match="kind='sdf' requires a non-null sdf_reference"):
        DomainSpec(kind="sdf", bounds=())


def test_dimension_from_bounds() -> None:
    d = DomainSpec(kind="rectangular", bounds=((0.0, 1.0), (0.0, 2.0), (-1.0, 1.0)))
    assert d.dimension == 3


# --- CertificationBudget --------------------------------------------------


def test_budget_max_wall_positive() -> None:
    with pytest.raises(ValueError):
        CertificationBudget(max_wall_s=0.0)


def test_budget_defaults_allow_no_heuristic_fallback() -> None:
    """Spec §3: silent downgrade is out of scope; default is fail-closed."""
    b = CertificationBudget(max_wall_s=1.0)
    assert b.allow_heuristic_fallback is False


# --- CertifiedModel -------------------------------------------------------


def test_certified_model_accepts_any_callable() -> None:
    """``model_fn`` is opaque by design — the verifier casts to its native form."""
    m = CertifiedModel(backend="numpy", model_fn=lambda x: x, params=None)
    assert m.backend == "numpy"


# --- VerifierUnavailableError --------------------------------------------


def test_unavailable_error_message_names_extra() -> None:
    err = VerifierUnavailableError(backend="jax_verify", extra="jax")
    msg = str(err)
    assert "jax_verify" in msg
    assert "'jax'" in msg
    assert "pip install" in msg
    assert err.backend == "jax_verify"
    assert err.extra == "jax"


def test_unavailable_error_carries_detail() -> None:
    err = VerifierUnavailableError(backend="jax_verify", extra="jax", detail="unmaintained")
    assert "unmaintained" in str(err)


# --- HardwareMeta ---------------------------------------------------------


def test_hw_meta_all_optional_versions() -> None:
    """Base install: no torch, no jax, no jax_verify — still validates."""
    hw = HardwareMeta(device="cpu", dtype="float64")
    assert hw.torch_version is None
    assert hw.jax_version is None
    assert hw.jax_verify_version is None


def test_hw_meta_frozen() -> None:
    hw = HardwareMeta(device="cpu", dtype="float64")
    with pytest.raises((TypeError, ValueError)):
        hw.device = "cuda"  # type: ignore[misc]
