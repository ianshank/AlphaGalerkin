"""Gate 2 — ``Certificate`` schema round-trip and forward-compat migration.

Covers:

* Lossless JSON round-trip via ``model_dump`` / ``model_validate`` (the
  Pydantic subclass field-loss precedent from ``src/refinement/config.py``).
* Enum enforcement on ``track`` and ``rigor``.
* Every cost / provenance field required by the spec §3 is present.
* ``migrate_certificate_document`` unversioned → v1 rewrites and rejects
  newer schemas (mirroring ``ScenarioBaselineDocument`` precedent).
* AC5 stability-consistency validator: ``unbounded_with_warning`` iff
  ``stability_constant is None``.
* NaN / Inf rejection for all numeric fields.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.pde.certificate import (
    CERTIFICATE_DOCUMENT_SCHEMA_VERSION,
    Certificate,
    migrate_certificate_document,
    new_certificate_id,
)


def _minimal_valid_kwargs(**overrides: object) -> dict[str, object]:
    """Return the smallest kwargs dict that instantiates ``Certificate``.

    A single helper keeps every test focused on the one field it exercises.
    """
    kwargs: dict[str, object] = {
        "certificate_id": new_certificate_id(),
        "pde_type": "poisson",
        "scenario_name": "unit_test",
        "track": "A",
        "rigor": "rigorous",
        "bound_value": 0.5,
        "residual_norm": 0.5,
        "stability_constant": 1.0,
        "stability_source": "analytic",
        "cert_wall_s": 0.01,
    }
    kwargs.update(overrides)
    return kwargs


# --- Round-trip -----------------------------------------------------------


def test_json_roundtrip_lossless() -> None:
    cert = Certificate(**_minimal_valid_kwargs())  # type: ignore[arg-type]
    dumped = cert.model_dump()
    restored = Certificate.model_validate(dumped)
    assert restored == cert


def test_all_required_fields_present() -> None:
    cert = Certificate(**_minimal_valid_kwargs())  # type: ignore[arg-type]
    dumped = cert.model_dump()
    for field in (
        "schema_version",
        "certificate_id",
        "pde_type",
        "scenario_name",
        "track",
        "rigor",
        "norm",
        "bound_value",
        "residual_norm",
        "stability_source",
        "cert_wall_s",
        "cert_peak_mem_mb",
    ):
        assert field in dumped, f"required field {field!r} missing"


def test_extra_fields_ignored_forward_compat() -> None:
    """A newer binary can add fields; older readers must drop them silently."""
    dumped = Certificate(**_minimal_valid_kwargs()).model_dump()  # type: ignore[arg-type]
    dumped["some_future_field"] = {"nested": [1, 2, 3]}
    restored = Certificate.model_validate(dumped)
    assert not hasattr(restored, "some_future_field")


# --- Enum enforcement ----------------------------------------------------


@pytest.mark.parametrize("bad_track", ["C", "a", "", "AB", 0])
def test_track_enum_enforced(bad_track: object) -> None:
    with pytest.raises(ValidationError):
        Certificate(**_minimal_valid_kwargs(track=bad_track))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_rigor", ["strict", "loose", "", None])
def test_rigor_enum_enforced(bad_rigor: object) -> None:
    with pytest.raises(ValidationError):
        Certificate(**_minimal_valid_kwargs(rigor=bad_rigor))  # type: ignore[arg-type]


# --- Stability consistency (AC5) -----------------------------------------


def test_unbounded_requires_none_constant() -> None:
    with pytest.raises(ValidationError, match="unbounded_with_warning"):
        Certificate(
            **_minimal_valid_kwargs(
                stability_constant=1.0,
                stability_source="unbounded_with_warning",
            )
        )  # type: ignore[arg-type]


def test_non_unbounded_requires_numeric_constant() -> None:
    with pytest.raises(ValidationError, match="numeric"):
        Certificate(
            **_minimal_valid_kwargs(
                stability_constant=None,
                stability_source="analytic",
            )
        )  # type: ignore[arg-type]


def test_unbounded_certificate_valid() -> None:
    cert = Certificate(
        **_minimal_valid_kwargs(
            stability_constant=None,
            stability_source="unbounded_with_warning",
        )
    )  # type: ignore[arg-type]
    assert cert.stability_constant is None


# --- NaN / Inf rejection --------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_bound_value_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        Certificate(**_minimal_valid_kwargs(bound_value=bad))  # type: ignore[arg-type]


def test_stability_constant_finite_when_set() -> None:
    with pytest.raises(ValidationError):
        Certificate(**_minimal_valid_kwargs(stability_constant=math.inf))  # type: ignore[arg-type]


def test_negative_bound_rejected() -> None:
    with pytest.raises(ValidationError):
        Certificate(**_minimal_valid_kwargs(bound_value=-0.01))  # type: ignore[arg-type]


# --- Migration ------------------------------------------------------------


def test_migrate_unversioned_document() -> None:
    raw = dict(_minimal_valid_kwargs())
    assert "schema_version" not in raw
    migrated = migrate_certificate_document(raw)
    assert migrated["schema_version"] == CERTIFICATE_DOCUMENT_SCHEMA_VERSION
    # Original dict must not be mutated.
    assert "schema_version" not in raw


def test_migrate_current_version_noop() -> None:
    raw = _minimal_valid_kwargs()
    raw["schema_version"] = CERTIFICATE_DOCUMENT_SCHEMA_VERSION
    migrated = migrate_certificate_document(raw)
    assert migrated["schema_version"] == CERTIFICATE_DOCUMENT_SCHEMA_VERSION


def test_migrate_rejects_future_version() -> None:
    raw = _minimal_valid_kwargs()
    raw["schema_version"] = CERTIFICATE_DOCUMENT_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than this binary"):
        migrate_certificate_document(raw)


def test_migrate_rejects_non_int_version() -> None:
    raw = _minimal_valid_kwargs()
    raw["schema_version"] = "1"
    with pytest.raises(ValueError, match="must be int"):
        migrate_certificate_document(raw)


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.integers()
        | st.text(max_size=20)
        | st.floats(allow_nan=False, allow_infinity=False),
        max_size=6,
    )
)
def test_migrate_idempotent_hypothesis(extra: dict[str, object]) -> None:
    """Migrating twice yields the same result as once (Hypothesis-fuzzed).

    Mirrors the ``ScenarioBaselineDocument`` idempotence property. Any
    additional field must survive both passes unchanged.
    """
    raw = dict(_minimal_valid_kwargs())
    raw.update(extra)
    # Skip the pathological case where the fuzzer supplied a non-int schema_version.
    if "schema_version" in raw and not isinstance(raw["schema_version"], int):
        return
    once = migrate_certificate_document(raw)
    twice = migrate_certificate_document(once)
    assert once == twice


# --- Frozen model ---------------------------------------------------------


def test_certificate_is_frozen() -> None:
    """Certificates are artifacts; mutation would break provenance chains."""
    cert = Certificate(**_minimal_valid_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        cert.bound_value = 999.0  # type: ignore[misc]
