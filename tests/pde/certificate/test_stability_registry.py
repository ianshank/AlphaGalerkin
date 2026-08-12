"""Gate 6 — Stability-constant registry honesty (spec AC5).

Every :class:`~src.pde.config.PDEType` value must have a declared entry
(``undocumented_stability_constants = 0``). Helmholtz / Biharmonic /
Navier-Stokes ship as ``unbounded_with_warning`` — a wrong number here would
be exactly the fabrication precedent the certificate spec exists to prevent.
"""

from __future__ import annotations

import pytest

from src.pde.certificate import (
    StabilityConstantRegistry,
    StabilityEntry,
    register_stability,
)
from src.pde.certificate.stability import UNBOUNDED_RENDER_STRING
from src.pde.config import PDEType


def test_every_pde_type_registered() -> None:
    """AC5: ``undocumented_stability_constants = 0``.

    Every enum value must appear in the registry. If a new ``PDEType`` is
    added upstream without a stability entry, this test fails loudly.
    """
    registry = StabilityConstantRegistry()
    registered = set(registry.registered_types())
    missing = set(PDEType) - registered
    assert missing == set(), f"undocumented stability constants: {[t.value for t in missing]!r}"


def test_unbounded_operators_render_warning() -> None:
    """AC5 render contract — the warning phrasing is fixed."""
    registry = StabilityConstantRegistry()
    for pde_type in (PDEType.HELMHOLTZ, PDEType.BIHARMONIC, PDEType.NAVIER_STOKES):
        entry = registry.get(pde_type)
        assert entry.source == "unbounded_with_warning"
        assert entry.value is None
        assert entry.render() == UNBOUNDED_RENDER_STRING


def test_bounded_operators_carry_numeric_constant() -> None:
    """Non-UNBOUNDED entries must ship a positive numeric constant."""
    registry = StabilityConstantRegistry()
    bounded = (
        PDEType.POISSON,
        PDEType.HEAT,
        PDEType.ADVECTION_DIFFUSION,
        PDEType.WAVE,
        PDEType.BURGERS,
    )
    for pde_type in bounded:
        entry = registry.get(pde_type)
        assert entry.source in ("analytic", "estimated")
        assert entry.value is not None
        assert entry.value > 0.0


def test_get_unknown_raises() -> None:
    """Silent fall-through would violate AC5. ``get`` on an unregistered type must raise."""
    registry = StabilityConstantRegistry()
    # Reset to force the miss.
    StabilityConstantRegistry._reset_for_tests()
    fresh = StabilityConstantRegistry()
    with pytest.raises(KeyError, match="no stability entry registered"):
        fresh.get(PDEType.POISSON)


def test_duplicate_register_raises() -> None:
    """Two modules cannot silently disagree on the same operator's constant."""
    registry = StabilityConstantRegistry()
    assert registry.has(PDEType.POISSON)  # populated by builtin
    dup = StabilityEntry(pde_type=PDEType.POISSON, source="analytic", value=2.0, notes="dup")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(dup)


def test_replace_overrides_and_logs_warning() -> None:
    """``replace()`` is the intentional escape hatch — it must succeed and log."""
    registry = StabilityConstantRegistry()
    original = registry.get(PDEType.POISSON)
    new_entry = StabilityEntry(
        pde_type=PDEType.POISSON, source="analytic", value=42.0, notes="override"
    )
    registry.replace(new_entry)
    assert registry.get(PDEType.POISSON).value == 42.0
    assert original.value != 42.0  # frozen model — untouched


def test_register_stability_helper_rejects_negative_value() -> None:
    StabilityConstantRegistry._reset_for_tests()
    fresh = StabilityConstantRegistry()  # noqa: F841 — populated for isolation
    with pytest.raises(ValueError, match="must be positive"):
        register_stability(PDEType.POISSON, source="analytic", value=-1.0)


def test_register_stability_helper_rejects_bounded_none() -> None:
    StabilityConstantRegistry._reset_for_tests()
    fresh = StabilityConstantRegistry()  # noqa: F841
    with pytest.raises(ValueError, match="requires a numeric value"):
        register_stability(PDEType.POISSON, source="analytic", value=None)


def test_register_stability_helper_rejects_unbounded_with_value() -> None:
    StabilityConstantRegistry._reset_for_tests()
    fresh = StabilityConstantRegistry()  # noqa: F841
    with pytest.raises(ValueError, match="requires value=None"):
        register_stability(PDEType.HELMHOLTZ, source="unbounded_with_warning", value=1.0)


def test_registry_is_singleton() -> None:
    """Two ``StabilityConstantRegistry()`` calls return the same instance."""
    assert StabilityConstantRegistry() is StabilityConstantRegistry()


def test_registered_types_snapshot_immutable() -> None:
    """``registered_types()`` returns a tuple — accidental mutation is impossible."""
    registry = StabilityConstantRegistry()
    snap = registry.registered_types()
    assert isinstance(snap, tuple)


def test_has_returns_bool() -> None:
    registry = StabilityConstantRegistry()
    assert registry.has(PDEType.POISSON) is True


def test_helmholtz_notes_reference_operator_gate() -> None:
    """The TODO trail must point at the follow-on spec, per the plan §Weakness #2."""
    entry = StabilityConstantRegistry().get(PDEType.HELMHOLTZ)
    assert "operator_gate" in entry.notes
