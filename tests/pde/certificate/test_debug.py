"""Human-readable inspector for :class:`CertifiedResidualBound`."""

from __future__ import annotations

from src.pde.certificate.debug import inspect_bound
from src.pde.certificate.types import CertifiedResidualBound, HardwareMeta

_BOUND_DEFAULTS: dict[str, object] = {
    "upper_bound": 0.5,
    "rigor": "heuristic",
    "backend": "heuristic_grid",
    "domain_coverage": "grid_sampled",
    "compile_wall_s": 0.0,
    "cert_wall_s": 0.42,
    "steady_state_wall_s": 0.42,
    "failure_reason": None,
    "notes": "",
}


def _bound(**overrides: object) -> CertifiedResidualBound:
    kwargs: dict[str, object] = {
        **_BOUND_DEFAULTS,
        "hardware_meta": HardwareMeta(device="cpu", dtype="float64"),
    }
    kwargs.update(overrides)
    return CertifiedResidualBound(**kwargs)  # type: ignore[arg-type]


def test_compact_summary_single_line() -> None:
    s = inspect_bound(_bound(), verbose=False)
    assert "\n" not in s
    assert "upper=" in s
    assert "heuristic" in s


def test_verbose_summary_multi_line() -> None:
    s = inspect_bound(_bound(), verbose=True)
    assert "\n" in s
    assert "upper_bound" in s
    assert "cert_wall_s" in s
    assert "device" in s


def test_verbose_includes_failure_reason_when_present() -> None:
    s = inspect_bound(
        _bound(rigor="failed", upper_bound=0.0, failure_reason="budget_exceeded: x"),
        verbose=True,
    )
    assert "failure_reason" in s
    assert "budget_exceeded" in s


def test_verbose_includes_notes_when_present() -> None:
    s = inspect_bound(_bound(notes="budget_exceeded"), verbose=True)
    assert "notes" in s
