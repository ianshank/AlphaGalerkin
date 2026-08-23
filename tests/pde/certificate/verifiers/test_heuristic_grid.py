"""Heuristic-grid verifier — behaviour, cost accounting, and failure paths."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from numpy.typing import NDArray

from src.pde.certificate.types import (
    CertificationBudget,
    CertifiedModel,
    DomainSpec,
)
from src.pde.certificate.verifiers.heuristic_grid import HeuristicGridResidualVerifier


def _unit_square(resolution: int = 8) -> DomainSpec:
    return DomainSpec(
        kind="rectangular",
        bounds=((0.0, 1.0), (0.0, 1.0)),
        grid_resolution=resolution,
    )


def _relaxed_budget() -> CertificationBudget:
    return CertificationBudget(max_wall_s=60.0)


# --- Happy path -----------------------------------------------------------


def test_constant_residual_returns_that_constant() -> None:
    """r(x) = c → sample max = c on any grid."""

    def const_residual(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(points.shape[0], 0.375, dtype=np.float64)

    v = HeuristicGridResidualVerifier()
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=const_residual, params=None),
        domain=_unit_square(),
        budget=_relaxed_budget(),
    )
    assert bound.rigor == "heuristic"
    assert bound.domain_coverage == "grid_sampled"
    assert bound.backend == "heuristic_grid"
    assert math.isclose(bound.upper_bound, 0.375, rel_tol=1e-12)
    assert bound.failure_reason is None


def test_absolute_value_taken() -> None:
    """r(x) = -1 everywhere → bound = |−1| = 1."""

    def neg_residual(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(points.shape[0], -1.0, dtype=np.float64)

    v = HeuristicGridResidualVerifier()
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=neg_residual, params=None),
        domain=_unit_square(),
        budget=_relaxed_budget(),
    )
    assert math.isclose(bound.upper_bound, 1.0)


def test_grid_resolution_from_domain() -> None:
    """``DomainSpec.grid_resolution`` overrides the verifier default."""
    seen: list[int] = []

    def counting_residual(points: NDArray[np.float64]) -> NDArray[np.float64]:
        seen.append(int(points.shape[0]))
        return np.zeros(points.shape[0])

    v = HeuristicGridResidualVerifier()
    v.certify(
        model=CertifiedModel(backend="numpy", model_fn=counting_residual, params=None),
        domain=_unit_square(resolution=5),
        budget=_relaxed_budget(),
    )
    # 2D grid, 5 samples per axis → 25 total points.
    assert seen == [25]


def test_default_resolution_when_domain_unset() -> None:
    """Falls back to ``HeuristicGridResidualVerifier`` default when domain is silent."""
    seen: list[int] = []

    def counting_residual(points: NDArray[np.float64]) -> NDArray[np.float64]:
        seen.append(int(points.shape[0]))
        return np.zeros(points.shape[0])

    v = HeuristicGridResidualVerifier(default_grid_resolution=6)
    domain = DomainSpec(kind="rectangular", bounds=((0.0, 1.0), (0.0, 1.0)))
    v.certify(
        model=CertifiedModel(backend="numpy", model_fn=counting_residual, params=None),
        domain=domain,
        budget=_relaxed_budget(),
    )
    assert seen == [36]  # 6 × 6


def test_default_resolution_positive() -> None:
    with pytest.raises(ValueError):
        HeuristicGridResidualVerifier(default_grid_resolution=0)


# --- Cost accounting ------------------------------------------------------


def test_wall_clock_fields_populated() -> None:
    """Spec §4 AC5 — every bound carries the three cost fields."""

    def r(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(points.shape[0])

    v = HeuristicGridResidualVerifier()
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=r, params=None),
        domain=_unit_square(),
        budget=_relaxed_budget(),
    )
    assert bound.compile_wall_s == 0.0  # no JIT in the heuristic path
    assert bound.cert_wall_s >= 0.0
    assert bound.steady_state_wall_s == bound.cert_wall_s


# --- Structural rigor guard ----------------------------------------------


def test_heuristic_verifier_cannot_forge_rigorous_bound() -> None:
    """A grid-sampled bound cannot promote itself to rigor='rigorous'.

    The type validator refuses at :class:`CertifiedResidualBound` construction
    — this test is a check that the *verifier* never even tries.
    """

    def r(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(points.shape[0])

    v = HeuristicGridResidualVerifier()
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=r, params=None),
        domain=_unit_square(),
        budget=_relaxed_budget(),
    )
    assert bound.rigor == "heuristic"
    assert bound.domain_coverage == "grid_sampled"


# --- Domain-kind handling -------------------------------------------------


def test_sdf_domain_produces_failed_bound_in_ws1() -> None:
    """SDF coverage lands in WS2; WS1 fails closed per §4 AC1 (no raise)."""

    def r(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(points.shape[0])

    v = HeuristicGridResidualVerifier()
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=r, params=None),
        domain=DomainSpec(kind="sdf", sdf_reference="analytical_helix"),
        budget=_relaxed_budget(),
    )
    assert bound.rigor == "failed"
    assert bound.failure_reason is not None
    assert "only supports rectangular" in bound.failure_reason


# --- Failure paths --------------------------------------------------------


def test_budget_overrun_fails_closed_by_default() -> None:
    """Spec §4 AC1 — default budget behaviour is fail-closed."""

    def r(points: NDArray[np.float64]) -> NDArray[np.float64]:
        # Force wall-clock over the tiny budget below.
        time.sleep(0.05)
        return np.zeros(points.shape[0])

    v = HeuristicGridResidualVerifier()
    tight = CertificationBudget(max_wall_s=0.01, allow_heuristic_fallback=False)
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=r, params=None),
        domain=_unit_square(),
        budget=tight,
    )
    assert bound.rigor == "failed"
    assert bound.failure_reason is not None
    assert "budget_exceeded" in bound.failure_reason


def test_budget_overrun_with_fallback_returns_heuristic_with_notes() -> None:
    """``allow_heuristic_fallback=True`` opts in to the tagged-heuristic path."""

    def r(points: NDArray[np.float64]) -> NDArray[np.float64]:
        time.sleep(0.05)
        return np.full(points.shape[0], 0.1)

    v = HeuristicGridResidualVerifier()
    tight = CertificationBudget(max_wall_s=0.01, allow_heuristic_fallback=True)
    bound = v.certify(
        model=CertifiedModel(backend="numpy", model_fn=r, params=None),
        domain=_unit_square(),
        budget=tight,
    )
    assert bound.rigor == "heuristic"
    assert bound.notes == "budget_exceeded"


def test_model_backend_mismatch_raises() -> None:
    v = HeuristicGridResidualVerifier()
    with pytest.raises((ValueError, TypeError)):  # Pydantic rejects the unknown backend first
        v.certify(
            model=CertifiedModel(backend="wat", model_fn=lambda x: x, params=None),  # type: ignore[arg-type]
            domain=_unit_square(),
            budget=_relaxed_budget(),
        )
