"""Tests for conserved-quantity integrals, drift, and the conservation audit."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.cli import run_audit_conservation
from src.reentry.conservation import (
    CONSERVATION_DRIFT_FLOOR,
    conservation_drift,
    conserved_integrals_1d,
)


class TestConservedIntegrals1D:
    def test_uniform_field(self) -> None:
        # Uniform ρ=2, u=3, p=4 over 10 cells of width 0.1 (length 1.0), γ=1.4.
        n = 10
        dx = 0.1
        rho = np.full(n, 2.0)
        u = np.full(n, 3.0)
        p = np.full(n, 4.0)
        integrals = conserved_integrals_1d(rho, u, p, dx, gamma=1.4)

        # mass = ρ·L = 2·1.0
        assert integrals["mass"] == pytest.approx(2.0)
        # momentum = ρu·L = 6·1.0
        assert integrals["momentum"] == pytest.approx(6.0)
        # energy density = p/(γ-1) + ½ρu² = 4/0.4 + 0.5·2·9 = 10 + 9 = 19; ·L
        assert integrals["energy"] == pytest.approx(19.0)

    def test_zero_velocity_zero_momentum(self) -> None:
        n = 5
        integrals = conserved_integrals_1d(np.ones(n), np.zeros(n), np.ones(n), dx=0.2, gamma=1.4)
        assert integrals["momentum"] == pytest.approx(0.0)

    def test_invalid_dx_raises(self) -> None:
        with pytest.raises(ValueError, match="dx must be positive"):
            conserved_integrals_1d(np.ones(3), np.zeros(3), np.ones(3), dx=0.0, gamma=1.4)

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma must exceed 1"):
            conserved_integrals_1d(np.ones(3), np.zeros(3), np.ones(3), dx=0.1, gamma=1.0)


class TestConservationDrift:
    def test_zero_drift(self) -> None:
        initial = {"mass": 1.0, "energy": 2.0}
        drift = conservation_drift(initial, dict(initial))
        assert drift["mass"]["absolute"] == 0.0
        assert drift["mass"]["relative"] == 0.0

    def test_relative_and_absolute(self) -> None:
        drift = conservation_drift({"mass": 10.0}, {"mass": 11.0})
        assert drift["mass"]["absolute"] == pytest.approx(1.0)
        assert drift["mass"]["relative"] == pytest.approx(0.1)

    def test_zero_baseline_uses_floor(self) -> None:
        # Baseline 0 -> relative divides by the floor (no ZeroDivisionError).
        drift = conservation_drift({"momentum": 0.0}, {"momentum": 0.5})
        assert drift["momentum"]["absolute"] == pytest.approx(0.5)
        assert drift["momentum"]["relative"] == pytest.approx(0.5 / CONSERVATION_DRIFT_FLOOR)

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            conservation_drift({"mass": 1.0}, {"energy": 1.0})


class TestConservationAudit:
    def test_audit_passes_on_sod(self) -> None:
        # Finite-volume Roe scheme conserves mass + energy to ~machine precision
        # while the waves stay interior at t=0.2.
        assert run_audit_conservation() is True

    def test_audit_tight_tolerance_still_passes(self) -> None:
        # Mass drift is exactly 0 and energy ~3e-16, so even 1e-10 passes.
        assert run_audit_conservation(rtol=1e-10) is True

    def test_audit_fails_under_impossible_tolerance(self) -> None:
        # Energy drift is ~3e-16; a sub-machine tolerance must fail.
        assert run_audit_conservation(rtol=1e-20) is False
