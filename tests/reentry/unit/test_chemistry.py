"""Tests for the reentry finite-rate chemistry subpackage.

Covers the previously-untested ``src/reentry/chemistry`` modules: the
``ChemicalMechanism`` protocol, the Park (1993) 5-species air mechanism, and
the stiff operator-split integrator (exercised with a lightweight fake
mechanism so the integrator is tested independently of the Park physics).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.chemistry.mechanism import ChemicalMechanism
from src.reentry.chemistry.park1993 import (
    EXCHANGE_N2_O,
    Park1993Mechanism,
    ReactionRate,
)
from src.reentry.chemistry.stiffness import ChemistryIntegrator
from src.reentry.config.chemistry import ChemistryConfig


def _park() -> Park1993Mechanism:
    return Park1993Mechanism(ChemistryConfig(name="park_test"))


class _FakeMechanism:
    """Minimal mass-conserving A->B mechanism for integrator tests."""

    rate_constant: float = 1.0

    @property
    def n_species(self) -> int:
        return 2

    @property
    def n_reactions(self) -> int:
        return 1

    @property
    def species_names(self) -> list[str]:
        return ["A", "B"]

    def source_terms(self, density, t_tr, t_ve, mass_fractions):  # type: ignore[no-untyped-def]
        ya = mass_fractions[:, 0]
        rate = self.rate_constant * density * ya  # kg/m^3/s
        omega = np.zeros_like(mass_fractions)
        omega[:, 0] = -rate
        omega[:, 1] = rate
        return omega

    def energy_exchange_rate(self, density, t_tr, t_ve, mass_fractions):  # type: ignore[no-untyped-def]
        return np.zeros(density.shape[0], dtype=np.float64)


class TestChemicalMechanismProtocol:
    def test_park_satisfies_protocol(self) -> None:
        assert isinstance(_park(), ChemicalMechanism)

    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(_FakeMechanism(), ChemicalMechanism)


class TestPark1993Properties:
    def test_species_and_reaction_counts(self) -> None:
        mech = _park()
        assert mech.n_species == 5
        assert mech.n_reactions == 17
        assert mech.species_names == ["N2", "O2", "NO", "N", "O"]


class TestPark1993RateHelpers:
    def test_controlling_temperature_equal_temps(self) -> None:
        mech = _park()
        t = np.array([5000.0, 8000.0])
        t_a = mech._controlling_temperature(t, t)
        np.testing.assert_allclose(t_a, t, rtol=1e-12)

    def test_forward_rate_positive_finite(self) -> None:
        mech = _park()
        t_a = np.array([4000.0, 10000.0])
        k = mech._forward_rate(EXCHANGE_N2_O, t_a)
        assert np.all(k > 0) and np.all(np.isfinite(k))

    def test_equilibrium_constant_in_unit_interval(self) -> None:
        mech = _park()
        keq = mech._equilibrium_constant(113200.0, np.array([3000.0, 6000.0]))
        assert np.all((keq > 0) & (keq <= 1.0))

    def test_reaction_rate_is_frozen(self) -> None:
        rr = ReactionRate(C=1.0, n=0.0, theta=100.0)
        with pytest.raises(Exception):
            rr.C = 2.0  # type: ignore[misc]


class TestPark1993SourceTerms:
    def _state(self, temperature: float):
        density = np.array([1.0e-2], dtype=np.float64)
        t = np.array([temperature], dtype=np.float64)
        # Air-like initial composition (mostly N2/O2).
        y = np.array([[0.75, 0.23, 0.0, 0.0, 0.02]], dtype=np.float64)
        return density, t, y

    def test_shape_and_finite(self) -> None:
        mech = _park()
        density, t, y = self._state(8000.0)
        omega = mech.source_terms(density, t, t, y)
        assert omega.shape == (1, 5)
        assert np.all(np.isfinite(omega))

    def test_mass_is_conserved(self) -> None:
        # Each reaction conserves mass, so the per-point species sum is ~0.
        mech = _park()
        density, t, y = self._state(8000.0)
        omega = mech.source_terms(density, t, t, y)
        scale = np.abs(omega).max() + 1.0
        assert abs(float(omega.sum())) <= 1e-6 * scale

    def test_low_temperature_rates_negligible(self) -> None:
        # At 300 K the dissociation Arrhenius factors are ~exp(-large) ~ 0.
        mech = _park()
        density, t, y = self._state(300.0)
        omega = mech.source_terms(density, t, t, y)
        assert np.all(np.abs(omega) < 1e-3)


class TestPark1993EnergyExchange:
    def test_equilibrium_gives_zero_exchange(self) -> None:
        mech = _park()
        density = np.array([1.0e-2])
        t = np.array([6000.0])
        y = np.array([[0.75, 0.23, 0.0, 0.0, 0.02]])
        q_tv = mech.energy_exchange_rate(density, t, t, y)
        np.testing.assert_allclose(q_tv, 0.0, atol=1e-9)

    def test_off_equilibrium_is_finite_nonzero(self) -> None:
        mech = _park()
        density = np.array([1.0e-2])
        y = np.array([[0.75, 0.23, 0.0, 0.0, 0.02]])
        q_tv = mech.energy_exchange_rate(density, np.array([8000.0]), np.array([4000.0]), y)
        assert np.all(np.isfinite(q_tv))
        assert abs(float(q_tv[0])) > 0.0


class TestChemistryIntegrator:
    def _inputs(self):
        density = np.array([1.0e-2], dtype=np.float64)
        t = np.array([6000.0], dtype=np.float64)
        y = np.array([[0.9, 0.1]], dtype=np.float64)
        return density, t, y

    def test_lsoda_advances_and_normalizes(self) -> None:
        integrator = ChemistryIntegrator(_FakeMechanism(), method="lsoda")
        density, t, y = self._inputs()
        y_new = integrator.integrate(density, t, t, y, dt=0.1)
        assert y_new.shape == (1, 2)
        np.testing.assert_allclose(y_new.sum(axis=1), 1.0, rtol=1e-9)
        assert y_new[0, 0] < y[0, 0]  # A decayed into B

    def test_backward_euler_advances(self) -> None:
        integrator = ChemistryIntegrator(_FakeMechanism(), method="backward_euler")
        density, t, y = self._inputs()
        y_new = integrator.integrate(density, t, t, y, dt=0.01)
        np.testing.assert_allclose(y_new.sum(axis=1), 1.0, rtol=1e-9)
        assert np.all((y_new >= 0.0) & (y_new <= 1.0))

    def test_lsoda_failure_falls_back_to_initial(self, monkeypatch) -> None:
        from src.reentry.chemistry import stiffness as stiff_mod

        class _FailSol:
            success = False
            message = "forced failure"

        monkeypatch.setattr(stiff_mod, "solve_ivp", lambda *a, **k: _FailSol())
        integrator = ChemistryIntegrator(_FakeMechanism(), method="lsoda")
        density, t, y = self._inputs()
        y_new = integrator.integrate(density, t, t, y, dt=0.1)
        # Fallback keeps the (normalized) initial composition — no crash.
        np.testing.assert_allclose(y_new, y, rtol=1e-9)

    def test_lsoda_exception_falls_back(self, monkeypatch) -> None:
        from src.reentry.chemistry import stiffness as stiff_mod

        def _boom(*_a, **_k):
            raise RuntimeError("integrator blew up")

        monkeypatch.setattr(stiff_mod, "solve_ivp", _boom)
        integrator = ChemistryIntegrator(_FakeMechanism(), method="lsoda")
        density, t, y = self._inputs()
        y_new = integrator.integrate(density, t, t, y, dt=0.1)
        np.testing.assert_allclose(y_new, y, rtol=1e-9)

    def test_backward_euler_hits_max_substeps(self) -> None:
        # A large rate constant prevents fixed-point convergence within the cap;
        # the loop must still terminate and return a finite, normalized result.
        mech = _FakeMechanism()
        mech.rate_constant = 1.0e6
        integrator = ChemistryIntegrator(mech, method="backward_euler", max_substeps=3)
        density, t, y = self._inputs()
        y_new = integrator.integrate(density, t, t, y, dt=1.0)
        assert np.all(np.isfinite(y_new))
        np.testing.assert_allclose(y_new.sum(axis=1), 1.0, rtol=1e-9)
