"""Tests for equation of state implementations."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.config.gas import GasConfig
from src.reentry.gas.eos import CaloricallyPerfectEOS, EquationOfState, ThermallyPerfectEOS


@pytest.fixture
def gas_config() -> GasConfig:
    return GasConfig(
        name="test",
        species=["N2"],
        gamma=1.4,
        molecular_weights={"N2": 0.0280134},
        formation_enthalpies={"N2": 0.0},
        theta_v={"N2": 3395.0},
    )


@pytest.fixture
def gas_config_5sp() -> GasConfig:
    return GasConfig(name="test_5sp")


class TestCaloricallyPerfectEOS:
    def test_satisfies_protocol(self, gas_config: GasConfig) -> None:
        eos = CaloricallyPerfectEOS(gas_config)
        assert isinstance(eos, EquationOfState)

    def test_ideal_gas_law(self, gas_config: GasConfig) -> None:
        eos = CaloricallyPerfectEOS(gas_config)
        rho = np.array([1.225])
        yi = np.array([[1.0]])

        # p = rho * R * T => T = p / (rho * R)
        # R_N2 = 8.314 / 0.028 ≈ 296.9
        p = np.array([101325.0])
        t = eos.temperature(rho, p, yi)
        assert 270 < t[0] < 290  # Standard atmosphere ~288K

    def test_sound_speed(self, gas_config: GasConfig) -> None:
        eos = CaloricallyPerfectEOS(gas_config)
        rho = np.array([1.225])
        p = np.array([101325.0])
        yi = np.array([[1.0]])

        a = eos.sound_speed(rho, p, yi)
        assert 330 < a[0] < 350  # ~340 m/s at STP

    def test_pressure_from_internal_energy(self, gas_config: GasConfig) -> None:
        eos = CaloricallyPerfectEOS(gas_config)
        rho = np.array([1.0])
        yi = np.array([[1.0]])
        # e = p / (rho * (gamma-1))
        p_expected = 100000.0
        e = p_expected / (rho[0] * 0.4)
        p = eos.pressure(rho, np.array([e]), yi)
        np.testing.assert_allclose(p[0], p_expected, rtol=1e-10)

    def test_energy_roundtrip(self, gas_config: GasConfig) -> None:
        eos = CaloricallyPerfectEOS(gas_config)
        rho = np.array([2.0])
        p_in = np.array([200000.0])
        yi = np.array([[1.0]])

        e = eos.internal_energy(rho, p_in, yi)
        p_out = eos.pressure(rho, e, yi)
        np.testing.assert_allclose(p_out, p_in, rtol=1e-10)


class TestThermallyPerfectEOS:
    def test_satisfies_protocol(self, gas_config: GasConfig) -> None:
        eos = ThermallyPerfectEOS(gas_config)
        assert isinstance(eos, EquationOfState)

    def test_temperature_computation(self, gas_config: GasConfig) -> None:
        eos = ThermallyPerfectEOS(gas_config)
        rho = np.array([1.225])
        p = np.array([101325.0])
        yi = np.array([[1.0]])
        t = eos.temperature(rho, p, yi)
        # Same as ideal gas at low temperature
        assert 270 < t[0] < 290

    def test_high_temperature_differs_from_calorically_perfect(self, gas_config: GasConfig) -> None:
        """At high T, vibrational modes change gamma."""
        cpg = CaloricallyPerfectEOS(gas_config)
        tpg = ThermallyPerfectEOS(gas_config)

        rho = np.array([0.01])
        p = np.array([50000.0])
        yi = np.array([[1.0]])

        a_cpg = cpg.sound_speed(rho, p, yi)
        a_tpg = tpg.sound_speed(rho, p, yi)

        # At high temp, vibrational excitation lowers effective gamma
        # so ThermallyPerfect sound speed differs
        assert a_cpg[0] != a_tpg[0]
