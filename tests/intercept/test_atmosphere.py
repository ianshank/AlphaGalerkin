"""Tests for ISA atmosphere model and wind profiles."""

from __future__ import annotations

import torch

from src.intercept.atmosphere import (
    ISA_P0,
    ISA_RHO0,
    ISA_T0,
    ISA_TROPOPAUSE_ALT,
    ISA_TROPOPAUSE_T,
    ISAAtmosphere,
    WindModel,
)
from src.intercept.config import AtmosphereConfig, WindProfileType


class TestISAAtmosphere:
    def setup_method(self) -> None:
        self.atmo = ISAAtmosphere()

    def test_sea_level_temperature(self) -> None:
        alt = torch.tensor(0.0, dtype=torch.float64)
        T = self.atmo.temperature(alt)
        assert torch.allclose(T, torch.tensor(ISA_T0, dtype=torch.float64), atol=0.01)

    def test_sea_level_density(self) -> None:
        alt = torch.tensor(0.0, dtype=torch.float64)
        rho = self.atmo.density(alt)
        assert torch.allclose(rho, torch.tensor(ISA_RHO0, dtype=torch.float64), atol=0.001)

    def test_sea_level_pressure(self) -> None:
        alt = torch.tensor(0.0, dtype=torch.float64)
        P = self.atmo.pressure(alt)
        assert torch.allclose(P, torch.tensor(ISA_P0, dtype=torch.float64), atol=1.0)

    def test_temperature_decreases_with_altitude(self) -> None:
        alts = torch.tensor([0.0, 1000.0, 5000.0, 10000.0], dtype=torch.float64)
        temps = self.atmo.temperature(alts)
        # Temperature should decrease in troposphere
        for i in range(len(alts) - 1):
            assert temps[i] > temps[i + 1]

    def test_tropopause_temperature(self) -> None:
        alt = torch.tensor(ISA_TROPOPAUSE_ALT, dtype=torch.float64)
        T = self.atmo.temperature(alt)
        assert torch.allclose(T, torch.tensor(ISA_TROPOPAUSE_T, dtype=torch.float64), atol=0.5)

    def test_stratosphere_constant_temperature(self) -> None:
        alt1 = torch.tensor(12000.0, dtype=torch.float64)
        alt2 = torch.tensor(15000.0, dtype=torch.float64)
        T1 = self.atmo.temperature(alt1)
        T2 = self.atmo.temperature(alt2)
        assert torch.allclose(T1, T2, atol=0.01)

    def test_density_decreases_with_altitude(self) -> None:
        alts = torch.tensor([0.0, 5000.0, 10000.0, 15000.0], dtype=torch.float64)
        rhos = self.atmo.density(alts)
        for i in range(len(alts) - 1):
            assert rhos[i] > rhos[i + 1]

    def test_pressure_decreases_with_altitude(self) -> None:
        alts = torch.tensor([0.0, 5000.0, 10000.0, 15000.0], dtype=torch.float64)
        pressures = self.atmo.pressure(alts)
        for i in range(len(alts) - 1):
            assert pressures[i] > pressures[i + 1]

    def test_speed_of_sound_sea_level(self) -> None:
        alt = torch.tensor(0.0, dtype=torch.float64)
        a = self.atmo.speed_of_sound(alt)
        # ISA sea-level speed of sound ~ 340.3 m/s
        assert torch.allclose(a, torch.tensor(340.3, dtype=torch.float64), atol=0.5)

    def test_mach_number(self) -> None:
        speed = torch.tensor(340.0, dtype=torch.float64)
        alt = torch.tensor(0.0, dtype=torch.float64)
        mach = self.atmo.mach_number(speed, alt)
        assert torch.allclose(mach, torch.tensor(1.0, dtype=torch.float64), atol=0.01)

    def test_dynamic_pressure(self) -> None:
        speed = torch.tensor(100.0, dtype=torch.float64)
        alt = torch.tensor(0.0, dtype=torch.float64)
        q = self.atmo.dynamic_pressure(speed, alt)
        expected = 0.5 * ISA_RHO0 * 100.0**2
        assert torch.allclose(q, torch.tensor(expected, dtype=torch.float64), atol=10.0)

    def test_negative_altitude_clamped(self) -> None:
        alt = torch.tensor(-100.0, dtype=torch.float64)
        T = self.atmo.temperature(alt)
        T_zero = self.atmo.temperature(torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(T, T_zero, atol=0.01)

    def test_batched_computation(self) -> None:
        alts = torch.linspace(0, 15000, 100, dtype=torch.float64)
        rho = self.atmo.density(alts)
        assert rho.shape == (100,)
        assert (rho > 0).all()

    def test_temperature_offset(self) -> None:
        config = AtmosphereConfig(name="hot", temperature_offset_k=10.0)
        atmo = ISAAtmosphere(config)
        alt = torch.tensor(0.0, dtype=torch.float64)
        T = atmo.temperature(alt)
        assert torch.allclose(T, torch.tensor(ISA_T0 + 10.0, dtype=torch.float64), atol=0.01)


class TestWindModel:
    def test_zero_wind(self) -> None:
        wind = WindModel()
        alt = torch.tensor(100.0, dtype=torch.float64)
        w = wind.get_wind(alt)
        assert torch.allclose(w, torch.zeros(3, dtype=torch.float64))

    def test_constant_wind(self) -> None:
        config = AtmosphereConfig(
            name="windy",
            wind_profile=WindProfileType.CONSTANT,
            wind_speed_ms=10.0,
            wind_direction_rad=0.0,  # From North
        )
        wind = WindModel(config)
        alt = torch.tensor(100.0, dtype=torch.float64)
        w = wind.get_wind(alt)
        # Wind from North -> negative North component
        assert w[..., 0] < 0
        assert torch.allclose(
            torch.norm(w[..., :2]),
            torch.tensor(10.0, dtype=torch.float64),
            atol=0.1,
        )

    def test_logarithmic_increases_with_altitude(self) -> None:
        config = AtmosphereConfig(
            name="log_wind",
            wind_profile=WindProfileType.LOGARITHMIC,
            wind_speed_ms=10.0,
            wind_direction_rad=0.0,
        )
        wind = WindModel(config)
        w_low = wind.get_wind(torch.tensor(10.0, dtype=torch.float64))
        w_high = wind.get_wind(torch.tensor(100.0, dtype=torch.float64))
        assert torch.norm(w_high) > torch.norm(w_low)

    def test_batched_wind(self) -> None:
        config = AtmosphereConfig(
            name="test",
            wind_profile=WindProfileType.CONSTANT,
            wind_speed_ms=5.0,
        )
        wind = WindModel(config)
        alts = torch.linspace(0, 1000, 10, dtype=torch.float64)
        w = wind.get_wind(alts)
        assert w.shape == (10, 3)
