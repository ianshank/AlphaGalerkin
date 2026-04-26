"""Tests for fire physics models (fuel, ignition, radiation, terrain)."""

from __future__ import annotations

import numpy as np
import pytest

from src.firefighting.config.fire import FireConfig
from src.firefighting.config.terrain import TerrainConfig
from src.firefighting.config.wind import WindConfig
from src.firefighting.fire.convection import ConvectiveHeatTransfer
from src.firefighting.fire.fuel import FuelModel, FuelState
from src.firefighting.fire.ignition import IgnitionModel
from src.firefighting.fire.perimeter import LevelSetPerimeter
from src.firefighting.fire.radiation import RadiativeHeatTransfer
from src.firefighting.fire.terrain import TerrainEffects
from src.firefighting.fire.wind import WindField


@pytest.fixture
def fire_config() -> FireConfig:
    return FireConfig(name="test")


class TestFuelModel:
    def test_no_combustion_below_ignition(self, fire_config: FireConfig) -> None:
        model = FuelModel(fire_config)
        temp = np.full((10, 10), 300.0)  # Ambient
        fuel = np.ones((10, 10))
        rate = model.fuel_consumption_rate(temp, fuel)
        np.testing.assert_allclose(rate, 0.0)

    def test_combustion_above_ignition(self, fire_config: FireConfig) -> None:
        model = FuelModel(fire_config)
        temp = np.full((10, 10), 700.0)  # Above ignition
        fuel = np.ones((10, 10))
        rate = model.fuel_consumption_rate(temp, fuel)
        assert np.all(rate > 0)

    def test_ignition_mask(self, fire_config: FireConfig) -> None:
        model = FuelModel(fire_config)
        temp = np.full((10, 10), 300.0)
        temp[5, 5] = 700.0
        fuel = np.ones((10, 10))
        mask = model.ignition_mask(temp, fuel)
        assert mask[5, 5] is np.True_
        assert not mask[0, 0]

    def test_no_combustion_without_fuel(self, fire_config: FireConfig) -> None:
        model = FuelModel(fire_config)
        temp = np.full((10, 10), 700.0)
        fuel = np.zeros((10, 10))
        mask = model.ignition_mask(temp, fuel)
        assert not np.any(mask)


class TestFuelState:
    def test_available_fuel(self) -> None:
        fuel = FuelState(
            loading=np.ones((5, 5)),
            moisture=np.full((5, 5), 0.08),
            consumed=np.full((5, 5), 0.3),
        )
        np.testing.assert_allclose(fuel.available, 0.7)

    def test_effective_loading(self) -> None:
        fuel = FuelState(
            loading=np.full((5, 5), 2.0),
            moisture=np.full((5, 5), 0.08),
            consumed=np.full((5, 5), 0.5),
        )
        np.testing.assert_allclose(fuel.effective_loading, 1.0)


class TestIgnitionModel:
    def test_no_ignition_at_ambient(self, fire_config: FireConfig) -> None:
        model = IgnitionModel(fire_config)
        temp = np.full((10, 10), 300.0)
        fuel = np.ones((10, 10))
        model.initialize((10, 10))
        result = model.update(temp, fuel, dt=1.0)
        assert not np.any(result)


class TestRadiativeHeatTransfer:
    def test_no_radiation_without_burning(self, fire_config: FireConfig) -> None:
        rad = RadiativeHeatTransfer(fire_config)
        temp = np.full((10, 10), 300.0)
        burning = np.zeros((10, 10), dtype=bool)
        q = rad.compute(temp, burning, dx=10.0, dy=10.0)
        np.testing.assert_allclose(q, 0.0)

    def test_radiation_from_burning_cell(self, fire_config: FireConfig) -> None:
        rad = RadiativeHeatTransfer(fire_config)
        temp = np.full((10, 10), 300.0)
        temp[5, 5] = 1200.0
        burning = np.zeros((10, 10), dtype=bool)
        burning[5, 5] = True
        q = rad.compute(temp, burning, dx=10.0, dy=10.0)
        # Neighbors of (5,5) should receive radiation
        assert q[4, 5] > 0 or q[6, 5] > 0


class TestConvectiveHeatTransfer:
    def test_no_transport_without_wind(self, fire_config: FireConfig) -> None:
        conv = ConvectiveHeatTransfer(fire_config)
        temp = np.full((10, 10), 300.0)
        wind_u = np.zeros((10, 10))
        wind_v = np.zeros((10, 10))
        dt = conv.compute(temp, wind_u, wind_v, dx=10.0, dy=10.0)
        np.testing.assert_allclose(dt, 0.0)


class TestTerrainEffects:
    def test_flat_terrain_no_effect(self) -> None:
        config = TerrainConfig(name="test", flat_terrain=True)
        terrain = TerrainEffects(config)
        elevation = np.zeros((10, 10))
        factor = terrain.compute_slope_factor(elevation, dx=10.0, dy=10.0)
        np.testing.assert_allclose(factor, 1.0)

    def test_uphill_increases_spread(self) -> None:
        config = TerrainConfig(name="test")
        terrain = TerrainEffects(config)
        # Create sloped terrain
        x = np.arange(10)
        elevation = np.outer(np.ones(10), x * 10.0)  # 10m per cell
        factor = terrain.compute_slope_factor(elevation, dx=10.0, dy=10.0)
        # Slope factor should be > 1 where slope exists
        assert factor.max() > 1.0


class TestWindField:
    def test_uniform_field(self) -> None:
        config = WindConfig(name="test", default_speed_m_s=10.0, default_direction_deg=0.0)
        wind = WindField(config)
        wu, wv = wind.uniform_field((10, 10))
        assert wu.shape == (10, 10)
        assert wv.shape == (10, 10)


class TestLevelSetPerimeter:
    def test_initialize_from_mask(self) -> None:
        ls = LevelSetPerimeter(dx=10.0, dy=10.0)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        phi = ls.initialize_from_mask(mask)
        assert phi[10, 10] < 0  # Inside fire
        assert phi[0, 0] > 0  # Outside fire

    def test_burned_area(self) -> None:
        ls = LevelSetPerimeter(dx=10.0, dy=10.0)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True  # 4x4 = 16 cells
        ls.initialize_from_mask(mask)
        area = ls.get_burned_area_m2()
        # ~16 cells * 100 m^2 = ~1600 m^2
        assert area > 0
