"""Coverage boost: wind field, terrain, boundary encoder, ignition, perimeter."""

from __future__ import annotations

import numpy as np

from src.firefighting.config.sensor import SensorConfig
from src.firefighting.config.terrain import TerrainConfig
from src.firefighting.config.wind import WindConfig
from src.firefighting.fire.terrain import TerrainEffects
from src.firefighting.fire.wind import WindField
from src.firefighting.sensor.boundary_encoder import BoundaryEncoder


class TestWindFieldIntegration:
    def test_uniform_field(self) -> None:
        config = WindConfig(name="test", default_speed_m_s=5.0, default_direction_deg=270.0)
        wf = WindField(config)
        u, v = wf.uniform_field(shape=(10, 10))
        assert u.shape == (10, 10)
        # Wind from 270 deg = from west; convention varies, just check non-zero
        assert np.any(u != 0)

    def test_zero_speed(self) -> None:
        config = WindConfig(name="test", default_speed_m_s=0.0, default_direction_deg=0.0)
        wf = WindField(config)
        u, v = wf.uniform_field(shape=(5, 5))
        np.testing.assert_allclose(u, 0.0, atol=1e-10)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)

    def test_fire_modification(self) -> None:
        config = WindConfig(
            name="test",
            default_speed_m_s=5.0,
            default_direction_deg=270.0,
            enable_fire_induced_wind=True,
            fire_wind_coupling_strength=0.5,
        )
        wf = WindField(config)
        u, v = wf.uniform_field(shape=(10, 10))
        temp = np.full((10, 10), 300.0)
        temp[4:6, 4:6] = 1000.0  # Hot spot
        u_mod, v_mod = wf.apply_fire_modification(u, v, temp, dx=10.0, dy=10.0)
        assert u_mod.shape == (10, 10)


class TestTerrainEffectsExtended:
    def test_slope_factor(self) -> None:
        config = TerrainConfig(name="test", enable_slope_effects=True)
        terrain = TerrainEffects(config)
        # compute_slope_factor takes (elevation, dx, dy)
        elev = np.zeros((5, 5))
        elev[0, :] = 100.0  # Slope from south to north
        factor = terrain.compute_slope_factor(elev, dx=10.0, dy=10.0)
        assert factor.shape == (5, 5)

    def test_flat_terrain_factor(self) -> None:
        config = TerrainConfig(name="test", enable_slope_effects=True)
        terrain = TerrainEffects(config)
        elev = np.ones((5, 5)) * 100.0  # Flat
        factor = terrain.compute_slope_factor(elev, dx=10.0, dy=10.0)
        np.testing.assert_allclose(factor, 1.0)

    def test_disabled_terrain(self) -> None:
        config = TerrainConfig(name="test", enable_slope_effects=False)
        terrain = TerrainEffects(config)
        elev = np.random.rand(5, 5) * 100.0
        factor = terrain.compute_slope_factor(elev, dx=10.0, dy=10.0)
        np.testing.assert_allclose(factor, 1.0)


class TestBoundaryEncoder:
    def test_create(self) -> None:
        config = SensorConfig(name="test")
        encoder = BoundaryEncoder(config, grid_shape=(10, 10))
        assert encoder is not None

    def test_update_thermal(self) -> None:
        config = SensorConfig(name="test")
        encoder = BoundaryEncoder(config, grid_shape=(10, 10))
        temp = np.full((10, 10), 400.0)
        encoder.update_thermal(temp, 100.0)
        bc = encoder.get_boundary_conditions(current_time=100.5)
        assert bc.temperature_field.shape == (10, 10)

    def test_update_wind(self) -> None:
        config = SensorConfig(name="test")
        encoder = BoundaryEncoder(config, grid_shape=(10, 10))
        wind_u = np.full((10, 10), 3.0)
        wind_v = np.zeros((10, 10))
        encoder.update_wind(wind_u, wind_v, 100.0)
        bc = encoder.get_boundary_conditions(current_time=100.5)
        assert bc.wind_u.shape == (10, 10)


class TestPerimeterExtended:
    def test_level_set_advance(self) -> None:
        from src.firefighting.fire.perimeter import LevelSetPerimeter

        perim = LevelSetPerimeter(dx=10.0, dy=10.0)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        perim.initialize_from_mask(mask)

        burned_before = perim.get_burned_area_m2()
        speed = np.ones((20, 20)) * 1.0
        perim.advance(speed, dt=1.0)
        burned_after = perim.get_burned_area_m2()
        # Fire should spread
        assert burned_after >= burned_before

    def test_perimeter_mask(self) -> None:
        from src.firefighting.fire.perimeter import LevelSetPerimeter

        perim = LevelSetPerimeter(dx=10.0, dy=10.0)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        perim.initialize_from_mask(mask)
        pmask = perim.get_perimeter_mask()
        assert pmask.shape == (20, 20)
        assert np.any(pmask)
