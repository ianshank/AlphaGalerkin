"""Tests for sensor decoders and wind interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from src.firefighting.config.sensor import SensorConfig
from src.firefighting.sensor.thermal import ThermalCameraDecoder, ThermalFrame
from src.firefighting.sensor.wind import WindFieldInterpolator, WindObservation


class TestThermalCameraDecoder:
    def test_decode_generic(self) -> None:
        config = SensorConfig(name="test", thermal_fov_deg=45.0)
        decoder = ThermalCameraDecoder(config)
        raw = np.full((480, 640), 350.0)  # 350 K everywhere
        frame = decoder.decode_frame(raw, timestamp=100.0)
        assert frame.shape == (480, 640)
        np.testing.assert_allclose(frame.temperature_K, 350.0)
        assert np.all(frame.confidence >= 0.5)

    def test_hot_spots(self) -> None:
        frame = ThermalFrame(
            temperature_K=np.array([[300, 500], [400, 800]], dtype=np.float64),
            timestamp=0.0,
            confidence=np.ones((2, 2)),
        )
        spots = frame.hot_spots(threshold_K=400.0)
        assert spots[0, 0] is np.False_
        assert spots[0, 1] is np.True_
        assert spots[1, 0] is np.True_
        assert spots[1, 1] is np.True_

    def test_max_temperature(self) -> None:
        frame = ThermalFrame(
            temperature_K=np.array([[300, 500], [400, 800]], dtype=np.float64),
            timestamp=0.0,
            confidence=np.ones((2, 2)),
        )
        assert frame.max_temperature == 800.0

    def test_project_to_grid(self) -> None:
        config = SensorConfig(name="test", thermal_fov_deg=90.0)
        decoder = ThermalCameraDecoder(config)
        raw = np.full((10, 10), 400.0)
        frame = decoder.decode_frame(raw, timestamp=0.0)

        temp_grid, conf_grid = decoder.project_to_grid(
            frame,
            grid_shape=(20, 20),
            domain_size_x_m=200.0,
            domain_size_y_m=200.0,
            drone_x_m=100.0,
            drone_y_m=100.0,
            altitude_m=100.0,
        )
        # Some cells should have data, others NaN
        assert temp_grid.shape == (20, 20)
        assert np.any(~np.isnan(temp_grid))


class TestWindFieldInterpolator:
    def test_no_observations_uses_default(self) -> None:
        interp = WindFieldInterpolator(default_speed_m_s=10.0, default_direction_deg=0.0)
        grid_x = np.ones((5, 5)) * 50.0
        grid_y = np.ones((5, 5)) * 50.0
        wu, wv = interp.interpolate([], grid_x, grid_y)
        # Direction 0 (north) => v = speed, u = 0
        np.testing.assert_allclose(wu, 0.0, atol=1e-10)
        np.testing.assert_allclose(wv, 10.0, atol=1e-10)

    def test_single_observation(self) -> None:
        interp = WindFieldInterpolator()
        obs = [WindObservation(x=50, y=50, speed_m_s=15.0, direction_deg=90.0, timestamp=0.0)]
        grid_x = np.array([[50.0]])
        grid_y = np.array([[50.0]])
        wu, wv = interp.interpolate(obs, grid_x, grid_y)
        # Direction 90 (east) => u = speed, v ~ 0
        assert wu[0, 0] == pytest.approx(15.0, abs=0.1)
        assert abs(wv[0, 0]) < 0.1

    def test_multiple_observations_blend(self) -> None:
        interp = WindFieldInterpolator()
        obs = [
            WindObservation(x=0, y=50, speed_m_s=10.0, direction_deg=90.0, timestamp=0.0),
            WindObservation(x=100, y=50, speed_m_s=10.0, direction_deg=270.0, timestamp=0.0),
        ]
        grid_x = np.array([[50.0]])  # Midpoint
        grid_y = np.array([[50.0]])
        wu, wv = interp.interpolate(obs, grid_x, grid_y)
        # Opposite directions at equal distance => should cancel
        assert abs(wu[0, 0]) < 0.1
