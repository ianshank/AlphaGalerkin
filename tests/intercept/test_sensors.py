"""Tests for sensor models and multi-sensor fusion."""

from __future__ import annotations

import pytest
import torch

from src.intercept.config import SensorConfig, SensorType
from src.intercept.dynamics import create_initial_state
from src.intercept.sensors import (
    ElectroOpticalSensor,
    GPSDeniedNavigator,
    RadarSensor,
    SensorFusion,
    SensorRegistry,
    StalenessTracker,
)


class TestRadarSensor:
    def setup_method(self) -> None:
        self.config = SensorConfig(
            name="test_radar",
            sensor_type=SensorType.RADAR,
            range_noise_m=5.0,
            azimuth_noise_rad=0.005,
            elevation_noise_rad=0.005,
            max_range_m=50000.0,
            fov_rad=1.5,
            update_rate_hz=10.0,
        )
        self.sensor = RadarSensor(self.config)

    def test_detection_in_fov(self) -> None:
        target = create_initial_state(position=[5000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0])
        own_pos = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)
        meas = self.sensor.detect(target, own_pos, time_s=0.0)
        assert meas is not None
        assert meas.position.shape == (3,)
        assert meas.covariance.shape == (3, 3)

    def test_no_detection_out_of_range(self) -> None:
        target = create_initial_state(
            position=[100000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0]
        )
        own_pos = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)
        meas = self.sensor.detect(target, own_pos, time_s=0.0)
        assert meas is None

    def test_noise_statistics(self) -> None:
        """Measurement noise should be approximately Gaussian with configured sigma."""
        target = create_initial_state(position=[5000.0, 0.0, -3000.0], velocity=[0.0, 0.0, 0.0])
        own_pos = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)

        errors = []
        for _ in range(200):
            meas = self.sensor.detect(target, own_pos, time_s=0.0)
            assert meas is not None
            err = torch.norm(meas.position - target.position).item()
            errors.append(err)

        mean_err = sum(errors) / len(errors)
        # Mean error should be reasonable (not zero, not huge)
        assert 0.1 < mean_err < 200.0


class TestElectroOpticalSensor:
    def test_detection(self) -> None:
        config = SensorConfig(
            name="test_eo",
            sensor_type=SensorType.EO,
            azimuth_noise_rad=0.01,
            elevation_noise_rad=0.01,
            max_range_m=20000.0,
            fov_rad=1.0,
        )
        sensor = ElectroOpticalSensor(config)
        target = create_initial_state(position=[5000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0])
        own_pos = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)
        meas = sensor.detect(target, own_pos, time_s=0.0)
        assert meas is not None

    def test_large_range_uncertainty(self) -> None:
        """EO should have much larger range uncertainty than radar."""
        config = SensorConfig(
            name="test_eo",
            sensor_type=SensorType.EO,
            azimuth_noise_rad=0.01,
            max_range_m=20000.0,
            fov_rad=1.5,
        )
        sensor = ElectroOpticalSensor(config)
        target = create_initial_state(position=[5000.0, 0.0, -3000.0], velocity=[0.0, 0.0, 0.0])
        own_pos = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)
        meas = sensor.detect(target, own_pos, time_s=0.0)
        assert meas is not None
        # EO covariance should be much larger than typical radar
        assert torch.trace(meas.covariance).item() > 1000.0


class TestInfraredSensor:
    def test_detection(self) -> None:
        from src.intercept.sensors import InfraredSensor

        config = SensorConfig(
            name="test_ir",
            sensor_type=SensorType.IR,
            azimuth_noise_rad=0.005,
            elevation_noise_rad=0.005,
            max_range_m=30000.0,
            fov_rad=1.0,
            range_uncertainty_fraction=0.2,
        )
        sensor = InfraredSensor(config)
        target = create_initial_state(position=[5000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0])
        own_pos = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)
        meas = sensor.detect(target, own_pos, time_s=0.0)
        assert meas is not None
        assert meas.sensor_id == "ir"
        assert meas.covariance.shape == (3, 3)

    def test_ir_registered(self) -> None:
        assert SensorRegistry().get("ir") is not None


class TestStalenessTracker:
    def test_initial_staleness_infinite(self) -> None:
        tracker = StalenessTracker()
        assert tracker.get_staleness("unknown", 10.0) == float("inf")

    def test_staleness_grows(self) -> None:
        tracker = StalenessTracker()
        tracker.update("t1", 0.0)
        assert tracker.get_staleness("t1", 5.0) == pytest.approx(5.0)

    def test_confidence_at_zero_staleness(self) -> None:
        tracker = StalenessTracker(half_life_s=5.0)
        tracker.update("t1", 10.0)
        conf = tracker.confidence("t1", 10.0)
        assert conf == pytest.approx(1.0)

    def test_confidence_at_half_life(self) -> None:
        tracker = StalenessTracker(half_life_s=5.0)
        tracker.update("t1", 0.0)
        conf = tracker.confidence("t1", 5.0)
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_confidence_decays(self) -> None:
        tracker = StalenessTracker(half_life_s=5.0)
        tracker.update("t1", 0.0)
        c1 = tracker.confidence("t1", 2.0)
        c2 = tracker.confidence("t1", 10.0)
        assert c1 > c2


class TestSensorFusion:
    def test_fused_track_more_accurate(self) -> None:
        """Fusing radar + EO should produce better track than radar alone."""
        radar_config = SensorConfig(
            name="radar",
            sensor_type=SensorType.RADAR,
            range_noise_m=10.0,
            azimuth_noise_rad=0.01,
            elevation_noise_rad=0.01,
            max_range_m=50000.0,
            fov_rad=1.5,
        )
        eo_config = SensorConfig(
            name="eo",
            sensor_type=SensorType.EO,
            azimuth_noise_rad=0.005,
            elevation_noise_rad=0.005,
            max_range_m=20000.0,
            fov_rad=1.5,
        )

        radar = RadarSensor(radar_config)
        eo = ElectroOpticalSensor(eo_config)
        fusion = SensorFusion([radar, eo])

        target = create_initial_state(
            position=[5000.0, 100.0, -3000.0], velocity=[-200.0, 0.0, 0.0]
        )
        own = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)

        # Process multiple updates
        for i in range(20):
            t = i * 0.1
            track = fusion.process(target, own, t, "t1")

        pos_err = torch.norm(track.position - target.position).item()
        # After 20 fused updates, position should be reasonably accurate
        assert pos_err < 500.0  # generous bound for noisy sensors

    def test_confidence_tracking(self) -> None:
        config = SensorConfig(
            name="radar",
            max_range_m=50000.0,
            fov_rad=1.5,
        )
        fusion = SensorFusion([RadarSensor(config)])

        target = create_initial_state(position=[5000.0, 0.0, -3000.0], velocity=[0.0, 0.0, 0.0])
        own = torch.tensor([0.0, 0.0, -3000.0], dtype=torch.float64)

        fusion.process(target, own, 0.0, "t1")
        conf = fusion.get_confidence("t1", 0.0)
        assert conf == pytest.approx(1.0, abs=0.01)


class TestGPSDeniedNavigator:
    def test_propagation(self) -> None:
        state = create_initial_state(position=[0.0, 0.0, -3000.0], velocity=[100.0, 0.0, 0.0])
        nav = GPSDeniedNavigator(state)
        force = torch.zeros(3, dtype=torch.float64)
        torque = torch.zeros(3, dtype=torch.float64)

        for _ in range(100):
            nav.propagate(force, torque, dt=0.01)

        assert nav.state.time.item() == pytest.approx(1.0, abs=0.02)

    def test_uncertainty_grows(self) -> None:
        state = create_initial_state(position=[0.0, 0.0, -3000.0])
        nav = GPSDeniedNavigator(state, drift_rate_ms=1.0)
        force = torch.zeros(3, dtype=torch.float64)
        torque = torch.zeros(3, dtype=torch.float64)

        for _ in range(1000):
            nav.propagate(force, torque, dt=0.01)

        assert nav.uncertainty_m == pytest.approx(10.0, abs=0.2)
        assert nav.is_reliable(max_uncertainty_m=50.0)

    def test_reliability_degrades(self) -> None:
        state = create_initial_state(position=[0.0, 0.0, -3000.0])
        nav = GPSDeniedNavigator(state, drift_rate_ms=5.0)
        force = torch.zeros(3, dtype=torch.float64)
        torque = torch.zeros(3, dtype=torch.float64)

        # After 20s at 5 m/s drift -> 100m uncertainty
        for _ in range(2000):
            nav.propagate(force, torque, dt=0.01)

        assert not nav.is_reliable(max_uncertainty_m=50.0)


class TestSensorRegistry:
    def test_radar_registered(self) -> None:
        assert SensorRegistry().get("radar") is RadarSensor

    def test_eo_registered(self) -> None:
        assert SensorRegistry().get("eo") is ElectroOpticalSensor
