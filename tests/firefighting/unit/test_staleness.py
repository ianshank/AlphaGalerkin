"""Tests for sensor staleness tracking."""

from __future__ import annotations

import pytest

from src.firefighting.config.sensor import SensorConfig
from src.firefighting.sensor.staleness import StalenessTracker


@pytest.fixture
def sensor_config() -> SensorConfig:
    return SensorConfig(
        name="test",
        stale_threshold_s=5.0,
        confidence_decay_rate=0.5,
        min_confidence=0.1,
    )


class TestStalenessTracker:
    def test_fresh_data_full_confidence(self, sensor_config: SensorConfig) -> None:
        tracker = StalenessTracker(sensor_config)
        tracker.update(timestamp=100.0)
        assert tracker.confidence(current_time=101.0) == 1.0

    def test_stale_data_decays(self, sensor_config: SensorConfig) -> None:
        tracker = StalenessTracker(sensor_config)
        tracker.update(timestamp=100.0)
        # 10s past stale threshold
        conf = tracker.confidence(current_time=115.0)
        assert conf < 1.0
        assert conf >= sensor_config.min_confidence

    def test_very_stale_hits_floor(self, sensor_config: SensorConfig) -> None:
        tracker = StalenessTracker(sensor_config)
        tracker.update(timestamp=0.0)
        conf = tracker.confidence(current_time=1000.0)
        assert conf == sensor_config.min_confidence

    def test_uninitialized_returns_minimum(self, sensor_config: SensorConfig) -> None:
        tracker = StalenessTracker(sensor_config)
        assert tracker.confidence(current_time=100.0) == sensor_config.min_confidence

    def test_is_stale(self, sensor_config: SensorConfig) -> None:
        tracker = StalenessTracker(sensor_config)
        tracker.update(timestamp=100.0)
        assert not tracker.is_stale(current_time=103.0)  # 3s < 5s threshold
        assert tracker.is_stale(current_time=106.0)  # 6s > 5s threshold
