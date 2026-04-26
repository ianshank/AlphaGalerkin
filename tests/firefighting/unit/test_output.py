"""Tests for output modules: perimeter export, confidence, telemetry."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.firefighting.output.confidence import ConfidenceEstimator, ConfidenceMap
from src.firefighting.output.perimeter import PerimeterExporter
from src.firefighting.output.telemetry import PredictionTelemetry, TelemetryLogger


class TestPerimeterExporter:
    def test_export_empty_fire(self) -> None:
        exporter = PerimeterExporter()
        mask = np.zeros((10, 10), dtype=bool)
        result = exporter.export(mask, dx=10.0, dy=10.0, timestamp=100.0)
        assert result.burned_area_m2 == 0.0
        assert result.geojson["geometry"]["coordinates"] == []

    def test_export_small_fire(self) -> None:
        exporter = PerimeterExporter()
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:7, 4:7] = True  # 3x3 fire
        result = exporter.export(mask, dx=10.0, dy=10.0, timestamp=200.0)
        assert result.burned_area_m2 == pytest.approx(900.0)
        assert len(result.geojson["geometry"]["coordinates"]) > 0

    def test_geojson_valid(self) -> None:
        exporter = PerimeterExporter()
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        result = exporter.export(mask, dx=10.0, dy=10.0, timestamp=0.0)
        json_str = exporter.to_json_string(result)
        parsed = json.loads(json_str)
        assert parsed["type"] == "Feature"
        assert parsed["geometry"]["type"] == "MultiPoint"


class TestConfidenceEstimator:
    def test_full_confidence_with_fresh_data(self) -> None:
        estimator = ConfidenceEstimator()
        obs_mask = np.ones((10, 10), dtype=bool)
        conf = estimator.compute(
            sensor_confidence=1.0,
            observation_mask=obs_mask,
            prediction_age_s=0.0,
        )
        assert conf.mean_confidence > 0.9

    def test_low_confidence_far_from_observations(self) -> None:
        estimator = ConfidenceEstimator()
        obs_mask = np.zeros((20, 20), dtype=bool)
        obs_mask[10, 10] = True  # Single observation point
        conf = estimator.compute(
            sensor_confidence=1.0,
            observation_mask=obs_mask,
        )
        # Cells far from observation should have lower confidence
        assert conf.values[0, 0] < conf.values[10, 10]

    def test_confidence_map_properties(self) -> None:
        values = np.random.rand(10, 10)
        conf = ConfidenceMap(values=values)
        assert 0 <= conf.mean_confidence <= 1
        assert conf.min_confidence >= 0
        assert 0 <= conf.low_confidence_fraction <= 1


class TestTelemetryLogger:
    def test_log_prediction(self) -> None:
        logger_obj = TelemetryLogger(mission_id="test_mission")
        telemetry = PredictionTelemetry(
            timestamp=100.0,
            prediction_step=1,
            burned_area_m2=5000.0,
            max_temperature_K=1200.0,
            perimeter_length_m=500.0,
            mean_confidence=0.85,
            sensor_staleness_s=2.0,
            inference_latency_ms=350.0,
            memory_usage_mb=2048.0,
            wind_speed_m_s=5.0,
            wind_direction_deg=270.0,
        )
        # Should not raise
        logger_obj.log_prediction(telemetry)

    def test_telemetry_to_dict(self) -> None:
        t = PredictionTelemetry(
            timestamp=0,
            prediction_step=0,
            burned_area_m2=0,
            max_temperature_K=0,
            perimeter_length_m=0,
            mean_confidence=0,
            sensor_staleness_s=0,
            inference_latency_ms=0,
            memory_usage_mb=0,
            wind_speed_m_s=0,
            wind_direction_deg=0,
        )
        d = t.to_dict()
        assert "timestamp" in d
        assert "burned_area_m2" in d
