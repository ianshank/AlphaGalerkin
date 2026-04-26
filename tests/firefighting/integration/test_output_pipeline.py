"""Integration test: fire solver → output pipeline.

Verifies the full output chain: solver result → perimeter export →
confidence estimation → telemetry logging.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.firefighting.config.fire import FireConfig
from src.firefighting.config.sensor import SensorConfig
from src.firefighting.config.solver import FireSolverConfig
from src.firefighting.output.confidence import ConfidenceEstimator
from src.firefighting.output.perimeter import PerimeterExporter
from src.firefighting.output.telemetry import PredictionTelemetry, TelemetryLogger
from src.firefighting.sensor.staleness import StalenessTracker
from src.firefighting.sensor.thermal import ThermalCameraDecoder
from src.firefighting.solver.coupled import CoupledFireSolver


class TestOutputPipeline:
    """Tests the full output pipeline from solver result to exports."""

    @pytest.fixture
    def fire_result(self):
        config = FireSolverConfig(
            name="test",
            nx=20,
            ny=20,
            domain_size_x_m=200.0,
            domain_size_y_m=200.0,
            dt_s=0.5,
            prediction_horizon_s=10.0,
            max_steps=30,
        )
        fire_config = FireConfig(name="test")
        solver = CoupledFireSolver(config, fire_config)
        state = solver.create_initial_state(
            ignition_center=(100.0, 100.0),
            ignition_radius_m=15.0,
        )
        wind_u = np.full((20, 20), 2.0)
        wind_v = np.zeros((20, 20))
        return solver.run(state, wind_u, wind_v, t_final=10.0)

    def test_perimeter_export(self, fire_result) -> None:
        exporter = PerimeterExporter()
        burned = fire_result.final_state.temperature > 400.0
        result = exporter.export(burned, dx=10.0, dy=10.0, timestamp=10.0)
        assert result.geojson["type"] == "Feature"
        json_str = exporter.to_json_string(result)
        assert "fire_perimeter" in json_str

    def test_confidence_with_observation(self, fire_result) -> None:
        estimator = ConfidenceEstimator()
        # Simulate partial observations
        obs = np.zeros((20, 20), dtype=bool)
        obs[8:12, 8:12] = True
        conf = estimator.compute(sensor_confidence=0.85, observation_mask=obs)
        # Near observations = high confidence
        assert conf.values[10, 10] > conf.values[0, 0]

    def test_telemetry_from_result(self, fire_result) -> None:
        telem = PredictionTelemetry(
            timestamp=10.0,
            prediction_step=20,
            burned_area_m2=fire_result.burned_area_m2,
            max_temperature_K=fire_result.max_temperature_K,
            perimeter_length_m=100.0,
            mean_confidence=0.8,
            sensor_staleness_s=1.0,
            inference_latency_ms=350.0,
            memory_usage_mb=2048.0,
            wind_speed_m_s=2.0,
            wind_direction_deg=270.0,
        )
        logger_obj = TelemetryLogger(mission_id="pipeline_test")
        logger_obj.log_prediction(telem)  # Should not raise
        assert telem.burned_area_m2 == fire_result.burned_area_m2


class TestSensorIntegration:
    """Tests sensor data feeding into the prediction pipeline."""

    def test_thermal_to_staleness(self) -> None:
        config = SensorConfig(name="test", thermal_fov_deg=60.0)
        decoder = ThermalCameraDecoder(config)

        # Decode thermal frame
        raw = np.full((32, 32), 320.0)
        raw[14:18, 14:18] = 900.0
        frame = decoder.decode_frame(raw, timestamp=50.0)

        # Staleness tracking
        tracker = StalenessTracker(config)
        tracker.update(timestamp=50.0)
        assert tracker.confidence(current_time=50.5) > 0.95
        assert tracker.confidence(current_time=80.0) < tracker.confidence(current_time=50.5)
