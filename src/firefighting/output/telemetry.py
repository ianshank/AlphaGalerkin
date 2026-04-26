"""Structured telemetry logging for firefighting predictions.

Emits structured log events for real-time monitoring by
incident command and post-flight analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import structlog

logger = structlog.get_logger("firefighting.telemetry")


@dataclass
class PredictionTelemetry:
    """Telemetry snapshot for a single prediction cycle."""

    timestamp: float
    prediction_step: int
    burned_area_m2: float
    max_temperature_K: float  # noqa: N815
    perimeter_length_m: float
    mean_confidence: float
    sensor_staleness_s: float
    inference_latency_ms: float
    memory_usage_mb: float
    wind_speed_m_s: float
    wind_direction_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryLogger:
    """Emits structured telemetry events via structlog.

    Logs are queryable by incident management systems and
    can be streamed to ground station dashboards.
    """

    def __init__(self, mission_id: str = "default") -> None:
        self.mission_id = mission_id
        self._cycle_count = 0

    def log_prediction(self, telemetry: PredictionTelemetry) -> None:
        """Log a prediction cycle telemetry snapshot."""
        self._cycle_count += 1
        logger.info(
            "prediction_cycle",
            mission_id=self.mission_id,
            cycle=self._cycle_count,
            **telemetry.to_dict(),
        )

    def log_warning(self, event: str, **kwargs: object) -> None:
        """Log a warning event (e.g., sensor dropout, high latency)."""
        logger.warning(
            event,
            mission_id=self.mission_id,
            cycle=self._cycle_count,
            **kwargs,
        )

    def log_sensor_dropout(self, sensor_type: str, staleness_s: float) -> None:
        """Log sensor data dropout."""
        self.log_warning(
            "sensor_dropout",
            sensor_type=sensor_type,
            staleness_s=staleness_s,
        )

    def log_latency_exceeded(self, actual_ms: float, budget_ms: float) -> None:
        """Log latency budget exceeded."""
        self.log_warning(
            "latency_exceeded",
            actual_ms=actual_ms,
            budget_ms=budget_ms,
            overshoot_ms=actual_ms - budget_ms,
        )
