"""Sensor data staleness tracking and confidence decay.

Monitors freshness of sensor data and computes confidence
scores that decay exponentially when data becomes stale.
This enables graceful degradation when sensors drop out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from src.firefighting.config.sensor import SensorConfig


@runtime_checkable
class BoundaryConditionSource(Protocol):
    """Sensor-driven boundary condition provider."""

    def staleness_seconds(self) -> float:
        """Seconds since last valid update."""
        ...

    def confidence(self) -> float:
        """Current confidence score [0, 1]."""
        ...


@dataclass
class SensorReading:
    """A timestamped sensor measurement."""

    timestamp: float  # Unix epoch seconds
    data: np.ndarray  # Sensor-specific data
    source: str  # Sensor identifier


@dataclass
class StalenessTracker:
    """Tracks data freshness for a single sensor channel.

    Confidence = 1.0 when data is fresh (age < stale_threshold).
    Confidence decays exponentially once data goes stale:
        confidence = max(min_confidence, exp(-decay_rate * (age - threshold)))
    """

    config: SensorConfig
    _last_update: float = field(default_factory=time.time)
    _is_initialized: bool = False

    def update(self, timestamp: float | None = None) -> None:
        """Record a new sensor reading.

        Args:
            timestamp: Reading timestamp. Uses current time if None.

        """
        self._last_update = timestamp if timestamp is not None else time.time()
        self._is_initialized = True

    def staleness_seconds(self, current_time: float | None = None) -> float:
        """Time since last update in seconds."""
        if not self._is_initialized:
            return float("inf")
        now = current_time if current_time is not None else time.time()
        return max(0.0, now - self._last_update)

    def confidence(self, current_time: float | None = None) -> float:
        """Current confidence score [min_confidence, 1.0].

        Returns 1.0 if data is fresh, decays exponentially after stale threshold.
        """
        if not self._is_initialized:
            return self.config.min_confidence

        age = self.staleness_seconds(current_time)
        if age <= self.config.stale_threshold_s:
            return 1.0

        excess = age - self.config.stale_threshold_s
        decay = np.exp(-self.config.confidence_decay_rate * excess)
        return max(self.config.min_confidence, float(decay))

    def is_stale(self, current_time: float | None = None) -> bool:
        """Whether data has exceeded the stale threshold."""
        return self.staleness_seconds(current_time) > self.config.stale_threshold_s
