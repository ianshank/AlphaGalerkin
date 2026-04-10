"""Prediction confidence map generation.

Computes spatially varying confidence scores for fire spread
predictions based on sensor data quality, model uncertainty,
and distance from known fire perimeter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ConfidenceMap:
    """Spatially varying prediction confidence.

    Attributes:
        values: Confidence scores [0, 1] at each grid cell (ny, nx).
        mean_confidence: Domain-averaged confidence.
        min_confidence: Minimum confidence in the domain.
        low_confidence_fraction: Fraction of cells below 0.5 confidence.

    """

    values: NDArray[np.float64]

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.values))

    @property
    def min_confidence(self) -> float:
        return float(np.min(self.values))

    @property
    def low_confidence_fraction(self) -> float:
        return float(np.mean(self.values < 0.5))


class ConfidenceEstimator:
    """Estimates prediction confidence across the simulation domain.

    Combines multiple confidence factors:
    1. Sensor data freshness (from staleness tracker)
    2. Distance from last known observation
    3. Time since last sensor update
    4. Proximity to fire front (higher near perimeter)
    """

    def __init__(
        self,
        sensor_weight: float = 0.4,
        distance_weight: float = 0.3,
        time_weight: float = 0.3,
        decay_distance_m: float = 1000.0,
    ) -> None:
        self.sensor_weight = sensor_weight
        self.distance_weight = distance_weight
        self.time_weight = time_weight
        self.decay_distance_m = decay_distance_m

    def compute(
        self,
        sensor_confidence: float,
        observation_mask: NDArray[np.bool_] | None = None,
        prediction_age_s: float = 0.0,
        max_prediction_age_s: float = 1800.0,
        dx: float = 10.0,
        dy: float = 10.0,
    ) -> ConfidenceMap:
        """Compute confidence map.

        Args:
            sensor_confidence: Overall sensor confidence [0, 1].
            observation_mask: Cells with direct sensor observations (ny, nx).
            prediction_age_s: Seconds since prediction started.
            max_prediction_age_s: Maximum prediction horizon.
            dx: Cell width in meters.
            dy: Cell height in meters.

        Returns:
            ConfidenceMap with spatially varying confidence.

        """
        if observation_mask is None:
            # No observation data — uniform sensor confidence
            ny, nx = 10, 10  # Default
            conf = np.full((ny, nx), sensor_confidence)
            return ConfidenceMap(values=conf)

        ny, nx = observation_mask.shape

        # Factor 1: Sensor confidence (uniform)
        f_sensor = np.full((ny, nx), sensor_confidence)

        # Factor 2: Distance from observed cells
        if np.any(observation_mask):
            f_distance = self._distance_confidence(
                observation_mask,
                dx,
                dy,
                self.decay_distance_m,
            )
        else:
            f_distance = np.full((ny, nx), 0.5)

        # Factor 3: Temporal decay
        time_frac = min(prediction_age_s / max(max_prediction_age_s, 1.0), 1.0)
        f_time = np.full((ny, nx), 1.0 - 0.5 * time_frac)

        # Weighted combination
        confidence = (
            self.sensor_weight * f_sensor
            + self.distance_weight * f_distance
            + self.time_weight * f_time
        )
        confidence = np.clip(confidence, 0.0, 1.0)

        return ConfidenceMap(values=confidence)

    @staticmethod
    def _distance_confidence(
        obs_mask: NDArray[np.bool_],
        dx: float,
        dy: float,
        decay_distance_m: float = 1000.0,
    ) -> NDArray[np.float64]:
        """Confidence decays with distance from observed cells."""
        ny, nx = obs_mask.shape
        # Simple distance transform via iteration
        dist = np.full((ny, nx), float(max(ny, nx)), dtype=np.float64)
        dist[obs_mask] = 0.0

        # Forward pass
        for j in range(ny):
            for i in range(nx):
                if j > 0:
                    dist[j, i] = min(dist[j, i], dist[j - 1, i] + 1)
                if i > 0:
                    dist[j, i] = min(dist[j, i], dist[j, i - 1] + 1)

        # Backward pass
        for j in range(ny - 1, -1, -1):
            for i in range(nx - 1, -1, -1):
                if j < ny - 1:
                    dist[j, i] = min(dist[j, i], dist[j + 1, i] + 1)
                if i < nx - 1:
                    dist[j, i] = min(dist[j, i], dist[j, i + 1] + 1)

        # Convert to meters and then to confidence
        dist_m = dist * max(dx, dy)
        return np.clip(1.0 - dist_m / decay_distance_m, 0.3, 1.0)
