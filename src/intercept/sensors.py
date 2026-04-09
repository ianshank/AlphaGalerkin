"""Sensor models and multi-sensor fusion.

Provides sensor simulation (radar, EO, IR) with configurable noise,
multi-sensor fusion via sequential EKF updates, staleness tracking
with exponential confidence decay, and GPS-denied dead reckoning.

All sensors registered via create_registry for extensibility.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import structlog
import torch
from torch import Tensor

from src.intercept.config import SensorConfig
from src.intercept.dynamics import RigidBody6DOF, RigidBodyState
from src.intercept.frames import FrameTransform
from src.intercept.tracking import (
    ExtendedKalmanFilter,
    Measurement,
    TrackState,
    create_initial_track,
)
from src.templates.registry import create_registry

logger = structlog.get_logger(__name__)


class BaseSensor(ABC):
    """Abstract base class for sensor models."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config

    @abstractmethod
    def detect(
        self,
        target_state: RigidBodyState,
        own_position: Tensor,
        time_s: float,
    ) -> Measurement | None:
        """Attempt to detect a target.

        Args:
            target_state: True target state.
            own_position: Sensor position in NED (3,).
            time_s: Current time in seconds.

        Returns:
            Noisy measurement or None if target not detectable.

        """
        ...

    def in_fov(self, target_position: Tensor, own_position: Tensor) -> bool:
        """Check if target is within sensor field of view.

        Args:
            target_position: Target position in NED (3,).
            own_position: Sensor position in NED (3,).

        Returns:
            True if target is detectable.

        """
        rel = target_position - own_position
        range_m = torch.norm(rel).item()
        if range_m > self.config.max_range_m:
            return False

        _, az, el = FrameTransform.range_and_bearing(rel.unsqueeze(0))
        # Simple FOV check: elevation within fov_rad
        if abs(el.item()) > self.config.fov_rad:
            return False
        return True


SensorRegistry, register_sensor = create_registry("Sensor", BaseSensor)


@register_sensor("radar")
class RadarSensor(BaseSensor):
    """Radar sensor with range + angular measurements.

    Produces position measurements in Cartesian NED with noise
    derived from range/angle measurement uncertainties.
    """

    def detect(
        self,
        target_state: RigidBodyState,
        own_position: Tensor,
        time_s: float,
    ) -> Measurement | None:
        if not self.in_fov(target_state.position, own_position):
            return None

        rel = target_state.position - own_position
        range_m = torch.norm(rel).item()

        # Add noise to spherical coordinates
        range_noise = torch.randn(1, dtype=torch.float64).item() * self.config.range_noise_m
        az_noise = torch.randn(1, dtype=torch.float64).item() * self.config.azimuth_noise_rad
        el_noise = torch.randn(1, dtype=torch.float64).item() * self.config.elevation_noise_rad

        true_range, true_az, true_el = FrameTransform.range_and_bearing(rel.unsqueeze(0))
        noisy_range = true_range.item() + range_noise
        noisy_az = true_az.item() + az_noise
        noisy_el = true_el.item() + el_noise

        # Convert back to Cartesian
        horiz = noisy_range * math.cos(noisy_el)
        meas_n = horiz * math.cos(noisy_az)
        meas_e = horiz * math.sin(noisy_az)
        meas_d = -noisy_range * math.sin(noisy_el)

        meas_pos = own_position + torch.tensor([meas_n, meas_e, meas_d], dtype=torch.float64)

        # Build covariance from measurement noise
        r_var = self.config.range_noise_m**2
        cross_range_var = (range_m * self.config.azimuth_noise_rad) ** 2
        cov = torch.diag(
            torch.tensor(
                [r_var + cross_range_var, r_var + cross_range_var, r_var],
                dtype=torch.float64,
            )
        )

        return Measurement(
            position=meas_pos,
            covariance=cov,
            timestamp=time_s,
            sensor_id="radar",
        )


@register_sensor("eo")
class ElectroOpticalSensor(BaseSensor):
    """Electro-optical sensor (bearing-only).

    Produces angular measurements only. Range is inferred from
    track fusion with other sensors. Position measurement has
    very large range uncertainty.
    """

    def detect(
        self,
        target_state: RigidBodyState,
        own_position: Tensor,
        time_s: float,
    ) -> Measurement | None:
        if not self.in_fov(target_state.position, own_position):
            return None

        rel = target_state.position - own_position
        range_m = torch.norm(rel).item()

        # Bearing-only: add angular noise, use true range with large uncertainty
        az_noise = torch.randn(1, dtype=torch.float64).item() * self.config.azimuth_noise_rad * 2
        el_noise = torch.randn(1, dtype=torch.float64).item() * self.config.elevation_noise_rad * 2

        true_range, true_az, true_el = FrameTransform.range_and_bearing(rel.unsqueeze(0))
        noisy_az = true_az.item() + az_noise
        noisy_el = true_el.item() + el_noise

        horiz = range_m * math.cos(noisy_el)
        meas_n = horiz * math.cos(noisy_az)
        meas_e = horiz * math.sin(noisy_az)
        meas_d = -range_m * math.sin(noisy_el)

        meas_pos = own_position + torch.tensor([meas_n, meas_e, meas_d], dtype=torch.float64)

        # Large range uncertainty for bearing-only
        range_var = (range_m * 0.3) ** 2  # 30% range uncertainty
        cross_var = (range_m * self.config.azimuth_noise_rad * 2) ** 2
        cov = torch.diag(
            torch.tensor(
                [range_var + cross_var, range_var + cross_var, range_var], dtype=torch.float64
            )
        )

        return Measurement(
            position=meas_pos,
            covariance=cov,
            timestamp=time_s,
            sensor_id="eo",
        )


@register_sensor("ir")
class InfraredSensor(BaseSensor):
    """Infrared sensor (bearing-only, better for hot targets).

    Similar to EO but with tighter angular noise for targets
    with thermal signatures (motor plume, hot surfaces).
    """

    def detect(
        self,
        target_state: RigidBodyState,
        own_position: Tensor,
        time_s: float,
    ) -> Measurement | None:
        if not self.in_fov(target_state.position, own_position):
            return None

        rel = target_state.position - own_position
        range_m = torch.norm(rel).item()

        # Tighter angular noise than EO
        az_noise = torch.randn(1, dtype=torch.float64).item() * self.config.azimuth_noise_rad
        el_noise = torch.randn(1, dtype=torch.float64).item() * self.config.elevation_noise_rad

        true_range, true_az, true_el = FrameTransform.range_and_bearing(rel.unsqueeze(0))
        noisy_az = true_az.item() + az_noise
        noisy_el = true_el.item() + el_noise

        horiz = range_m * math.cos(noisy_el)
        meas_n = horiz * math.cos(noisy_az)
        meas_e = horiz * math.sin(noisy_az)
        meas_d = -range_m * math.sin(noisy_el)

        meas_pos = own_position + torch.tensor([meas_n, meas_e, meas_d], dtype=torch.float64)

        range_var = (range_m * 0.2) ** 2
        cross_var = (range_m * self.config.azimuth_noise_rad) ** 2
        cov = torch.diag(
            torch.tensor(
                [range_var + cross_var, range_var + cross_var, range_var], dtype=torch.float64
            )
        )

        return Measurement(
            position=meas_pos,
            covariance=cov,
            timestamp=time_s,
            sensor_id="ir",
        )


class StalenessTracker:
    """Tracks data freshness per track with confidence decay."""

    def __init__(self, half_life_s: float = 5.0) -> None:
        self.half_life_s = half_life_s
        self._last_update: dict[str, float] = {}

    def update(self, track_id: str, timestamp: float) -> None:
        """Record a measurement update for a track."""
        self._last_update[track_id] = timestamp

    def get_staleness(self, track_id: str, current_time: float) -> float:
        """Get seconds since last update."""
        if track_id not in self._last_update:
            return float("inf")
        return current_time - self._last_update[track_id]

    def confidence(self, track_id: str, current_time: float) -> float:
        """Compute confidence as exponential decay of staleness.

        Returns:
            Confidence in [0, 1]. 0.5 at half_life_s.

        """
        staleness = self.get_staleness(track_id, current_time)
        if staleness == float("inf"):
            return 0.0
        decay_rate = math.log(2) / self.half_life_s
        return math.exp(-decay_rate * staleness)


class SensorFusion:
    """Fuses measurements from multiple sensors via sequential EKF updates."""

    def __init__(
        self,
        sensors: list[BaseSensor],
        ekf: ExtendedKalmanFilter | None = None,
    ) -> None:
        self.sensors = sensors
        self.ekf = ekf or ExtendedKalmanFilter()
        self.staleness_tracker = StalenessTracker()
        self._tracks: dict[str, TrackState] = {}

    def process(
        self,
        target_state: RigidBodyState,
        own_position: Tensor,
        time_s: float,
        track_id: str = "track_0",
    ) -> TrackState:
        """Process all sensors for a target and fuse measurements.

        Args:
            target_state: True target state (for sensor simulation).
            own_position: Sensor platform position.
            time_s: Current time.
            track_id: Track identifier.

        Returns:
            Updated fused track state.

        """
        # Initialize track if needed
        if track_id not in self._tracks:
            self._tracks[track_id] = create_initial_track(
                position=target_state.position.detach().cpu().tolist(),
                velocity=target_state.velocity.detach().cpu().tolist(),
                position_noise=100.0,
                velocity_noise=50.0,
                track_id=track_id,
            )

        track = self._tracks[track_id]

        # Predict forward
        dt = time_s - track.timestamp
        if dt > 0:
            track = self.ekf.predict(track, dt)

        # Collect and fuse measurements from all sensors
        updated = False
        for sensor in self.sensors:
            meas = sensor.detect(target_state, own_position, time_s)
            if meas is not None:
                track = self.ekf.update(track, meas)
                updated = True

        if updated:
            self.staleness_tracker.update(track_id, time_s)

        self._tracks[track_id] = track
        return track

    def get_confidence(self, track_id: str, current_time: float) -> float:
        """Get confidence for a track."""
        return self.staleness_tracker.confidence(track_id, current_time)


class GPSDeniedNavigator:
    """IMU-only dead reckoning for GPS-denied operation.

    Propagates state using 6-DOF dynamics with growing uncertainty
    when GPS is unavailable. Suitable for terminal phase guidance.
    """

    def __init__(
        self,
        initial_state: RigidBodyState,
        dynamics: RigidBody6DOF | None = None,
        drift_rate_ms: float = 0.5,
    ) -> None:
        self.state = initial_state.clone()
        self.dynamics = dynamics or RigidBody6DOF()
        self.drift_rate = drift_rate_ms
        self._elapsed_s = 0.0

    def propagate(
        self,
        force_ned: Tensor,
        torque_body: Tensor,
        dt: float,
    ) -> RigidBodyState:
        """Propagate state via dead reckoning.

        Args:
            force_ned: Estimated force from IMU.
            torque_body: Estimated torque from IMU.
            dt: Time step.

        Returns:
            Propagated state (with growing position error).

        """
        self.state = self.dynamics.step(self.state, force_ned, torque_body, dt)
        self._elapsed_s += dt
        return self.state

    @property
    def uncertainty_m(self) -> float:
        """Estimated position uncertainty in meters.

        Grows linearly with time at drift_rate m/s.
        """
        return self.drift_rate * self._elapsed_s

    def is_reliable(self, max_uncertainty_m: float = 50.0) -> bool:
        """Check if dead reckoning is still within acceptable uncertainty."""
        return self.uncertainty_m < max_uncertainty_m
