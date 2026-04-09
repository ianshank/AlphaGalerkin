"""Track state estimation via Extended Kalman Filter.

Provides EKF for fusing noisy sensor measurements into smooth
track states (position + velocity) with uncertainty quantification.

The EKF uses a constant-velocity process model by default, with
optional acceleration estimation for maneuvering targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
from torch import Tensor

logger = structlog.get_logger(__name__)


@dataclass
class TrackState:
    """Estimated track state with uncertainty.

    Attributes:
        position: Estimated position in NED (m) (3,).
        velocity: Estimated velocity in NED (m/s) (3,).
        covariance: State covariance matrix (6, 6).
        timestamp: Time of last update (s).
        staleness: Time since last measurement update (s).
        track_id: Unique track identifier.

    """

    position: Tensor
    velocity: Tensor
    covariance: Tensor
    timestamp: float = 0.0
    staleness: float = 0.0
    track_id: str = ""

    @property
    def state_vector(self) -> Tensor:
        """6-state vector [px, py, pz, vx, vy, vz]."""
        return torch.cat([self.position, self.velocity], dim=-1)

    @property
    def position_uncertainty(self) -> Tensor:
        """3-sigma position uncertainty (m) (3,)."""
        return 3.0 * torch.sqrt(torch.diagonal(self.covariance)[:3])

    @property
    def velocity_uncertainty(self) -> Tensor:
        """3-sigma velocity uncertainty (m/s) (3,)."""
        return 3.0 * torch.sqrt(torch.diagonal(self.covariance)[3:6])

    def clone(self) -> TrackState:
        """Deep copy."""
        return TrackState(
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            covariance=self.covariance.clone(),
            timestamp=self.timestamp,
            staleness=self.staleness,
            track_id=self.track_id,
        )


@dataclass
class Measurement:
    """Sensor measurement.

    Attributes:
        position: Measured position in NED (m) (3,).
        covariance: Measurement noise covariance (3, 3).
        timestamp: Measurement time (s).
        sensor_id: Sensor identifier.

    """

    position: Tensor
    covariance: Tensor
    timestamp: float = 0.0
    sensor_id: str = ""


class ExtendedKalmanFilter:
    """Extended Kalman Filter for track state estimation.

    Uses a constant-velocity process model:
        x(k+1) = F * x(k) + w
    where x = [pos, vel] (6-state) and F includes dt-based integration.

    Measurement model is direct position observation:
        z = H * x + v
    where H extracts position from state.
    """

    def __init__(
        self,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize EKF.

        Args:
            process_noise_pos: Position process noise std dev (m).
            process_noise_vel: Velocity process noise std dev (m/s).
                Higher values track maneuvering targets better.
            dtype: Tensor dtype.

        """
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.dtype = dtype

    def _transition_matrix(self, dt: float, device: torch.device) -> Tensor:
        """Constant-velocity state transition matrix F (6x6)."""
        F = torch.eye(6, dtype=self.dtype, device=device)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _process_noise(self, dt: float, device: torch.device) -> Tensor:
        """Process noise covariance Q (6x6).

        Discrete white noise model for constant velocity.
        """
        q_pos = self.process_noise_pos**2
        q_vel = self.process_noise_vel**2
        Q = torch.zeros(6, 6, dtype=self.dtype, device=device)
        # Position noise grows with dt^2 * velocity noise
        for i in range(3):
            Q[i, i] = q_pos + q_vel * dt**2 / 3.0
            Q[i, i + 3] = q_vel * dt / 2.0
            Q[i + 3, i] = q_vel * dt / 2.0
            Q[i + 3, i + 3] = q_vel
        return Q

    def _measurement_matrix(self, device: torch.device) -> Tensor:
        """Measurement matrix H (3x6): extracts position."""
        H = torch.zeros(3, 6, dtype=self.dtype, device=device)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def predict(self, track: TrackState, dt: float) -> TrackState:
        """Predict track state forward by dt seconds.

        Args:
            track: Current track state.
            dt: Time step in seconds.

        Returns:
            Predicted track state with grown uncertainty.

        """
        device = track.position.device
        F = self._transition_matrix(dt, device)
        Q = self._process_noise(dt, device)

        x = track.state_vector
        P = track.covariance

        # Predict state
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        return TrackState(
            position=x_pred[:3],
            velocity=x_pred[3:6],
            covariance=P_pred,
            timestamp=track.timestamp + dt,
            staleness=track.staleness + dt,
            track_id=track.track_id,
        )

    def update(self, track: TrackState, measurement: Measurement) -> TrackState:
        """Update track state with a new measurement.

        Args:
            track: Predicted track state.
            measurement: New position measurement.

        Returns:
            Updated track state with reduced uncertainty.

        """
        device = track.position.device
        H = self._measurement_matrix(device)
        R = measurement.covariance.to(device=device, dtype=self.dtype)

        x = track.state_vector
        P = track.covariance

        # Innovation
        z = measurement.position.to(device=device, dtype=self.dtype)
        y = z - H @ x  # innovation

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman gain
        K = P @ H.T @ torch.linalg.inv(S)

        # Update state
        x_upd = x + K @ y
        I = torch.eye(6, dtype=self.dtype, device=device)
        P_upd = (I - K @ H) @ P

        # Symmetrize covariance
        P_upd = 0.5 * (P_upd + P_upd.T)

        return TrackState(
            position=x_upd[:3],
            velocity=x_upd[3:6],
            covariance=P_upd,
            timestamp=measurement.timestamp,
            staleness=0.0,
            track_id=track.track_id,
        )

    def predict_and_update(self, track: TrackState, measurement: Measurement) -> TrackState:
        """Combined predict + update step.

        Args:
            track: Current track state.
            measurement: New measurement.

        Returns:
            Updated track state.

        """
        dt = measurement.timestamp - track.timestamp
        if dt > 0:
            predicted = self.predict(track, dt)
        else:
            predicted = track
        return self.update(predicted, measurement)


class ConfidenceEnvelope:
    """Projects track uncertainty forward in time.

    Computes the expanding 3-sigma uncertainty ellipsoid for a track
    as prediction time increases without new measurements.
    """

    def __init__(self, ekf: ExtendedKalmanFilter | None = None) -> None:
        self.ekf = ekf or ExtendedKalmanFilter()

    def project(
        self,
        track: TrackState,
        lookahead_times: list[float],
    ) -> list[TrackState]:
        """Project track state and uncertainty forward.

        Args:
            track: Current track state.
            lookahead_times: List of lookahead times in seconds.

        Returns:
            List of predicted track states at each time.

        """
        results = []
        for t in lookahead_times:
            predicted = self.ekf.predict(track, t)
            results.append(predicted)
        return results

    def cep_at_time(self, track: TrackState, lookahead_s: float) -> float:
        """Compute Circular Error Probable at a given lookahead time.

        CEP is the radius of a circle containing 50% of predictions.
        For Gaussian errors: CEP ≈ 0.674 * sqrt(sigma_x^2 + sigma_y^2) / sqrt(2).

        Args:
            track: Current track state.
            lookahead_s: Lookahead time in seconds.

        Returns:
            CEP in meters.

        """
        predicted = self.ekf.predict(track, lookahead_s)
        pos_var = torch.diagonal(predicted.covariance)[:3]
        # CEP from horizontal position variances (N and E)
        horizontal_var = pos_var[0] + pos_var[1]
        cep = 0.674 * torch.sqrt(horizontal_var).item()
        return cep


def create_initial_track(
    position: list[float],
    velocity: list[float] | None = None,
    position_noise: float = 10.0,
    velocity_noise: float = 5.0,
    track_id: str = "track_0",
    dtype: torch.dtype = torch.float64,
) -> TrackState:
    """Create an initial track state from first detection.

    Args:
        position: Initial position in NED (m).
        velocity: Initial velocity estimate (m/s), defaults to zero.
        position_noise: Initial position uncertainty (m).
        velocity_noise: Initial velocity uncertainty (m/s).
        track_id: Track identifier.
        dtype: Tensor dtype.

    Returns:
        Initial track state with covariance.

    """
    pos = torch.tensor(position, dtype=dtype)
    vel = torch.tensor(velocity or [0.0, 0.0, 0.0], dtype=dtype)

    P = torch.zeros(6, 6, dtype=dtype)
    for i in range(3):
        P[i, i] = position_noise**2
        P[i + 3, i + 3] = velocity_noise**2

    return TrackState(
        position=pos,
        velocity=vel,
        covariance=P,
        track_id=track_id,
    )
