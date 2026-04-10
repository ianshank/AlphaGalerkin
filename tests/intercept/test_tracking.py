"""Tests for EKF track state estimation."""

from __future__ import annotations

import pytest
import torch

from src.intercept.tracking import (
    ConfidenceEnvelope,
    ExtendedKalmanFilter,
    Measurement,
    create_initial_track,
)


class TestTrackState:
    def test_create_initial_track(self) -> None:
        track = create_initial_track(
            position=[1000.0, 2000.0, -500.0],
            velocity=[100.0, 0.0, -10.0],
            track_id="t1",
        )
        assert track.position.shape == (3,)
        assert track.velocity.shape == (3,)
        assert track.covariance.shape == (6, 6)
        assert track.track_id == "t1"

    def test_state_vector(self) -> None:
        track = create_initial_track(position=[1.0, 2.0, 3.0], velocity=[4.0, 5.0, 6.0])
        sv = track.state_vector
        assert sv.shape == (6,)
        assert torch.allclose(sv[:3], track.position)
        assert torch.allclose(sv[3:], track.velocity)

    def test_uncertainty(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, 0.0], position_noise=10.0)
        pu = track.position_uncertainty
        assert pu.shape == (3,)
        # 3-sigma uncertainty should be 30m
        assert torch.allclose(pu, torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64), atol=0.1)

    def test_clone(self) -> None:
        track = create_initial_track(position=[1.0, 2.0, 3.0])
        clone = track.clone()
        clone.position[0] = 999.0
        assert track.position[0].item() != 999.0


class TestExtendedKalmanFilter:
    def setup_method(self) -> None:
        self.ekf = ExtendedKalmanFilter(process_noise_pos=0.1, process_noise_vel=1.0)

    def test_predict_moves_forward(self) -> None:
        track = create_initial_track(
            position=[0.0, 0.0, -1000.0],
            velocity=[100.0, 0.0, 0.0],
        )
        predicted = self.ekf.predict(track, dt=1.0)
        # Position should advance by velocity * dt
        assert predicted.position[0].item() == pytest.approx(100.0, abs=1.0)
        assert predicted.timestamp == 1.0

    def test_predict_grows_uncertainty(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, 0.0])
        predicted = self.ekf.predict(track, dt=5.0)
        # Uncertainty should grow
        orig_trace = torch.trace(track.covariance).item()
        pred_trace = torch.trace(predicted.covariance).item()
        assert pred_trace > orig_trace

    def test_update_reduces_uncertainty(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, 0.0], position_noise=100.0)
        # High-quality measurement
        meas = Measurement(
            position=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            covariance=torch.eye(3, dtype=torch.float64) * 1.0,
            timestamp=0.0,
        )
        updated = self.ekf.update(track, meas)
        orig_trace = torch.trace(track.covariance).item()
        upd_trace = torch.trace(updated.covariance).item()
        assert upd_trace < orig_trace

    def test_update_moves_toward_measurement(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, 0.0], position_noise=100.0)
        meas = Measurement(
            position=torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64),
            covariance=torch.eye(3, dtype=torch.float64) * 1.0,
        )
        updated = self.ekf.update(track, meas)
        # Updated position should be closer to measurement
        assert updated.position[0].item() > 0.0

    def test_convergence_on_constant_velocity(self) -> None:
        """EKF should converge on a constant-velocity target."""
        true_vel = torch.tensor([50.0, 20.0, -5.0], dtype=torch.float64)
        true_pos = torch.tensor([1000.0, 500.0, -2000.0], dtype=torch.float64)

        track = create_initial_track(
            position=[900.0, 400.0, -1900.0],  # offset initial guess
            velocity=[0.0, 0.0, 0.0],
            position_noise=50.0,
            velocity_noise=20.0,
        )

        dt = 0.1
        for i in range(100):
            t = i * dt
            actual_pos = true_pos + true_vel * t
            noise = torch.randn(3, dtype=torch.float64) * 5.0
            meas = Measurement(
                position=actual_pos + noise,
                covariance=torch.eye(3, dtype=torch.float64) * 25.0,
                timestamp=t,
            )
            track = self.ekf.predict_and_update(track, meas)

        # After convergence, position error should be small
        final_true = true_pos + true_vel * 100 * dt
        pos_err = torch.norm(track.position - final_true).item()
        assert pos_err < 20.0, f"Position error too large: {pos_err}"

        # Velocity should be close to true
        vel_err = torch.norm(track.velocity - true_vel).item()
        assert vel_err < 10.0, f"Velocity error too large: {vel_err}"

    def test_staleness_tracking(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, 0.0])
        predicted = self.ekf.predict(track, dt=5.0)
        assert predicted.staleness == 5.0

        meas = Measurement(
            position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            covariance=torch.eye(3, dtype=torch.float64),
            timestamp=5.0,
        )
        updated = self.ekf.update(predicted, meas)
        assert updated.staleness == 0.0


class TestConfidenceEnvelope:
    def test_cep_grows_with_time(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, -1000.0])
        envelope = ConfidenceEnvelope()

        cep_5 = envelope.cep_at_time(track, 5.0)
        cep_10 = envelope.cep_at_time(track, 10.0)
        assert cep_10 > cep_5

    def test_project_returns_correct_count(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, -1000.0])
        envelope = ConfidenceEnvelope()
        results = envelope.project(track, [1.0, 2.0, 5.0, 10.0])
        assert len(results) == 4

    def test_cep_at_zero_is_initial_uncertainty(self) -> None:
        track = create_initial_track(position=[0.0, 0.0, -1000.0], position_noise=10.0)
        envelope = ConfidenceEnvelope()
        cep_0 = envelope.cep_at_time(track, 0.001)
        # Should be close to initial uncertainty
        assert cep_0 < 20.0
