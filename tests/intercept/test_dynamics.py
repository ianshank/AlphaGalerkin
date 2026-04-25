"""Tests for 6-DOF rigid body dynamics.

Key validation: zero-drag ballistic trajectory matches analytical parabola.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.intercept.config import DynamicsConfig, GravityModel, IntegrationMethod
from src.intercept.dynamics import (
    RigidBody6DOF,
    RigidBodyState,
    create_initial_state,
    gravity_ned,
)

G0 = 9.80665


class TestRigidBodyState:
    def test_create_initial_state(self) -> None:
        state = create_initial_state(
            position=[100.0, 200.0, -1000.0],
            velocity=[50.0, 0.0, 0.0],
            mass=100.0,
        )
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.quaternion.shape == (4,)
        assert state.angular_velocity.shape == (3,)
        assert state.mass.item() == 100.0
        assert state.time.item() == 0.0

    def test_to_tensor(self) -> None:
        state = create_initial_state()
        t = state.to_tensor()
        assert t.shape == (13,)

    def test_from_tensor_roundtrip(self) -> None:
        state = create_initial_state(
            position=[1.0, 2.0, 3.0],
            velocity=[4.0, 5.0, 6.0],
        )
        t = state.to_tensor()
        state2 = RigidBodyState.from_tensor(t, state.mass, state.time)
        assert torch.allclose(state.position, state2.position)
        assert torch.allclose(state.velocity, state2.velocity)
        assert torch.allclose(state.quaternion, state2.quaternion)

    def test_clone(self) -> None:
        state = create_initial_state(position=[1.0, 2.0, 3.0])
        clone = state.clone()
        clone.position[0] = 999.0
        assert state.position[0] != 999.0

    def test_speed(self) -> None:
        state = create_initial_state(velocity=[3.0, 4.0, 0.0])
        assert torch.allclose(state.speed, torch.tensor(5.0, dtype=torch.float64))

    def test_altitude(self) -> None:
        state = create_initial_state(position=[0.0, 0.0, -1000.0])
        assert torch.allclose(state.altitude, torch.tensor(1000.0, dtype=torch.float64))

    def test_euler_angles(self) -> None:
        state = create_initial_state(euler_deg=[10.0, 20.0, 30.0])
        # Should create non-identity quaternion
        assert not torch.allclose(
            state.quaternion,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )


class TestGravity:
    def test_constant_gravity(self) -> None:
        alt = torch.tensor(0.0, dtype=torch.float64)
        g = gravity_ned(alt, GravityModel.CONSTANT)
        assert g.shape == (3,)
        assert torch.allclose(g[2], torch.tensor(G0, dtype=torch.float64))
        assert torch.allclose(g[0], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(g[1], torch.tensor(0.0, dtype=torch.float64))

    def test_wgs84_decreases_with_altitude(self) -> None:
        g_sea = gravity_ned(torch.tensor(0.0, dtype=torch.float64), GravityModel.WGS84)
        g_high = gravity_ned(torch.tensor(10000.0, dtype=torch.float64), GravityModel.WGS84)
        assert g_high[2] < g_sea[2]

    def test_batched_gravity(self) -> None:
        alts = torch.tensor([0.0, 1000.0, 5000.0], dtype=torch.float64)
        g = gravity_ned(alts, GravityModel.CONSTANT)
        assert g.shape == (3, 3)


class TestRigidBody6DOF:
    def test_free_fall(self) -> None:
        """Object dropped from 1000m should hit ground at t = sqrt(2h/g)."""
        config = DynamicsConfig(
            name="test",
            integration_method=IntegrationMethod.RK4,
            dt=0.001,
        )
        dynamics = RigidBody6DOF(config)
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],  # 1000m altitude
            velocity=[0.0, 0.0, 0.0],
            mass=1.0,
        )
        zero_force = torch.zeros(3, dtype=torch.float64)
        zero_torque = torch.zeros(3, dtype=torch.float64)

        # Analytical: t = sqrt(2 * 1000 / 9.80665) ~ 14.28s
        t_analytical = math.sqrt(2.0 * 1000.0 / G0)
        n_steps = int(t_analytical / config.dt)

        for _ in range(n_steps):
            state = dynamics.step(state, zero_force, zero_torque)

        # Should be near ground level (altitude ~ 0)
        assert abs(state.altitude.item()) < 1.0  # within 1m of ground

    def test_ballistic_trajectory_analytical(self) -> None:
        """Zero-drag ballistic trajectory must match analytical parabola.

        This is the Phase 1 KEY VALIDATION.
        Launch at 45 degrees, v0 = 100 m/s. After t seconds:
          x(t) = v0 * cos(45) * t
          z(t) = z0 + v0 * sin(45) * t - 0.5 * g * t^2
        """
        config = DynamicsConfig(
            name="test",
            integration_method=IntegrationMethod.RK4,
            dt=0.01,
        )
        dynamics = RigidBody6DOF(config)

        v0 = 100.0
        angle = math.pi / 4  # 45 degrees
        vn = v0 * math.cos(angle)
        vd = -v0 * math.sin(angle)  # NED: up is negative d

        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],  # 1000m altitude
            velocity=[vn, 0.0, vd],
            mass=1.0,
        )

        zero_force = torch.zeros(3, dtype=torch.float64)
        zero_torque = torch.zeros(3, dtype=torch.float64)

        # Check at t = 5s and t = 10s
        for target_time in [5.0, 10.0]:
            test_state = state.clone()
            n_steps = int(target_time / config.dt)

            for _ in range(n_steps):
                test_state = dynamics.step(test_state, zero_force, zero_torque)

            t = target_time
            # Analytical position (NED)
            analytical_n = vn * t
            analytical_d = -1000.0 + vd * t + 0.5 * G0 * t**2  # gravity adds to d

            # Check position error < 1e-4 m (should be ~1e-6 with RK4)
            err_n = abs(test_state.position[0].item() - analytical_n)
            err_d = abs(test_state.position[2].item() - analytical_d)

            assert err_n < 1e-4, f"North error at t={t}: {err_n}"
            assert err_d < 1e-4, f"Down error at t={t}: {err_d}"

    def test_quaternion_norm_preserved(self) -> None:
        """Quaternion norm should remain 1.0 over 1000 integration steps."""
        config = DynamicsConfig(name="test", dt=0.01)
        dynamics = RigidBody6DOF(config)

        state = create_initial_state(
            euler_deg=[10.0, 20.0, 30.0],
            velocity=[100.0, 0.0, -10.0],
        )

        zero_force = torch.zeros(3, dtype=torch.float64)
        torque = torch.tensor([0.1, -0.05, 0.02], dtype=torch.float64)

        for _ in range(1000):
            state = dynamics.step(state, zero_force, torque)
            quat_norm = torch.norm(state.quaternion)
            assert torch.allclose(
                quat_norm,
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
            ), f"Quaternion norm drifted to {quat_norm.item()}"

    def test_constant_velocity_no_force(self) -> None:
        """With zero force and zero gravity, velocity should be constant."""
        config = DynamicsConfig(
            name="test",
            dt=0.01,
            g0=0.0,  # disable gravity for this test
        )
        dynamics = RigidBody6DOF(config)
        state = create_initial_state(velocity=[100.0, 50.0, -20.0])

        zero_force = torch.zeros(3, dtype=torch.float64)
        zero_torque = torch.zeros(3, dtype=torch.float64)

        for _ in range(100):
            state = dynamics.step(state, zero_force, zero_torque)

        assert torch.allclose(
            state.velocity,
            torch.tensor([100.0, 50.0, -20.0], dtype=torch.float64),
            atol=1e-8,
        )

    def test_euler_integration(self) -> None:
        """Euler integrator should work but be less accurate than RK4."""
        config = DynamicsConfig(
            name="test",
            integration_method=IntegrationMethod.EULER,
            dt=0.01,
        )
        dynamics = RigidBody6DOF(config)
        state = create_initial_state(velocity=[100.0, 0.0, 0.0])
        zero_force = torch.zeros(3, dtype=torch.float64)
        zero_torque = torch.zeros(3, dtype=torch.float64)

        state = dynamics.step(state, zero_force, zero_torque)
        assert state.time.item() == pytest.approx(0.01)

    def test_propagate(self) -> None:
        """Propagate should return trajectory list."""
        dynamics = RigidBody6DOF()
        state = create_initial_state(velocity=[100.0, 0.0, 0.0])
        zero_force = torch.zeros(3, dtype=torch.float64)
        zero_torque = torch.zeros(3, dtype=torch.float64)

        trajectory = dynamics.propagate(state, zero_force, zero_torque, duration=1.0)
        assert len(trajectory) > 1
        assert trajectory[0].time.item() == 0.0
        assert trajectory[-1].time.item() == pytest.approx(1.0, abs=0.02)

    def test_applied_force(self) -> None:
        """Applied force should cause acceleration in force direction."""
        config = DynamicsConfig(name="test", dt=0.01, g0=0.0)
        dynamics = RigidBody6DOF(config)
        state = create_initial_state(mass=10.0)

        force = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64)  # 100N North
        zero_torque = torch.zeros(3, dtype=torch.float64)

        state = dynamics.step(state, force, zero_torque)
        # a = F/m = 10 m/s^2, v = a*dt = 0.1 m/s
        assert state.velocity[0].item() > 0.0
        assert torch.allclose(
            state.velocity[0],
            torch.tensor(0.1, dtype=torch.float64),
            atol=1e-6,
        )

    def test_torque_causes_rotation(self) -> None:
        """Applied torque should cause angular acceleration."""
        config = DynamicsConfig(name="test", dt=0.01, g0=0.0)
        inertia = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        dynamics = RigidBody6DOF(config, inertia=inertia)
        state = create_initial_state()

        zero_force = torch.zeros(3, dtype=torch.float64)
        torque = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        state = dynamics.step(state, zero_force, torque)
        # omega_dot = torque / I = 1 rad/s^2
        # omega = omega_dot * dt = 0.01 rad/s
        assert state.angular_velocity[0].item() > 0.0
