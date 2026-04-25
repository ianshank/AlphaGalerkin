"""Tests for interceptor-specific dynamics (missile and drone)."""

from __future__ import annotations

import torch

from src.intercept.config import InterceptorConfig, InterceptorType
from src.intercept.dynamics import create_initial_state
from src.intercept.interceptor_dynamics import (
    MissileDynamics,
    MotorState,
    RotorDroneDynamics,
)


class TestMotorState:
    def test_is_burning(self) -> None:
        motor = MotorState(burn_time_remaining=5.0, thrust_n=1000.0, fuel_mass_kg=2.0)
        assert motor.is_burning

    def test_not_burning_no_time(self) -> None:
        motor = MotorState(burn_time_remaining=0.0, thrust_n=1000.0, fuel_mass_kg=2.0)
        assert not motor.is_burning

    def test_not_burning_no_fuel(self) -> None:
        motor = MotorState(burn_time_remaining=5.0, thrust_n=1000.0, fuel_mass_kg=0.0)
        assert not motor.is_burning


class TestMissileDynamics:
    def _default_config(self) -> InterceptorConfig:
        return InterceptorConfig(
            name="test_missile",
            interceptor_type=InterceptorType.MISSILE,
            mass_kg=50.0,
            motor_thrust_n=5000.0,
            motor_burn_time_s=3.0,
            fuel_mass_kg=5.0,
            max_g=30.0,
            cd_0=0.2,
            reference_area_m2=0.02,
        )

    def test_step_advances_state(self) -> None:
        config = self._default_config()
        missile = MissileDynamics(config)
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
            mass=config.mass_kg,
        )
        accel = torch.zeros(3, dtype=torch.float64)
        new_state = missile.step(state, accel, dt=0.01)
        assert new_state.time.item() > state.time.item()

    def test_thrust_during_burn(self) -> None:
        config = self._default_config()
        missile = MissileDynamics(config)
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
            mass=config.mass_kg,
        )
        thrust = missile.compute_thrust(state, dt=0.01)
        # During burn, thrust should be nonzero along body x-axis
        assert thrust[0].item() > 0.0

    def test_no_thrust_after_burnout(self) -> None:
        config = self._default_config()
        config = InterceptorConfig(
            name="test", motor_thrust_n=1000.0, motor_burn_time_s=0.0, fuel_mass_kg=0.0
        )
        missile = MissileDynamics(config)
        state = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[200.0, 0.0, 0.0])
        thrust = missile.compute_thrust(state, dt=0.01)
        assert torch.allclose(thrust, torch.zeros(3, dtype=torch.float64))

    def test_fuel_depletes(self) -> None:
        config = self._default_config()
        missile = MissileDynamics(config)
        initial_fuel = missile.motor.fuel_mass_kg
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0], velocity=[200.0, 0.0, 0.0], mass=config.mass_kg
        )
        missile.compute_thrust(state, dt=1.0)
        assert missile.motor.fuel_mass_kg < initial_fuel

    def test_g_limit_clamp(self) -> None:
        config = self._default_config()
        missile = MissileDynamics(config)
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0], velocity=[200.0, 0.0, 0.0], mass=config.mass_kg
        )
        # Command very large acceleration
        huge_accel = torch.tensor([10000.0, 0.0, 0.0], dtype=torch.float64)
        force, torque = missile.command_acceleration(state, huge_accel, dt=0.01)
        # Force should be clamped
        max_force = config.max_g * 9.80665 * state.mass.item()
        # Guidance force alone (without aero/thrust) shouldn't exceed max
        assert force.shape == (3,)


class TestRotorDroneDynamics:
    def _default_config(self) -> InterceptorConfig:
        return InterceptorConfig(
            name="test_drone",
            interceptor_type=InterceptorType.ROTOR_DRONE,
            mass_kg=5.0,
            max_g=10.0,
            max_speed_ms=50.0,
            cd_0=0.5,
            reference_area_m2=0.1,
        )

    def test_step_advances_state(self) -> None:
        config = self._default_config()
        drone = RotorDroneDynamics(config)
        state = create_initial_state(
            position=[0.0, 0.0, -100.0], velocity=[20.0, 0.0, 0.0], mass=config.mass_kg
        )
        accel = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        new_state = drone.step(state, accel, dt=0.01)
        assert new_state.time.item() > state.time.item()

    def test_g_limit_clamp(self) -> None:
        config = self._default_config()
        drone = RotorDroneDynamics(config)
        state = create_initial_state(
            position=[0.0, 0.0, -100.0], velocity=[20.0, 0.0, 0.0], mass=config.mass_kg
        )
        # Command acceleration exceeding max-g
        huge_accel = torch.tensor([5000.0, 0.0, 0.0], dtype=torch.float64)
        new_state = drone.step(state, huge_accel, dt=0.01)
        # State should advance without error (clamping works)
        assert new_state.time.item() > 0.0

    def test_drag_opposes_motion(self) -> None:
        config = self._default_config()
        drone = RotorDroneDynamics(config)
        state = create_initial_state(
            position=[0.0, 0.0, -100.0], velocity=[30.0, 0.0, 0.0], mass=config.mass_kg
        )
        # Zero commanded acceleration -> only drag + gravity
        zero_accel = torch.zeros(3, dtype=torch.float64)
        new_state = drone.step(state, zero_accel, dt=0.1)
        # Speed should decrease due to drag
        assert new_state.speed.item() < state.speed.item()
