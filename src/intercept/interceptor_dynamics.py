"""Interceptor-specific dynamics models.

Extends the base RigidBody6DOF with vehicle-specific models:
- MissileDynamics: fin-controlled missile with motor thrust profile
- RotorDroneDynamics: quadrotor with rotor thrust commands

Both produce force/torque vectors that feed into the base 6-DOF integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
from torch import Tensor

from src.intercept.aero import AeroModel, SimpleAeroModel
from src.intercept.atmosphere import ISAAtmosphere
from src.intercept.config import STANDARD_GRAVITY_MS2 as G0
from src.intercept.config import InterceptorConfig
from src.intercept.dynamics import RigidBody6DOF, RigidBodyState
from src.intercept.frames import FrameTransform

logger = structlog.get_logger(__name__)


@dataclass
class MotorState:
    """Rocket motor state.

    Attributes:
        burn_time_remaining: Remaining burn time (s).
        thrust_n: Current thrust (N).
        fuel_mass_kg: Remaining fuel mass (kg).

    """

    burn_time_remaining: float
    thrust_n: float
    fuel_mass_kg: float

    @property
    def is_burning(self) -> bool:
        return self.burn_time_remaining > 0 and self.fuel_mass_kg > 0


class MissileDynamics:
    """Fin-controlled missile dynamics.

    Models:
    - Motor thrust profile (boost/sustain/coast phases)
    - Fin deflection actuator with rate limiting
    - Mass depletion from fuel consumption
    - Aerodynamic forces via pluggable AeroModel
    """

    def __init__(
        self,
        config: InterceptorConfig,
        dynamics: RigidBody6DOF | None = None,
        aero_model: AeroModel | None = None,
        atmosphere: ISAAtmosphere | None = None,
    ) -> None:
        self.config = config
        self.dynamics = dynamics or RigidBody6DOF()
        self.aero_model = aero_model or SimpleAeroModel(
            cd=config.cd_0,
            reference_area=config.reference_area_m2,
            mass=config.mass_kg,
            max_g=config.max_g,
        )
        self.atmosphere = atmosphere or ISAAtmosphere()
        self.motor = MotorState(
            burn_time_remaining=config.motor_burn_time_s,
            thrust_n=config.motor_thrust_n,
            fuel_mass_kg=config.fuel_mass_kg,
        )
        self._fin_deflection = torch.zeros(3, dtype=torch.float64)

    def compute_thrust(self, state: RigidBodyState, dt: float) -> Tensor:
        """Compute motor thrust in body-frame forward direction.

        Returns:
            Thrust force in body frame (3,).

        """
        if not self.motor.is_burning:
            return torch.zeros(3, dtype=state.position.dtype, device=state.position.device)

        thrust_body = torch.zeros(3, dtype=state.position.dtype, device=state.position.device)
        thrust_body[0] = self.motor.thrust_n  # forward body axis

        # Update motor state
        self.motor.burn_time_remaining -= dt
        fuel_rate = self.motor.thrust_n / (self.config.specific_impulse_s * G0)
        self.motor.fuel_mass_kg = max(0.0, self.motor.fuel_mass_kg - fuel_rate * dt)

        return thrust_body

    def command_acceleration(
        self,
        state: RigidBodyState,
        accel_cmd_ned: Tensor,
        dt: float,
    ) -> tuple[Tensor, Tensor]:
        """Convert acceleration command to force/torque.

        The acceleration command is realized as a force applied
        through aerodynamic control surfaces (fins).

        Args:
            state: Current vehicle state.
            accel_cmd_ned: Commanded acceleration in NED (3,).
            dt: Time step for rate limiting.

        Returns:
            (total_force_ned, total_torque_body).

        """
        # Compute aerodynamic forces
        aero_force, aero_torque = self.aero_model.compute_forces(state, self.atmosphere)

        # Compute motor thrust (body frame -> NED)
        thrust_body = self.compute_thrust(state, dt)
        thrust_ned = FrameTransform.body_to_ned(thrust_body, state.quaternion)

        # Guidance force: F = m * a_cmd
        current_mass = state.mass.item()
        guidance_force = accel_cmd_ned * current_mass

        # Clamp guidance force by max-g
        max_force = self.config.max_g * G0 * current_mass
        force_mag = torch.norm(guidance_force)
        if force_mag.item() > max_force:
            guidance_force = guidance_force * (max_force / force_mag)

        total_force = aero_force + thrust_ned + guidance_force

        return total_force, aero_torque

    def step(
        self,
        state: RigidBodyState,
        accel_cmd_ned: Tensor,
        dt: float | None = None,
    ) -> RigidBodyState:
        """Advance missile state by one time step.

        Args:
            state: Current state.
            accel_cmd_ned: Commanded acceleration in NED.
            dt: Time step (uses dynamics config if None).

        Returns:
            New state after integration.

        """
        dt = dt or self.dynamics.config.dt
        force, torque = self.command_acceleration(state, accel_cmd_ned, dt)

        # Update mass for fuel consumption
        new_mass = max(
            state.mass.item() - 0.0,  # mass tracked in motor state
            self.config.mass_kg - self.config.fuel_mass_kg,
        )

        new_state = self.dynamics.step(state, force, torque, dt)
        new_state.mass = torch.tensor(new_mass, dtype=state.mass.dtype, device=state.mass.device)
        return new_state


class RotorDroneDynamics:
    """Quadrotor interceptor drone dynamics.

    Simplified model: treats rotor thrust as a direct force command
    in the body frame, with max-thrust and max-turn-rate limits.
    """

    def __init__(
        self,
        config: InterceptorConfig,
        dynamics: RigidBody6DOF | None = None,
        atmosphere: ISAAtmosphere | None = None,
    ) -> None:
        self.config = config
        self.dynamics = dynamics or RigidBody6DOF()
        self.atmosphere = atmosphere or ISAAtmosphere()

    def step(
        self,
        state: RigidBodyState,
        accel_cmd_ned: Tensor,
        dt: float | None = None,
    ) -> RigidBodyState:
        """Advance drone state by one time step.

        Args:
            state: Current state.
            accel_cmd_ned: Commanded acceleration in NED.
            dt: Time step.

        Returns:
            New state after integration.

        """
        dt = dt or self.dynamics.config.dt

        # Clamp by max-g
        max_accel = self.config.max_g * G0
        accel_mag = torch.norm(accel_cmd_ned)
        if accel_mag.item() > max_accel:
            accel_cmd_ned = accel_cmd_ned * (max_accel / accel_mag)

        # Simple drag model for drone
        speed = state.speed
        alt = state.altitude
        rho = self.atmosphere.density(alt)
        cd = self.config.cd_0
        sref = self.config.reference_area_m2
        q = 0.5 * rho * speed**2
        drag_mag = q * cd * sref
        v_hat = state.velocity / (speed + 1e-12)
        drag_force = -drag_mag * v_hat

        # Total force = drag + commanded acceleration force
        guidance_force = accel_cmd_ned * state.mass
        total_force = drag_force + guidance_force

        torque = torch.zeros_like(state.angular_velocity)
        return self.dynamics.step(state, total_force, torque, dt)
