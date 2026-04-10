"""6-DOF rigid body dynamics with quaternion-based attitude integration.

Implements translational (F=ma) and rotational (Euler's equations)
dynamics with RK4 integration. All operations are PyTorch tensor ops
for autodiff compatibility with physics-informed losses.

State vector (13D):
- position (3): NED coordinates in meters
- velocity (3): NED velocity in m/s
- quaternion (4): body-to-NED orientation [w, x, y, z]
- angular_velocity (3): body-frame angular velocity in rad/s
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
from torch import Tensor

from src.intercept.config import STANDARD_GRAVITY_MS2 as G0
from src.intercept.config import DynamicsConfig, GravityModel, IntegrationMethod
from src.intercept.frames import QuaternionOps

logger = structlog.get_logger(__name__)


@dataclass
class RigidBodyState:
    """Complete 6-DOF rigid body state.

    All tensors support batched operations with leading batch dimensions.

    Attributes:
        position: NED position in meters (..., 3).
        velocity: NED velocity in m/s (..., 3).
        quaternion: Body-to-NED rotation quaternion [w,x,y,z] (..., 4).
        angular_velocity: Body-frame angular velocity in rad/s (..., 3).
        mass: Vehicle mass in kg (scalar or ...).
        time: Simulation time in seconds (scalar or ...).

    """

    position: Tensor
    velocity: Tensor
    quaternion: Tensor
    angular_velocity: Tensor
    mass: Tensor
    time: Tensor

    def to_tensor(self) -> Tensor:
        """Flatten state to a single tensor (..., 13)."""
        return torch.cat(
            [
                self.position,
                self.velocity,
                self.quaternion,
                self.angular_velocity,
            ],
            dim=-1,
        )

    @staticmethod
    def from_tensor(t: Tensor, mass: Tensor, time: Tensor) -> RigidBodyState:
        """Reconstruct state from flattened tensor (..., 13)."""
        return RigidBodyState(
            position=t[..., :3],
            velocity=t[..., 3:6],
            quaternion=t[..., 6:10],
            angular_velocity=t[..., 10:13],
            mass=mass,
            time=time,
        )

    def clone(self) -> RigidBodyState:
        """Deep copy of state."""
        return RigidBodyState(
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            quaternion=self.quaternion.clone(),
            angular_velocity=self.angular_velocity.clone(),
            mass=self.mass.clone(),
            time=self.time.clone(),
        )

    @property
    def speed(self) -> Tensor:
        """Scalar speed in m/s (...)."""
        return torch.norm(self.velocity, dim=-1)

    @property
    def altitude(self) -> Tensor:
        """Altitude (negative of NED down component) in meters (...)."""
        return -self.position[..., 2]


def gravity_ned(
    altitude_m: Tensor,
    model: GravityModel = GravityModel.CONSTANT,
    g0: float = G0,
) -> Tensor:
    """Compute gravity vector in NED frame.

    Args:
        altitude_m: Altitude in meters (...).
        model: Gravity model to use.
        g0: Standard gravitational acceleration.

    Returns:
        Gravity vector in NED [0, 0, g] (..., 3).

    """
    if model == GravityModel.WGS84:
        # Gravity decreases with altitude: g = g0 * (Re / (Re + h))^2
        re = 6_371_000.0  # mean Earth radius
        g = g0 * (re / (re + altitude_m)) ** 2
    else:
        g = torch.full_like(altitude_m, g0)

    gravity = torch.zeros(*altitude_m.shape, 3, device=altitude_m.device, dtype=altitude_m.dtype)
    gravity[..., 2] = g  # NED: down is positive
    return gravity


class RigidBody6DOF:
    """6-DOF rigid body dynamics integrator.

    Integrates translational and rotational equations of motion
    using configurable methods (Euler, RK4).

    The dynamics are:
        Translation: m * a = F_gravity + F_aero + F_thrust
        Rotation: I * omega_dot = torque - omega x (I * omega)
        Quaternion: q_dot = 0.5 * q * [0, omega]
    """

    def __init__(
        self,
        config: DynamicsConfig | None = None,
        inertia: Tensor | None = None,
    ) -> None:
        """Initialize 6-DOF dynamics.

        Args:
            config: Dynamics configuration.
            inertia: Moment of inertia tensor (3,) diagonal or (3,3) full.
                     Defaults to unit inertia.

        """
        self.config = config or DynamicsConfig(name="default_dynamics")
        if inertia is not None:
            if inertia.dim() == 1:
                self._inertia_diag = inertia
                self._inertia_full = torch.diag(inertia)
            else:
                self._inertia_diag = torch.diagonal(inertia)
                self._inertia_full = inertia
        else:
            self._inertia_diag = torch.ones(3, dtype=torch.float64)
            self._inertia_full = torch.eye(3, dtype=torch.float64)

    def _derivatives(
        self,
        state: RigidBodyState,
        force_ned: Tensor,
        torque_body: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute state derivatives.

        Args:
            state: Current state.
            force_ned: Total external force in NED frame (..., 3).
            torque_body: Total external torque in body frame (..., 3).

        Returns:
            (pos_dot, vel_dot, quat_dot, omega_dot).

        """
        # Translation: v_dot = F/m + gravity
        grav = gravity_ned(
            state.altitude,
            model=self.config.gravity_model,
            g0=self.config.g0,
        )
        vel_dot = force_ned / state.mass.unsqueeze(-1) + grav

        # Position derivative
        pos_dot = state.velocity

        # Quaternion kinematics: q_dot = 0.5 * q * [0, omega]
        omega_quat = torch.zeros(
            *state.angular_velocity.shape[:-1],
            4,
            device=state.angular_velocity.device,
            dtype=state.angular_velocity.dtype,
        )
        omega_quat[..., 1:] = state.angular_velocity
        quat_dot = 0.5 * QuaternionOps.multiply(state.quaternion, omega_quat)

        # Rotational dynamics: I * omega_dot = torque - omega x (I * omega)
        inertia = self._inertia_diag.to(
            device=state.angular_velocity.device,
            dtype=state.angular_velocity.dtype,
        )
        i_omega = inertia * state.angular_velocity
        gyroscopic = torch.cross(state.angular_velocity, i_omega, dim=-1)
        omega_dot = (torque_body - gyroscopic) / inertia

        return pos_dot, vel_dot, quat_dot, omega_dot

    def step(
        self,
        state: RigidBodyState,
        force_ned: Tensor,
        torque_body: Tensor,
        dt: float | None = None,
    ) -> RigidBodyState:
        """Advance state by one time step.

        Args:
            state: Current rigid body state.
            force_ned: Total external force in NED (..., 3).
            torque_body: Total external torque in body frame (..., 3).
            dt: Time step override (uses config.dt if None).

        Returns:
            New state after integration.

        """
        dt = dt or self.config.dt

        if self.config.integration_method == IntegrationMethod.RK4:
            return self._step_rk4(state, force_ned, torque_body, dt)
        return self._step_euler(state, force_ned, torque_body, dt)

    def _step_euler(
        self,
        state: RigidBodyState,
        force_ned: Tensor,
        torque_body: Tensor,
        dt: float,
    ) -> RigidBodyState:
        """Forward Euler integration step."""
        pos_dot, vel_dot, quat_dot, omega_dot = self._derivatives(state, force_ned, torque_body)

        new_pos = state.position + pos_dot * dt
        new_vel = state.velocity + vel_dot * dt
        new_quat = QuaternionOps.normalize(state.quaternion + quat_dot * dt)
        new_omega = state.angular_velocity + omega_dot * dt

        return RigidBodyState(
            position=new_pos,
            velocity=new_vel,
            quaternion=new_quat,
            angular_velocity=new_omega,
            mass=state.mass.clone(),
            time=state.time + dt,
        )

    def _step_rk4(
        self,
        state: RigidBodyState,
        force_ned: Tensor,
        torque_body: Tensor,
        dt: float,
    ) -> RigidBodyState:
        """4th-order Runge-Kutta integration step.

        Forces/torques are held constant over the step (suitable for
        guidance loop rates >= 50 Hz).
        """

        def get_derivs(s: RigidBodyState) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            return self._derivatives(s, force_ned, torque_body)

        def make_state(
            base: RigidBodyState,
            dp: Tensor,
            dv: Tensor,
            dq: Tensor,
            dw: Tensor,
            h: float,
        ) -> RigidBodyState:
            return RigidBodyState(
                position=base.position + dp * h,
                velocity=base.velocity + dv * h,
                quaternion=QuaternionOps.normalize(base.quaternion + dq * h),
                angular_velocity=base.angular_velocity + dw * h,
                mass=base.mass,
                time=base.time + h,
            )

        # k1
        dp1, dv1, dq1, dw1 = get_derivs(state)

        # k2
        s2 = make_state(state, dp1, dv1, dq1, dw1, dt * 0.5)
        dp2, dv2, dq2, dw2 = get_derivs(s2)

        # k3
        s3 = make_state(state, dp2, dv2, dq2, dw2, dt * 0.5)
        dp3, dv3, dq3, dw3 = get_derivs(s3)

        # k4
        s4 = make_state(state, dp3, dv3, dq3, dw3, dt)
        dp4, dv4, dq4, dw4 = get_derivs(s4)

        # Combine
        new_pos = state.position + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        new_vel = state.velocity + (dt / 6.0) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
        new_quat = QuaternionOps.normalize(
            state.quaternion + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
        )
        new_omega = state.angular_velocity + (dt / 6.0) * (dw1 + 2 * dw2 + 2 * dw3 + dw4)

        return RigidBodyState(
            position=new_pos,
            velocity=new_vel,
            quaternion=new_quat,
            angular_velocity=new_omega,
            mass=state.mass.clone(),
            time=state.time + dt,
        )

    def propagate(
        self,
        state: RigidBodyState,
        force_ned: Tensor,
        torque_body: Tensor,
        duration: float,
        dt: float | None = None,
    ) -> list[RigidBodyState]:
        """Propagate state forward for a given duration.

        Args:
            state: Initial state.
            force_ned: Constant force in NED (..., 3).
            torque_body: Constant torque in body frame (..., 3).
            duration: Total propagation time in seconds.
            dt: Time step (uses config.dt if None).

        Returns:
            List of states at each time step (including initial).

        """
        dt = dt or self.config.dt
        n_steps = int(duration / dt)
        trajectory = [state]

        current = state
        for _ in range(n_steps):
            current = self.step(current, force_ned, torque_body, dt)
            trajectory.append(current)

        return trajectory


def create_initial_state(
    position: list[float] | None = None,
    velocity: list[float] | None = None,
    euler_deg: list[float] | None = None,
    mass: float = 100.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> RigidBodyState:
    """Convenience factory for creating initial states.

    Args:
        position: NED position [n, e, d] in meters.
        velocity: NED velocity [vn, ve, vd] in m/s.
        euler_deg: Euler angles [roll, pitch, yaw] in degrees.
        mass: Vehicle mass in kg.
        dtype: Tensor dtype.
        device: Tensor device.

    Returns:
        Initialized RigidBodyState.

    """
    pos = torch.tensor(position or [0.0, 0.0, 0.0], dtype=dtype, device=device)
    vel = torch.tensor(velocity or [0.0, 0.0, 0.0], dtype=dtype, device=device)

    if euler_deg is not None:
        euler_rad = torch.tensor(euler_deg, dtype=dtype, device=device) * (torch.pi / 180.0)
        quat = QuaternionOps.from_euler(euler_rad[0], euler_rad[1], euler_rad[2])
    else:
        quat = QuaternionOps.identity(device=device, dtype=dtype)

    omega = torch.zeros(3, dtype=dtype, device=device)
    m = torch.tensor(mass, dtype=dtype, device=device)
    t = torch.tensor(0.0, dtype=dtype, device=device)

    return RigidBodyState(
        position=pos,
        velocity=vel,
        quaternion=quat,
        angular_velocity=omega,
        mass=m,
        time=t,
    )
