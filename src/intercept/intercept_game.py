"""Core MCTS game integration for intercept engagements.

InterceptGame models a 1v1 engagement as a sequential decision game.
The interceptor is the agent; the threat follows a prescribed trajectory.

Satisfies the GameInterface protocol from src/mcts/search.py,
enabling direct reuse of the battle-tested MCTS engine.

Architecture follows the PDEGameAdapter pattern from src/pde/mcts_adapter.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from src.intercept.aero import AeroModel, SimpleAeroModel
from src.intercept.atmosphere import ISAAtmosphere
from src.intercept.config import STANDARD_GRAVITY_MS2 as G0
from src.intercept.config import (
    EngagementPhase,
    GuidanceConfig,
    InterceptorConfig,
    MCTSInterceptConfig,
)
from src.intercept.dynamics import RigidBody6DOF, RigidBodyState

if TYPE_CHECKING:
    from src.intercept.guidance import GuidanceLaw

logger = structlog.get_logger(__name__)


@dataclass
class InterceptState:
    """Complete state of an intercept engagement.

    Attributes:
        interceptor: Interceptor vehicle state.
        threat: Threat vehicle state.
        time: Current simulation time (s).
        miss_estimate: Current estimated miss distance (m).
        range_m: Current range between vehicles (m).
        closing_velocity: Rate of range decrease (m/s).
        phase: Engagement phase.
        steps: Number of simulation steps taken.

    """

    interceptor: RigidBodyState
    threat: RigidBodyState
    time: float
    miss_estimate: float = float("inf")
    range_m: float = float("inf")
    closing_velocity: float = 0.0
    phase: EngagementPhase = EngagementPhase.MIDCOURSE
    steps: int = 0


class InterceptGameAdapter:
    """Adapts an intercept engagement for MCTS search.

    Satisfies the GameInterface protocol from src/mcts/search.py:
    - get_state() -> NDArray[np.float32]
    - get_legal_actions() -> list[int]
    - apply_action(action: int) -> None
    - is_terminal() -> bool
    - get_winner() -> int
    - clone() -> InterceptGameAdapter

    Action space: 3D acceleration grid. Each axis discretized into
    `grid_size` levels from -max_g to +max_g, plus a coast action.
    Total actions = grid_size^3 + 1.
    """

    def __init__(
        self,
        interceptor_state: RigidBodyState,
        threat_state: RigidBodyState,
        interceptor_config: InterceptorConfig | None = None,
        mcts_config: MCTSInterceptConfig | None = None,
        dynamics: RigidBody6DOF | None = None,
        threat_aero: AeroModel | None = None,
        interceptor_aero: AeroModel | None = None,
        atmosphere: ISAAtmosphere | None = None,
    ) -> None:
        self._interceptor_config = interceptor_config or InterceptorConfig(name="default")
        self._mcts_config = mcts_config or MCTSInterceptConfig(name="default")
        self._dynamics = dynamics or RigidBody6DOF()
        self._threat_aero = threat_aero or SimpleAeroModel()
        self._interceptor_aero = interceptor_aero or SimpleAeroModel(
            cd=self._interceptor_config.cd_0,
            reference_area=self._interceptor_config.reference_area_m2,
        )
        self._atmosphere = atmosphere or ISAAtmosphere()

        self._state = InterceptState(
            interceptor=interceptor_state.clone(),
            threat=threat_state.clone(),
            time=interceptor_state.time.item(),
        )
        self._update_geometry()

        grid = self._mcts_config.action_grid_size
        self._n_actions = grid**3 + 1  # +1 for coast
        self._grid_size = grid
        self._max_accel = self._interceptor_config.max_g * G0
        self._dt = self._mcts_config.rollout_dt_s
        self._max_steps = int(self._mcts_config.time_horizon_s / self._dt)
        self._kill_radius = self._interceptor_config.kill_radius_m
        self._divergence_vel = self._mcts_config.divergence_velocity_ms
        self._divergence_steps = self._mcts_config.divergence_min_steps

    def _update_geometry(self) -> None:
        """Update range, closing velocity, miss estimate."""
        rel_pos = self._state.threat.position - self._state.interceptor.position
        rel_vel = self._state.threat.velocity - self._state.interceptor.velocity
        self._state.range_m = torch.norm(rel_pos).item()

        los = rel_pos / (self._state.range_m + 1e-12)
        self._state.closing_velocity = -torch.dot(rel_vel, los).item()

        # Time-to-go and ZEM
        if self._state.closing_velocity > 0:
            tgo = self._state.range_m / self._state.closing_velocity
            zem = rel_pos + rel_vel * tgo
            self._state.miss_estimate = torch.norm(zem).item()
        else:
            self._state.miss_estimate = float("inf")

        # Phase detection
        if self._state.range_m < 500.0:
            self._state.phase = EngagementPhase.TERMINAL

    def _decode_action(self, action: int) -> NDArray[np.float64]:
        """Decode action index to acceleration vector in NED."""
        if action == self._n_actions - 1:
            return np.zeros(3, dtype=np.float64)

        g = self._grid_size
        # Map action index to 3D grid coordinates
        iz = action % g
        iy = (action // g) % g
        ix = action // (g * g)

        # Map grid coordinates to [-max_accel, +max_accel]
        ax = self._max_accel * (2.0 * ix / (g - 1) - 1.0) if g > 1 else 0.0
        ay = self._max_accel * (2.0 * iy / (g - 1) - 1.0) if g > 1 else 0.0
        az = self._max_accel * (2.0 * iz / (g - 1) - 1.0) if g > 1 else 0.0

        accel = np.array([ax, ay, az], dtype=np.float64)

        # Clamp total magnitude
        mag = np.linalg.norm(accel)
        if mag > self._max_accel:
            accel = accel * (self._max_accel / mag)

        return accel

    # --- GameInterface protocol ---

    def get_state(self) -> NDArray[np.float32]:
        """Encode engagement state as feature vector (15 features).

        Features: relative_pos(3), relative_vel(3), LOS_rate(3),
                  tgo(1), energy_ratio(1), miss_estimate(1),
                  threat_speed(1), interceptor_speed(1), range(1).
        """
        rel_pos = self._state.threat.position - self._state.interceptor.position
        rel_vel = self._state.threat.velocity - self._state.interceptor.velocity
        range_m = torch.norm(rel_pos) + 1e-12

        # LOS rate
        los_rate = torch.cross(rel_pos, rel_vel, dim=-1) / (range_m**2)

        # Time-to-go
        closing = -torch.dot(rel_vel, rel_pos / range_m)
        tgo = range_m / closing.clamp(min=1.0)

        features = torch.cat(
            [
                rel_pos,
                rel_vel,
                los_rate,
                tgo.unsqueeze(0),
                torch.tensor([self._state.closing_velocity / 1000.0], dtype=rel_pos.dtype),
                torch.tensor(
                    [min(self._state.miss_estimate, 1000.0) / 1000.0], dtype=rel_pos.dtype
                ),
                self._state.threat.speed.unsqueeze(0) / 1000.0,
                self._state.interceptor.speed.unsqueeze(0) / 1000.0,
                torch.tensor([self._state.range_m / 10000.0], dtype=rel_pos.dtype),
            ]
        )
        return features.detach().cpu().float().numpy()

    def get_legal_actions(self) -> list[int]:
        """All actions are legal (g-limit enforced by clamping)."""
        return list(range(self._n_actions))

    def apply_action(self, action: int) -> None:
        """Apply interceptor guidance action and advance both vehicles."""
        accel = self._decode_action(action)
        accel_t = torch.tensor(
            accel,
            dtype=self._state.interceptor.position.dtype,
            device=self._state.interceptor.position.device,
        )

        # Advance interceptor with guidance command
        int_aero_f, int_aero_t = self._interceptor_aero.compute_forces(
            self._state.interceptor, self._atmosphere
        )
        int_force = int_aero_f + accel_t * self._state.interceptor.mass
        self._state.interceptor = self._dynamics.step(
            self._state.interceptor, int_force, int_aero_t, self._dt
        )

        # Advance threat ballistically (no guidance)
        thr_force, thr_torque = self._threat_aero.compute_forces(
            self._state.threat, self._atmosphere
        )
        self._state.threat = self._dynamics.step(
            self._state.threat, thr_force, thr_torque, self._dt
        )

        self._state.steps += 1
        self._state.time += self._dt
        self._update_geometry()

    def is_terminal(self) -> bool:
        """Check if engagement has ended."""
        # Hit
        if self._state.range_m < self._kill_radius:
            return True
        # Max steps
        if self._state.steps >= self._max_steps:
            return True
        # Diverging (target moving away)
        if (
            self._state.closing_velocity < self._divergence_vel
            and self._state.steps > self._divergence_steps
        ):
            return True
        # Crashed
        if self._state.interceptor.altitude.item() < 0:
            return True
        return False

    def get_winner(self) -> int:
        """Map engagement outcome to {-1, 0, 1}.

        +1: successful intercept (range < kill radius)
        -1: miss or crash
         0: ambiguous (still in progress at horizon)
        """
        if self._state.range_m < self._kill_radius:
            return 1
        if self._state.interceptor.altitude.item() < 0:
            return -1
        if self._state.closing_velocity < self._divergence_vel:
            return -1
        if self._state.steps >= self._max_steps:
            if self._state.miss_estimate < self._kill_radius * 5:
                return 0  # close but not definitive
            return -1
        return 0

    def clone(self) -> InterceptGameAdapter:
        """Deep copy for MCTS tree expansion."""
        new = InterceptGameAdapter.__new__(InterceptGameAdapter)
        new._interceptor_config = self._interceptor_config
        new._mcts_config = self._mcts_config
        new._dynamics = self._dynamics
        new._threat_aero = self._threat_aero
        new._interceptor_aero = self._interceptor_aero
        new._atmosphere = self._atmosphere
        new._state = InterceptState(
            interceptor=self._state.interceptor.clone(),
            threat=self._state.threat.clone(),
            time=self._state.time,
            miss_estimate=self._state.miss_estimate,
            range_m=self._state.range_m,
            closing_velocity=self._state.closing_velocity,
            phase=self._state.phase,
            steps=self._state.steps,
        )
        new._n_actions = self._n_actions
        new._grid_size = self._grid_size
        new._max_accel = self._max_accel
        new._dt = self._dt
        new._max_steps = self._max_steps
        new._kill_radius = self._kill_radius
        return new


def run_engagement(
    interceptor_state: RigidBodyState,
    threat_state: RigidBodyState,
    guidance_law: GuidanceLaw,
    guidance_config: GuidanceConfig | None = None,
    interceptor_config: InterceptorConfig | None = None,
    dynamics: RigidBody6DOF | None = None,
    atmosphere: ISAAtmosphere | None = None,
    max_time: float = 60.0,
    dt: float = 0.02,
    divergence_velocity_ms: float = -100.0,
) -> list[InterceptState]:
    """Run a closed-loop engagement simulation.

    Uses the provided guidance law to steer the interceptor
    toward the threat. Records state history.

    Args:
        interceptor_state: Initial interceptor state.
        threat_state: Initial threat state.
        guidance_law: Object with compute(own, threat, config) method.
        guidance_config: Guidance law configuration.
        interceptor_config: Interceptor configuration.
        dynamics: 6-DOF dynamics integrator.
        atmosphere: Atmosphere model.
        max_time: Maximum engagement duration (s).
        dt: Simulation time step (s).
        divergence_velocity_ms: Closing velocity threshold for divergence (m/s).

    Returns:
        List of InterceptState snapshots.

    """
    guidance_config = guidance_config or GuidanceConfig(name="default")
    interceptor_config = interceptor_config or InterceptorConfig(name="default")
    dynamics = dynamics or RigidBody6DOF()
    atmosphere = atmosphere or ISAAtmosphere()
    threat_aero = SimpleAeroModel()
    interceptor_aero = SimpleAeroModel(
        cd=interceptor_config.cd_0,
        reference_area=interceptor_config.reference_area_m2,
    )

    int_state = interceptor_state.clone()
    thr_state = threat_state.clone()
    kill_radius = interceptor_config.kill_radius_m

    history: list[InterceptState] = []
    n_steps = int(max_time / dt)

    for step in range(n_steps):
        rel_pos = thr_state.position - int_state.position
        range_m = torch.norm(rel_pos).item()

        rel_vel = thr_state.velocity - int_state.velocity
        los = rel_pos / (range_m + 1e-12)
        closing = -torch.dot(rel_vel, los).item()

        state = InterceptState(
            interceptor=int_state.clone(),
            threat=thr_state.clone(),
            time=int_state.time.item(),
            range_m=range_m,
            closing_velocity=closing,
            steps=step,
        )
        history.append(state)

        # Check termination
        if range_m < kill_radius:
            logger.info("engagement_hit", range_m=range_m, step=step)
            break
        if int_state.altitude.item() < 0 or thr_state.altitude.item() < 0:
            break
        if closing < divergence_velocity_ms and step > 10:
            break

        # Compute guidance command
        cmd = guidance_law.compute(int_state, thr_state, guidance_config)
        if cmd.should_breakoff:
            logger.info("engagement_breakoff", miss=cmd.miss_distance, step=step)
            break

        # Advance interceptor
        int_force_aero, int_torque = interceptor_aero.compute_forces(int_state, atmosphere)
        guidance_force = cmd.acceleration * int_state.mass
        int_state = dynamics.step(int_state, int_force_aero + guidance_force, int_torque, dt)

        # Advance threat (ballistic)
        thr_force, thr_torque = threat_aero.compute_forces(thr_state, atmosphere)
        thr_state = dynamics.step(thr_state, thr_force, thr_torque, dt)

    return history
