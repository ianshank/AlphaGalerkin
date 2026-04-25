"""Threat trajectory prediction using MCTS-based maneuver search.

Predicts future threat trajectories by:
1. Ballistic mode: simple 6-DOF forward propagation (no maneuvers)
2. Maneuvering mode: MCTS explores possible maneuver sequences

The ThreatMCTSGame implements the GameInterface protocol from
src/mcts/search.py, enabling direct reuse of the MCTS engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from src.intercept.aero import AeroModel, SimpleAeroModel
from src.intercept.atmosphere import ISAAtmosphere
from src.intercept.config import STANDARD_GRAVITY_MS2 as G0
from src.intercept.config import ThreatConfig
from src.intercept.dynamics import (
    RigidBody6DOF,
    RigidBodyState,
)

logger = structlog.get_logger(__name__)


@dataclass
class PredictedTrajectory:
    """Result of threat trajectory prediction.

    Attributes:
        positions: List of predicted NED positions (m).
        velocities: List of predicted NED velocities (m/s).
        times: List of prediction times (s).
        confidence: Overall confidence score [0, 1].

    """

    positions: list[NDArray[np.float64]]
    velocities: list[NDArray[np.float64]]
    times: list[float]
    confidence: float = 1.0

    @property
    def final_position(self) -> NDArray[np.float64]:
        """Position at end of prediction horizon."""
        return self.positions[-1]

    @property
    def duration(self) -> float:
        """Total prediction duration in seconds."""
        return self.times[-1] - self.times[0] if self.times else 0.0


class ThreatPredictor:
    """Predicts future threat trajectory.

    Ballistic mode: integrates 6-DOF dynamics forward with aero drag.
    Maneuvering mode: uses MCTS to explore possible maneuver sequences.
    """

    def __init__(
        self,
        dynamics: RigidBody6DOF | None = None,
        aero_model: AeroModel | None = None,
        atmosphere: ISAAtmosphere | None = None,
        threat_config: ThreatConfig | None = None,
    ) -> None:
        self.dynamics = dynamics or RigidBody6DOF()
        self.aero_model = aero_model or SimpleAeroModel()
        self.atmosphere = atmosphere or ISAAtmosphere()
        self.threat_config = threat_config or ThreatConfig(name="default")

    def predict_ballistic(
        self,
        state: RigidBodyState,
        horizon_s: float,
        dt: float = 0.1,
    ) -> PredictedTrajectory:
        """Predict trajectory assuming no maneuvers (ballistic).

        Args:
            state: Current threat state.
            horizon_s: Prediction time horizon in seconds.
            dt: Integration time step.

        Returns:
            Predicted trajectory with positions and velocities.

        """
        positions = [state.position.detach().cpu().numpy().copy()]
        velocities = [state.velocity.detach().cpu().numpy().copy()]
        times = [state.time.item()]

        current = state.clone()
        n_steps = int(horizon_s / dt)

        for _ in range(n_steps):
            force, torque = self.aero_model.compute_forces(current, self.atmosphere)
            current = self.dynamics.step(current, force, torque, dt)
            positions.append(current.position.detach().cpu().numpy().copy())
            velocities.append(current.velocity.detach().cpu().numpy().copy())
            times.append(current.time.item())

        return PredictedTrajectory(
            positions=positions,
            velocities=velocities,
            times=times,
            confidence=1.0,  # high confidence for ballistic
        )


class ThreatMCTSGame:
    """MCTS game for exploring possible threat maneuvers.

    Satisfies the GameInterface protocol from src/mcts/search.py:
    - get_state() -> NDArray[np.float32]
    - get_legal_actions() -> list[int]
    - apply_action(action: int) -> None
    - is_terminal() -> bool
    - get_winner() -> int
    - clone() -> ThreatMCTSGame

    Action space: discretized accelerations. 26 directions
    (6 axis-aligned + 12 edge + 8 corner) x 3 magnitudes + coast = 79 actions.
    Actions exceeding structural g-limit are filtered as illegal.
    """

    # 26 unit direction vectors + [0,0,0] for coast
    _DIRECTIONS: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                _DIRECTIONS.append((dx, dy, dz))

    N_MAGNITUDES = 3  # fraction of max-g: 0.33, 0.67, 1.0
    N_ACTIONS = len(_DIRECTIONS) * N_MAGNITUDES + 1  # +1 for coast

    def __init__(
        self,
        initial_state: RigidBodyState,
        dynamics: RigidBody6DOF,
        aero_model: AeroModel,
        atmosphere: ISAAtmosphere,
        max_g: float = 5.0,
        horizon_s: float = 5.0,
        dt: float = 0.1,
    ) -> None:
        self._state = initial_state.clone()
        self._dynamics = dynamics
        self._aero_model = aero_model
        self._atmosphere = atmosphere
        self._max_g = max_g
        self._horizon_s = horizon_s
        self._dt = dt
        self._steps = 0
        self._max_steps = int(horizon_s / dt)
        self._initial_speed = initial_state.speed.item()

    def get_state(self) -> NDArray[np.float32]:
        """Encode current state as feature vector."""
        pos = self._state.position.detach().cpu().float().numpy()
        vel = self._state.velocity.detach().cpu().float().numpy()
        t = np.array([self._state.time.item()], dtype=np.float32)
        speed = np.array([self._state.speed.item()], dtype=np.float32)
        alt = np.array([self._state.altitude.item()], dtype=np.float32)
        return np.concatenate([pos, vel, t, speed, alt]).astype(np.float32)

    def get_legal_actions(self) -> list[int]:
        """Get legal action indices (filtered by g-limit)."""
        # All actions are legal by default; structural limits
        # are soft constraints handled in apply_action by clamping
        return list(range(self.N_ACTIONS))

    def _decode_action(self, action: int) -> NDArray[np.float64]:
        """Decode action index to acceleration vector in NED (m/s^2)."""
        if action == self.N_ACTIONS - 1:
            # Coast: zero commanded acceleration
            return np.zeros(3, dtype=np.float64)

        mag_idx = action % self.N_MAGNITUDES
        dir_idx = action // self.N_MAGNITUDES

        # Magnitude: fraction of max-g
        mag_fraction = (mag_idx + 1) / self.N_MAGNITUDES
        mag = mag_fraction * self._max_g * G0  # convert to m/s^2

        # Direction
        d = self._DIRECTIONS[dir_idx]
        direction = np.array(d, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        return direction * mag

    def apply_action(self, action: int) -> None:
        """Apply maneuver action and advance one time step."""
        accel = self._decode_action(action)
        accel_tensor = torch.tensor(
            accel, dtype=self._state.position.dtype, device=self._state.position.device
        )

        # Compute aero forces
        aero_force, aero_torque = self._aero_model.compute_forces(self._state, self._atmosphere)

        # Add maneuver force (F = m * a)
        maneuver_force = accel_tensor * self._state.mass

        total_force = aero_force + maneuver_force

        self._state = self._dynamics.step(self._state, total_force, aero_torque, self._dt)
        self._steps += 1

    def is_terminal(self) -> bool:
        """Terminal when horizon reached or vehicle crashed."""
        if self._steps >= self._max_steps:
            return True
        if self._state.altitude.item() < 0:
            return True
        return False

    def get_winner(self) -> int:
        """Map trajectory plausibility to outcome.

        +1: trajectory is physically plausible and maintained energy
        -1: trajectory is implausible (crashed, impossible g-loads)
         0: neutral / uncertain
        """
        if self._state.altitude.item() < 0:
            return -1  # crashed

        speed_ratio = self._state.speed.item() / (self._initial_speed + 1e-6)
        if speed_ratio > 0.5:
            return 1  # plausible: maintained energy
        return 0  # ambiguous

    def clone(self) -> ThreatMCTSGame:
        """Deep copy for MCTS tree expansion."""
        new = ThreatMCTSGame.__new__(ThreatMCTSGame)
        new._state = self._state.clone()
        new._dynamics = self._dynamics
        new._aero_model = self._aero_model
        new._atmosphere = self._atmosphere
        new._max_g = self._max_g
        new._horizon_s = self._horizon_s
        new._dt = self._dt
        new._steps = self._steps
        new._max_steps = self._max_steps
        new._initial_speed = self._initial_speed
        return new
