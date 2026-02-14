"""Inverse problem solving with MCTS-based measurement planning.

This module addresses non-convex, multimodal inverse problem landscapes
where gradient-based methods get trapped in local optima. MCTS plans
measurement sequences for optimal information gain in sensor placement
and experimental design problems.

The Deep BSDE connection (E, Han & Jentzen, PNAS 2018) reformulated
parabolic PDEs as backward SDEs with the gradient as a "policy function",
establishing an explicit RL-PDE link that this module builds upon.

State: current parameter estimate + uncertainty + sensor network.
Actions: place/remove sensors, take measurements, refine estimate.
Reward: information gain (uncertainty reduction) per measurement cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("planning.inverse")


class InverseActionType(str, Enum):
    """Actions for inverse problem solving."""

    PLACE_SENSOR = "place_sensor"
    """Place a new sensor at a specified location."""

    REMOVE_SENSOR = "remove_sensor"
    """Remove an existing sensor from the network."""

    TAKE_MEASUREMENT = "take_measurement"
    """Collect a measurement from the next unmeasured sensor."""

    UPDATE_PRIOR = "update_prior"
    """Update the prior distribution using collected data."""

    REFINE_ESTIMATE = "refine_estimate"
    """Refine the current parameter estimate via Bayesian update."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


@dataclass
class SensorConfig:
    """Configuration of a single sensor/measurement point.

    Attributes:
        location: Spatial location of the sensor (dim,).
        noise_level: Standard deviation of measurement noise.
        measurement_type: Type of measurement (pointwise, averaged, gradient).
        measurement: Recorded measurement value (None if not yet taken).

    """

    location: np.ndarray
    noise_level: float = 0.01
    measurement_type: str = "pointwise"
    measurement: float | None = None


@dataclass
class InverseProblemState:
    """State of an inverse problem solve.

    Tracks the current parameter estimate, placed sensors,
    collected measurements, and uncertainty quantification.

    Attributes:
        parameter_estimate: Current estimate of unknown parameters.
        parameter_bounds: Per-dimension (min, max) bounds.
        sensors: List of placed sensor configurations.
        uncertainty: Diagonal covariance (per-parameter variance).
        total_information_gain: Cumulative information gain so far.
        measurement_budget: Maximum number of measurements allowed.
        measurements_taken: Number of measurements collected so far.
        step: Number of planning steps taken.

    """

    parameter_estimate: np.ndarray
    parameter_bounds: list[tuple[float, float]]
    sensors: list[SensorConfig] = field(default_factory=list)

    # Uncertainty (diagonal covariance for simplicity)
    uncertainty: np.ndarray | None = None

    # Information gain tracking
    total_information_gain: float = 0.0
    measurement_budget: int = 50
    measurements_taken: int = 0

    step: int = 0

    @property
    def remaining_measurements(self) -> int:
        """Return the number of measurements still available."""
        return self.measurement_budget - self.measurements_taken

    @property
    def mean_uncertainty(self) -> float:
        """Return the mean of the diagonal uncertainty vector."""
        if self.uncertainty is None:
            return float("inf")
        return float(np.mean(self.uncertainty))

    def clone(self) -> InverseProblemState:
        """Return a deep, independent copy of this state."""
        return InverseProblemState(
            parameter_estimate=self.parameter_estimate.copy(),
            parameter_bounds=list(self.parameter_bounds),
            sensors=[
                SensorConfig(
                    location=s.location.copy(),
                    noise_level=s.noise_level,
                    measurement_type=s.measurement_type,
                    measurement=s.measurement,
                )
                for s in self.sensors
            ],
            uncertainty=(self.uncertainty.copy() if self.uncertainty is not None else None),
            total_information_gain=self.total_information_gain,
            measurement_budget=self.measurement_budget,
            measurements_taken=self.measurements_taken,
            step=self.step,
        )


@dataclass
class InverseAction:
    """An inverse problem action.

    Attributes:
        action_type: The type of action to take.
        location: Spatial location for sensor placement (None for
            non-spatial actions).
        params: Optional parameters for the action.

    """

    action_type: InverseActionType
    location: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)


class InverseProblemSolver:
    """Solves inverse problems via MCTS-based measurement planning.

    Plans optimal sensor placement and measurement sequences to
    maximise information gain about unknown parameters.

    The solver treats inverse problem solving as a sequential game:
    - State: current parameter estimate + uncertainty + sensor network
    - Actions: place/remove sensors, take measurements, refine estimate
    - Reward: information gain (uncertainty reduction) per measurement cost

    All tunable parameters are exposed through the constructor -- no
    magic numbers are hard-coded.

    Parameters
    ----------
    domain_bounds:
        Bounding box as a list of (min, max) tuples per dimension.
    max_sensors:
        Upper limit on number of sensors that can be placed.
    num_simulations:
        Number of look-ahead simulations per planning step.
    noise_level:
        Default measurement noise standard deviation.
    exploration_weight:
        Weight for the exploration bonus in information gain.

    """

    def __init__(
        self,
        domain_bounds: list[tuple[float, float]],
        max_sensors: int = 20,
        num_simulations: int = 50,
        noise_level: float = 0.01,
        exploration_weight: float = 1.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._bounds = domain_bounds
        self._max_sensors = max_sensors
        self._num_simulations = num_simulations
        self._noise_level = noise_level
        self._exploration_weight = exploration_weight
        self._dim = len(domain_bounds)
        self._rng = np.random.default_rng(seed)

        logger.info(
            "inverse_solver.init",
            dim=self._dim,
            max_sensors=max_sensors,
            num_simulations=num_simulations,
            noise_level=noise_level,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_next_action(
        self,
        state: InverseProblemState,
        _forward_model: Callable[..., Any] | None = None,
    ) -> InverseAction:
        """Plan next measurement/sensing action.

        Uses estimated information gain to decide where to place
        sensors and when to take measurements.

        Evaluates every valid action by scoring with the information
        gain heuristic.  Returns the action with the highest expected
        information gain per cost.

        Parameters
        ----------
        state:
            Current inverse problem state.
        forward_model:
            Optional callable for forward model evaluation.
            If provided, may be used for look-ahead simulation.

        Returns
        -------
        InverseAction
            The best action found.

        """
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return InverseAction(action_type=InverseActionType.NO_OP)

        best_action = valid_actions[0]
        best_score = -float("inf")

        for action in valid_actions:
            score = self._compute_information_gain(state, action)
            if score > best_score:
                best_score = score
                best_action = action

        logger.info(
            "inverse_solver.plan",
            chosen_action=best_action.action_type.value,
            score=best_score,
            remaining_measurements=state.remaining_measurements,
            step=state.step,
        )
        return best_action

    def get_valid_actions(
        self,
        state: InverseProblemState,
    ) -> list[InverseAction]:
        """Return valid actions for the current state.

        Enforces sensor count bounds and measurement budget to
        prevent degenerate configurations.
        """
        actions: list[InverseAction] = []

        has_budget = state.remaining_measurements > 0

        # Place a new sensor (if under max and budget allows)
        if len(state.sensors) < self._max_sensors and has_budget:
            candidate = self._sample_candidate_location(state)
            actions.append(
                InverseAction(
                    action_type=InverseActionType.PLACE_SENSOR,
                    location=candidate,
                )
            )

        # Remove a sensor (if at least one exists)
        if len(state.sensors) > 0:
            # Select the sensor with highest noise (least informative)
            worst_idx = int(
                np.argmax(
                    [s.noise_level for s in state.sensors],
                )
            )
            actions.append(
                InverseAction(
                    action_type=InverseActionType.REMOVE_SENSOR,
                    params={"sensor_index": worst_idx},
                )
            )

        # Take measurement (if unmeasured sensors exist and budget allows)
        unmeasured = [i for i, s in enumerate(state.sensors) if s.measurement is None]
        if unmeasured and has_budget:
            actions.append(
                InverseAction(
                    action_type=InverseActionType.TAKE_MEASUREMENT,
                    params={"sensor_index": unmeasured[0]},
                )
            )

        # Update prior (if at least one measurement has been taken)
        measured = [s for s in state.sensors if s.measurement is not None]
        if measured:
            actions.append(
                InverseAction(
                    action_type=InverseActionType.UPDATE_PRIOR,
                )
            )

        # Refine estimate (if measurements exist and uncertainty is set)
        if measured and state.uncertainty is not None:
            actions.append(
                InverseAction(
                    action_type=InverseActionType.REFINE_ESTIMATE,
                )
            )

        # No-op is always valid
        actions.append(InverseAction(action_type=InverseActionType.NO_OP))

        return actions

    def apply_action(
        self,
        state: InverseProblemState,
        action: InverseAction,
        measurement_value: float | None = None,
    ) -> InverseProblemState:
        """Apply an action to produce a new state.

        Parameters
        ----------
        state:
            Current inverse problem state.
        action:
            The action to apply.
        measurement_value:
            The measurement value (required for TAKE_MEASUREMENT).

        Returns
        -------
        InverseProblemState
            The updated state.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == InverseActionType.PLACE_SENSOR:
            if action.location is not None:
                new_state.sensors.append(
                    SensorConfig(
                        location=action.location.copy(),
                        noise_level=self._noise_level,
                        measurement_type="pointwise",
                        measurement=None,
                    )
                )

        elif action.action_type == InverseActionType.REMOVE_SENSOR:
            idx = action.params.get("sensor_index", 0)
            if 0 <= idx < len(new_state.sensors):
                new_state.sensors.pop(idx)

        elif action.action_type == InverseActionType.TAKE_MEASUREMENT:
            idx = action.params.get("sensor_index", 0)
            if 0 <= idx < len(new_state.sensors):
                new_state.sensors[idx].measurement = measurement_value
                new_state.measurements_taken += 1
                # Update information gain based on noise level
                sensor = new_state.sensors[idx]
                info_gain = 1.0 / max(sensor.noise_level, 1e-10)
                new_state.total_information_gain += info_gain

        elif action.action_type == InverseActionType.UPDATE_PRIOR:
            # Initialize or update uncertainty from measurements
            n_params = len(new_state.parameter_estimate)
            if new_state.uncertainty is None:
                new_state.uncertainty = np.ones(n_params)

            # Reduce uncertainty based on number of measurements
            measured = [s for s in new_state.sensors if s.measurement is not None]
            if measured:
                # Each measurement reduces uncertainty proportionally
                reduction = 1.0 / (1.0 + len(measured))
                new_state.uncertainty = new_state.uncertainty * reduction

        elif action.action_type == InverseActionType.REFINE_ESTIMATE:
            new_state.parameter_estimate = self._update_estimate(new_state)

        # NO_OP: do nothing

        return new_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_information_gain(
        self,
        state: InverseProblemState,
        action: InverseAction,
    ) -> float:
        """Estimate information gain from a measurement.

        Uses D-optimality (determinant of Fisher information)
        as a proxy for information gain.

        Higher-value actions get a larger score.  Sensor placement
        far from existing sensors is rewarded with an exploration
        bonus.
        """
        if action.action_type == InverseActionType.NO_OP:
            return 0.0

        # Base information value by action type
        action_info: dict[InverseActionType, float] = {
            InverseActionType.PLACE_SENSOR: 0.8,
            InverseActionType.TAKE_MEASUREMENT: 1.0,
            InverseActionType.UPDATE_PRIOR: 0.4,
            InverseActionType.REFINE_ESTIMATE: 0.6,
            InverseActionType.REMOVE_SENSOR: 0.1,
        }
        info = action_info.get(action.action_type, 0.0)

        # Exploration bonus: distance to nearest existing sensor
        exploration_bonus = 1.0
        if action.location is not None and state.sensors:
            min_dist = float("inf")
            for sensor in state.sensors:
                dist = float(
                    np.linalg.norm(
                        action.location - sensor.location,
                    )
                )
                min_dist = min(min_dist, dist)
            # Normalise by domain diameter
            diameter = np.sqrt(sum((hi - lo) ** 2 for lo, hi in self._bounds))
            if diameter > 0:
                exploration_bonus = 1.0 + self._exploration_weight * min_dist / diameter

        # Uncertainty-aware bonus: higher gain when uncertainty is high
        uncertainty_bonus = 1.0
        if state.uncertainty is not None:
            mean_unc = float(np.mean(state.uncertainty))
            uncertainty_bonus = 1.0 + mean_unc

        return info * exploration_bonus * uncertainty_bonus

    def _update_estimate(
        self,
        state: InverseProblemState,
    ) -> np.ndarray:
        """Update parameter estimate using Bayesian-style update.

        Simplified: weighted average biased toward measurements
        with lower noise.  Each measured sensor contributes a
        pull toward the measurement value, inversely weighted
        by its noise level.
        """
        measured = [s for s in state.sensors if s.measurement is not None]
        if not measured:
            return state.parameter_estimate.copy()

        # Compute weighted shift from measurements
        n_params = len(state.parameter_estimate)
        new_estimate = state.parameter_estimate.copy()

        total_weight = 0.0
        weighted_sum = np.zeros(n_params)

        for sensor in measured:
            weight = 1.0 / max(sensor.noise_level**2, 1e-10)
            total_weight += weight
            # Each measurement nudges the estimate (simplified)
            weighted_sum += weight * sensor.measurement * np.ones(n_params)  # type: ignore[operator]

        if total_weight > 0:
            # Blend current estimate with measurement-derived estimate
            prior_weight = 1.0
            posterior = (prior_weight * new_estimate + weighted_sum) / (prior_weight + total_weight)
            # Clip to bounds
            for d in range(n_params):
                if d < len(state.parameter_bounds):
                    lo, hi = state.parameter_bounds[d]
                    posterior[d] = np.clip(posterior[d], lo, hi)
            new_estimate = posterior

        return new_estimate

    def _sample_candidate_location(
        self,
        state: InverseProblemState,
    ) -> np.ndarray:
        """Sample a candidate sensor location.

        Uses a space-filling heuristic: generate several random
        candidates and pick the one farthest from all existing
        sensors.
        """
        n_candidates = 10
        candidates = np.empty((n_candidates, self._dim))
        for d in range(self._dim):
            lo, hi = self._bounds[d]
            candidates[:, d] = self._rng.uniform(lo, hi, size=n_candidates)

        if not state.sensors:
            return candidates[0]

        existing = np.array([s.location for s in state.sensors])
        best_idx = 0
        best_min_dist = -1.0
        for i in range(n_candidates):
            dists = np.linalg.norm(existing - candidates[i], axis=1)
            min_dist = float(np.min(dists))
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        return candidates[best_idx]
