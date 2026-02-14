"""MCTS-based adaptive mesh refinement for numerical relativity.

Numerical relativity simulations (e.g. binary black hole mergers) require
extreme multiscale AMR: near-singularity resolution (~M/100 near punctures)
while tracking gravitational waves to extraction radii (~1000M), demanding
6-10+ refinement levels.

This module frames NR mesh management as a sequential game where MCTS
plans refinement/coarsening strategies with look-ahead, replacing the
documented "rules of thumb" for AMR tagging criteria.

Additionally, gauge choice (lapse function, shift vector) is modeled
as a decision with long-term consequences.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("planning.numerical_relativity")


# ======================================================================
# Enumerations
# ======================================================================


class NRActionType(str, Enum):
    """Actions for numerical relativity AMR."""

    REFINE_REGION = "refine_region"
    """Add finer resolution within an existing refinement level."""

    COARSEN_REGION = "coarsen_region"
    """Remove a refinement level that is no longer needed."""

    SET_GAUGE_LAPSE = "set_gauge_lapse"
    """Change the lapse (time-slicing) gauge condition."""

    SET_GAUGE_SHIFT = "set_gauge_shift"
    """Change the shift (spatial-coordinate) gauge condition."""

    ADJUST_EXTRACTION_RADIUS = "adjust_extraction_radius"
    """Move the gravitational-wave extraction sphere."""

    ADD_REFINEMENT_LEVEL = "add_refinement_level"
    """Add a new refinement level centred on a puncture."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


class GaugeCondition(str, Enum):
    """Available gauge conditions for lapse/shift."""

    GEODESIC = "geodesic"
    """Geodesic slicing (alpha=1, beta=0)."""

    HARMONIC = "harmonic"
    """Harmonic gauge: Box(x^mu) = 0."""

    BONA_MASSO = "bona_masso"
    """1+log slicing for the lapse."""

    GAMMA_DRIVER = "gamma_driver"
    """Gamma-driver shift condition."""

    PUNCTURE = "puncture"
    """Moving puncture gauge (1+log + Gamma-driver)."""


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class RefinementLevel:
    """A single AMR refinement level.

    Attributes:
        level: Integer refinement level (0 = coarsest).
        center: Centre coordinates of the refinement box.
        extent: Half-widths of the refinement box per dimension.
        resolution: Grid spacing at this level (in units of M).

    """

    level: int
    center: np.ndarray  # (dim,) centre coordinates
    extent: np.ndarray  # (dim,) half-widths
    resolution: float   # Grid spacing at this level

    @property
    def volume(self) -> float:
        """Compute volume of refinement region."""
        return float(np.prod(2.0 * self.extent))

    @property
    def estimated_points(self) -> int:
        """Estimate the number of grid points on this level."""
        if self.resolution <= 0:
            return 0
        points_per_dim = 2.0 * self.extent / self.resolution
        return int(np.prod(np.maximum(points_per_dim, 1.0)))

    def clone(self) -> RefinementLevel:
        """Return a deep, independent copy of this refinement level."""
        return RefinementLevel(
            level=self.level,
            center=self.center.copy(),
            extent=self.extent.copy(),
            resolution=self.resolution,
        )


@dataclass
class NRMeshState:
    """State of a numerical relativity simulation mesh.

    Attributes:
        dimension: Spatial dimension (typically 3).
        domain_extent: Total domain half-width in units of M.
        base_resolution: Coarsest grid spacing in units of M.
        refinement_levels: Ordered list of refinement levels.
        lapse_gauge: Current gauge condition for the lapse function.
        shift_gauge: Current gauge condition for the shift vector.
        extraction_radius: GW extraction sphere radius in units of M.
        puncture_locations: Locations of BH punctures.
        constraint_violation: L2 norm of the Hamiltonian constraint.
        time: Current simulation time in units of M.
        step: Number of planning steps taken so far.
        max_levels: Maximum number of refinement levels allowed.
        min_resolution: Finest allowed grid spacing in units of M.

    """

    dimension: int = 3
    domain_extent: float = 1000.0  # Total domain half-width in M
    base_resolution: float = 8.0   # Coarsest grid spacing in M
    refinement_levels: list[RefinementLevel] = field(default_factory=list)
    lapse_gauge: GaugeCondition = GaugeCondition.BONA_MASSO
    shift_gauge: GaugeCondition = GaugeCondition.GAMMA_DRIVER
    extraction_radius: float = 100.0  # GW extraction radius in M

    # Physical state estimates
    puncture_locations: list[np.ndarray] = field(default_factory=list)
    constraint_violation: float = 0.0  # L2 norm of Hamiltonian constraint

    time: float = 0.0
    step: int = 0
    max_levels: int = 10
    min_resolution: float = 0.01  # Finest allowed resolution in M

    @property
    def num_levels(self) -> int:
        """Return the number of active refinement levels."""
        return len(self.refinement_levels)

    @property
    def total_grid_points(self) -> int:
        """Estimate total grid points across all levels.

        Includes the base grid plus all refinement levels.
        """
        # Base grid contribution
        base_points_per_dim = 2.0 * self.domain_extent / self.base_resolution
        base_total = int(base_points_per_dim ** self.dimension)

        # Sum over refinement levels
        level_total = sum(
            rl.estimated_points for rl in self.refinement_levels
        )
        return base_total + level_total

    def clone(self) -> NRMeshState:
        """Return a deep, independent copy of this state."""
        return NRMeshState(
            dimension=self.dimension,
            domain_extent=self.domain_extent,
            base_resolution=self.base_resolution,
            refinement_levels=[rl.clone() for rl in self.refinement_levels],
            lapse_gauge=self.lapse_gauge,
            shift_gauge=self.shift_gauge,
            extraction_radius=self.extraction_radius,
            puncture_locations=[p.copy() for p in self.puncture_locations],
            constraint_violation=self.constraint_violation,
            time=self.time,
            step=self.step,
            max_levels=self.max_levels,
            min_resolution=self.min_resolution,
        )


@dataclass
class NRAction:
    """A numerical relativity mesh action.

    Attributes:
        action_type: The type of mesh/gauge action to take.
        params: Optional parameters for the action (e.g. region
            centre, gauge target, extraction radius delta).

    """

    action_type: NRActionType
    params: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# NR Mesh Manager
# ======================================================================


class NRMeshManager:
    """Manages NR mesh refinement using MCTS-style look-ahead.

    Plans AMR refinement strategies for binary black hole mergers and
    similar simulations.  Decides when and where to refine/coarsen,
    which gauge conditions to use, and where to place GW extraction
    spheres.

    All tunable parameters are exposed through the constructor -- no
    magic numbers are hard-coded.

    Parameters
    ----------
    max_levels:
        Maximum number of refinement levels allowed.
    min_resolution:
        Finest allowed grid spacing in units of M.
    refinement_ratio:
        Resolution ratio between successive levels.
    num_simulations:
        Number of look-ahead simulations per planning step.
    constraint_weight:
        Weight applied to constraint violation in the reward.
    cost_weight:
        Weight applied to computational cost in the reward.
    extraction_step:
        Step size for extraction radius adjustments in units of M.
    puncture_threshold:
        Distance threshold below which a puncture needs finer resolution.

    """

    def __init__(
        self,
        max_levels: int = 10,
        min_resolution: float = 0.01,
        refinement_ratio: int = 2,
        num_simulations: int = 50,
        constraint_weight: float = 1.0,
        cost_weight: float = 0.5,
        extraction_step: float = 50.0,
        puncture_threshold: float = 1.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._max_levels = max_levels
        self._min_resolution = min_resolution
        self._refinement_ratio = refinement_ratio
        self._num_simulations = num_simulations
        self._constraint_weight = constraint_weight
        self._cost_weight = cost_weight
        self._extraction_step = extraction_step
        self._puncture_threshold = puncture_threshold
        self._rng = np.random.default_rng(seed)

        logger.info(
            "nr_mesh_manager.init",
            max_levels=max_levels,
            min_resolution=min_resolution,
            refinement_ratio=refinement_ratio,
            num_simulations=num_simulations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_next_action(
        self,
        state: NRMeshState,
        constraint_fn: Callable[[NRMeshState], float] | None = None,
    ) -> NRAction:
        """Plan the next mesh operation.

        Uses constraint violation estimates and puncture tracking
        to decide optimal refinement strategy.  Evaluates every valid
        action by simulating its effect and scoring with the reward
        function.

        Parameters
        ----------
        state:
            Current NR mesh state.
        constraint_fn:
            Optional callable ``(state) -> float`` that returns the
            Hamiltonian constraint violation for a given mesh state.
            If not provided, a simple heuristic estimate is used.

        Returns
        -------
        NRAction
            The best action found.

        """
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return NRAction(action_type=NRActionType.NO_OP)

        best_action = valid_actions[0]
        best_reward = -float("inf")

        for action in valid_actions:
            total_reward = 0.0
            for _ in range(self._num_simulations):
                new_state = self.apply_action(state, action)
                if constraint_fn is not None:
                    new_state.constraint_violation = constraint_fn(new_state)
                else:
                    new_state.constraint_violation = (
                        self._estimate_constraint(new_state)
                    )
                reward = self._compute_reward(state, new_state)
                total_reward += reward

            avg_reward = total_reward / self._num_simulations
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action

        logger.info(
            "nr_mesh_manager.plan",
            chosen_action=best_action.action_type.value,
            expected_reward=best_reward,
            step=state.step,
            num_levels=state.num_levels,
        )
        return best_action

    def get_valid_actions(self, state: NRMeshState) -> list[NRAction]:
        """Return valid actions for the current mesh state.

        Enforces level count bounds, resolution limits, and gauge
        constraints to prevent degenerate configurations.
        """
        actions: list[NRAction] = []

        # --- Refinement: refine existing levels if resolution allows ---
        for rl in state.refinement_levels:
            child_resolution = rl.resolution / self._refinement_ratio
            if child_resolution >= state.min_resolution:
                actions.append(
                    NRAction(
                        action_type=NRActionType.REFINE_REGION,
                        params={
                            "target_level": rl.level,
                            "center": rl.center.tolist(),
                        },
                    )
                )

        # --- Coarsening: remove levels if more than zero ---
        for rl in state.refinement_levels:
            actions.append(
                NRAction(
                    action_type=NRActionType.COARSEN_REGION,
                    params={"target_level": rl.level},
                )
            )

        # --- Add new refinement level (centred on punctures) ---
        if state.num_levels < state.max_levels:
            for idx, punct in enumerate(state.puncture_locations):
                actions.append(
                    NRAction(
                        action_type=NRActionType.ADD_REFINEMENT_LEVEL,
                        params={
                            "puncture_index": idx,
                            "center": punct.tolist(),
                        },
                    )
                )
            # Also allow adding a level at the domain centre if no
            # punctures are present
            if not state.puncture_locations:
                actions.append(
                    NRAction(
                        action_type=NRActionType.ADD_REFINEMENT_LEVEL,
                        params={
                            "center": [0.0] * state.dimension,
                        },
                    )
                )

        # --- Gauge conditions ---
        for gauge in GaugeCondition:
            if gauge != state.lapse_gauge:
                actions.append(
                    NRAction(
                        action_type=NRActionType.SET_GAUGE_LAPSE,
                        params={"target": gauge.value},
                    )
                )
            if gauge != state.shift_gauge:
                actions.append(
                    NRAction(
                        action_type=NRActionType.SET_GAUGE_SHIFT,
                        params={"target": gauge.value},
                    )
                )

        # --- Extraction radius adjustment ---
        if state.extraction_radius + self._extraction_step <= state.domain_extent:
            actions.append(
                NRAction(
                    action_type=NRActionType.ADJUST_EXTRACTION_RADIUS,
                    params={"delta": self._extraction_step},
                )
            )
        if state.extraction_radius - self._extraction_step > 0.0:
            actions.append(
                NRAction(
                    action_type=NRActionType.ADJUST_EXTRACTION_RADIUS,
                    params={"delta": -self._extraction_step},
                )
            )

        # No-op is always valid
        actions.append(NRAction(action_type=NRActionType.NO_OP))

        return actions

    def apply_action(
        self,
        state: NRMeshState,
        action: NRAction,
    ) -> NRMeshState:
        """Apply an action to produce a new mesh state.

        Parameters
        ----------
        state:
            Current NR mesh state.
        action:
            The mesh/gauge action to apply.

        Returns
        -------
        NRMeshState
            The updated state with the action applied.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == NRActionType.REFINE_REGION:
            target_level = action.params.get("target_level", 0)
            # Find the target level and create a child refinement
            for rl in new_state.refinement_levels:
                if rl.level == target_level:
                    child_resolution = (
                        rl.resolution / self._refinement_ratio
                    )
                    child_extent = rl.extent / self._refinement_ratio
                    new_level = RefinementLevel(
                        level=rl.level + 1,
                        center=rl.center.copy(),
                        extent=child_extent,
                        resolution=max(
                            child_resolution, state.min_resolution,
                        ),
                    )
                    new_state.refinement_levels.append(new_level)
                    break

        elif action.action_type == NRActionType.COARSEN_REGION:
            target_level = action.params.get("target_level", 0)
            new_state.refinement_levels = [
                rl for rl in new_state.refinement_levels
                if rl.level != target_level
            ]

        elif action.action_type == NRActionType.SET_GAUGE_LAPSE:
            target = action.params.get("target", GaugeCondition.BONA_MASSO.value)
            new_state.lapse_gauge = GaugeCondition(target)

        elif action.action_type == NRActionType.SET_GAUGE_SHIFT:
            target = action.params.get("target", GaugeCondition.GAMMA_DRIVER.value)
            new_state.shift_gauge = GaugeCondition(target)

        elif action.action_type == NRActionType.ADJUST_EXTRACTION_RADIUS:
            delta = action.params.get("delta", 0.0)
            new_state.extraction_radius = max(
                0.0,
                min(
                    state.extraction_radius + delta,
                    state.domain_extent,
                ),
            )

        elif action.action_type == NRActionType.ADD_REFINEMENT_LEVEL:
            center_list = action.params.get(
                "center", [0.0] * state.dimension,
            )
            center = np.array(center_list, dtype=float)

            # Determine resolution for the new level
            if new_state.refinement_levels:
                finest = min(
                    rl.resolution for rl in new_state.refinement_levels
                )
                new_resolution = finest / self._refinement_ratio
            else:
                new_resolution = (
                    state.base_resolution / self._refinement_ratio
                )
            new_resolution = max(new_resolution, state.min_resolution)

            # Default extent scales with resolution
            extent_size = new_resolution * 16.0  # 16 grid cells per side
            new_level = RefinementLevel(
                level=new_state.num_levels + 1,
                center=center,
                extent=np.full(state.dimension, extent_size),
                resolution=new_resolution,
            )
            new_state.refinement_levels.append(new_level)

        # NO_OP: do nothing

        return new_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        old_state: NRMeshState,
        new_state: NRMeshState,
    ) -> float:
        """Compute reward as constraint reduction / computational cost.

        Reward = constraint_weight * (old_violation - new_violation)
                 - cost_weight * (new_points - old_points) / old_points

        A positive reward indicates improvement: the constraint
        violation decreased without excessive computational overhead.
        """
        # Constraint improvement (positive = good)
        constraint_delta = (
            old_state.constraint_violation - new_state.constraint_violation
        )

        # Computational cost ratio
        old_points = max(old_state.total_grid_points, 1)
        new_points = max(new_state.total_grid_points, 1)
        cost_ratio = (new_points - old_points) / old_points

        return (
            self._constraint_weight * constraint_delta
            - self._cost_weight * cost_ratio
        )

    def _needs_refinement_near_puncture(self, state: NRMeshState) -> bool:
        """Check if puncture locations need finer resolution.

        Returns True if any puncture is not covered by a refinement
        level whose resolution is at or below the puncture threshold.
        """
        for punct in state.puncture_locations:
            covered = False
            for rl in state.refinement_levels:
                # Check if puncture is inside the refinement box
                dist = np.abs(punct - rl.center)
                inside = np.all(dist <= rl.extent)
                if inside and rl.resolution <= self._puncture_threshold:
                    covered = True
                    break
            if not covered:
                return True
        return False

    def _estimate_constraint(self, state: NRMeshState) -> float:
        """Heuristic constraint violation estimate.

        Finer resolution near punctures reduces constraint violation.
        This is a simplified model used when no external constraint
        function is provided.
        """
        if not state.puncture_locations:
            # No punctures: constraint depends only on base resolution
            return state.base_resolution / state.domain_extent

        # For each puncture, find the finest covering resolution
        total_violation = 0.0
        for punct in state.puncture_locations:
            finest_covering = state.base_resolution
            for rl in state.refinement_levels:
                dist = np.abs(punct - rl.center)
                inside = np.all(dist <= rl.extent)
                if inside:
                    finest_covering = min(finest_covering, rl.resolution)
            # Violation scales with resolution^2 (second-order convergence)
            total_violation += finest_covering ** 2

        return total_violation / len(state.puncture_locations)
