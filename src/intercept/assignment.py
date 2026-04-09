"""Swarm-level threat-interceptor assignment.

Provides assignment solvers for optimally pairing interceptors
to threats, with triage when outnumbered and reassignment on
kill/breakoff events.

Solvers:
- HungarianAssigner: optimal assignment via scipy
- AuctionAssigner: Bertsekas auction algorithm
- GreedyAssigner: nearest-first for speed baseline

All registered via create_registry for plug-in extensibility.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from src.intercept.config import AssignmentConfig
from src.intercept.dynamics import RigidBodyState
from src.intercept.tracking import TrackState
from src.templates.registry import create_registry

logger = structlog.get_logger(__name__)


@dataclass
class AssignmentResult:
    """Result of an assignment computation.

    Attributes:
        assignments: Mapping of interceptor_id -> threat_id.
        unassigned_threats: Threat IDs with no assigned interceptor.
        unassigned_interceptors: Interceptor IDs with no assignment.
        cost: Total assignment cost.
        computation_time_ms: Wall-clock time for computation.

    """

    assignments: dict[str, str]
    unassigned_threats: list[str] = field(default_factory=list)
    unassigned_interceptors: list[str] = field(default_factory=list)
    cost: float = 0.0
    computation_time_ms: float = 0.0


class AssignmentSolver(ABC):
    """Abstract base class for assignment solvers."""

    @abstractmethod
    def solve(
        self,
        cost_matrix: NDArray[np.float64],
        threat_ids: list[str],
        interceptor_ids: list[str],
        config: AssignmentConfig,
    ) -> AssignmentResult:
        """Solve the assignment problem.

        Args:
            cost_matrix: Cost matrix (n_interceptors, n_threats).
            threat_ids: Threat identifiers.
            interceptor_ids: Interceptor identifiers.
            config: Assignment configuration.

        Returns:
            AssignmentResult with pairings.

        """
        ...


AssignmentRegistry, register_assignment = create_registry("AssignmentSolver", AssignmentSolver)


def build_cost_matrix(
    threats: list[TrackState],
    interceptors: list[RigidBodyState],
    range_weight: float = 1.0,
    closing_weight: float = 0.5,
) -> NDArray[np.float64]:
    """Build cost matrix for assignment.

    Cost = range_weight * range + closing_weight * (1 / closing_vel).
    Lower cost = better pairing.

    Args:
        threats: List of threat track states.
        interceptors: List of interceptor states.
        range_weight: Weight for range component.
        closing_weight: Weight for closing velocity component.

    Returns:
        Cost matrix (n_interceptors, n_threats).

    """
    n_int = len(interceptors)
    n_thr = len(threats)
    cost = np.zeros((n_int, n_thr), dtype=np.float64)

    for i, intc in enumerate(interceptors):
        for j, thr in enumerate(threats):
            rel_pos = thr.position - intc.position
            range_m = torch.norm(rel_pos).item()

            rel_vel = thr.velocity - intc.velocity
            los = rel_pos / (range_m + 1e-12)
            closing = -torch.dot(rel_vel, los).item()

            range_cost = range_weight * range_m
            closing_cost = closing_weight * (1000.0 / (max(closing, 1.0)))

            cost[i, j] = range_cost + closing_cost

    return cost


@register_assignment("hungarian")
class HungarianAssigner(AssignmentSolver):
    """Optimal assignment via Hungarian algorithm.

    Uses scipy.optimize.linear_sum_assignment for O(n^3) optimal
    assignment. Handles rectangular matrices (more threats than
    interceptors or vice versa).
    """

    def solve(
        self,
        cost_matrix: NDArray[np.float64],
        threat_ids: list[str],
        interceptor_ids: list[str],
        config: AssignmentConfig,
    ) -> AssignmentResult:
        from scipy.optimize import linear_sum_assignment

        t0 = time.monotonic()

        n_int, n_thr = cost_matrix.shape
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: dict[str, str] = {}
        total_cost = 0.0
        assigned_threats = set()

        for r, c in zip(row_ind, col_ind, strict=True):
            assignments[interceptor_ids[r]] = threat_ids[c]
            total_cost += cost_matrix[r, c]
            assigned_threats.add(threat_ids[c])

        unassigned_threats = [t for t in threat_ids if t not in assigned_threats]
        assigned_ints = set(assignments.keys())
        unassigned_interceptors = [i for i in interceptor_ids if i not in assigned_ints]

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        return AssignmentResult(
            assignments=assignments,
            unassigned_threats=unassigned_threats,
            unassigned_interceptors=unassigned_interceptors,
            cost=total_cost,
            computation_time_ms=elapsed_ms,
        )


@register_assignment("greedy")
class GreedyAssigner(AssignmentSolver):
    """Nearest-first greedy assignment.

    Fast O(n*m) assignment that pairs each interceptor with the
    cheapest remaining threat. Not optimal but very fast.
    """

    def solve(
        self,
        cost_matrix: NDArray[np.float64],
        threat_ids: list[str],
        interceptor_ids: list[str],
        config: AssignmentConfig,
    ) -> AssignmentResult:
        t0 = time.monotonic()

        n_int, n_thr = cost_matrix.shape
        assignments: dict[str, str] = {}
        assigned_threats: set[int] = set()
        total_cost = 0.0

        # Sort interceptors by minimum cost to any threat
        int_order = np.argsort(cost_matrix.min(axis=1))

        for i in int_order:
            if len(assigned_threats) >= n_thr:
                break
            # Find cheapest unassigned threat
            best_j = -1
            best_cost = float("inf")
            for j in range(n_thr):
                if j not in assigned_threats and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                assignments[interceptor_ids[i]] = threat_ids[best_j]
                assigned_threats.add(best_j)
                total_cost += best_cost

        unassigned_threats = [threat_ids[j] for j in range(n_thr) if j not in assigned_threats]
        assigned_ints = set(assignments.keys())
        unassigned_interceptors = [i for i in interceptor_ids if i not in assigned_ints]

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        return AssignmentResult(
            assignments=assignments,
            unassigned_threats=unassigned_threats,
            unassigned_interceptors=unassigned_interceptors,
            cost=total_cost,
            computation_time_ms=elapsed_ms,
        )


@register_assignment("auction")
class AuctionAssigner(AssignmentSolver):
    """Bertsekas auction algorithm for assignment.

    Iterative bidding algorithm. Each interceptor bids on its
    preferred threat; prices adjust until equilibrium. Good for
    distributed systems where interceptors decide locally.
    """

    def solve(
        self,
        cost_matrix: NDArray[np.float64],
        threat_ids: list[str],
        interceptor_ids: list[str],
        config: AssignmentConfig,
        epsilon: float = 0.01,
        max_iter: int = 1000,
    ) -> AssignmentResult:
        t0 = time.monotonic()

        n_int, n_thr = cost_matrix.shape
        # Convert cost to benefit (negate) for auction
        benefit = -cost_matrix

        prices = np.zeros(n_thr)
        assignment = np.full(n_int, -1, dtype=int)
        threat_owner = np.full(n_thr, -1, dtype=int)

        for _ in range(max_iter):
            unassigned = [i for i in range(n_int) if assignment[i] == -1]
            if not unassigned:
                break

            for i in unassigned:
                # Compute net values
                values = benefit[i] - prices
                # Find best and second-best
                sorted_idx = np.argsort(-values)
                best_j = sorted_idx[0]
                best_val = values[best_j]
                second_val = values[sorted_idx[1]] if len(sorted_idx) > 1 else best_val - 1

                # Bid
                bid_increment = best_val - second_val + epsilon

                # Check if threat is already assigned
                prev_owner = threat_owner[best_j]
                if prev_owner >= 0:
                    assignment[prev_owner] = -1

                assignment[i] = best_j
                threat_owner[best_j] = i
                prices[best_j] += bid_increment

        # Build result
        assignments: dict[str, str] = {}
        total_cost = 0.0
        assigned_threats: set[int] = set()

        for i in range(n_int):
            j = assignment[i]
            if j >= 0 and j < n_thr:
                assignments[interceptor_ids[i]] = threat_ids[j]
                total_cost += cost_matrix[i, j]
                assigned_threats.add(j)

        unassigned_threats = [threat_ids[j] for j in range(n_thr) if j not in assigned_threats]
        assigned_ints = set(assignments.keys())
        unassigned_interceptors = [i for i in interceptor_ids if i not in assigned_ints]

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        return AssignmentResult(
            assignments=assignments,
            unassigned_threats=unassigned_threats,
            unassigned_interceptors=unassigned_interceptors,
            cost=total_cost,
            computation_time_ms=elapsed_ms,
        )


class TriageLogic:
    """Prioritize threats when outnumbered.

    Ranks threats by severity and drops lowest-priority threats
    when there are more threats than interceptors.
    """

    @staticmethod
    def prioritize(
        threats: list[TrackState],
        defended_position: NDArray[np.float64] | None = None,
    ) -> list[int]:
        """Rank threats by severity (highest first).

        Severity = 1 / (range_to_defended * time_to_arrive).
        Closer and faster threats rank higher.

        Args:
            threats: List of threat track states.
            defended_position: Position to defend in NED (m).

        Returns:
            Indices sorted by decreasing severity.

        """
        if defended_position is None:
            defended_position = np.zeros(3, dtype=np.float64)

        def_pos = torch.tensor(defended_position, dtype=torch.float64)
        severities = []

        for thr in threats:
            rel = def_pos - thr.position
            range_m = torch.norm(rel).item()
            speed = torch.norm(thr.velocity).item()
            time_to_arrive = range_m / (speed + 1e-6)
            severity = 1.0 / (range_m * time_to_arrive + 1e-6)
            severities.append(severity)

        return list(np.argsort(severities)[::-1])

    @staticmethod
    def triage(
        threats: list[TrackState],
        n_interceptors: int,
        defended_position: NDArray[np.float64] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Select which threats to engage and which to drop.

        Args:
            threats: All detected threats.
            n_interceptors: Number of available interceptors.
            defended_position: Position to defend.

        Returns:
            (engaged_indices, dropped_indices).

        """
        priority = TriageLogic.prioritize(threats, defended_position)
        engaged = priority[:n_interceptors]
        dropped = priority[n_interceptors:]
        return engaged, dropped


class ReassignmentManager:
    """Manages dynamic reassignment on engagement events."""

    def __init__(
        self,
        solver: AssignmentSolver,
        config: AssignmentConfig,
    ) -> None:
        self.solver = solver
        self.config = config
        self._last_assignment_time = -float("inf")

    def reassign(
        self,
        threats: list[TrackState],
        interceptors: list[RigidBodyState],
        threat_ids: list[str],
        interceptor_ids: list[str],
        current_time: float,
    ) -> AssignmentResult | None:
        """Recompute assignment if debounce interval has passed.

        Args:
            threats: Current threat states.
            interceptors: Current interceptor states.
            threat_ids: Threat identifiers.
            interceptor_ids: Interceptor identifiers.
            current_time: Current simulation time.

        Returns:
            New assignment or None if too soon to reassign.

        """
        if current_time - self._last_assignment_time < self.config.reassignment_interval_s:
            return None

        cost = build_cost_matrix(threats, interceptors)
        result = self.solver.solve(cost, threat_ids, interceptor_ids, self.config)
        self._last_assignment_time = current_time

        logger.info(
            "reassignment_computed",
            n_assigned=len(result.assignments),
            n_unassigned_threats=len(result.unassigned_threats),
            cost=result.cost,
            time_ms=result.computation_time_ms,
        )

        return result
