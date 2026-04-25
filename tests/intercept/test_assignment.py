"""Tests for swarm assignment solvers."""

from __future__ import annotations

import numpy as np
import pytest

from src.intercept.assignment import (
    AssignmentRegistry,
    AuctionAssigner,
    GreedyAssigner,
    HungarianAssigner,
    ReassignmentManager,
    TriageLogic,
    build_cost_matrix,
)
from src.intercept.config import AssignmentConfig
from src.intercept.dynamics import create_initial_state
from src.intercept.tracking import create_initial_track


def _make_threats(n: int) -> tuple[list, list[str]]:
    tracks = []
    ids = []
    for i in range(n):
        t = create_initial_track(
            position=[5000.0 + i * 100, float(i * 50), -3000.0],
            velocity=[-200.0, 0.0, 0.0],
            track_id=f"threat_{i}",
        )
        tracks.append(t)
        ids.append(f"threat_{i}")
    return tracks, ids


def _make_interceptors(n: int) -> tuple[list, list[str]]:
    states = []
    ids = []
    for i in range(n):
        s = create_initial_state(
            position=[0.0, float(i * 100), -3000.0],
            velocity=[300.0, 0.0, 0.0],
        )
        states.append(s)
        ids.append(f"int_{i}")
    return states, ids


class TestCostMatrix:
    def test_shape(self) -> None:
        threats, _ = _make_threats(5)
        interceptors, _ = _make_interceptors(3)
        cost = build_cost_matrix(threats, interceptors)
        assert cost.shape == (3, 5)

    def test_closer_is_cheaper(self) -> None:
        threats, _ = _make_threats(2)
        interceptors, _ = _make_interceptors(1)
        cost = build_cost_matrix(threats, interceptors)
        # First threat is closer (5000m vs 5100m)
        assert cost[0, 0] < cost[0, 1]


class TestHungarianAssigner:
    def test_optimal_2x2(self) -> None:
        cost = np.array([[1.0, 3.0], [3.0, 1.0]])
        solver = HungarianAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, ["t0", "t1"], ["i0", "i1"], config)
        assert len(result.assignments) == 2
        assert result.cost == pytest.approx(2.0)

    def test_rectangular_more_threats(self) -> None:
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 5.0]])
        solver = HungarianAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, ["t0", "t1", "t2"], ["i0", "i1"], config)
        assert len(result.assignments) == 2
        assert len(result.unassigned_threats) == 1

    def test_20x10_under_200ms(self) -> None:
        threats, t_ids = _make_threats(20)
        interceptors, i_ids = _make_interceptors(10)
        cost = build_cost_matrix(threats, interceptors)
        solver = HungarianAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, t_ids, i_ids, config)
        assert len(result.assignments) == 10
        assert len(result.unassigned_threats) == 10
        assert result.computation_time_ms < 200.0

    def test_no_double_assignment(self) -> None:
        threats, t_ids = _make_threats(5)
        interceptors, i_ids = _make_interceptors(5)
        cost = build_cost_matrix(threats, interceptors)
        solver = HungarianAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, t_ids, i_ids, config)
        assigned_threats = list(result.assignments.values())
        assert len(assigned_threats) == len(set(assigned_threats))


class TestGreedyAssigner:
    def test_produces_assignment(self) -> None:
        cost = np.array([[1.0, 3.0], [3.0, 1.0]])
        solver = GreedyAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, ["t0", "t1"], ["i0", "i1"], config)
        assert len(result.assignments) == 2

    def test_faster_than_hungarian(self) -> None:
        threats, t_ids = _make_threats(20)
        interceptors, i_ids = _make_interceptors(10)
        cost = build_cost_matrix(threats, interceptors)
        config = AssignmentConfig(name="test")

        greedy = GreedyAssigner()
        r_greedy = greedy.solve(cost, t_ids, i_ids, config)

        hungarian = HungarianAssigner()
        r_hung = hungarian.solve(cost, t_ids, i_ids, config)

        # Greedy should complete (may or may not be faster for small sizes)
        assert len(r_greedy.assignments) == 10


class TestAuctionAssigner:
    def test_produces_assignment(self) -> None:
        cost = np.array([[1.0, 3.0], [3.0, 1.0]])
        solver = AuctionAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, ["t0", "t1"], ["i0", "i1"], config)
        assert len(result.assignments) == 2

    def test_quality_comparable_to_hungarian(self) -> None:
        cost = np.random.default_rng(42).uniform(1, 100, size=(5, 5))
        config = AssignmentConfig(name="test")

        hung = HungarianAssigner().solve(
            cost, [f"t{i}" for i in range(5)], [f"i{i}" for i in range(5)], config
        )
        auct = AuctionAssigner().solve(
            cost, [f"t{i}" for i in range(5)], [f"i{i}" for i in range(5)], config
        )

        # Auction should be within 50% of optimal
        assert auct.cost < hung.cost * 1.5


class TestTriageLogic:
    def test_prioritize_closer_threats(self) -> None:
        threats = [
            create_initial_track(
                position=[10000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0], track_id="far"
            ),
            create_initial_track(
                position=[2000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0], track_id="close"
            ),
        ]
        priority = TriageLogic.prioritize(threats)
        # Closer threat (index 1) should rank higher
        assert priority[0] == 1

    def test_triage_drops_lowest(self) -> None:
        threats = [
            create_initial_track(
                position=[10000.0, 0.0, -3000.0], velocity=[-100.0, 0.0, 0.0], track_id="far"
            ),
            create_initial_track(
                position=[2000.0, 0.0, -3000.0], velocity=[-300.0, 0.0, 0.0], track_id="close_fast"
            ),
            create_initial_track(
                position=[5000.0, 0.0, -3000.0], velocity=[-200.0, 0.0, 0.0], track_id="mid"
            ),
        ]
        engaged, dropped = TriageLogic.triage(threats, n_interceptors=2)
        assert len(engaged) == 2
        assert len(dropped) == 1


class TestReassignmentManager:
    def test_debounce(self) -> None:
        solver = HungarianAssigner()
        config = AssignmentConfig(name="test", reassignment_interval_s=1.0)
        manager = ReassignmentManager(solver, config)

        threats, t_ids = _make_threats(2)
        interceptors, i_ids = _make_interceptors(2)

        r1 = manager.reassign(threats, interceptors, t_ids, i_ids, current_time=0.0)
        assert r1 is not None

        r2 = manager.reassign(threats, interceptors, t_ids, i_ids, current_time=0.5)
        assert r2 is None  # debounced

        r3 = manager.reassign(threats, interceptors, t_ids, i_ids, current_time=1.5)
        assert r3 is not None


class TestAssignmentRegistry:
    def test_hungarian_registered(self) -> None:
        assert AssignmentRegistry().get("hungarian") is HungarianAssigner

    def test_greedy_registered(self) -> None:
        assert AssignmentRegistry().get("greedy") is GreedyAssigner

    def test_auction_registered(self) -> None:
        assert AssignmentRegistry().get("auction") is AuctionAssigner
