"""Unit tests for ``src.poc.visualization.pareto``."""

from __future__ import annotations

from src.poc.visualization.pareto import collect_points, compute_pareto_front


class TestComputeParetoFront:
    def test_empty(self):
        assert compute_pareto_front([]) == []

    def test_single_point(self):
        assert compute_pareto_front([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_all_on_front(self):
        pts = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]
        front = compute_pareto_front(pts)
        assert set(front) == set(pts)
        # Sorted ascending by x
        assert front == sorted(front)

    def test_dominated_points_excluded(self):
        pts = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        front = compute_pareto_front(pts)
        assert front == [(1.0, 1.0)]

    def test_mixed(self):
        pts = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (2.5, 5.0)]
        front = compute_pareto_front(pts)
        # (2.5, 5.0) is dominated by every other point; the rest are incomparable
        assert (2.5, 5.0) not in front
        assert (1.0, 4.0) in front
        assert (2.0, 3.0) in front
        assert (3.0, 2.0) in front

    def test_duplicates_preserved(self):
        """When duplicates exist we keep at least one representative on the front."""
        pts = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]
        front = compute_pareto_front(pts)
        assert (1.0, 1.0) in front
        assert (2.0, 2.0) not in front


class TestCollectPoints:
    def test_flattens_parallel_lists(self):
        methods = {
            "a": {"t": [1.0, 2.0], "e": [0.1, 0.2]},
            "b": {"t": [3.0], "e": [0.3]},
        }
        points = list(collect_points(methods, "t", "e"))
        assert points == [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]

    def test_empty_methods(self):
        assert list(collect_points({}, "x", "y")) == []
