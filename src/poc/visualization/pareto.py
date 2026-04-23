"""Pareto frontier computation utilities.

Extracted from :class:`src.poc.visualization.plots.ParetoFrontierPlot`
so the domination logic is independently testable and reusable by
other plot types, CSV post-processors, and benchmark runners.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def compute_pareto_front(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the Pareto-minimal subset of ``points``.

    A point ``(x, y)`` is Pareto-minimal if no other point dominates it —
    where ``(x', y')`` dominates ``(x, y)`` iff ``x' <= x`` and
    ``y' <= y`` with at least one strict inequality.  Typical use is
    (compute_cost, error) or (latency, loss) pairs where smaller is
    better in both axes.

    Args:
        points: Iterable of 2-tuples.  Duplicates are preserved.

    Returns:
        List of Pareto-minimal points sorted ascending by the first
        coordinate for plotting convenience.

    Examples:
        >>> compute_pareto_front([(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)])
        [(1.0, 2.0), (2.0, 1.0)]

    """
    unique_points = list(points)
    front: list[tuple[float, float]] = []
    for t, e in unique_points:
        dominated = any(
            (ot < t and oe <= e) or (ot <= t and oe < e)
            for ot, oe in unique_points
            if (ot, oe) != (t, e)
        )
        if not dominated:
            front.append((t, e))
    front.sort()
    return front


def collect_points(
    methods: dict[str, dict[str, list[float]]],
    x_key: str,
    y_key: str,
) -> Iterable[tuple[float, float]]:
    """Flatten a ``{method: {x_key: [...], y_key: [...]}}`` payload to points.

    Args:
        methods: Dict mapping method name to a dict of parallel lists.
        x_key: Key whose list provides the x-axis coordinates.
        y_key: Key whose list provides the y-axis coordinates.

    Yields:
        ``(x, y)`` tuples in method-declaration order.

    """
    for method_data in methods.values():
        xs = method_data[x_key]
        ys = method_data[y_key]
        yield from zip(xs, ys, strict=True)
