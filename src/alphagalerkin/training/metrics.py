"""Training metric collectors (Observer pattern)."""

from __future__ import annotations

from collections import defaultdict


class MetricCollector:
    """Collects and aggregates training metrics.

    Implements the Observer pattern for metric reporting: any
    component can call ``record`` to push a named scalar, and
    the collector maintains both a full history and a per-
    iteration snapshot.

    Example::

        metrics = MetricCollector()
        metrics.record("loss", 0.42)
        metrics.record("loss", 0.38)
        avg = metrics.get_average("loss", window=2)
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._iteration_metrics: dict[str, float] = {}

    # ---------------------------------------------------------------
    # Recording
    # ---------------------------------------------------------------

    def record(self, name: str, value: float) -> None:
        """Record a single metric value.

        Parameters
        ----------
        name:
            Metric name (e.g. ``"training/loss"``).
        value:
            Scalar value to record.

        """
        self._metrics[name].append(value)
        self._iteration_metrics[name] = value

    # ---------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------

    def get_latest(self, name: str) -> float | None:
        """Return the most recently recorded value for *name*.

        Returns ``None`` if no values have been recorded.
        """
        values = self._metrics.get(name)
        if values:
            return values[-1]
        return None

    def get_average(
        self,
        name: str,
        window: int = 10,
    ) -> float | None:
        """Return a windowed running average of *name*.

        Parameters
        ----------
        name:
            Metric name.
        window:
            Number of most-recent values to average over.

        Returns
        -------
        float | None
            The average, or ``None`` if no values exist.

        """
        values = self._metrics.get(name)
        if not values:
            return None
        recent = values[-window:]
        return sum(recent) / len(recent)

    def get_iteration_summary(self) -> dict[str, float]:
        """Return all metrics recorded since the last call.

        Each call drains the per-iteration accumulator, so
        calling it twice in a row yields an empty dict.
        """
        summary = dict(self._iteration_metrics)
        self._iteration_metrics.clear()
        return summary

    def get_full_history(self) -> dict[str, list[float]]:
        """Return the complete metric history for all names."""
        return dict(self._metrics)

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._metrics.clear()
        self._iteration_metrics.clear()
