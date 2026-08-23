"""Statistics helpers for the perf benchmark.

Pure functions and a small dataclass — no side effects, no I/O. Kept
separate from ``benchmark.py`` so the math is unit-testable without
spinning up a codec.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import structlog

logger = structlog.get_logger(__name__)

# Percentile points reported in every cell. Surfacing them as a constant
# (rather than scattering 50 / 90 / 99 across the codebase) means the same
# percentile set is rendered into baselines, reports, and regression
# checks. Adding p95 in the future is a one-line edit here plus a new
# ``BaselineEntry`` field.
DEFAULT_PERCENTILES: tuple[int, ...] = (50, 90, 99)


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics over a list of per-iteration latencies.

    All fields are in milliseconds. ``count`` is the number of measurement
    samples (excludes warmup). Percentile fields are keyed by integer
    percentile so adding p95 later does not break existing readers.
    """

    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    percentiles_ms: dict[int, float]

    def percentile(self, p: int) -> float:
        """Return the p-th percentile, or raise if not recorded."""
        if p not in self.percentiles_ms:
            raise KeyError(
                f"percentile p{p} not recorded; available: {sorted(self.percentiles_ms)}",
            )
        return self.percentiles_ms[p]

    def to_dict(self) -> dict[str, float | int | dict[int, float]]:
        return asdict(self)


def percentile(values: Sequence[float], p: int) -> float:
    """Linear-interpolated percentile.

    Avoids importing numpy purely for one helper; the benchmark already
    pulls torch and that's enough.

    Args:
    ----
        values: sample list (need not be sorted, must be non-empty).
        p: percentile in ``[0, 100]``.

    Returns:
    -------
        Linearly-interpolated percentile in the same units as ``values``.

    """
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"percentile p={p} must be in [0, 100]")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo_idx = int(math.floor(rank))
    hi_idx = int(math.ceil(rank))
    if lo_idx == hi_idx:
        return float(sorted_values[lo_idx])
    weight = rank - lo_idx
    return float(
        sorted_values[lo_idx] * (1.0 - weight) + sorted_values[hi_idx] * weight,
    )


def summarize_latencies(
    latencies_ms: Iterable[float],
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
) -> LatencyStats:
    """Reduce a list of latencies to a ``LatencyStats``.

    Empty input is a programmer error and raises — the caller should gate
    on ``n_repeats >= 1`` (Pydantic does this) before invoking us.
    """
    samples = list(latencies_ms)
    if not samples:
        raise ValueError("cannot summarize empty latency list")
    if any(x < 0 for x in samples):
        raise ValueError("negative latencies are not physically meaningful")

    mean = sum(samples) / len(samples)
    if len(samples) == 1:
        std = 0.0
    else:
        std = math.sqrt(
            sum((x - mean) ** 2 for x in samples) / (len(samples) - 1),
        )

    pct_map = {p: percentile(samples, p) for p in percentiles}

    return LatencyStats(
        count=len(samples),
        mean_ms=mean,
        min_ms=min(samples),
        max_ms=max(samples),
        std_ms=std,
        percentiles_ms=pct_map,
    )


def throughput_fps(
    latencies_ms: Sequence[float],
    frames_per_iter: int,
) -> float:
    """Mean throughput in frames/second across a list of iteration latencies.

    Uses the mean (not the harmonic mean) because that's what users plan
    around — "what frame rate can I expect on average?". P-tail latency
    is reported separately via ``LatencyStats``.
    """
    if frames_per_iter < 1:
        raise ValueError(f"frames_per_iter must be >= 1, got {frames_per_iter}")
    if not latencies_ms:
        raise ValueError("cannot compute throughput from empty latency list")
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    if mean_ms <= 0.0:
        # Degenerate but possible on toy inputs (sub-microsecond clock
        # resolution). Return +inf so a downstream gate does not silently
        # treat 0/0 as a regression.
        return math.inf
    return (frames_per_iter * 1000.0) / mean_ms


def regression_pct(baseline: float, observed: float, *, higher_is_better: bool) -> float:
    """Percent regression of ``observed`` vs ``baseline``.

    Returns positive when ``observed`` is *worse* than ``baseline``, regardless
    of metric direction. Throughput uses ``higher_is_better=True``; latency
    and VRAM use ``higher_is_better=False``.

    Edge case: a zero baseline is not comparable; we return 0 and emit a
    debug log so a CI gate does not spuriously trip.
    """
    if baseline <= 0.0:
        logger.debug(
            "regression_pct.zero_baseline",
            baseline=baseline,
            observed=observed,
        )
        return 0.0
    delta = observed - baseline
    if not higher_is_better:
        delta = -delta
    return -100.0 * delta / baseline
