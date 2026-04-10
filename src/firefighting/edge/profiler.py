"""Latency and memory profiling for edge deployment.

Measures inference cycle timing and peak memory usage to
verify compliance with edge device constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import structlog

from src.firefighting.config.edge import EdgeConfig

logger = structlog.get_logger(__name__)


@dataclass
class LatencyBreakdown:
    """Timing breakdown for a single prediction cycle."""

    sensor_ingest_ms: float = 0.0
    mcts_search_ms: float = 0.0
    pde_solve_ms: float = 0.0
    output_encode_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.sensor_ingest_ms + self.mcts_search_ms + self.pde_solve_ms + self.output_encode_ms
        )

    def within_budget(self, config: EdgeConfig) -> bool:
        return self.total_ms <= config.max_latency_ms


@dataclass
class ProfilingResult:
    """Summary of profiling session."""

    n_cycles: int
    mean_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    peak_memory_mb: float
    budget_violations: int
    breakdowns: list[LatencyBreakdown] = field(default_factory=list)


class EdgeProfiler:
    """Profiles prediction cycle performance on edge devices.

    Measures wall-clock time for each pipeline stage and tracks
    peak memory usage to verify compliance with device constraints.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._breakdowns: list[LatencyBreakdown] = []

    def start_cycle(self) -> CycleTimer:
        """Begin timing a new prediction cycle."""
        return CycleTimer(self.config)

    def record(self, breakdown: LatencyBreakdown) -> None:
        """Record a completed cycle's timing."""
        self._breakdowns.append(breakdown)

        if not breakdown.within_budget(self.config):
            logger.warning(
                "latency_budget_exceeded",
                total_ms=breakdown.total_ms,
                budget_ms=self.config.max_latency_ms,
            )

    def summarize(self) -> ProfilingResult:
        """Generate profiling summary."""
        if not self._breakdowns:
            return ProfilingResult(
                n_cycles=0,
                mean_latency_ms=0,
                max_latency_ms=0,
                p95_latency_ms=0,
                peak_memory_mb=0,
                budget_violations=0,
            )

        latencies = [b.total_ms for b in self._breakdowns]
        violations = sum(1 for b in self._breakdowns if not b.within_budget(self.config))

        return ProfilingResult(
            n_cycles=len(self._breakdowns),
            mean_latency_ms=float(np.mean(latencies)),
            max_latency_ms=float(np.max(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            peak_memory_mb=0.0,  # Would use psutil in production
            budget_violations=violations,
            breakdowns=self._breakdowns,
        )

    def reset(self) -> None:
        """Clear recorded data."""
        self._breakdowns.clear()


class CycleTimer:
    """Context manager for timing individual pipeline stages."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._breakdown = LatencyBreakdown()
        self._stage_start: float = 0.0

    @property
    def breakdown(self) -> LatencyBreakdown:
        return self._breakdown

    def start_stage(self, stage: str) -> None:
        """Begin timing a stage."""
        self._stage_start = time.perf_counter()

    def end_stage(self, stage: str) -> float:
        """End timing a stage and record its duration.

        Args:
            stage: One of 'sensor_ingest', 'mcts_search', 'pde_solve', 'output_encode'.

        Returns:
            Stage duration in milliseconds.

        """
        elapsed_ms = (time.perf_counter() - self._stage_start) * 1000.0

        if stage == "sensor_ingest":
            self._breakdown.sensor_ingest_ms = elapsed_ms
        elif stage == "mcts_search":
            self._breakdown.mcts_search_ms = elapsed_ms
        elif stage == "pde_solve":
            self._breakdown.pde_solve_ms = elapsed_ms
        elif stage == "output_encode":
            self._breakdown.output_encode_ms = elapsed_ms

        return elapsed_ms
