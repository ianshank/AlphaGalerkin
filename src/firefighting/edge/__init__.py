"""Edge deployment utilities for drone compute.

Provides memory budget management, latency profiling, and
optimized inference runtime for Jetson Orin Nano.
"""

from src.firefighting.edge.memory import MemoryBudgetManager, MemorySnapshot
from src.firefighting.edge.profiler import EdgeProfiler, LatencyBreakdown, ProfilingResult

__all__ = [
    "MemoryBudgetManager",
    "MemorySnapshot",
    "EdgeProfiler",
    "LatencyBreakdown",
    "ProfilingResult",
]
