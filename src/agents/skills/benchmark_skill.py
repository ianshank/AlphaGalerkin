"""Declarative Benchmark Skill for autonomous agents and performance pipelines."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.templates.logging import create_logger_class

BenchmarkSkillLogger = create_logger_class("BenchmarkSkill")


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark sweep."""

    name: str
    warmup_rounds: int = 2
    timing_rounds: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""

    name: str
    mean_duration_s: float
    median_duration_s: float
    std_duration_s: float
    min_duration_s: float
    max_duration_s: float
    throughput_items_per_s: float
    total_rounds: int
    raw_durations_s: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkSkill:
    """Reusable skill for profiling throughput, latency, and scaling."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.logger = BenchmarkSkillLogger(component=config.name)

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        item_count: int = 1,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run warmups followed by timed benchmark rounds."""
        self.logger.info("benchmark_started", name=self.config.name)

        # Warmup phase
        for _ in range(self.config.warmup_rounds):
            func(*args, **kwargs)

        # Timed execution phase
        durations: list[float] = []
        for _ in range(self.config.timing_rounds):
            start = time.perf_counter()
            func(*args, **kwargs)
            duration = time.perf_counter() - start
            durations.append(duration)

        mean_dur = statistics.mean(durations) if durations else 0.0
        median_dur = statistics.median(durations) if durations else 0.0
        std_dur = statistics.stdev(durations) if len(durations) > 1 else 0.0
        min_dur = min(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0
        throughput = (item_count / mean_dur) if mean_dur > 0 else 0.0

        result = BenchmarkResult(
            name=self.config.name,
            mean_duration_s=mean_dur,
            median_duration_s=median_dur,
            std_duration_s=std_dur,
            min_duration_s=min_dur,
            max_duration_s=max_dur,
            throughput_items_per_s=throughput,
            total_rounds=len(durations),
            raw_durations_s=durations,
            metadata=self.config.metadata,
        )

        self.logger.info(
            "benchmark_finished",
            mean_s=mean_dur,
            throughput=throughput,
            name=self.config.name,
        )
        return result
