"""Reusable Agent Skills for AlphaGalerkin.

Provides modular, declarative capabilities that can be composed and invoked
by autonomous agents or pipelines:
- BenchmarkSkill: Orchestrates performance, scaling, and throughput sweeps.
- SelfPlaySkill: Generates self-play rollouts and extracts training experiences.
"""

from __future__ import annotations

from src.agents.skills.benchmark_skill import BenchmarkConfig, BenchmarkResult, BenchmarkSkill
from src.agents.skills.self_play_skill import SelfPlayConfig, SelfPlayResult, SelfPlaySkill

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSkill",
    "SelfPlayConfig",
    "SelfPlayResult",
    "SelfPlaySkill",
]
