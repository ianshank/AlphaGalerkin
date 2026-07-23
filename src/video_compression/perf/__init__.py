"""Performance benchmarking for the AlphaGalerkin video codec.

Public surface:

- ``PerfBenchmarkConfig`` — sweep configuration.
- ``PerfBenchmark`` — executable benchmark.
- ``run_benchmark`` — convenience wrapper returning an ``ExecutionResult``.
- ``BaselineRegistry`` — load/save/diff baselines for regression gating.
- ``BenchmarkReport`` / ``CellResult`` — report types.

See ``scripts/benchmark_codec.py`` for a CLI entry point.
"""

from __future__ import annotations

from src.video_compression.perf.baseline import (
    BaselineDocument,
    BaselineEntry,
    BaselineRegistry,
    CellDiff,
    RegressionReport,
    baseline_entry_from_cell,
)
from src.video_compression.perf.benchmark import (
    BenchmarkReport,
    CellResult,
    PerfBenchmark,
    baseline_from_report,
    cell_key,
    report_from_result,
    run_benchmark,
)
from src.video_compression.perf.config import (
    PERF_BASELINE_DOCUMENT_SCHEMA_VERSION,
    PERF_BASELINE_ENTRY_SCHEMA_VERSION,
    PERF_BENCHMARK_CONFIG_SCHEMA_VERSION,
    BenchmarkPhase,
    PerfBenchmarkConfig,
    Precision,
    ResolutionSpec,
    RuntimeBackend,
    RuntimeProfile,
)
from src.video_compression.perf.device import (
    device_label,
    list_cuda_devices,
    resolve_device,
)
from src.video_compression.perf.metrics import (
    DEFAULT_PERCENTILES,
    LatencyStats,
    percentile,
    regression_pct,
    summarize_latencies,
    throughput_fps,
)
from src.video_compression.perf.subjects import (
    BenchmarkSubject,
    CodecForwardSubject,
    create_subject,
)

__all__ = [
    "BaselineDocument",
    "BaselineEntry",
    "BaselineRegistry",
    "BenchmarkPhase",
    "BenchmarkReport",
    "BenchmarkSubject",
    "CellDiff",
    "CellResult",
    "CodecForwardSubject",
    "DEFAULT_PERCENTILES",
    "LatencyStats",
    "PERF_BASELINE_DOCUMENT_SCHEMA_VERSION",
    "PERF_BASELINE_ENTRY_SCHEMA_VERSION",
    "PERF_BENCHMARK_CONFIG_SCHEMA_VERSION",
    "PerfBenchmark",
    "PerfBenchmarkConfig",
    "Precision",
    "RegressionReport",
    "ResolutionSpec",
    "RuntimeBackend",
    "RuntimeProfile",
    "baseline_entry_from_cell",
    "baseline_from_report",
    "cell_key",
    "create_subject",
    "device_label",
    "list_cuda_devices",
    "percentile",
    "regression_pct",
    "report_from_result",
    "resolve_device",
    "run_benchmark",
    "summarize_latencies",
    "throughput_fps",
]
