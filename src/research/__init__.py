"""Research and Transfer Validation module for AlphaGalerkin.

Provides:
- Experiment management and tracking
- Transfer learning validation
- Benchmarking utilities
- Model comparison tools
- Results reporting and visualization
"""

from __future__ import annotations

from src.research.config import (
    ExperimentConfig,
    BenchmarkConfig,
    TransferConfig,
    ComparisonConfig,
)
from src.research.experiment import (
    Experiment,
    ExperimentRun,
    ExperimentTracker,
)
from src.research.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSuite,
)
from src.research.validator import (
    TransferValidator,
    TransferResult,
    create_transfer_validator,
)
from src.research.comparison import (
    ModelComparison,
    ComparisonResult,
    ModelMetrics,
)
from src.research.reporter import (
    Reporter,
    ReportFormat,
    create_reporter,
)

__all__ = [
    # Configuration
    "ExperimentConfig",
    "BenchmarkConfig",
    "TransferConfig",
    "ComparisonConfig",
    # Experiment
    "Experiment",
    "ExperimentRun",
    "ExperimentTracker",
    # Benchmark
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    # Transfer validation
    "TransferValidator",
    "TransferResult",
    "create_transfer_validator",
    # Comparison
    "ModelComparison",
    "ComparisonResult",
    "ModelMetrics",
    # Reporter
    "Reporter",
    "ReportFormat",
    "create_reporter",
]
