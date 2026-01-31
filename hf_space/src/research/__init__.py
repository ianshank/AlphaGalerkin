"""Research and Transfer Validation module for AlphaGalerkin.

Provides:
- Experiment management and tracking
- Transfer learning validation
- Benchmarking utilities
- Model comparison tools
- Results reporting and visualization
"""

from __future__ import annotations

from src.research.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSuite,
)
from src.research.comparison import (
    ComparisonResult,
    ModelComparison,
    ModelMetrics,
)
from src.research.config import (
    BenchmarkConfig,
    ComparisonConfig,
    ExperimentConfig,
    TransferConfig,
)
from src.research.experiment import (
    Experiment,
    ExperimentRun,
    ExperimentTracker,
)
from src.research.reporter import (
    Reporter,
    ReportFormat,
    create_reporter,
)
from src.research.validator import (
    TransferResult,
    TransferValidator,
    create_transfer_validator,
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
