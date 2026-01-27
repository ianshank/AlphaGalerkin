"""Notebook Utilities for AlphaGalerkin Demos.

This module provides reusable, configurable utilities for Jupyter notebooks
demonstrating AlphaGalerkin functionality.

Features:
- Centralized configuration with no hard-coded values
- Reusable visualization functions
- Benchmarking utilities with proper timing
- Error handling and logging support
"""

from notebooks.utils.config import (
    DemoConfig,
    create_demo_config,
    get_default_board_sizes,
)
from notebooks.utils.benchmark import (
    BenchmarkResult,
    benchmark_attention,
    benchmark_model_throughput,
    format_benchmark_table,
)
from notebooks.utils.visualization import (
    plot_fourier_features,
    plot_attention_comparison,
    plot_poisson_samples,
    plot_go_board,
    plot_policy_heatmap,
)
from notebooks.utils.helpers import (
    setup_environment,
    create_sample_board,
    safe_model_forward,
)

__all__ = [
    # Config
    "DemoConfig",
    "create_demo_config",
    "get_default_board_sizes",
    # Benchmarking
    "BenchmarkResult",
    "benchmark_attention",
    "benchmark_model_throughput",
    "format_benchmark_table",
    # Visualization
    "plot_fourier_features",
    "plot_attention_comparison",
    "plot_poisson_samples",
    "plot_go_board",
    "plot_policy_heatmap",
    # Helpers
    "setup_environment",
    "create_sample_board",
    "safe_model_forward",
]
