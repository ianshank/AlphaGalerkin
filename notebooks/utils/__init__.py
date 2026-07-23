"""Notebook Utilities for AlphaGalerkin Demos.

This module provides reusable, configurable utilities for Jupyter notebooks
demonstrating AlphaGalerkin functionality.

Features:
- Centralized configuration with no hard-coded values
- Reusable visualization functions
- Benchmarking utilities with proper timing
- Error handling and logging support
"""

from notebooks.utils.benchmark import (
    BenchmarkResult,
    benchmark_attention,
    benchmark_model_throughput,
    benchmark_module,
    format_benchmark_table,
    format_throughput_table,
)
from notebooks.utils.config import (
    BenchmarkConfig,
    DemoConfig,
    GoBoardConfig,
    ModelConfig,
    PhysicsConfig,
    VisualizationConfig,
    create_demo_config,
    get_board_labels,
    get_default_board_sizes,
)
from notebooks.utils.helpers import (
    EnvironmentInfo,
    ModelForwardResult,
    create_sample_board,
    create_sample_board_from_config,
    format_model_summary,
    safe_model_forward,
    setup_environment,
    validate_board_sizes,
)
from notebooks.utils.visualization import (
    plot_attention_comparison,
    plot_fourier_features,
    plot_go_board,
    plot_multi_board_visualization,
    plot_poisson_samples,
    plot_policy_heatmap,
)

__all__ = [
    # Config classes
    "DemoConfig",
    "ModelConfig",
    "BenchmarkConfig",
    "VisualizationConfig",
    "PhysicsConfig",
    "GoBoardConfig",
    # Config functions
    "create_demo_config",
    "get_default_board_sizes",
    "get_board_labels",
    # Benchmarking
    "BenchmarkResult",
    "benchmark_module",
    "benchmark_attention",
    "benchmark_model_throughput",
    "format_benchmark_table",
    "format_throughput_table",
    # Visualization
    "plot_fourier_features",
    "plot_attention_comparison",
    "plot_poisson_samples",
    "plot_go_board",
    "plot_policy_heatmap",
    "plot_multi_board_visualization",
    # Helper classes
    "EnvironmentInfo",
    "ModelForwardResult",
    # Helper functions
    "setup_environment",
    "create_sample_board",
    "create_sample_board_from_config",
    "safe_model_forward",
    "format_model_summary",
    "validate_board_sizes",
]
