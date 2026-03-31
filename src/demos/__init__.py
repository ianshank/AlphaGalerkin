"""Demo modules for AlphaGalerkin Hugging Face Space.

This package contains interactive demonstrations of AlphaGalerkin's capabilities:
- Physics zero-shot transfer visualization
- Architecture and attention visualization
- Performance benchmarking
- Enhanced Go gameplay with analysis

All demos follow AlphaGalerkin patterns:
- Pydantic configuration with no hardcoded values
- Structured logging with context binding
- Reusable visualization components

Demo modules with heavy dependencies (torch, scipy) are lazily imported.
Import them directly when needed:
    from src.demos.physics_demo import PhysicsDemo
    from src.demos.benchmark_demo import BenchmarkDemo
    from src.demos.architecture_demo import ArchitectureDemo
"""

from typing import Any

import structlog

# Core configuration - always available
from src.demos.config import (
    ArchitectureDemoConfig,
    BenchmarkDemoConfig,
    ColorScheme,
    DemoConfig,
    GameDemoConfig,
    PhysicsDemoConfig,
    VisualizationConfig,
)

logger = structlog.get_logger(__name__)

_VISUALIZATION_NAMES = frozenset(
    {
        "AttentionVisualizer",
        "BoardVisualizer",
        "ChartVisualizer",
        "FieldVisualizer",
        "PlotResult",
        "figure_to_image",
        "get_colormap",
    }
)


# Lazy imports for demo modules that require torch or matplotlib
def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import for heavy demo modules."""
    if name in _VISUALIZATION_NAMES:
        logger.debug("lazy_import_visualizations", requested_name=name)
        from src.demos import visualizations as _viz

        return getattr(_viz, name)

    if name in ("PhysicsDemo", "TransferResult", "create_physics_demo_tab"):
        logger.debug("lazy_import_physics_demo", requested_name=name)
        from src.demos.physics_demo import (
            PhysicsDemo,
            TransferResult,
            create_physics_demo_tab,
        )

        return {
            "PhysicsDemo": PhysicsDemo,
            "TransferResult": TransferResult,
            "create_physics_demo_tab": create_physics_demo_tab,
        }[name]

    if name in ("BenchmarkDemo", "BenchmarkResult", "BenchmarkSuite", "create_benchmark_demo_tab"):
        logger.debug("lazy_import_benchmark_demo", requested_name=name)
        from src.demos.benchmark_demo import (
            BenchmarkDemo,
            BenchmarkResult,
            BenchmarkSuite,
            create_benchmark_demo_tab,
        )

        return {
            "BenchmarkDemo": BenchmarkDemo,
            "BenchmarkResult": BenchmarkResult,
            "BenchmarkSuite": BenchmarkSuite,
            "create_benchmark_demo_tab": create_benchmark_demo_tab,
        }[name]

    if name in ("ArchitectureDemo", "create_architecture_demo_tab"):
        logger.debug("lazy_import_architecture_demo", requested_name=name)
        from src.demos.architecture_demo import (
            ArchitectureDemo,
            create_architecture_demo_tab,
        )

        return {
            "ArchitectureDemo": ArchitectureDemo,
            "create_architecture_demo_tab": create_architecture_demo_tab,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration (always available)
    "ArchitectureDemoConfig",
    "BenchmarkDemoConfig",
    "ColorScheme",
    "DemoConfig",
    "GameDemoConfig",
    "PhysicsDemoConfig",
    "VisualizationConfig",
    # Visualization utilities (require matplotlib)
    "AttentionVisualizer",
    "BoardVisualizer",
    "ChartVisualizer",
    "FieldVisualizer",
    "PlotResult",
    "figure_to_image",
    "get_colormap",
    # Physics demo (lazy, requires torch)
    "PhysicsDemo",
    "TransferResult",
    "create_physics_demo_tab",
    # Benchmark demo (lazy, requires torch)
    "BenchmarkDemo",
    "BenchmarkResult",
    "BenchmarkSuite",
    "create_benchmark_demo_tab",
    # Architecture demo (lazy, requires torch)
    "ArchitectureDemo",
    "create_architecture_demo_tab",
]
