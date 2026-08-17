"""Configuration utilities for AlphaGalerkin demo notebooks.

Provides centralized, validated configuration with no hard-coded values.
All demo parameters are configurable through DemoConfig.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    # Fallback to standard logging if structlog not available
    logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for demo model instantiation."""

    d_model: int = 64
    n_heads: int = 4
    n_galerkin_layers: int = 2
    n_softmax_layers: int = 1
    n_fourier_features: int = 32
    input_channels: int = 17  # Standard Go input channels

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Check positivity first to avoid ZeroDivisionError
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        # Now safe to check divisibility
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking operations."""

    n_warmup_runs: int = 5
    n_timed_runs: int = 50
    batch_size: int = 4
    n_evals: int = 100


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    figure_width: float = 14.0
    figure_height: float = 6.0
    colormap_diverging: str = "RdBu"
    colormap_sequential: str = "viridis"
    colormap_hot: str = "Reds"
    font_size_title: int = 12
    font_size_label: int = 10
    font_size_tick: int = 9


@dataclass
class PhysicsConfig:
    """Configuration for physics demonstration."""

    n_charges: int = 3
    boundary_value: float = 0.0
    charge_std: float = 1.0


@dataclass
class GoBoardConfig:
    """Configuration for Go board visualization."""

    black_stone_positions: list[tuple[int, int]] = field(
        default_factory=lambda: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
    )
    white_stone_positions: list[tuple[int, int]] = field(
        default_factory=lambda: [(3, 4), (4, 3), (4, 4), (5, 5)]
    )
    stone_radius: float = 0.4
    board_color: float = 0.82  # YlOrBr colormap value
    grid_alpha: float = 0.5
    grid_linewidth: float = 0.5


@dataclass
class DemoConfig:
    """Master configuration for AlphaGalerkin demo notebooks.

    Centralizes all configurable parameters to avoid hard-coded values
    throughout the notebook.

    Note:
        seq_lengths is auto-computed from board_sizes if not explicitly provided.
        If you provide custom seq_lengths, it must have the same length as board_sizes.

    """

    # Board sizes for demonstrations
    board_sizes: Sequence[int] = field(default_factory=lambda: [5, 9, 13, 19])

    # Sequence lengths for attention benchmarks (auto-computed from board_sizes if None)
    # These are the squared board sizes used for attention sequence length testing
    seq_lengths: Sequence[int] | None = None

    # Physics demo board sizes
    physics_board_sizes: Sequence[int] = field(default_factory=lambda: [9, 13, 19, 25])

    # Random seed for reproducibility
    random_seed: int = 42

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    go_board: GoBoardConfig = field(default_factory=GoBoardConfig)

    def __post_init__(self) -> None:
        """Validate configuration consistency and auto-compute derived values."""
        # Auto-compute seq_lengths from board_sizes if not provided
        if self.seq_lengths is None:
            object.__setattr__(self, "seq_lengths", [s * s for s in self.board_sizes])
        else:
            # Validate that custom seq_lengths matches board_sizes length
            if len(self.seq_lengths) != len(self.board_sizes):
                raise ValueError(
                    f"seq_lengths length ({len(self.seq_lengths)}) must match "
                    f"board_sizes length ({len(self.board_sizes)}). "
                    f"Either provide matching lengths or omit seq_lengths to auto-compute."
                )
            # Validate values match expected squares
            expected_seq = [s * s for s in self.board_sizes]
            if list(self.seq_lengths) != expected_seq:
                logger.warning(
                    "seq_lengths_custom",
                    message="Custom seq_lengths provided; may not match board_sizes squared",
                    expected=expected_seq,
                    actual=list(self.seq_lengths),
                )


def create_demo_config(**overrides: object) -> DemoConfig:
    """Create a DemoConfig with optional overrides.

    Args:
        **overrides: Field overrides for the configuration.

    Returns:
        Configured DemoConfig instance.

    Example:
        >>> config = create_demo_config(random_seed=123, board_sizes=[9, 19])
        >>> config.random_seed
        123

    """
    logger.debug("creating_demo_config", overrides=list(overrides.keys()))
    return DemoConfig(**overrides)


def get_default_board_sizes() -> list[int]:
    """Get the default board sizes for demonstrations.

    Returns:
        List of standard board sizes: [5, 9, 13, 19]

    """
    return [5, 9, 13, 19]


def get_board_labels(sizes: Sequence[int]) -> list[str]:
    """Generate board size labels (e.g., '9×9').

    Args:
        sizes: Sequence of board sizes.

    Returns:
        List of formatted labels.

    """
    return [f"{s}×{s}" for s in sizes]
