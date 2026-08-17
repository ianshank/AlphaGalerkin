"""Configuration schemas for AlphaGalerkin HF Space demos.

All demo parameters are configurable through Pydantic models.
No hardcoded values - everything is validated and documented.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ColorScheme(str, Enum):
    """Available color schemes for visualizations."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"


class VisualizationConfig(BaseModel):
    """Configuration for visualization rendering.

    Controls appearance of plots, heatmaps, and board displays.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Figure dimensions
    figure_width: float = Field(
        default=8.0,
        gt=2.0,
        le=20.0,
        description="Figure width in inches",
    )
    figure_height: float = Field(
        default=6.0,
        gt=2.0,
        le=20.0,
        description="Figure height in inches",
    )
    dpi: int = Field(
        default=100,
        ge=50,
        le=300,
        description="Dots per inch for rendering",
    )

    # Colors and styling
    color_scheme: ColorScheme = Field(
        default=ColorScheme.VIRIDIS,
        description="Color scheme for heatmaps",
    )
    background_color: str = Field(
        default="#f0f0f0",
        description="Background color (hex or named color)",
    )
    grid_color: str = Field(
        default="#333333",
        description="Grid line color",
    )
    font_size: int = Field(
        default=10,
        ge=6,
        le=24,
        description="Base font size for labels",
    )

    # Board visualization specific
    board_wood_color: str = Field(
        default="#e3c586",
        description="Wood color for Go board background",
    )
    black_stone_color: str = Field(
        default="#000000",
        description="Color for black stones",
    )
    white_stone_color: str = Field(
        default="#ffffff",
        description="Color for white stones",
    )
    stone_border_color: str = Field(
        default="#000000",
        description="Border color for stones",
    )
    last_move_marker_color: str = Field(
        default="#ff0000",
        description="Marker color for last move",
    )

    # Animation settings
    animation_interval_ms: int = Field(
        default=100,
        ge=10,
        le=2000,
        description="Milliseconds between animation frames",
    )

    @field_validator("background_color", "grid_color", "board_wood_color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        """Validate color string format."""
        v = v.strip()
        # Accept hex colors or named colors
        if v.startswith("#") and len(v) not in (4, 7):
            raise ValueError(f"Invalid hex color: {v}")
        return v


class PhysicsDemoConfig(BaseModel):
    """Configuration for physics zero-shot transfer demo.

    Controls grid sizes, samples, and visualization for Poisson equation demos.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Grid sizes for demonstration
    train_grid_size: int = Field(
        default=9,
        ge=5,
        le=32,
        description="Grid size for training visualization",
    )
    eval_grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes for zero-shot transfer evaluation",
    )
    max_grid_size: int = Field(
        default=32,
        ge=9,
        le=64,
        description="Maximum allowed grid size",
    )

    # Poisson solver parameters
    n_charges: int | None = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of point charges (None for continuous)",
    )
    charge_std: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Standard deviation of charge magnitudes",
    )
    use_spectral_solver: bool = Field(
        default=True,
        description="Use spectral method (FFT) for solving",
    )

    # Model parameters for demo
    n_demo_samples: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of samples for demo visualization",
    )

    # MSE threshold for success
    mse_threshold: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="MSE threshold for zero-shot transfer success",
    )

    # Visualization
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
    )

    @field_validator("eval_grid_sizes")
    @classmethod
    def validate_eval_sizes(cls, v: list[int]) -> list[int]:
        """Validate and sort evaluation grid sizes."""
        if not v:
            raise ValueError("eval_grid_sizes cannot be empty")
        for size in v:
            if size < 5 or size > 64:
                raise ValueError(f"Grid size {size} must be between 5 and 64")
        return sorted(set(v))

    @model_validator(mode="after")
    def validate_sizes_consistency(self) -> PhysicsDemoConfig:
        """Ensure eval sizes don't exceed max_grid_size."""
        for size in self.eval_grid_sizes:
            if size > self.max_grid_size:
                raise ValueError(f"Eval size {size} exceeds max_grid_size {self.max_grid_size}")
        return self


class BenchmarkDemoConfig(BaseModel):
    """Configuration for performance benchmarking demo.

    Controls benchmark parameters for FNet vs Softmax comparison.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Benchmark sizes (N = board_size^2 = sequence length)
    benchmark_sizes: list[int] = Field(
        default_factory=lambda: [81, 169, 361, 625, 900],
        description="Sequence lengths (board_size^2) to benchmark",
    )
    max_sequence_length: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Maximum sequence length for benchmarks",
    )

    # Benchmark parameters
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for benchmarking",
    )
    n_warmup_runs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of warmup runs before timing",
    )
    n_benchmark_runs: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of runs for timing average",
    )

    # Model dimensions
    d_model: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Model embedding dimension",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of attention heads",
    )

    # Device settings
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Device for benchmarking",
    )
    use_amp: bool = Field(
        default=True,
        description="Use automatic mixed precision",
    )

    # Visualization
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
    )

    @field_validator("benchmark_sizes")
    @classmethod
    def validate_benchmark_sizes(cls, v: list[int]) -> list[int]:
        """Validate benchmark sizes are reasonable."""
        if not v:
            raise ValueError("benchmark_sizes cannot be empty")
        return sorted(set(v))


class GameDemoConfig(BaseModel):
    """Configuration for Go game demo.

    Controls board size, MCTS parameters, and display options.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Board configuration
    board_size: int = Field(
        default=9,
        ge=5,
        le=19,
        description="Board size for gameplay",
    )
    komi: float = Field(
        default=6.5,
        ge=0.0,
        le=10.0,
        description="Komi (compensation for White)",
    )

    # MCTS parameters
    n_simulations: int = Field(
        default=60,
        ge=10,
        le=1600,
        description="MCTS simulations per move",
    )
    c_puct: float = Field(
        default=1.5,
        gt=0.0,
        le=10.0,
        description="PUCT exploration constant",
    )
    dirichlet_alpha: float = Field(
        default=0.03,
        gt=0.0,
        lt=1.0,
        description="Dirichlet noise alpha",
    )
    dirichlet_epsilon: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dirichlet noise mixing coefficient",
    )

    # AI vs AI settings
    ai_move_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Delay between AI moves in AI vs AI mode",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for move selection",
    )

    # Analysis features
    show_policy_heatmap: bool = Field(
        default=True,
        description="Show policy probability heatmap",
    )
    show_value_estimate: bool = Field(
        default=True,
        description="Show value (win probability) estimate",
    )
    show_move_suggestions: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of top move suggestions to show",
    )

    # Visualization
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
    )


class ArchitectureDemoConfig(BaseModel):
    """Configuration for architecture visualization demo.

    Controls attention pattern and Fourier feature visualization.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Sample parameters
    sample_board_size: int = Field(
        default=9,
        ge=5,
        le=19,
        description="Board size for sample visualization",
    )
    n_attention_heads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of attention heads to visualize",
    )

    # Fourier feature visualization
    n_fourier_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of samples for Fourier feature plots",
    )
    fourier_frequency_range: tuple[float, float] = Field(
        default=(0.1, 10.0),
        description="Range of frequencies to visualize",
    )

    # Attention visualization
    attention_layer_to_show: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Which attention layer to visualize",
    )
    show_galerkin_vs_softmax: bool = Field(
        default=True,
        description="Compare Galerkin and Softmax attention patterns",
    )

    # LBB stability visualization
    show_lbb_metrics: bool = Field(
        default=True,
        description="Show LBB stability condition metrics",
    )

    # Visualization
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
    )


class DemoConfig(BaseModel):
    """Root configuration combining all demo configurations.

    This is the single source of truth for all demo settings.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        protected_namespaces=(),  # allow model_checkpoint_path field
    )

    # Sub-configurations
    game: GameDemoConfig = Field(
        default_factory=GameDemoConfig,
        description="Go game demo configuration",
    )
    physics: PhysicsDemoConfig = Field(
        default_factory=PhysicsDemoConfig,
        description="Physics transfer demo configuration",
    )
    benchmark: BenchmarkDemoConfig = Field(
        default_factory=BenchmarkDemoConfig,
        description="Performance benchmark configuration",
    )
    architecture: ArchitectureDemoConfig = Field(
        default_factory=ArchitectureDemoConfig,
        description="Architecture visualization configuration",
    )

    # Global settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Default device for computations",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )

    # Model paths (relative to app root)
    model_checkpoint_path: str = Field(
        default="checkpoint.pt",
        description="Path to model checkpoint",
    )
    physics_model_path: str | None = Field(
        default=None,
        description="Path to physics model checkpoint (optional)",
    )

    @classmethod
    def from_env(cls) -> DemoConfig:
        """Create config from environment variables.

        Environment variables:
        - DEMO_DEBUG: Set to 'true' for debug mode
        - DEMO_LOG_LEVEL: Logging level
        - DEMO_DEVICE: Device preference
        - DEMO_SEED: Random seed

        Returns:
            DemoConfig instance with environment overrides.

        """
        import os

        overrides = {}

        if os.getenv("DEMO_DEBUG", "").lower() == "true":
            overrides["debug"] = True

        if log_level := os.getenv("DEMO_LOG_LEVEL"):
            overrides["log_level"] = log_level.upper()

        if device := os.getenv("DEMO_DEVICE"):
            overrides["device"] = device

        if seed := os.getenv("DEMO_SEED"):
            overrides["seed"] = int(seed)

        if model_path := os.getenv("DEMO_MODEL_PATH"):
            overrides["model_checkpoint_path"] = model_path

        return cls(**overrides)
