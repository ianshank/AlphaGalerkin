"""Configuration models for the AlphaGalerkin E2E Dashboard.

All runtime-tunable constants live here as Pydantic-validated fields so that
no magic numbers appear inside tab or app logic.  Create a custom config by
subclassing or by constructing with keyword overrides:

    cfg = DashboardConfig(app=AppConfig(port=8080))
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# UI / Server
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Top-level Gradio server configuration."""

    host: str = Field(default="0.0.0.0", description="Bind address for the Gradio server")
    port: int = Field(default=7860, ge=1024, le=65535, description="TCP port to listen on")
    share: bool = Field(default=False, description="Create a public Gradio share link")
    debug: bool = Field(default=False, description="Enable Gradio debug mode")
    css_tab_font_size: str = Field(default="14px", description="Tab button font-size CSS value")
    css_tab_padding: str = Field(default="8px 16px", description="Tab button padding CSS value")
    plot_dpi: int = Field(default=110, ge=72, le=300, description="DPI for all matplotlib plots")


# ---------------------------------------------------------------------------
# Go game tab
# ---------------------------------------------------------------------------


class GameConfig(BaseModel):
    """Configuration for the Go Game tab."""

    board_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Available board sizes (9=training, larger=zero-shot transfer)",
    )
    default_board_size: int = Field(default=9, ge=5, description="Initially selected board size")
    ai_temperature_vs_human: float = Field(
        default=0.0,
        ge=0.0,
        description="MCTS temperature when playing against a human (0=deterministic)",
    )
    ai_temperature_self_play: float = Field(
        default=0.1,
        ge=0.0,
        description="MCTS temperature for AI vs AI self-play",
    )
    board_image_height_px: int = Field(
        default=460, ge=200, description="Board image widget height in pixels"
    )
    fallback_board_size_px: int = Field(
        default=400, ge=100, description="Pixel dimension of the fallback blank board image"
    )


# ---------------------------------------------------------------------------
# PDE solver tab
# ---------------------------------------------------------------------------


class PDEConfig(BaseModel):
    """Configuration for the Poisson PDE Solver tab."""

    grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19, 25, 32],
        description="Dropdown options for N in the N×N grid",
    )
    default_grid_size: int = Field(default=9, ge=3, description="Initially selected grid size")
    comparison_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes used in the resolution-comparison plot",
    )
    charge_patterns: list[str] = Field(
        default_factory=lambda: ["Point Charge", "Dipole", "Quadrupole", "Ring", "Random"],
        description="Available charge-pattern options",
    )
    default_pattern: str = Field(default="Point Charge", description="Initially selected pattern")
    strength_min: float = Field(default=-2.0, description="Minimum charge strength slider value")
    strength_max: float = Field(default=2.0, description="Maximum charge strength slider value")
    default_strength: float = Field(default=1.0, description="Default charge strength")
    position_min: float = Field(default=0.1, description="Minimum normalised position (0–1)")
    position_max: float = Field(default=0.9, description="Maximum normalised position (0–1)")
    epsilon: float = Field(
        default=1e-9, gt=0, description="Small value to guard against division by zero"
    )
    ring_num_charges: int = Field(
        default=8, ge=3, description="Number of charges in a ring pattern"
    )


# ---------------------------------------------------------------------------
# PoC scenario tab
# ---------------------------------------------------------------------------


class ComplexityRunConfig(BaseModel):
    """Runtime defaults for the Complexity benchmark scenario."""

    default_grid_sizes_str: str = Field(
        default="9,13,19,25", description="Comma-separated grid sizes shown in the text box"
    )
    default_d_model: int = Field(default=64, ge=16, description="Default d_model slider value")
    default_iterations: int = Field(
        default=15, ge=10, description="Default number of timed iterations"
    )
    n_warmup: int = Field(
        default=2, ge=1, description="Warmup iterations (not timed) before the benchmark"
    )
    min_grid_sizes: int = Field(
        default=3, ge=2, description="Minimum number of distinct grid sizes required"
    )
    fallback_grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19, 25],
        description="Grid sizes to use when the user provides fewer than min_grid_sizes",
    )


class StabilityRunConfig(BaseModel):
    """Runtime defaults for the LBB Stability scenario."""

    default_resolutions_str: str = Field(
        default="5,9,13", description="Comma-separated resolutions shown in the text box"
    )
    default_d_model: int = Field(default=64, ge=16, description="Default d_model slider value")
    default_training_steps: int = Field(
        default=100, ge=100, description="Default number of training steps to monitor"
    )
    n_forward_passes: int = Field(
        default=20, ge=5, description="Number of forward passes for initialisation stability"
    )
    lbb_threshold: float = Field(
        default=1e-6, gt=0, description="Minimum acceptable LBB constant β"
    )
    max_lbb_violations: int = Field(
        default=0, ge=0, description="Maximum allowed LBB violations during training"
    )
    min_resolutions: int = Field(
        default=2, ge=2, description="Minimum number of distinct resolutions required"
    )
    fallback_resolutions: list[int] = Field(
        default_factory=lambda: [5, 9, 13],
        description="Resolutions to use when user provides fewer than min_resolutions",
    )


class TransferMilestone(BaseModel):
    """Validated milestone data for the zero-shot transfer scenario."""

    train_resolution: int = Field(default=9, description="Resolution used for training")
    mse_threshold: float = Field(
        default=0.05, gt=0, description="Pass/fail MSE threshold from the PoC spec"
    )
    achieved_mse: dict[int, float] = Field(
        default_factory=lambda: {9: 0.000041, 13: 0.000098, 19: 0.000209},
        description="Achieved MSE values per evaluation resolution (from 2026-01-26 run)",
    )
    milestone_date: str = Field(default="2026-01-26", description="Date milestone was achieved")


class PoCConfig(BaseModel):
    """Configuration for the PoC Scenario Runner tab."""

    complexity: ComplexityRunConfig = Field(default_factory=ComplexityRunConfig)
    stability: StabilityRunConfig = Field(default_factory=StabilityRunConfig)
    transfer: TransferMilestone = Field(default_factory=TransferMilestone)


# ---------------------------------------------------------------------------
# Training dashboard tab
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Configuration for the Training Dashboard tab."""

    # Architecture sliders
    d_model_min: int = Field(default=64, ge=16, description="d_model slider minimum")
    d_model_max: int = Field(default=512, le=4096, description="d_model slider maximum")
    d_model_default: int = Field(default=256, description="d_model slider default")
    d_model_step: int = Field(default=64, ge=1, description="d_model slider step size")

    galerkin_layers_min: int = Field(default=1, description="Minimum Galerkin layers")
    galerkin_layers_max: int = Field(default=12, description="Maximum Galerkin layers")
    galerkin_layers_default: int = Field(default=6, description="Default Galerkin layers")

    softmax_layers_min: int = Field(default=1, description="Minimum Softmax layers")
    softmax_layers_max: int = Field(default=6, description="Maximum Softmax layers")
    softmax_layers_default: int = Field(default=2, description="Default Softmax layers")

    fourier_min: int = Field(default=32, description="Minimum Fourier features")
    fourier_max: int = Field(default=256, description="Maximum Fourier features")
    fourier_default: int = Field(default=128, description="Default Fourier features")
    fourier_step: int = Field(default=32, description="Fourier features step size")

    # Training curve sliders
    steps_min: int = Field(default=1000, description="Minimum total training steps")
    steps_max: int = Field(default=50000, description="Maximum total training steps")
    steps_default: int = Field(default=10000, description="Default total training steps")
    steps_step: int = Field(default=1000, description="Steps slider step size")

    default_lr: float = Field(default=3e-4, gt=0, description="Default peak learning rate")
    default_policy_weight: float = Field(
        default=1.0, gt=0, description="Default policy loss weight"
    )
    default_value_weight: float = Field(
        default=1.0, gt=0, description="Default value loss weight"
    )
    default_lbb_weight: float = Field(default=0.1, ge=0, description="Default LBB loss weight")

    # Simulated curve parameters (representative training dynamics)
    policy_loss_scale: float = Field(default=2.5, gt=0, description="Policy loss initial scale")
    value_loss_scale: float = Field(default=0.8, gt=0, description="Value loss initial scale")
    lbb_loss_scale: float = Field(default=0.3, gt=0, description="LBB loss initial scale")
    policy_decay_fraction: float = Field(
        default=0.3, gt=0, lt=1, description="Policy decay as fraction of total steps"
    )
    value_decay_fraction: float = Field(
        default=0.25, gt=0, lt=1, description="Value decay as fraction of total steps"
    )
    lbb_decay_fraction: float = Field(
        default=0.4, gt=0, lt=1, description="LBB decay as fraction of total steps"
    )
    warmup_fraction: float = Field(
        default=0.05, gt=0, lt=0.5, description="Warmup steps as fraction of total steps"
    )
    lbb_const_asymptote: float = Field(
        default=0.05, gt=0, description="LBB constant asymptotic value in simulated curve"
    )
    lbb_const_amplitude: float = Field(
        default=0.08, ge=0, description="LBB constant initial rise amplitude"
    )
    lbb_const_noise_scale: float = Field(
        default=0.005, ge=0, description="Noise scale on the simulated LBB constant curve"
    )
    curve_n_points: int = Field(
        default=200, ge=10, description="Number of plot points in the training curve"
    )
    random_seed: int = Field(
        default=42, description="Random seed for reproducible simulated curves"
    )
    lbb_min_threshold: float = Field(
        default=1e-6, gt=0, description="Threshold line shown on LBB stability chart"
    )


# ---------------------------------------------------------------------------
# Top-level composite config
# ---------------------------------------------------------------------------


class DashboardConfig(BaseModel):
    """Root configuration for the AlphaGalerkin E2E Dashboard."""

    app: AppConfig = Field(default_factory=AppConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    pde: PDEConfig = Field(default_factory=PDEConfig)
    poc: PoCConfig = Field(default_factory=PoCConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# Module-level singleton — import and use directly in tab modules.
DEFAULT_CONFIG: Final[DashboardConfig] = DashboardConfig()

__all__ = [
    "AppConfig",
    "GameConfig",
    "PDEConfig",
    "ComplexityRunConfig",
    "StabilityRunConfig",
    "TransferMilestone",
    "PoCConfig",
    "TrainingConfig",
    "DashboardConfig",
    "DEFAULT_CONFIG",
]
