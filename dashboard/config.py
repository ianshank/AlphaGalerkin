"""Configuration models for the AlphaGalerkin E2E Dashboard.

All runtime-tunable constants live here as Pydantic-validated fields so that
no magic numbers appear inside tab or app logic.  Create a custom config by
subclassing or by constructing with keyword overrides:

    cfg = DashboardConfig(app=AppConfig(port=8080))
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field, field_validator

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

    @field_validator("achieved_mse")
    @classmethod
    def validate_achieved_mse(cls, v: dict[int, float]) -> dict[int, float]:
        """Ensure the map is non-empty and all MSE values are strictly positive."""
        if not v:
            raise ValueError("achieved_mse must contain at least one entry")
        non_positive = {k: val for k, val in v.items() if val <= 0}
        if non_positive:
            raise ValueError(f"achieved_mse values must be > 0; invalid entries: {non_positive}")
        return v


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
    default_value_weight: float = Field(default=1.0, gt=0, description="Default value loss weight")
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


# ---------------------------------------------------------------------------
# Reentry TPS tab
# ---------------------------------------------------------------------------


class ReentryConfig(BaseModel):
    """Configuration for the Reentry Thermal Protection System tab."""

    grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19, 25, 32],
        description="Dropdown options for N in the N×N grid",
    )
    default_grid_size: int = Field(default=13, ge=3, description="Initially selected grid size")
    comparison_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes used in resolution-comparison plot",
    )
    kappa_min: float = Field(default=0.01, gt=0, description="Min thermal diffusivity (m²/s)")
    kappa_max: float = Field(default=0.5, description="Max thermal diffusivity")
    default_kappa: float = Field(default=0.1, description="Default thermal diffusivity")
    surface_temp_min: float = Field(default=1000.0, description="Min surface temperature (K)")
    surface_temp_max: float = Field(default=3500.0, description="Max surface temperature (K)")
    default_surface_temp: float = Field(default=2500.0, description="Default surface temperature")
    interior_temp: float = Field(default=300.0, description="Initial interior temperature (K)")
    bondline_temp_limit: float = Field(
        default=450.0, description="Max allowable bondline temperature (K)"
    )
    velocity_min: float = Field(default=3.0, description="Min reentry velocity (km/s)")
    velocity_max: float = Field(default=12.0, description="Max reentry velocity (km/s)")
    default_velocity: float = Field(default=7.5, description="Default reentry velocity")
    total_time_min: float = Field(default=0.1, gt=0, description="Min simulation time (s)")
    total_time_max: float = Field(default=5.0, description="Max simulation time (s)")
    default_total_time: float = Field(default=1.0, description="Default simulation time")
    n_time_snapshots: int = Field(default=5, ge=2, description="Time snapshots to display")
    epsilon: float = Field(default=1e-9, gt=0, description="Guard against division by zero")


# ---------------------------------------------------------------------------
# Wildfire spread tab
# ---------------------------------------------------------------------------


class WildfireConfig(BaseModel):
    """Configuration for the Wildfire Spread Simulation tab."""

    grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19, 25, 32],
        description="Dropdown options for N in the N×N grid",
    )
    default_grid_size: int = Field(default=19, ge=3, description="Initially selected grid size")
    comparison_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes for resolution-comparison plot",
    )
    ignition_patterns: list[str] = Field(
        default_factory=lambda: ["Center", "Edge", "Corner", "Line", "Random"],
        description="Available ignition pattern options",
    )
    default_ignition: str = Field(default="Center", description="Initially selected pattern")
    wind_speed_min: float = Field(default=0.0, description="Min wind speed (m/s)")
    wind_speed_max: float = Field(default=20.0, description="Max wind speed (m/s)")
    default_wind_speed: float = Field(default=5.0, description="Default wind speed")
    wind_direction_min: float = Field(default=0.0, description="Min wind direction (degrees)")
    wind_direction_max: float = Field(default=360.0, description="Max wind direction (degrees)")
    default_wind_direction: float = Field(default=45.0, description="Default wind direction")
    diffusion_min: float = Field(default=0.01, gt=0, description="Min thermal diffusion")
    diffusion_max: float = Field(default=0.5, description="Max thermal diffusion")
    default_diffusion: float = Field(default=0.1, description="Default diffusion rate")
    fuel_density_min: float = Field(default=0.1, gt=0, description="Min fuel density")
    fuel_density_max: float = Field(default=2.0, description="Max fuel density")
    default_fuel_density: float = Field(default=1.0, description="Default fuel density")
    ignition_threshold: float = Field(default=0.5, description="Temperature for ignition")
    total_time_min: float = Field(default=0.5, gt=0, description="Min simulation time")
    total_time_max: float = Field(default=10.0, description="Max simulation time")
    default_total_time: float = Field(default=3.0, description="Default simulation time")
    n_time_snapshots: int = Field(default=6, ge=2, description="Time snapshots to display")
    epsilon: float = Field(default=1e-9, gt=0, description="Guard against division by zero")


# ---------------------------------------------------------------------------
# Missile defense tab
# ---------------------------------------------------------------------------


class MissileDefenseConfig(BaseModel):
    """Configuration for the Missile Defense Intercept Analysis tab."""

    grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19, 25, 32],
        description="Dropdown options for flow-field resolution N×N",
    )
    default_grid_size: int = Field(default=19, ge=3, description="Initially selected grid size")
    comparison_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes for resolution-comparison plot",
    )
    threat_angle_min: float = Field(default=20.0, description="Min launch angle (degrees)")
    threat_angle_max: float = Field(default=80.0, description="Max launch angle (degrees)")
    default_threat_angle: float = Field(default=45.0, description="Default threat launch angle")
    threat_velocity_min: float = Field(default=1.0, description="Min threat velocity (km/s)")
    threat_velocity_max: float = Field(default=7.0, description="Max threat velocity (km/s)")
    default_threat_velocity: float = Field(default=3.0, description="Default threat velocity")
    interceptor_x_min: float = Field(default=0.1, description="Min interceptor X (normalised)")
    interceptor_x_max: float = Field(default=0.9, description="Max interceptor X (normalised)")
    default_interceptor_x: float = Field(default=0.7, description="Default interceptor X")
    interceptor_y_min: float = Field(default=0.0, description="Min interceptor Y (normalised)")
    interceptor_y_max: float = Field(default=0.5, description="Max interceptor Y (normalised)")
    default_interceptor_y: float = Field(default=0.0, description="Default interceptor Y")
    interceptor_speed_min: float = Field(default=1.0, description="Min interceptor speed (km/s)")
    interceptor_speed_max: float = Field(default=10.0, description="Max interceptor speed (km/s)")
    default_interceptor_speed: float = Field(default=5.0, description="Default interceptor speed")
    dt: float = Field(default=0.01, gt=0, description="Time step for trajectory integration")
    max_time: float = Field(default=10.0, gt=0, description="Max simulation time")
    gravity: float = Field(default=9.81, gt=0, description="Gravitational acceleration (m/s²)")
    kill_radius: float = Field(default=0.05, gt=0, description="Kill radius for P_kill model")
    epsilon: float = Field(default=1e-9, gt=0, description="Guard against division by zero")


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
    reentry: ReentryConfig = Field(default_factory=ReentryConfig)
    wildfire: WildfireConfig = Field(default_factory=WildfireConfig)
    missile_defense: MissileDefenseConfig = Field(default_factory=MissileDefenseConfig)


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
    "ReentryConfig",
    "WildfireConfig",
    "MissileDefenseConfig",
    "DashboardConfig",
    "DEFAULT_CONFIG",
]
