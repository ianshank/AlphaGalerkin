"""Configuration models for the AlphaGalerkin E2E Dashboard.

All runtime-tunable constants live here as Pydantic-validated fields so that
no magic numbers appear inside tab or app logic.  Create a custom config by
subclassing or by constructing with keyword overrides:

    cfg = DashboardConfig(app=AppConfig(port=8080))
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BaseModel, Field, field_validator, model_validator

# Evaluation resolution the committed transfer benchmark reports baselines for.
# ``TransferMilestone``'s CNN baseline fields and ratio are specific to it, so the
# UI must pin its comparison here rather than deriving a target from the MSE map.
COMMITTED_TARGET_RESOLUTION: Final[int] = 19

# Relative tolerance when checking that the displayed transfer ratio agrees with the
# two displayed MSEs it is the quotient of. Loose enough to absorb the rounding in a
# hand-copied committed figure, tight enough that a genuine override mismatch fails.
RATIO_CONSISTENCY_RTOL: Final[float] = 1e-3

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
    """Committed zero-shot transfer result, operator vs. a retrained CNN baseline.

    Every value here is the committed benchmark figure, not a spike run. The operator
    transfers without retraining but is roughly an order of magnitude *less* accurate
    than a CNN retrained at the target resolution -- the value is zero retraining, not
    peak accuracy. See ``specs/transfer_baseline_compare.spec.md``.

    **Provenance, stated precisely.** All arms come from the *representative
    (median-ranked) seed* of the committed 3-seed run. The operator's
    ``COMMITTED_TARGET_RESOLUTION`` MSE is itself the 3-seed median; each baseline is
    that same seed's *paired* value, which is why ``transfer_ratio_19x19`` is an
    honest within-seed ratio. The per-metric medians of the CNN arms differ (retrained
    1.434e-04, zero-shot 3.153e-04) -- do **not** describe the baselines as medians.

    ``tests/dashboard/test_config.py`` and the charter's UI-claim guard both assert these
    defaults agree with ``config/baselines/transfer_ci.json``; update both together.
    """

    train_resolution: int = Field(default=9, description="Resolution used for training")
    mse_threshold: float = Field(
        default=0.05,
        gt=0,
        description=(
            "Legacy PoC pass/fail MSE threshold. Retained for reference only -- a ratio "
            "against this threshold is NOT a result and must not be rendered; the honest "
            "comparison is against the retrained-CNN baseline below."
        ),
    )
    # allow_inf_nan=False on the *value* type: the validators below reject non-positive
    # entries, but `nan <= 0` and `inf <= 0` are both False, so NaN and infinity would slip
    # through and render as a benchmark figure. The scalar fields below are already covered
    # by `gt=0` (both `nan > 0` and `inf > 0` fail pydantic's check for NaN); the constraint
    # is repeated there to make the intent explicit rather than implicit.
    achieved_mse: dict[int, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        default_factory=lambda: {
            9: 1.38227075e-04,
            13: 1.85244250e-03,
            19: 2.30064808e-03,
        },
        description=(
            "Operator zero-shot MSE per resolution, trained at 9x9 only. Representative "
            "(median-ranked) seed from results/transfer_baseline_compare.csv; the "
            "19x19 entry is also the 3-seed median and equals "
            "mse_alphagalerkin_zeroshot_19x19 in config/baselines/transfer_ci.json. "
            "Must contain COMMITTED_TARGET_RESOLUTION. Benchmark: "
            "specs/transfer_baseline_compare.spec.md"
        ),
    )
    cnn_retrained_mse_19x19: float = Field(
        default=1.62955009e-04,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Discrete CNN retrained at 19x19 -- the honest baseline the operator is measured "
            "against. Representative seed's paired value (NOT the 3-seed median, which is "
            "1.433924e-04). From config/baselines/transfer_ci.json::mse_cnn_retrained_19x19"
        ),
    )
    cnn_zeroshot_mse_19x19: float = Field(
        default=7.65602237e-05,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Discrete CNN evaluated zero-shot at 19x19. Representative seed's paired value "
            "(NOT the 3-seed median, which is 3.153264e-04). From "
            "config/baselines/transfer_ci.json::mse_cnn_zeroshot_19x19"
        ),
    )
    transfer_ratio_19x19: float = Field(
        default=14.118302311554318,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Operator zero-shot MSE divided by retrained-CNN MSE at 19x19, computed "
            "within the representative seed. Greater than 1 means the operator loses. "
            "From config/baselines/transfer_ci.json::transfer_mse_ratio_19x19"
        ),
    )
    milestone_date: str = Field(
        default="2026-07-22", description="Date the committed benchmark was recorded"
    )

    @field_validator("achieved_mse")
    @classmethod
    def validate_achieved_mse(cls, v: dict[int, float]) -> dict[int, float]:
        """Validate the MSE map, including presence of the committed target resolution.

        The baseline fields above are ``19x19``-specific by construction. If the map
        could omit 19, the UI would compare the operator at some other resolution
        against 19x19 baselines and label the result ``{target}x{target}`` -- exactly
        the mislabelled comparison this config exists to prevent.
        """
        if not v:
            raise ValueError("achieved_mse must contain at least one entry")
        non_positive = {k: val for k, val in v.items() if val <= 0}
        if non_positive:
            raise ValueError(f"achieved_mse values must be > 0; invalid entries: {non_positive}")
        if COMMITTED_TARGET_RESOLUTION not in v:
            raise ValueError(
                f"achieved_mse must contain the committed target resolution "
                f"{COMMITTED_TARGET_RESOLUTION}; the baseline fields "
                f"(cnn_retrained_mse_19x19, cnn_zeroshot_mse_19x19, transfer_ratio_19x19) "
                f"are specific to it. Got resolutions: {sorted(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_ratio_matches_operands(self) -> TransferMilestone:
        """The rendered ratio must equal the rendered operands that produce it.

        ``transfer_ratio_19x19`` is displayed beside ``achieved_mse[19]`` and
        ``cnn_retrained_mse_19x19``. Overriding one without the others would render a
        ratio that contradicts the two numbers printed next to it -- the same
        label-disagrees-with-operands defect this config exists to prevent.
        """
        operator = self.achieved_mse[COMMITTED_TARGET_RESOLUTION]
        derived = operator / self.cnn_retrained_mse_19x19
        if abs(derived - self.transfer_ratio_19x19) > RATIO_CONSISTENCY_RTOL * derived:
            raise ValueError(
                f"transfer_ratio_19x19={self.transfer_ratio_19x19!r} contradicts its own "
                f"operands: achieved_mse[{COMMITTED_TARGET_RESOLUTION}]={operator!r} / "
                f"cnn_retrained_mse_19x19={self.cnn_retrained_mse_19x19!r} = {derived!r}. "
                f"Override all three together, or none."
            )
        return self


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
