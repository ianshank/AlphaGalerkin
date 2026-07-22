"""Configuration for the ``transfer_baseline_compare`` PoC scenario.

Drives the honest zero-shot-transfer head-to-head: the resolution-independent
:class:`~src.experiments.physics_model.PhysicsOperator` (trained only at
``train_resolution``, applied zero-shot at ``target_resolution``) versus a discrete
CNN **retrained at** ``target_resolution``.

Every tunable is a typed Pydantic ``Field`` with bounds and a docstring; there are no
hardcoded numerical values in the scenario or harness.

The acceptance criterion is deliberately **comparative and falsifiable** — the opposite
of the fabricated "240x better than a fixed threshold" self-comparison it replaces. The
primary gate is ``transfer_mse_ratio_<t>x<t> < transfer_ratio_pass_threshold`` (does the
operator's zero-shot error beat a retrained CNN's at the target resolution?). The
matched-compute variant is recorded but not gated. See
``specs/transfer_baseline_compare.spec.md`` for the contract this config implements.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.poc.config import BaseScenarioConfig, MetricThreshold

SCENARIO_NAME = "transfer_baseline_compare"
"""Registry / YAML dispatch key for the scenario."""


class TransferBaselineCompareConfig(BaseScenarioConfig):
    """Config for the operator-vs-retrained-CNN zero-shot transfer comparison."""

    name: str = Field(default=SCENARIO_NAME, description="Scenario dispatch key.")
    description: str = Field(
        default="Zero-shot transfer: AlphaGalerkin operator vs a retrained discrete CNN.",
        description="Human-readable description.",
    )
    device: str = Field(
        default="cpu",
        description="Device string resolved via src.poc.device.resolve_device.",
    )

    # --- Resolutions ---------------------------------------------------- #
    train_resolution: int = Field(
        default=9,
        ge=3,
        le=25,
        description="Grid the operator is trained on (the CNN zero-shot arm also uses this).",
    )
    target_resolution: int = Field(
        default=19,
        ge=5,
        le=51,
        description="Zero-shot / retrain target grid; the headline ratio is at this size.",
    )
    secondary_resolutions: list[int] = Field(
        default_factory=lambda: [9, 13],
        description="Extra resolutions for the operator's recorded zero-shot curve.",
    )

    # --- Data ----------------------------------------------------------- #
    n_train_samples: int = Field(
        default=5000, ge=64, description="Training samples per arm (matched data volume)."
    )
    n_eval_samples: int = Field(
        default=500, ge=10, description="Samples in the shared held-out eval set."
    )
    n_charges: int = Field(default=5, ge=1, le=20, description="Point charges per PoissonSample.")
    charge_std: float = Field(
        default=1.0, gt=0.0, description="Charge-magnitude standard deviation."
    )
    eval_seed_base: int = Field(
        default=50000,
        ge=0,
        description="Held-out eval seed offset; eval seed = eval_seed_base + resolution.",
    )

    # --- Training (shared by both arms → matched training budget) ------- #
    batch_size: int = Field(default=32, ge=1, description="Training/eval batch size.")
    n_epochs: int = Field(default=100, ge=1, description="Training epochs per arm.")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="AdamW learning rate.")
    n_seeds: int = Field(
        default=5,
        ge=1,
        le=64,
        description=(
            "Independent seeds swept; the gated ratio is the MEDIAN across seeds "
            "(a single training run is high-variance)."
        ),
    )

    # --- Operator architecture (mirrors TransferScenarioConfig defaults) - #
    d_model: int = Field(default=128, ge=8, description="Operator model dimension.")
    n_heads: int = Field(default=4, ge=1, description="Operator attention heads.")
    n_layers: int = Field(default=4, ge=1, description="Operator Galerkin layers.")
    n_fourier_features: int = Field(default=64, ge=1, description="Operator Fourier feature count.")
    fourier_scale: float = Field(
        default=10.0, gt=0.0, description="Operator Fourier feature scale."
    )
    use_fnet: bool = Field(default=True, description="Enable FNet mixing in the operator.")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, description="Operator dropout.")

    # --- CNN baseline architecture -------------------------------------- #
    cnn_n_layers: int = Field(default=6, ge=0, le=32, description="CNN residual blocks.")
    cnn_kernel_size: int = Field(default=3, ge=1, le=7, description="CNN convolution kernel (odd).")
    cnn_channels: int | None = Field(
        default=None,
        ge=1,
        description="CNN channel width; None auto-matches the operator's parameter count.",
    )
    cnn_use_batchnorm: bool = Field(
        default=True, description="Use BatchNorm2d in CNN residual blocks."
    )
    cnn_dropout: float = Field(default=0.0, ge=0.0, lt=1.0, description="CNN dropout.")
    cnn_param_match_tolerance: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Relative band for auto-matching CNN params to the operator's.",
    )

    # --- Comparison / acceptance / artifacts ---------------------------- #
    matched_budget_mode: Literal["grad_steps", "wall_clock"] = Field(
        default="grad_steps",
        description=(
            "Matched-compute CNN budget. 'grad_steps' (CI default, deterministic) "
            "equals the primary CNN arm; 'wall_clock' retrains the CNN for the seconds "
            "the operator's training consumed (distinct end-to-end number)."
        ),
    )
    transfer_ratio_pass_threshold: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description=(
            "Primary gate ceiling: the operator-zero-shot / CNN-retrained MSE ratio at "
            "the target resolution must be STRICTLY below this. A value of 1.0 encodes "
            "the strong win claim (operator zero-shot strictly beats a retrained CNN); "
            "when a measured run shows the operator loses, calibrate this to a regression "
            "ceiling above the measured median (honest-loss handling, see the spec) so the "
            "gate flags a regression rather than asserting a false win."
        ),
    )
    output_dir: str = Field(
        default="results", description="Directory for the committed CSV/PNG artifacts."
    )
    artifact_basename: str = Field(
        default="transfer_baseline_compare",
        description="Basename for the CSV/PNG artifacts (no extension).",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("name")
    @classmethod
    def _lock_name(cls, v: str) -> str:
        if v != SCENARIO_NAME:
            raise ValueError(f"name must be {SCENARIO_NAME!r}, got {v!r}")
        return v

    @field_validator("cnn_kernel_size")
    @classmethod
    def _odd_kernel(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"cnn_kernel_size must be odd, got {v}")
        return v

    @field_validator("secondary_resolutions")
    @classmethod
    def _valid_secondary(cls, v: list[int]) -> list[int]:
        for res in v:
            if res < 3:
                raise ValueError(f"secondary resolutions must be >= 3, got {res}")
        return v

    @field_validator("artifact_basename")
    @classmethod
    def _basename_no_extension(cls, v: str) -> str:
        if not v:
            raise ValueError("artifact_basename must be non-empty")
        if v.endswith((".csv", ".png")):
            raise ValueError("artifact_basename must not include a file extension")
        return v

    @model_validator(mode="after")
    def _resolution_consistency(self) -> TransferBaselineCompareConfig:
        if self.target_resolution <= self.train_resolution:
            raise ValueError(
                "target_resolution must exceed train_resolution for a zero-shot "
                f"upscaling claim ({self.target_resolution} <= {self.train_resolution})"
            )
        return self

    @model_validator(mode="after")
    def _attention_dims_divisible(self) -> TransferBaselineCompareConfig:
        """d_model must be divisible by n_heads (GalerkinAttention splits heads evenly).

        Caught here at config-validation time rather than deep inside the operator build.
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self

    # ------------------------------------------------------------------ #
    # Derived helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def target_metric_name(self) -> str:
        """The resolution-suffixed gated metric name (matches the harness key)."""
        t = self.target_resolution
        return f"transfer_mse_ratio_{t}x{t}"

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Primary (target-resolution) acceptance threshold.

        The matched-compute ratio is recorded but intentionally *not* gated, so the
        scenario tests architecture quality rather than training-speed nondeterminism.
        The gate value should be calibrated from a measured run (see the spec).
        """
        return [
            MetricThreshold(
                name=self.target_metric_name,
                operator="<",
                value=self.transfer_ratio_pass_threshold,
                description=(
                    "Operator-zero-shot / CNN-retrained MSE ratio at the target "
                    f"resolution must be strictly < {self.transfer_ratio_pass_threshold}. "
                    "At the 1.0 default this asserts the operator strictly beats a "
                    "retrained CNN; when calibrated above a measured (losing) median it is "
                    "a regression ceiling, not a win claim."
                ),
            ),
        ]
