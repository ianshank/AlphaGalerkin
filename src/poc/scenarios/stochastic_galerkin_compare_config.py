"""Configuration for the ``stochastic_galerkin_compare`` PoC scenario.

Drives the two-arm Fokker-Planck/OU density benchmark
(``src.research.stochastic_galerkin_compare``): the deterministic
Galerkin-attention path (a small ``PhysicsOperator`` trained supervised on
density fields) versus the new stochastic Galerkin moment-projection path
(Strang propagation + density rendering, no training).

Honesty rule (spec Thresholds, ``docs/proposals/PRIOR_ART_REVIEW.md``
"novelty ≠ superiority"): the **only** gate is the stochastic arm's absolute
density MSE — on this benchmark the stochastic path is near-exact by
construction, so gating a stochastic-vs-deterministic ratio would be a
self-serving benchmark. The deterministic arm's MSE, the ratio, wall-clocks,
and parameter counts are recorded ungated.

Spec: specs/stochastic_galerkin_nke.spec.md (AC8, Thresholds).
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from src.poc.config import BaseScenarioConfig, MetricThreshold

SCENARIO_NAME = "stochastic_galerkin_compare"
"""Registry / YAML dispatch key for the scenario."""

DEFAULT_STOCHASTIC_MSE_GATE = 1e-6
"""Ceiling on the stochastic arm's density MSE on the shared grid.

Calibrated from the measured default-budget run (spec: Calibration
procedure): observed 2.3e-8 at grid 32 / strang_dt 0.1 — the 1e-6 gate
carries ~40× headroom while still failing on a broken moment solver. The
floor is set by the Strang splitting error in the rendered density, not by
the grid (rendering is exact evaluation).
"""

_SEED_PRIME_STRIDE = 1009
"""Stride between derived per-seed training runs (prime, mirrors scaling_law)."""


class StochasticGalerkinCompareConfig(BaseScenarioConfig):
    """Config for the deterministic-vs-stochastic Galerkin density benchmark."""

    name: str = Field(default=SCENARIO_NAME, description="Scenario dispatch key.")
    description: str = Field(
        default="Fokker-Planck/OU density: Galerkin attention vs stochastic Galerkin projection.",
        description="Human-readable description.",
    )
    device: str = Field(
        default="cpu",
        description="Device string resolved via src.poc.device.resolve_device.",
    )

    # --- Shared benchmark ------------------------------------------------ #
    grid_n: int = Field(default=32, ge=4, le=128, description="Shared eval grid is grid_n².")
    domain_half_width: float = Field(default=2.0, gt=0.0, description="Square domain is [-w, w]².")
    drift_matrix: list[list[float]] = Field(
        default_factory=lambda: [[-1.0, 0.3], [0.0, -0.8]],
        description="OU drift matrix A (2×2, stable).",
    )
    drift_bias: list[float] = Field(
        default_factory=lambda: [0.1, -0.2], description="OU drift bias b (length 2)."
    )
    diffusion: list[list[float]] = Field(
        default_factory=lambda: [[0.4, 0.0], [0.0, 0.3]],
        description="Diffusion factor g (2×m); Q = g gᵀ.",
    )
    t_end: float = Field(default=1.0, gt=0.0, description="Benchmark horizon T.")
    strang_dt: float = Field(default=0.1, gt=0.0, description="Stochastic-arm Strang step.")
    n_train_samples: int = Field(
        default=64, ge=1, description="Training ICs for the deterministic arm."
    )
    n_eval_samples: int = Field(default=16, ge=1, description="Shared held-out eval ICs.")
    m0_half_range: float = Field(
        default=0.5, gt=0.0, description="Initial means drawn from [-r, r]²."
    )
    p0_min: float = Field(default=0.1, gt=0.0, description="Min initial marginal variance.")
    p0_max: float = Field(default=0.3, gt=0.0, description="Max initial marginal variance.")
    eval_seed_base: int = Field(
        default=9973,
        ge=0,
        description="Eval ICs depend on THIS seed only (shared-eval-set fairness, AC8).",
    )

    # --- Deterministic arm (PhysicsOperator) budget ---------------------- #
    d_model: int = Field(default=32, ge=8, description="Operator model dimension.")
    n_heads: int = Field(default=2, ge=1, description="Operator attention heads.")
    n_layers: int = Field(default=2, ge=1, description="Operator Galerkin layers.")
    n_fourier_features: int = Field(default=16, ge=1, description="Operator Fourier features.")
    fourier_scale: float = Field(default=5.0, gt=0.0, description="Fourier feature scale.")
    use_fnet: bool = Field(default=False, description="Enable FNet mixing in the operator.")
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0, description="Operator dropout.")
    n_epochs: int = Field(default=40, ge=1, description="Deterministic-arm training epochs.")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="AdamW learning rate.")
    batch_size: int = Field(default=8, ge=1, description="Training batch size.")
    n_seeds: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "Training seeds swept for the deterministic arm (the stochastic arm is "
            "seed-independent); recorded metrics use the median run."
        ),
    )

    # --- Gate + artifacts ------------------------------------------------- #
    stochastic_mse_gate: float = Field(
        default=DEFAULT_STOCHASTIC_MSE_GATE,
        gt=0.0,
        description="The single gated threshold: stochastic_density_mse < this value.",
    )
    output_dir: str = Field(default="results", description="Artifact output directory.")
    artifact_basename: str = Field(
        default="stochastic_galerkin_compare",
        description="Basename for the CSV/PNG artifacts (no extension).",
    )

    @field_validator("name")
    @classmethod
    def _name_locked(cls, value: str) -> str:
        if value != SCENARIO_NAME:
            msg = f"name must be {SCENARIO_NAME!r} (YAML dispatch key); got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("artifact_basename")
    @classmethod
    def _basename_valid(cls, value: str) -> str:
        if not value:
            msg = "artifact_basename must be non-empty"
            raise ValueError(msg)
        if "." in value:
            msg = "artifact_basename must not include a file extension"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _cross_field(self) -> StochasticGalerkinCompareConfig:
        if self.strang_dt > self.t_end:
            msg = f"strang_dt ({self.strang_dt}) must not exceed t_end ({self.t_end})"
            raise ValueError(msg)
        if self.p0_min > self.p0_max:
            msg = f"p0_min ({self.p0_min}) must not exceed p0_max ({self.p0_max})"
            raise ValueError(msg)
        if len(self.drift_bias) != 2:
            msg = "the shared benchmark is 2D (drift_bias must have length 2)"
            raise ValueError(msg)
        if len(self.drift_matrix) != 2 or any(len(r) != 2 for r in self.drift_matrix):
            msg = "drift_matrix must be 2x2"
            raise ValueError(msg)
        if len(self.diffusion) != 2:
            msg = "diffusion must have 2 rows"
            raise ValueError(msg)
        return self

    def resolved_seeds(self) -> list[int]:
        """Derived per-run training seeds: seed + i·stride (prime stride)."""
        return [self.seed + i * _SEED_PRIME_STRIDE for i in range(self.n_seeds)]

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """The single gated metric (spec Thresholds table; AQA-asserted)."""
        return [
            MetricThreshold(
                name="stochastic_density_mse",
                operator="<",
                value=self.stochastic_mse_gate,
                description=(
                    "The stochastic Galerkin arm reproduces the analytic Fokker-Planck "
                    "density on the shared grid. The deterministic arm's MSE and the "
                    "stochastic/deterministic ratio are recorded UNGATED (novelty ≠ "
                    "superiority; gating a ratio the stochastic arm wins by construction "
                    "would be self-serving)."
                ),
            ),
        ]
