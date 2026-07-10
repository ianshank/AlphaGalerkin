"""Configuration for the λ-window scheduling ablation.

``HardnessProfileConfig`` parameterises the analytic (ground-truth) variance
profile. ``SchedulingParams`` is the frozen, scenario-independent knob bundle the
game and harness consume (mirrors ``lshape_amr_compare.ComparisonParams``).
``LambdaSchedulingConfig`` is the Pydantic ablation config with every knob
surfaced and typed, plus the acceptance thresholds.

Units: ``error_tolerance`` is in **kcal/mol** — the inherited ``1e-4`` default of
a generic refinement config is physically wrong for a free-energy standard error,
so the validator rejects anything outside ``[0.05, 1.0]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, field_validator, model_validator

from src.poc.config import MetricThreshold
from src.templates.config import BaseModuleConfig


class HardnessProfileConfig(BaseModuleConfig):
    """Analytic hardness ``baseline + peak_amplitude * gaussian(center, width)``."""

    baseline: float = Field(default=1.0, gt=0.0, description="Flat hardness floor.")
    peak_amplitude: float = Field(
        default=6.0, ge=0.0, description="Height of the hardness bump (mock transition)."
    )
    peak_center: float = Field(
        default=0.5, ge=0.0, le=1.0, description="λ location of the hardness peak."
    )
    peak_width: float = Field(
        default=0.08, gt=0.0, le=1.0, description="Gaussian width of the peak."
    )


@dataclass(frozen=True)
class SchedulingParams:
    """Scenario-independent knobs for the game + comparison harness."""

    n_initial_windows: int = 6
    max_windows: int = 16
    batch_samples: int = 200
    sample_budget: int = 4000
    batch_cost_ns: float = 1.0
    split_cost_ns: float = 0.5
    min_window_width: float = 0.02
    allow_split: bool = True
    sample_split_credit: float = 0.5
    max_steps: int = 40
    error_tolerance: float = 0.1  # kcal/mol
    reward_discount: float = 1.0  # gamma on shaped intermediate rewards in MCTS

    def __post_init__(self) -> None:
        """Validate the invariants direct harness callers must honour."""
        if self.n_initial_windows < 2:
            raise ValueError("n_initial_windows must be >= 2")
        if self.max_windows < self.n_initial_windows:
            raise ValueError("max_windows must be >= n_initial_windows")
        if not 0.0 < self.sample_split_credit <= 1.0:
            raise ValueError("sample_split_credit must be in (0, 1]")
        if self.batch_samples < 1:
            raise ValueError("batch_samples must be >= 1")
        if not 0.0 < self.reward_discount <= 1.0:
            raise ValueError("reward_discount must be in (0, 1]")


class LambdaSchedulingConfig(BaseModuleConfig):
    """Ablation config: MCTS vs greedy vs uniform λ-window scheduling."""

    # Scheduling knobs (mirror SchedulingParams; every value explicit).
    n_initial_windows: int = Field(default=6, ge=2, le=64)
    max_windows: int = Field(default=16, ge=2, le=256)
    batch_samples: int = Field(default=200, ge=1)
    sample_budget: int = Field(default=4000, ge=1)
    batch_cost_ns: float = Field(default=1.0, gt=0.0)
    split_cost_ns: float = Field(default=0.5, ge=0.0)
    min_window_width: float = Field(default=0.02, gt=0.0, lt=1.0)
    allow_split: bool = Field(default=True)
    sample_split_credit: float = Field(default=0.5, gt=0.0, le=1.0)
    max_steps: int = Field(default=40, ge=1, le=10_000)
    error_tolerance: float = Field(
        default=0.1,
        description="Convergence tolerance on ΔG standard error, in kcal/mol.",
    )
    reward_discount: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Discount gamma on shaped intermediate rewards in the MCTS arm.",
    )

    # Surrogate / ablation knobs.
    hardness: HardnessProfileConfig = Field(
        default_factory=lambda: HardnessProfileConfig(name="hardness")
    )
    surrogate_bias_sweep: list[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5],
        description="Bias levels for the MismatchedSurrogate sweep.",
    )
    surrogate_noise_amplitude: float = Field(default=0.0, ge=0.0)
    n_seeds: int = Field(default=5, ge=1, le=256)
    n_replicates: int = Field(default=8, ge=1, le=1024)
    n_simulations: int = Field(default=24, ge=1, le=4096)
    c_puct: float = Field(default=1.4, gt=0.0, le=10.0)

    # Acceptance.
    primary_bias: float = Field(
        default=0.25,
        ge=0.0,
        description="The bias level whose MCTS<greedy result is the binding gate.",
    )

    @field_validator("error_tolerance")
    @classmethod
    def _tol_kcal(cls, v: float) -> float:
        if not 0.05 <= v <= 1.0:
            raise ValueError(f"error_tolerance is kcal/mol and must be in [0.05, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> LambdaSchedulingConfig:
        if self.max_windows < self.n_initial_windows:
            raise ValueError("max_windows must be >= n_initial_windows")
        if self.primary_bias not in self.surrogate_bias_sweep:
            raise ValueError(
                f"primary_bias {self.primary_bias} must be one of surrogate_bias_sweep "
                f"{self.surrogate_bias_sweep}"
            )
        return self

    def to_params(self) -> SchedulingParams:
        """Project onto the frozen harness knob bundle."""
        return SchedulingParams(
            n_initial_windows=self.n_initial_windows,
            max_windows=self.max_windows,
            batch_samples=self.batch_samples,
            sample_budget=self.sample_budget,
            batch_cost_ns=self.batch_cost_ns,
            split_cost_ns=self.split_cost_ns,
            min_window_width=self.min_window_width,
            allow_split=self.allow_split,
            sample_split_credit=self.sample_split_credit,
            max_steps=self.max_steps,
            error_tolerance=self.error_tolerance,
            reward_discount=self.reward_discount,
        )

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Acceptance thresholds (config is the single source of truth)."""
        pb = f"{self.primary_bias:g}".replace(".", "p")
        return [
            MetricThreshold(
                name="dG_stderr_ratio_mcts_over_greedy_median",
                operator="<",
                value=1.0,
                description="At bias 0, MCTS ≤ greedy final ΔG stderr (median over seeds).",
            ),
            MetricThreshold(
                name=f"dG_stderr_ratio_at_bias_{pb}_median",
                operator="<",
                value=1.0,
                description=(
                    "The binding gate: planning must survive a "
                    f"{self.primary_bias:g} surrogate bias (median over seeds)."
                ),
            ),
            MetricThreshold(
                name="mcts_win_fraction",
                operator=">=",
                value=0.6,
                description="Fraction of seeds where MCTS beats greedy at primary bias.",
            ),
        ]
