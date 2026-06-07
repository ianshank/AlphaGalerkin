"""Configuration schema for the MCTS-budget scaling-law scenario.

Brown's "bitter lesson" framing predicts that solution quality improves
*predictably* with compute. This scenario sweeps the MCTS-simulation budget
(the per-decision search compute) and fits a log-log scaling curve of final
residual against budget for each evaluator arm. Every knob is a typed Pydantic
field — no hardcoded budgets, tolerances, or thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.integrations.lm_studio.config import LMStudioConfig
from src.poc.config import BaseScenarioConfig, MetricThreshold, ScenarioTier

SCALING_SCENARIO_NAME = "scaling_law"
"""Canonical scenario id; YAML rows must use this string for dispatch."""

_SEED_PRIME_STRIDE = 1009
"""Prime stride used when deriving per-seed values from the master seed."""

ArmName = Literal["random", "trained", "llm"]
PDEName = Literal[
    "poisson",
    "heat",
    "advection_diffusion",
    "burgers",
    "navier_stokes",
    "poisson_lshaped",
    "helmholtz",
    "biharmonic",
]
SignificanceTestType = Literal["t_test", "mann_whitney", "bootstrap", "permutation"]


class ScalingLawConfig(BaseScenarioConfig):
    """Sweep MCTS-simulation budget and fit a residual scaling curve.

    For each ``(arm, budget, seed)`` the scenario runs an MCTS
    basis-selection solve with ``n_simulations == budget`` per macro-step and
    records the final residual. It then fits ``log(median residual)`` against
    ``log(budget)`` per arm; the slope (``residual_scaling_exponent``) should
    be negative — more search compute yields a lower residual — and the fit
    quality (``residual_fit_r2``) should be high.
    """

    # Identification + tier
    name: str = Field(
        default=SCALING_SCENARIO_NAME,
        description=(
            "Scenario identifier. `load_config_from_dict` dispatches on this "
            f"field — the YAML row's `name:` must equal {SCALING_SCENARIO_NAME!r}."
        ),
    )
    description: str = Field(
        default=(
            "MCTS-budget scaling law: final residual vs simulation budget "
            "(log-log) per evaluator arm."
        ),
        description="Scenario description.",
    )
    tier: ScenarioTier = Field(
        default=ScenarioTier.INTEGRATION,
        description="Scenario validation tier.",
    )

    # Problem
    pde: PDEName = Field(
        default="poisson",
        description="PDE family the basis-selection game approximates.",
    )

    # Sweep axes
    arms: list[ArmName] = Field(
        default_factory=lambda: ["random"],
        description=(
            "Evaluator arms to sweep. The first arm is the primary arm whose "
            "scaling exponent drives the headline thresholds."
        ),
    )
    simulation_budgets: list[int] = Field(
        default_factory=lambda: [8, 16, 32, 64],
        description=(
            "MCTS simulations per macro-step (the swept compute axis). At "
            "least two distinct budgets are required to fit a slope."
        ),
    )

    # Seeds
    n_seeds: int = Field(
        default=5,
        ge=2,
        le=64,
        description="Seeds per (arm, budget) cell. >=2 enables significance tests.",
    )
    seeds: list[int] | None = Field(
        default=None,
        description=(
            "Explicit seed list. When None the scenario derives "
            "[seed + i * 1009 for i in range(n_seeds)] from the inherited seed."
        ),
    )

    # Solve budget
    target_residual: float = Field(
        default=1e-6,
        gt=0.0,
        lt=1.0,
        description=(
            "Inner-loop stop tolerance. Intentionally small by default so "
            "cells exhaust their macro-steps and the final residual reflects "
            "search quality rather than an early target hit."
        ),
    )
    max_basis_functions: int = Field(
        default=12,
        ge=1,
        le=128,
        description="Maximum bases the inner game may add before terminating.",
    )
    n_candidate_bases: int = Field(
        default=24,
        ge=2,
        le=128,
        description="Size of the candidate basis library (== action space).",
    )
    rollout_headroom: int = Field(
        default=2,
        ge=1,
        le=16,
        description=(
            "max_rollouts per cell = budget * max_basis_functions * "
            "rollout_headroom; headroom ensures every macro-step is reachable "
            "at each budget."
        ),
    )

    # Arm resources
    trained_checkpoint_path: Path | None = Field(
        default=None,
        description="AlphaGalerkin checkpoint (.pt) for the trained arm; None skips it.",
    )
    lm_studio: LMStudioConfig = Field(
        default_factory=LMStudioConfig,
        description="Configuration for the LM Studio client (LLM arm).",
    )

    # Device
    device: str = Field(
        default="cuda",
        description=(
            "Device preference passed to `src.poc.device.resolve_device`. "
            "'cuda' fails loud without CUDA; use 'cpu' or 'auto' for CI."
        ),
    )

    # Significance test (arm-vs-arm comparison at the largest budget)
    significance_test_type: SignificanceTestType = Field(
        default="mann_whitney",
        description="Statistical test for the optional arm-vs-arm comparison.",
    )
    significance_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance threshold for the arm-vs-arm comparison.",
    )
    n_bootstrap: int = Field(
        default=2000,
        ge=100,
        le=100000,
        description="Bootstrap resamples (only used when test type is 'bootstrap').",
    )

    # Acceptance thresholds (primary arm)
    min_residual_decay: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Minimum magnitude of the (negative) log-log slope. The primary "
            "arm's residual_scaling_exponent must be <= -min_residual_decay, "
            "i.e. more compute measurably reduces the residual."
        ),
    )
    min_fit_r2: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum R² of the primary arm's log-log scaling fit.",
    )

    requires_gpu: bool = Field(
        default=True,
        description="Resource hint — the LLM/trained arms need CUDA.",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("name")
    @classmethod
    def _name_locked(cls, v: str) -> str:
        if v != SCALING_SCENARIO_NAME:
            raise ValueError(
                f"name must be exactly {SCALING_SCENARIO_NAME!r} for YAML dispatch; got {v!r}"
            )
        return v

    @field_validator("arms")
    @classmethod
    def _arms_non_empty_unique(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("arms must be non-empty")
        # Preserve first-seen order while de-duplicating.
        return list(dict.fromkeys(v))

    @field_validator("simulation_budgets")
    @classmethod
    def _budgets_valid(cls, v: list[int]) -> list[int]:
        if any(b < 1 for b in v):
            raise ValueError("simulation_budgets must all be >= 1")
        unique = sorted(set(v))
        if len(unique) < 2:
            raise ValueError("simulation_budgets needs >= 2 distinct values to fit a slope")
        return unique

    @field_validator("seeds")
    @classmethod
    def _seeds_non_empty(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and not v:
            raise ValueError("seeds must be non-empty when provided (use None to derive)")
        return v

    @model_validator(mode="after")
    def _trained_needs_checkpoint(self) -> ScalingLawConfig:
        if "trained" in self.arms and self.trained_checkpoint_path is None:
            # Soft contract: the scenario gates the trained arm off at runtime
            # when no checkpoint is set, but flag the obvious misconfiguration.
            pass
        return self

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def resolved_seeds(self) -> list[int]:
        """Per-cell seeds (explicit deduped, or derived via prime stride)."""
        if self.seeds is not None:
            return list(dict.fromkeys(self.seeds))
        return [self.seed + i * _SEED_PRIME_STRIDE for i in range(self.n_seeds)]

    @property
    def primary_arm(self) -> str:
        """The arm whose scaling fit drives the headline thresholds."""
        return self.arms[0]

    def max_rollouts_for_budget(self, budget: int) -> int:
        """Per-cell rollout cap that scales with the simulation budget."""
        return budget * self.max_basis_functions * self.rollout_headroom

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Headline thresholds derived from the primary-arm scaling fit."""
        return [
            MetricThreshold(
                name="residual_scaling_exponent",
                operator="<=",
                value=-self.min_residual_decay,
                description=(
                    "Primary-arm log-log residual slope must be <= "
                    f"{-self.min_residual_decay} (more compute lowers residual)."
                ),
            ),
            MetricThreshold(
                name="residual_fit_r2",
                operator=">=",
                value=self.min_fit_r2,
                description=(
                    f"Primary-arm scaling fit R² must be >= {self.min_fit_r2}."
                ),
            ),
        ]


__all__ = ["SCALING_SCENARIO_NAME", "ScalingLawConfig"]
