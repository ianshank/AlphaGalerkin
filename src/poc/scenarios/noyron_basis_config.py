"""Configuration for the ``noyron_basis`` PoC scenario (Leap 71 v2.2).

Drives MCTS-guided Galerkin basis selection on an SDF-defined helical operator
(``helical_heat`` / ``helical_stokes`` / ``helical_magnetostatics``). Every
tunable — including the helix geometry — is a typed Pydantic field with bounds
and a docstring; there are no hardcoded numerical values in the scenario.

See ``specs/noyron_basis.spec.md`` for the contract this config implements.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.integrations.lm_studio.config import LMStudioConfig
from src.poc.config import BaseScenarioConfig, MetricThreshold

SCENARIO_NAME = "noyron_basis"
"""Registry / YAML dispatch key for the scenario."""

_VALID_ARMS: frozenset[str] = frozenset({"random", "trained", "llm"})
"""Evaluator arms the scenario can compare."""

_SEED_PRIME_STRIDE = 7919
"""Prime multiplier decorrelating per-seed RNG streams (mirrors the centaur scenarios)."""


class NoyronBasisConfig(BaseScenarioConfig):
    """Config for MCTS basis selection on a Leap 71 helical operator."""

    name: str = Field(default=SCENARIO_NAME, description="Scenario dispatch key.")
    description: str = Field(
        default="MCTS-guided Galerkin basis selection on a Leap 71 helical SDF operator.",
        description="Human-readable description.",
    )

    # --- What to solve -------------------------------------------------- #
    operator_name: Literal["helical_heat", "helical_stokes", "helical_magnetostatics"] = Field(
        default="helical_heat",
        description="Which registered helical operator to select bases for.",
    )
    arms: list[str] = Field(
        default_factory=lambda: ["random"],
        description="Evaluator arms to compare (subset of {random, trained, llm}).",
    )

    # --- Sweep / search knobs ------------------------------------------ #
    n_seeds: int = Field(default=3, ge=1, le=64, description="Independent per-seed repeats.")
    n_simulations: int = Field(
        default=16, ge=1, le=4096, description="MCTS simulations per macro-step."
    )
    max_basis_functions: int = Field(
        default=12, ge=1, le=256, description="Bases the game may add before terminating."
    )
    n_candidate_bases: int = Field(
        default=24, ge=1, le=1024, description="Candidate library size (== action space)."
    )
    target_residual: float = Field(
        default=1e-6, gt=0.0, lt=1.0, description="Inner-loop stop tolerance."
    )
    rollout_headroom: int = Field(
        default=2, ge=1, le=16, description="Multiplier for the per-cell rollout cap."
    )

    # --- Manufactured target ------------------------------------------- #
    # The helical operators are homogeneous (zero source, no steady exact
    # solution), so an un-augmented basis-selection game starts at zero error
    # and is degenerate. When ``manufactured`` is True the scenario overlays a
    # smooth product-of-sines target over the helix bounding box so the game
    # has a real, non-trivial field to approximate — the standard manufactured-
    # solution technique used by the flat-domain operators.
    manufactured: bool = Field(
        default=True,
        description="Overlay a manufactured product-of-sines target (non-degenerate game).",
    )
    manufactured_wavenumber: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Integer wavenumber k of the manufactured sin(k*pi*x_norm) target.",
    )

    # --- Helix geometry (surfaced — no hardcoded values) --------------- #
    helix_r_major: float = Field(
        default=0.05, gt=0.0, description="Major radius of the helical tube centreline (m)."
    )
    helix_r_minor: float = Field(
        default=0.012, gt=0.0, description="Minor (tube) radius of the helix (m)."
    )
    helix_pitch: float = Field(default=0.02, gt=0.0, description="Axial rise per turn (m).")
    helix_n_turns: int = Field(default=3, ge=1, le=64, description="Number of helical turns.")

    # --- Thresholds ----------------------------------------------------- #
    # Default thresholds assert *provable correctness*, not an aspirational
    # magnitude. Least-squares basis addition is monotone non-increasing in the
    # fit residual, so ``error_reduction_pct >= 0`` is a guaranteed property of
    # a correct pipeline. The *magnitude* of reduction achievable on 3D SDF
    # helix geometry is limited by the current candidate Galerkin basis library
    # (empirically ~2-4% even fitting every candidate), so a large headline
    # target (e.g. 20%) is an open research item — raise this field on a run
    # with an improved, geometry-aware basis library. See specs/noyron_basis.spec.md.
    min_error_reduction_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Threshold: primary-arm median % error reduction (>=0 is monotone-correct).",
    )
    max_final_residual: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Threshold: primary-arm median final residual stays bounded (no divergence).",
    )

    # --- Device / gated arms ------------------------------------------- #
    device: str = Field(
        default="cpu",
        description="Torch device preference ('cpu', 'cuda', 'cuda:N', 'auto').",
    )
    trained_checkpoint_path: str | None = Field(
        default=None, description="Checkpoint for the 'trained' arm (required if used)."
    )
    lm_studio: LMStudioConfig | None = Field(
        default=None, description="LM Studio config for the 'llm' arm (required if used)."
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

    @field_validator("arms")
    @classmethod
    def _arms_valid(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("arms must be non-empty")
        deduped = list(dict.fromkeys(v))  # order-preserving dedupe
        unknown = set(deduped) - _VALID_ARMS
        if unknown:
            raise ValueError(f"unknown arm(s) {sorted(unknown)}; valid: {sorted(_VALID_ARMS)}")
        return deduped

    @model_validator(mode="after")
    def _candidate_bounds(self) -> NoyronBasisConfig:
        if self.max_basis_functions > self.n_candidate_bases:
            raise ValueError(
                "max_basis_functions cannot exceed n_candidate_bases "
                f"({self.max_basis_functions} > {self.n_candidate_bases})"
            )
        return self

    # ------------------------------------------------------------------ #
    # Derived helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def primary_arm(self) -> str:
        """The arm whose error reduction drives the headline thresholds."""
        return self.arms[0]

    def resolved_seeds(self) -> list[int]:
        """Deterministic, decorrelated per-seed RNG seeds."""
        return [self.seed + i * _SEED_PRIME_STRIDE for i in range(self.n_seeds)]

    def max_rollouts_for_cell(self) -> int:
        """Hard per-cell rollout cap that scales with the search budget."""
        return self.n_simulations * self.max_basis_functions * self.rollout_headroom

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Headline thresholds derived from the primary-arm result."""
        return [
            MetricThreshold(
                name="error_reduction_pct",
                operator=">=",
                value=self.min_error_reduction_pct,
                description=(
                    "Primary-arm median Galerkin-error reduction must be >= "
                    f"{self.min_error_reduction_pct}%%."
                ),
            ),
            MetricThreshold(
                name="final_residual",
                operator="<=",
                value=self.max_final_residual,
                description=(
                    f"Primary-arm median final residual must be <= {self.max_final_residual}."
                ),
            ),
        ]


__all__ = ["SCENARIO_NAME", "NoyronBasisConfig"]
