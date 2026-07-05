"""Configuration for the ``lshape_amr_compare`` PoC scenario.

Drives the thesis-critical head-to-head on the standard L-shaped Poisson AMR
benchmark: an MCTS refinement policy versus classical Dörfler bulk marking, on
an identical masked solver / residual estimator / geometry / DOF accounting.

Every tunable is a typed Pydantic ``Field`` with bounds and a docstring; there
are no hardcoded numerical values in the scenario or harness.

The acceptance criterion is deliberately **comparative and falsifiable** — the
opposite of a monotone ``>= 0`` self-test. The primary gate is
``l2_error_ratio_at_matched_dof < 1.0`` (does the MCTS refinement *policy* beat
Dörfler at matched DOF?). The end-to-end matched-wall-clock ratio
(``error_per_dof_ratio_mcts_over_dorfler``) is recorded as a transparent
secondary metric — it is *expected* to exceed 1 for an untrained MCTS because
each refinement costs ``n_simulations`` real solves — but it does not gate, so
the scenario tests policy quality rather than search-implementation speed.

See ``specs/lshape_amr_compare.spec.md`` for the contract this config implements.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from src.poc.config import BaseScenarioConfig, MetricThreshold

SCENARIO_NAME = "lshape_amr_compare"
"""Registry / YAML dispatch key for the scenario."""


class LShapeAMRCompareConfig(BaseScenarioConfig):
    """Config for the L-shaped Poisson MCTS-vs-Dörfler AMR comparison."""

    name: str = Field(default=SCENARIO_NAME, description="Scenario dispatch key.")
    description: str = Field(
        default="L-shaped Poisson AMR: MCTS refinement policy vs Dörfler marking.",
        description="Human-readable description.",
    )
    device: str = Field(
        default="cpu",
        description="Device string resolved via src.poc.device.resolve_device.",
    )

    # --- Domain / discretisation --------------------------------------- #
    scale: float = Field(
        default=1.0, gt=0.0, le=100.0, description="L-shape half-width s (domain [-s,s]^2)."
    )
    initial_side: int = Field(
        default=4,
        ge=2,
        le=64,
        description=(
            "Elements per axis on the shared coarse grid (nodes = side+1). Must be "
            "even so the reentrant corner at the origin is a grid node."
        ),
    )
    max_dof: int = Field(
        default=400,
        ge=10,
        le=1_000_000,
        description="Active-DOF budget at which both arms stop refining.",
    )
    max_steps: int = Field(
        default=30, ge=1, le=10_000, description="Max refinement steps for the MCTS arm."
    )
    error_tolerance: float = Field(
        default=1e-6,
        gt=0.0,
        lt=1.0,
        description="Early-stop L2 error tolerance shared by both arms.",
    )

    # --- Dörfler arm ---------------------------------------------------- #
    marking_fraction: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Dörfler bulk-marking fraction theta (reused from AMRConfig).",
    )
    max_refinements: int = Field(
        default=30, ge=1, le=1000, description="Max Dörfler refinement levels."
    )

    # --- MCTS arm ------------------------------------------------------- #
    n_candidate_elements: int = Field(
        default=6,
        ge=1,
        le=256,
        description="Top-ranked refinable elements MCTS may choose between (== action space).",
    )
    n_simulations: int = Field(
        default=12, ge=1, le=4096, description="MCTS simulations per accepted refinement."
    )
    n_seeds: int = Field(
        default=5,
        ge=1,
        le=64,
        description=(
            "Independent seeds swept; the gated ratio is the MEDIAN across seeds "
            "(a single MCTS run is high-variance)."
        ),
    )
    value_scale: float = Field(
        default=4.0,
        gt=0.0,
        le=100.0,
        description="tanh steepness for the encoded lower-error-per-DOF leaf value.",
    )
    c_puct: float = Field(
        default=1.4, gt=0.0, le=10.0, description="PUCT exploration constant for MCTS."
    )
    add_noise: bool = Field(
        default=True, description="Add root Dirichlet exploration noise in MCTS."
    )

    # --- Acceptance + artifacts ---------------------------------------- #
    max_l2_ratio_at_matched_dof: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description=(
            "Primary gate: MCTS/Dörfler L2-error ratio at matched DOF must be strictly "
            "below this (1.0 == MCTS's refinement policy is at least as good as Dörfler)."
        ),
    )
    output_dir: str = Field(
        default="results",
        description="Directory for the committed CSV/PNG artifacts.",
    )
    artifact_basename: str = Field(
        default="lshape_mcts_vs_dorfler",
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

    @field_validator("initial_side")
    @classmethod
    def _even_initial_side(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(
                "initial_side must be even so the L-shape reentrant corner at the "
                f"origin aligns with a grid node, got {v}"
            )
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
    def _budget_consistency(self) -> LShapeAMRCompareConfig:
        # The coarse grid must offer at least one refinable element.
        if self.n_candidate_elements > self.initial_side * self.initial_side * 4:
            raise ValueError(
                "n_candidate_elements exceeds a sane bound for the coarse grid "
                f"({self.n_candidate_elements} > 4*initial_side^2)"
            )
        return self

    # ------------------------------------------------------------------ #
    # Derived helpers                                                     #
    # ------------------------------------------------------------------ #

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Primary (matched-DOF policy-quality) acceptance threshold.

        The matched-wall-clock ratio is recorded but intentionally *not* gated
        (see the module docstring), so the scenario tests refinement-policy
        quality rather than untrained-MCTS search speed.
        """
        return [
            MetricThreshold(
                name="l2_error_ratio_at_matched_dof",
                operator="<",
                value=self.max_l2_ratio_at_matched_dof,
                description=(
                    "MCTS/Dörfler L2-error ratio at matched DOF must be < "
                    f"{self.max_l2_ratio_at_matched_dof} (MCTS policy at least as good)."
                ),
            ),
        ]
