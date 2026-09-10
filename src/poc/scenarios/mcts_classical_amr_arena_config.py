"""Configuration for the ``mcts_classical_amr_arena`` PoC scenario.

Pre-registered MCTS vs classical (Dörfler / uniform) comparison on the
element-local ``skfem_tri`` substrate. Every tunable is a typed field.

The single gated metric is ``l2_error_ratio_at_matched_dof < 1.0`` (MCTS
quadrature L2 over Dörfler at the largest common DOF). Matched-solves and
wall-clock ratios are recorded and **ungated**. Adequacy-gate rates are a
precondition, not a look-ahead result.

See ``specs/mcts_classical_amr_arena.spec.md``.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, field_validator, model_validator

from src.poc.config import BaseScenarioConfig, MetricThreshold
from src.poc.scenarios._compare_lock import lock_scenario_name
from src.research.substrates.config import AdequacyGateConfig, SubstrateConfig

SCENARIO_NAME: Final[str] = "mcts_classical_amr_arena"
HEADLINE_EVALUATOR_NAME: Final[Literal["ResidualPriorErrorValueEvaluator"]] = (
    "ResidualPriorErrorValueEvaluator"
)
DEFAULT_SEED_STRIDE: Final[int] = 1009
DEFAULT_MARKING_FRACTION: Final[float] = 0.5
DEFAULT_POLICY_MAX_DOF: Final[int] = 600
DEFAULT_MAX_ACTION_SPACE: Final[int] = 16384


class MCTSClassicalAMRArenaConfig(BaseScenarioConfig):
    """Pinned contract for the MCTS-vs-classical AMR arena."""

    name: str = Field(default=SCENARIO_NAME, description="Scenario dispatch key.")
    description: str = Field(
        default=("MCTS vs Dörfler/uniform on SkfemTriSubstrate (element-local L-shape Poisson)."),
        description="Human-readable description.",
    )
    device: str = Field(
        default="cpu",
        description="Device string resolved via src.poc.device.resolve_device.",
    )
    substrate: SubstrateConfig = Field(
        default_factory=lambda: SubstrateConfig(
            name="arena_skfem",
            kind="skfem_tri",
            element_type="P1",
            initial_refinements=2,
            error_metric="quadrature",
            marking_variant="squared",
            solve_cache_max_entries=8192,
        ),
        description="Shared substrate for every arm (looked up by kind).",
    )
    operator_name: Literal["poisson", "lshape_poisson"] = Field(
        default="lshape_poisson",
        description="Exact-solution operator paired with the substrate.",
    )
    lshape_scale: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description="L-shape half-width (domain [-s, s]^2).",
    )
    marking_fraction: float = Field(
        default=DEFAULT_MARKING_FRACTION,
        gt=0.0,
        lt=1.0,
        description="Dörfler bulk fraction θ. Quote it with every published rate.",
    )
    max_dof: int = Field(
        default=DEFAULT_POLICY_MAX_DOF,
        ge=10,
        le=1_000_000,
        description="Policy-compare DOF budget (not the adequacy window).",
    )
    max_steps: int = Field(
        default=12,
        ge=1,
        le=10_000,
        description="Max single-element refine actions for the MCTS arm.",
    )
    max_refinements_classical: int = Field(
        default=16,
        ge=1,
        le=1000,
        description="Max Dörfler/uniform sweep levels in the policy window.",
    )
    error_tolerance: float = Field(
        default=1e-8,
        gt=0.0,
        lt=1.0,
        description="Early-stop L2 tolerance shared by policy arms.",
    )
    n_simulations: int = Field(
        default=8,
        ge=1,
        le=4096,
        description="MCTS simulations per accepted refinement.",
    )
    top_k_actions: int = Field(
        default=8,
        ge=0,
        le=1_000_000,
        description="Legal-set cap by residual (0 = every refinable unit).",
    )
    max_action_space: int = Field(
        default=DEFAULT_MAX_ACTION_SPACE,
        ge=1,
        le=1_000_000,
        description=(
            "Fixed action-index range. Must stay above n_units in the window "
            "or high-index triangles are silently dropped."
        ),
    )
    n_seeds: int = Field(
        default=3,
        ge=1,
        le=64,
        description="Independent MCTS seeds; gated ratio is the median.",
    )
    seed_stride: int = Field(
        default=DEFAULT_SEED_STRIDE,
        ge=1,
        le=1_000_000,
        description="Offset between consecutive resolved seeds.",
    )
    value_scale: float = Field(
        default=4.0,
        gt=0.0,
        le=100.0,
        description="tanh steepness for the trailing error-per-DOF leaf value.",
    )
    c_puct: float = Field(
        default=1.4,
        gt=0.0,
        le=10.0,
        description="PUCT exploration constant.",
    )
    add_noise: bool = Field(
        default=False,
        description="Root Dirichlet noise. Scored runs must keep this False.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Action-selection temperature. Scored runs must keep 0.",
    )
    search_mode: Literal["single_agent"] = Field(
        default="single_agent",
        description="MCTS backup semantics. Engine default is ZERO_SUM; do not omit.",
    )
    evaluator_name: Literal["ResidualPriorErrorValueEvaluator"] = Field(
        default=HEADLINE_EVALUATOR_NAME,
        description="Named headline leaf evaluator. Encoded/Random are forbidden.",
    )
    use_intermediate_rewards: bool = Field(
        default=False,
        description="Must stay False: scored arm bootstraps from leaf value only.",
    )
    require_adequacy_precondition: bool = Field(
        default=True,
        description="Abort if gate_violations() is nonempty on this substrate/θ.",
    )
    max_l2_ratio_at_matched_dof: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Primary gate: MCTS/Dörfler quadrature L2 ratio must be < this.",
    )
    output_dir: str = Field(default="results", description="Directory for CSV/PNG.")
    artifact_basename: str = Field(
        default="mcts_classical_amr_arena",
        description="Basename for CSV/PNG/run.json (no extension).",
    )

    @field_validator("name")
    @classmethod
    def _lock_name(cls, value: str) -> str:
        return lock_scenario_name(SCENARIO_NAME, value)

    @field_validator("evaluator_name")
    @classmethod
    def _lock_evaluator(cls, value: str) -> str:
        if value != HEADLINE_EVALUATOR_NAME:
            raise ValueError(f"evaluator_name must be {HEADLINE_EVALUATOR_NAME!r}, got {value!r}")
        return value

    @field_validator("artifact_basename")
    @classmethod
    def _basename_no_extension(cls, value: str) -> str:
        if not value:
            raise ValueError("artifact_basename must be non-empty")
        if value.endswith((".csv", ".png", ".json")):
            raise ValueError("artifact_basename must not include a file extension")
        return value

    @model_validator(mode="after")
    def _scored_flags_and_action_space(self) -> MCTSClassicalAMRArenaConfig:
        if self.add_noise:
            raise ValueError("scored arena requires add_noise=False")
        if self.temperature != 0.0:
            raise ValueError("scored arena requires temperature=0")
        if self.use_intermediate_rewards:
            raise ValueError("scored arena requires use_intermediate_rewards=False")
        if self.require_adequacy_precondition and (
            self.substrate.kind != "skfem_tri" or self.operator_name != "lshape_poisson"
        ):
            raise ValueError(
                "adequacy precondition is defined for skfem_tri + lshape_poisson; "
                "set require_adequacy_precondition=False for other hosts"
            )
        return self

    def resolved_seeds(self) -> list[int]:
        """Deterministic seed list: ``seed + i * seed_stride``."""
        return [int(self.seed) + i * int(self.seed_stride) for i in range(int(self.n_seeds))]

    def adequacy_gate(self) -> AdequacyGateConfig:
        """Pinned adequacy thresholds (θ quoted separately as marking_fraction)."""
        return AdequacyGateConfig(name="arena_adequacy_precondition")

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Single gated metric; matched-solves / wall-clock stay ungated."""
        return [
            MetricThreshold(
                name="l2_error_ratio_at_matched_dof",
                operator="<",
                value=self.max_l2_ratio_at_matched_dof,
                description=(
                    "MCTS/Dörfler quadrature-L2 ratio at matched DOF must be < "
                    f"{self.max_l2_ratio_at_matched_dof} (policy at least as good)."
                ),
            ),
        ]


__all__ = [
    "DEFAULT_MARKING_FRACTION",
    "DEFAULT_MAX_ACTION_SPACE",
    "DEFAULT_POLICY_MAX_DOF",
    "DEFAULT_SEED_STRIDE",
    "HEADLINE_EVALUATOR_NAME",
    "SCENARIO_NAME",
    "MCTSClassicalAMRArenaConfig",
]
