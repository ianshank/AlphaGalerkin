"""Configuration schema for the LLM-prior MCTS basis-selection ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.integrations.lm_studio.config import LMStudioConfig
from src.poc.config import BaseScenarioConfig, MetricThreshold, ScenarioTier
from src.seeding import derive_seeds

SCENARIO_NAME = "llm_prior_ablation"
"""Canonical scenario id; YAML rows must use this string for dispatch."""

_SEED_PRIME_STRIDE = 1009
"""Prime stride used when deriving per-seed values from the master seed."""

_PDES_INCOMPATIBLE_WITH_BASIS_SELECTION = frozenset(
    {"heat", "advection_diffusion", "navier_stokes"}
)
"""PDE registry names BasisSelectionGame cannot fit a scalar approximation to.

Two different reasons collapse into one rejection set: "heat" and
"advection_diffusion" (without a time argument) have no exact_solution() at
all; "navier_stokes" has one, but it is a vector velocity field of shape
(N, 2), while BasisSelectionGame's `solution - exact` fits a scalar (N,)
approximation, so it fails with a numpy broadcasting error instead of
ExactSolutionUnavailableError. Both failure modes are equally a config-time
rejection candidate -- fail fast, not partway through a run. Verified
exhaustively against every PDE name reachable from these three Literals
(`python -c` probe against `_centaur_common.build_pde_operator`, 2026-09-01);
"poisson", "burgers", "poisson_lshaped", "helmholtz", "biharmonic" all
return scalar (N,) and are fine.

Duplicated (not shared via an import) in scaling_law_config.py and
src/agents/config.py::ResearchPDEName's validator -- the latter is
deliberately decoupled from the heavy MCTS/PDE import surface, and this
module follows the same discipline rather than pulling in a shared
constant from a heavier module just for three string literals. See
docs/CODE_HYGIENE_AUDIT.md B24 (the original heat/advection_diffusion cut)
and its navier_stokes correction (a GitHub Copilot review finding on PR #140,
verified before fixing rather than trusted).
"""


class LLMPriorAblationConfig(BaseScenarioConfig):
    """Ablation scenario: random / trained / LLM evaluators on ID + OOD PDEs.

    Drives an MCTS basis-selection run for each (arm × PDE × seed), records
    rollouts-to-target-residual and final-residual per (arm, PDE), and
    computes the headline acceptance metrics. Acceptance thresholds are
    produced by :meth:`get_default_thresholds` and merged into the
    inherited ``thresholds`` list inside the scenario's ``setup``.
    """

    # Identification + tier
    name: str = Field(
        default=SCENARIO_NAME,
        description=(
            "Scenario identifier. Dispatch by `load_config_from_dict` is "
            "driven by this field — the YAML row's `name:` must equal "
            f"{SCENARIO_NAME!r} for the type_map lookup to resolve to "
            "this class."
        ),
    )
    description: str = Field(
        default=(
            "LLM-prior vs random vs trained MCTS basis selection on ID "
            "(Poisson) + OOD (Burgers) PDEs."
        ),
        description="Scenario description.",
    )
    tier: ScenarioTier = Field(
        default=ScenarioTier.INTEGRATION,
        description="Scenario validation tier.",
    )

    # PDE coverage
    # NOTE: "heat" and "advection_diffusion" (without a time argument) have no
    # exact_solution() for this game, and "navier_stokes" (only reachable via
    # ood_pde below) has one but it is vector-valued and incompatible with
    # this game's scalar approximation -- all three would otherwise raise
    # (ExactSolutionUnavailableError, or a numpy broadcasting ValueError for
    # navier_stokes) deep in the per-seed rollout loop
    # (docs/CODE_HYGIENE_AUDIT.md P0-1; src/pde/games/basis_selection.py)
    # rather than failing at construction. The `_id_pde_has_exact_solution`/
    # `_ood_pde_is_compatible` validators below (B24, later widened by a
    # Copilot review finding on PR #140 to cover navier_stokes) turn that into
    # an immediate, actionable config error instead -- fail fast, not mid-run
    # after already spending compute on other arms/seeds. Left in the Literals
    # (not narrowed) pending either a manufactured scalar solution for
    # heat/advection_diffusion or vector-valued basis selection for
    # navier_stokes, at which point the validators' rejection sets shrink
    # rather than the types changing.
    id_pde: Literal["poisson", "heat", "advection_diffusion"] = Field(
        default="poisson",
        description="In-distribution PDE — the trained evaluator is expected to win here.",
    )
    ood_pde: Literal[
        "burgers",
        "navier_stokes",
        "poisson_lshaped",
        "helmholtz",
        "biharmonic",
    ] = Field(
        default="burgers",
        description=(
            "Out-of-distribution PDE — exposes the trained evaluator's "
            "lack of generalisation. Default 'burgers' covers nonlinear "
            "shock structure the FNet residual encoding has not seen. "
            "'helmholtz' adds an oscillatory zeroth-order term and "
            "'biharmonic' a fourth-order operator — held-out residual "
            "structures the FNet was never trained on."
        ),
    )

    # MCTS budget
    n_mcts_simulations: int = Field(
        default=64,
        ge=4,
        le=4096,
        description="Number of MCTS simulations per macro-step (action selection).",
    )
    n_seeds: int = Field(
        default=10,
        ge=2,
        le=64,
        description="Number of seeds per (arm, PDE) cell. >=2 enables Mann-Whitney.",
    )
    seeds: list[int] | None = Field(
        default=None,
        description=(
            "Explicit seed list. When None the scenario derives "
            "[seed + i * 1009 for i in range(n_seeds)] from the inherited "
            "`seed` field — prime stride avoids low-frequency correlation."
        ),
    )
    target_residual: float = Field(
        default=1e-2,
        gt=0.0,
        lt=1.0,
        description="Stop the inner loop once `adapter.current_error` drops below this.",
    )
    max_rollouts: int = Field(
        default=4096,
        ge=1,
        description="Hard cap on accumulated MCTS simulations per (arm, PDE, seed) cell.",
    )

    # Arm selection
    run_random_arm: bool = Field(
        default=True,
        description="Run the RandomEvaluator baseline arm.",
    )
    run_trained_arm: bool = Field(
        default=True,
        description=(
            "Run the FNetEvaluator (trained) arm. Auto-skipped with a "
            "warning when `trained_checkpoint_path is None`."
        ),
    )
    run_llm_arm: bool = Field(
        default=True,
        description="Run the LMStudioEvaluator arm.",
    )
    trained_checkpoint_path: Path | None = Field(
        default=None,
        description=(
            "Path to an AlphaGalerkin checkpoint (.pt). Required for the "
            "trained arm; when None the trained arm is skipped."
        ),
    )

    # Device
    device: str = Field(
        default="cuda",
        description=(
            "Device preference passed to `src.poc.device.resolve_device`. "
            "Accepts 'cuda', 'cuda:N', 'cpu', or 'auto'. Default 'cuda' "
            "fails loud when CUDA is unavailable — the scenario is GPU-only."
        ),
    )

    # LLM sub-config
    lm_studio: LMStudioConfig = Field(
        default_factory=LMStudioConfig,
        description="Configuration for the LM Studio client.",
    )

    # Game knobs
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

    # Acceptance thresholds
    id_rollout_reduction_pct_min: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Minimum median-rollout reduction vs random on the ID PDE (percent).",
    )
    ood_llm_residual_max: float = Field(
        default=1e-2,
        gt=0.0,
        description="Maximum median final residual for the LLM arm on the OOD PDE.",
    )
    ood_trained_residual_min: float = Field(
        default=1e-1,
        gt=0.0,
        description=(
            "Minimum median final residual for the trained arm on the OOD "
            "PDE. The trained eval is expected to *fail* OOD; this is the "
            "lower bound that proves it."
        ),
    )
    llm_call_p95_latency_ms_max: float = Field(
        default=3000.0,
        gt=0.0,
        description=(
            "Maximum p95 LLM-call latency (ms). Recalibrated from the "
            "original 300 ms — Qwen-14B Q4 at 256 max_tokens runs ~1–3 s "
            "on premium GPUs."
        ),
    )
    significance_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Mann-Whitney significance threshold for ID rollout comparison.",
    )

    requires_gpu: bool = Field(
        default=True,
        description="Resource hint — this scenario refuses to run without CUDA.",
    )

    @field_validator("seeds")
    @classmethod
    def _seeds_non_empty(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and not v:
            raise ValueError("seeds must be non-empty when provided (use None to derive)")
        return v

    @field_validator("id_pde")
    @classmethod
    def _id_pde_has_exact_solution(cls, v: str) -> str:
        if v in _PDES_INCOMPATIBLE_WITH_BASIS_SELECTION:
            raise ValueError(
                f"id_pde={v!r} is incompatible with BasisSelectionGame (no "
                "exact_solution(), or a vector-valued one BasisSelectionGame "
                "cannot fit a scalar approximation to) and would fail partway "
                "through the ablation grid instead of failing here at "
                "construction (docs/CODE_HYGIENE_AUDIT.md P0-1, B24). Choose "
                "a different id_pde, or drop this rejection once a compatible "
                "solution exists for this operator."
            )
        return v

    @field_validator("ood_pde")
    @classmethod
    def _ood_pde_is_compatible(cls, v: str) -> str:
        if v in _PDES_INCOMPATIBLE_WITH_BASIS_SELECTION:
            raise ValueError(
                f"ood_pde={v!r} is incompatible with BasisSelectionGame -- "
                "for 'navier_stokes' specifically, exact_solution() returns "
                "a vector velocity field of shape (N, 2), while this game "
                "fits a scalar (N,) approximation, so `solution - exact` "
                "fails with a numpy broadcasting error instead of failing "
                "here at construction (docs/CODE_HYGIENE_AUDIT.md B24, a "
                "Copilot review finding on PR #140). Choose a different "
                "ood_pde, or drop this rejection once vector-valued basis "
                "selection exists."
            )
        return v

    @field_validator("name")
    @classmethod
    def _name_locked(cls, v: str) -> str:
        if v != SCENARIO_NAME:
            raise ValueError(
                f"name must be exactly {SCENARIO_NAME!r} so the YAML "
                f"dispatch in load_config_from_dict resolves correctly; "
                f"got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _at_least_one_arm(self) -> LLMPriorAblationConfig:
        if not any([self.run_random_arm, self.run_trained_arm, self.run_llm_arm]):
            raise ValueError(
                "at least one of run_random_arm / run_trained_arm / run_llm_arm must be True"
            )
        return self

    def resolved_seeds(self) -> list[int]:
        """Return the per-cell seed list (explicit or derived).

        Returns:
            When ``self.seeds`` is set, the explicit list with duplicates
            removed in first-seen order (so cell counts stay deterministic
            and aligned with the user's intent). When ``self.seeds`` is
            None, a list of length ``self.n_seeds`` derived from
            ``self.seed`` via a prime stride.

        """
        if self.seeds is not None:
            # dict.fromkeys preserves insertion order while deduplicating.
            return list(dict.fromkeys(self.seeds))
        return derive_seeds(self.seed, self.n_seeds, _SEED_PRIME_STRIDE)

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Acceptance thresholds derived from the headline fields.

        ``BaseScenario._evaluate_thresholds`` does not auto-call this
        method; the scenario's ``setup`` is responsible for installing
        the returned list into ``self.config.thresholds`` and for
        removing entries when their underlying arm is skipped.
        """
        return [
            MetricThreshold(
                name="id_rollout_reduction_pct",
                operator=">=",
                value=self.id_rollout_reduction_pct_min,
                description=(
                    "LLM-prior median-rollout reduction vs random on the "
                    f"ID PDE must be >= {self.id_rollout_reduction_pct_min}%."
                ),
            ),
            MetricThreshold(
                name="ood_llm_residual",
                operator="<=",
                value=self.ood_llm_residual_max,
                description=(
                    "LLM-arm median final residual on the OOD PDE must "
                    f"be <= {self.ood_llm_residual_max}."
                ),
            ),
            MetricThreshold(
                name="ood_trained_residual",
                operator=">",
                value=self.ood_trained_residual_min,
                description=(
                    "Trained-arm median final residual on the OOD PDE "
                    f"must exceed {self.ood_trained_residual_min} "
                    "(failure expected; proves the differentiator)."
                ),
            ),
            MetricThreshold(
                name="llm_call_p95_latency_ms",
                operator="<=",
                value=self.llm_call_p95_latency_ms_max,
                description=(
                    "p95 LLM-call latency must be at or below "
                    f"{self.llm_call_p95_latency_ms_max} ms."
                ),
            ),
        ]
