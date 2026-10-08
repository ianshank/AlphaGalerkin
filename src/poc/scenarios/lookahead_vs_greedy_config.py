"""Configuration for the ``lookahead_vs_greedy`` PoC scenario (peer-review Gate 1).

Every default below is a value pre-registered in
``specs/lookahead_vs_greedy.spec.md`` -- budgets, arms, seeds, the four GO
criteria and the break-even exponents -- and the validators enforce the
design's invariants: a ranked legal set (``top_k_actions >= 1``), room for the
tree to grow past one ply (``n_simulations > top_k_actions``), deterministic
action selection, and thresholds that can only be the pre-registered ones.

``get_default_thresholds()`` is the single source of the GO criteria: the
harness's verdict (``src.research.lookahead_vs_greedy_verdict``) and the PoC
framework's pass/fail both evaluate exactly these objects.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, field_validator, model_validator

from src.constants import DEFAULT_DIRICHLET_ALPHA, DEFAULT_DIRICHLET_EPSILON
from src.poc.config import BaseScenarioConfig, MetricThreshold
from src.poc.scenarios._compare_lock import lock_scenario_name
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    ADEQUACY_OPERATORS,
    DEFAULT_MAX_ACTION_SPACE,
    DEFAULT_SEED_STRIDE,
    HEADLINE_EVALUATOR_NAME,
    ZSHAPE_OPERATOR_NAME,
    MCTSClassicalAMRArenaConfig,
)
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME as ARENA_SCENARIO_NAME,
)
from src.research.lookahead_vs_greedy_metrics import (
    PRIMARY_DIVERGENCE_METRIC,
    PRIMARY_DOF_SAVING_METRIC,
    PRIMARY_RATIO_METRIC,
    ROBUST_MEDIAN_RATIO_METRIC,
    ROBUST_WINS_METRIC,
)
from src.research.substrates.config import SUBSTRATE_KIND_SKFEM_TRI, SubstrateConfig
from src.research.substrates.factory import ZSHAPE_SCALE, OperatorName

SCENARIO_NAME: Final[str] = "lookahead_vs_greedy"

# --- Pre-registered budgets and arms -----------------------------------------
DEFAULT_LOOKAHEAD_MAX_STEPS: Final[int] = 30
DEFAULT_CLASSICAL_MAX_DOF: Final[int] = 1000
DEFAULT_MAX_REFINEMENTS_CLASSICAL: Final[int] = 1000
DEFAULT_DORFLER_THETAS: Final[tuple[float, ...]] = (0.1, 0.3, 0.5)
DEFAULT_LOOKAHEAD_N_SIMULATIONS: Final[int] = 64
DEFAULT_LOOKAHEAD_TOP_K: Final[int] = 4
DEFAULT_ROBUST_N_SEEDS: Final[int] = 5
DEFAULT_BREAK_EVEN_ALPHAS: Final[tuple[float, ...]] = (1.0, 1.5)
#: The arena's values, unchanged (``config/scenarios/mcts_classical_amr_arena.yaml``).
DEFAULT_LOOKAHEAD_C_PUCT: Final[float] = 1.4
DEFAULT_LOOKAHEAD_VALUE_SCALE: Final[float] = 4.0
#: ``ResidualPriorErrorValueEvaluator``'s own default softmax temperature.
DEFAULT_PRIOR_TEMPERATURE: Final[float] = 1.0
DEFAULT_ADEQUACY_THETA: Final[float] = 0.5
DEFAULT_ERROR_TOLERANCE: Final[float] = 1e-8
DEFAULT_SOLVE_CACHE_ENTRIES: Final[int] = 8192
DEFAULT_INITIAL_REFINEMENTS: Final[int] = 2

# --- Pre-registered GO criteria ------------------------------------------------
DEFAULT_MAX_PRIMARY_RATIO: Final[float] = 0.98
DEFAULT_MIN_PRIMARY_DIVERGENCES: Final[int] = 1
DEFAULT_MAX_ROBUST_MEDIAN_RATIO: Final[float] = 1.0
DEFAULT_ROBUST_WIN_RATIO: Final[float] = 1.0
DEFAULT_MIN_ROBUST_WINS: Final[int] = 3
DEFAULT_MIN_DOF_SAVING_VS_GREEDY: Final[float] = 0.0

#: Testbed id of each operator (artifacts, logs, the verdict). ``poisson`` on the
#: unit square has no singularity; it is for mechanism smokes, never a verdict.
TESTBED_LABELS: Final[dict[str, str]] = {
    "lshape_poisson": "T1_lshape",
    "zshape_poisson": "T2_zshape",
    "poisson": "smoke_unit_square",
}


def _default_substrate() -> SubstrateConfig:
    """The arena's ``skfem_tri`` substrate, shared by every arm."""
    return SubstrateConfig(
        name="lookahead_skfem",
        kind="skfem_tri",
        element_type="P1",
        initial_refinements=DEFAULT_INITIAL_REFINEMENTS,
        error_metric="quadrature",
        marking_variant="squared",
        solve_cache_max_entries=DEFAULT_SOLVE_CACHE_ENTRIES,
    )


class LookaheadVsGreedyConfig(BaseScenarioConfig):
    """Pinned contract for one testbed of the Gate 1 go/no-go."""

    name: str = Field(default=SCENARIO_NAME, description="Scenario dispatch key.")
    description: str = Field(
        default="Gate 1: MCTS look-ahead vs the best classical marking policy.",
        description="Human-readable description.",
    )
    device: str = Field(default="cpu", description="Device string (numpy-only; recorded).")
    substrate: SubstrateConfig = Field(
        default_factory=_default_substrate,
        description="Shared substrate for every arm (same initial mesh and estimator).",
    )
    operator_name: OperatorName = Field(
        default="lshape_poisson",
        description="Testbed: lshape_poisson (T1), zshape_poisson (T2), poisson (smoke).",
    )
    lshape_scale: float = Field(
        default=1.0, gt=0.0, le=100.0, description="L-shape half-width; must be 1 for Z."
    )
    max_steps: int = Field(
        default=DEFAULT_LOOKAHEAD_MAX_STEPS,
        ge=1,
        le=10_000,
        description="Single-element refinements for greedy and both MCTS arms.",
    )
    classical_max_dof: int = Field(
        default=DEFAULT_CLASSICAL_MAX_DOF,
        ge=10,
        le=1_000_000,
        description="Classical DOF budget; also the game-driven episode stop.",
    )
    max_refinements_classical: int = Field(
        default=DEFAULT_MAX_REFINEMENTS_CLASSICAL,
        ge=1,
        le=1_000_000,
        description="Runaway guard on classical sweep levels.",
    )
    error_tolerance: float = Field(
        default=DEFAULT_ERROR_TOLERANCE, gt=0.0, lt=1.0, description="Shared early-stop L2."
    )
    dorfler_thetas: tuple[float, ...] = Field(
        default=DEFAULT_DORFLER_THETAS, description="Bulk fractions of the Dörfler arms."
    )
    n_simulations: int = Field(
        default=DEFAULT_LOOKAHEAD_N_SIMULATIONS,
        ge=1,
        le=4096,
        description="PUCT simulations per committed step (must exceed top_k_actions).",
    )
    top_k_actions: int = Field(
        default=DEFAULT_LOOKAHEAD_TOP_K,
        ge=1,
        le=1_000_000,
        description="Residual-ranked legal-set size. >= 1: divergence evidences search "
        "only on a ranked set (at 0, index order diverges by tie-break alone).",
    )
    max_action_space: int = Field(
        default=DEFAULT_MAX_ACTION_SPACE, ge=1, le=1_000_000, description="Action-index range."
    )
    c_puct: float = Field(
        default=DEFAULT_LOOKAHEAD_C_PUCT, gt=0.0, le=10.0, description="PUCT constant."
    )
    value_scale: float = Field(
        default=DEFAULT_LOOKAHEAD_VALUE_SCALE,
        gt=0.0,
        le=100.0,
        description="Steepness of the error-per-DOF leaf value.",
    )
    prior_temperature: float = Field(
        default=DEFAULT_PRIOR_TEMPERATURE,
        gt=0.0,
        le=100.0,
        description="Softmax temperature of the residual prior.",
    )
    dirichlet_alpha: float = Field(
        default=DEFAULT_DIRICHLET_ALPHA,
        gt=0.0,
        le=100.0,
        description="Root-noise concentration of the robustness arm (engine default).",
    )
    dirichlet_epsilon: float = Field(
        default=DEFAULT_DIRICHLET_EPSILON,
        gt=0.0,
        le=1.0,
        description="Root-noise mixing weight of the robustness arm (engine default).",
    )
    robust_n_seeds: int = Field(
        default=DEFAULT_ROBUST_N_SEEDS, ge=1, le=64, description="Robustness-arm seeds."
    )
    seed_stride: int = Field(
        default=DEFAULT_SEED_STRIDE,
        ge=1,
        le=1_000_000,
        description="Robustness seeds are seed + i * seed_stride.",
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Action selection; locked to 0."
    )
    search_mode: Literal["single_agent"] = Field(
        default="single_agent", description="MCTS backup; the engine default is ZERO_SUM."
    )
    evaluator_name: Literal["ResidualPriorErrorValueEvaluator"] = Field(
        default=HEADLINE_EVALUATOR_NAME, description="Headline leaf evaluator (locked)."
    )
    require_adequacy_precondition: bool = Field(
        default=True, description="Abort the testbed if its adequacy gate fails."
    )
    adequacy_theta: float = Field(
        default=DEFAULT_ADEQUACY_THETA,
        gt=0.0,
        lt=1.0,
        description="Dörfler θ of the adequacy measurement.",
    )
    max_primary_ratio: float = Field(
        default=DEFAULT_MAX_PRIMARY_RATIO, gt=0.0, le=10.0, description="C1 bar (<=)."
    )
    min_primary_divergences: int = Field(
        default=DEFAULT_MIN_PRIMARY_DIVERGENCES, ge=1, le=10_000, description="C2 bar (>=)."
    )
    max_robust_median_ratio: float = Field(
        default=DEFAULT_MAX_ROBUST_MEDIAN_RATIO, gt=0.0, le=10.0, description="C3 median bar (<)."
    )
    robust_win_ratio: float = Field(
        default=DEFAULT_ROBUST_WIN_RATIO,
        gt=0.0,
        le=10.0,
        description="A robustness seed beats the best classical arm iff its ratio < this.",
    )
    min_robust_wins: int = Field(
        default=DEFAULT_MIN_ROBUST_WINS, ge=1, le=64, description="C3 win-count bar (>=)."
    )
    min_dof_saving_vs_greedy: float = Field(
        default=DEFAULT_MIN_DOF_SAVING_VS_GREEDY,
        ge=0.0,
        le=1_000_000.0,
        description="C4 bar (>): DOF saving at equal accuracy, i.e. a finite K*.",
    )
    break_even_alphas: tuple[float, ...] = Field(
        default=DEFAULT_BREAK_EVEN_ALPHAS,
        description="Downstream solve-cost exponents for K* (cost ~ DOF**alpha).",
    )
    output_dir: str = Field(default="results", description="Directory for the artifacts.")
    artifact_basename: str = Field(
        default="lookahead_vs_greedy_lshape", description="Artifact basename (no extension)."
    )

    @field_validator("name")
    @classmethod
    def _lock_name(cls, value: str) -> str:
        return lock_scenario_name(SCENARIO_NAME, value)

    @field_validator("artifact_basename")
    @classmethod
    def _basename_no_extension(cls, value: str) -> str:
        if not value or value.endswith((".csv", ".png", ".json")):
            raise ValueError("artifact_basename must be non-empty and carry no file extension")
        return value

    @field_validator("dorfler_thetas")
    @classmethod
    def _thetas_are_fractions(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("dorfler_thetas must be non-empty and unique")
        if not all(0.0 < theta < 1.0 for theta in value):
            raise ValueError(f"every Dörfler θ must lie in (0, 1), got {value}")
        return value

    @field_validator("break_even_alphas")
    @classmethod
    def _alphas_are_positive(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or len(set(value)) != len(value) or not all(alpha > 0.0 for alpha in value):
            raise ValueError(f"break_even_alphas must be non-empty, unique and positive: {value}")
        return value

    @model_validator(mode="after")
    def _pre_registered_invariants(self) -> LookaheadVsGreedyConfig:
        if self.temperature != 0.0:
            raise ValueError("both MCTS arms select deterministically: temperature must be 0")
        if self.n_simulations <= self.top_k_actions:
            raise ValueError(
                f"n_simulations={self.n_simulations} must exceed top_k_actions="
                f"{self.top_k_actions}, or the tree has no room to grow past one ply"
            )
        if self.min_robust_wins > self.robust_n_seeds:
            raise ValueError(
                f"min_robust_wins={self.min_robust_wins} exceeds robust_n_seeds="
                f"{self.robust_n_seeds}; C3 could never pass"
            )
        if self.require_adequacy_precondition and (
            self.substrate.kind != SUBSTRATE_KIND_SKFEM_TRI
            or self.operator_name not in ADEQUACY_OPERATORS
        ):
            raise ValueError(
                "adequacy precondition is defined for skfem_tri + "
                f"{' / '.join(sorted(ADEQUACY_OPERATORS))}; "
                "set require_adequacy_precondition=False for other hosts"
            )
        if self.operator_name == ZSHAPE_OPERATOR_NAME and self.lshape_scale != ZSHAPE_SCALE:
            raise ValueError(
                f"the Z testbed is the unit Z-tetromino; lshape_scale must be {ZSHAPE_SCALE}"
            )
        if self.thresholds and self.thresholds != self.get_default_thresholds():
            raise ValueError(
                "thresholds are the pre-registered GO criteria, derived from the typed "
                "fields; set those fields, never the threshold list"
            )
        return self

    def testbed_label(self) -> str:
        """``T1_lshape`` / ``T2_zshape`` / ``smoke_unit_square``."""
        return TESTBED_LABELS[self.operator_name]

    def robust_seeds(self) -> list[int]:
        """Robustness seeds: ``seed + i * seed_stride``."""
        return [int(self.seed) + i * int(self.seed_stride) for i in range(self.robust_n_seeds)]

    def arena_config(self) -> MCTSClassicalAMRArenaConfig:
        """The arena config every game-driven arm's game is built from.

        It carries only the knobs ``build_arena_episode`` and
        ``run_arena_episode`` read -- the game, the budget and the stopping
        rules; every search knob lives on this config and reaches the search
        through the decision rule. It satisfies the arena's scored locks
        (``add_noise=False``, ``temperature=0``): root noise is a decision-rule
        argument, never a property of the game. ``max_dof`` is the classical
        budget, so only ``max_steps`` ends a game-driven episode.
        """
        return MCTSClassicalAMRArenaConfig(
            name=ARENA_SCENARIO_NAME,
            device=self.device,
            seed=self.seed,
            substrate=self.substrate,
            operator_name=self.operator_name,
            lshape_scale=self.lshape_scale,
            max_dof=self.classical_max_dof,
            max_steps=self.max_steps,
            error_tolerance=self.error_tolerance,
            top_k_actions=self.top_k_actions,
            max_action_space=self.max_action_space,
            value_scale=self.value_scale,
            require_adequacy_precondition=self.require_adequacy_precondition,
        )

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """The four pre-registered GO criteria (C3 in two parts); GO iff all pass."""
        return [
            MetricThreshold(
                name=PRIMARY_RATIO_METRIC,
                operator="<=",
                value=self.max_primary_ratio,
                description="C1: MCTS-primary L2 / best classical L2 at matched DOF.",
            ),
            MetricThreshold(
                name=PRIMARY_DIVERGENCE_METRIC,
                operator=">=",
                value=float(self.min_primary_divergences),
                description="C2: MCTS-primary decisions diverging from greedy (ranked set).",
            ),
            MetricThreshold(
                name=ROBUST_MEDIAN_RATIO_METRIC,
                operator="<",
                value=self.max_robust_median_ratio,
                description="C3: median over robustness seeds of the C1 ratio.",
            ),
            MetricThreshold(
                name=ROBUST_WINS_METRIC,
                operator=">=",
                value=float(self.min_robust_wins),
                description=f"C3: robustness seeds with ratio < {self.robust_win_ratio:g}.",
            ),
            MetricThreshold(
                name=PRIMARY_DOF_SAVING_METRIC,
                operator=">",
                value=self.min_dof_saving_vs_greedy,
                description="C4: DOF saving vs greedy at equal accuracy (finite K*).",
            ),
        ]


__all__ = [
    "DEFAULT_BREAK_EVEN_ALPHAS",
    "DEFAULT_CLASSICAL_MAX_DOF",
    "DEFAULT_DORFLER_THETAS",
    "DEFAULT_LOOKAHEAD_MAX_STEPS",
    "DEFAULT_LOOKAHEAD_N_SIMULATIONS",
    "DEFAULT_LOOKAHEAD_TOP_K",
    "DEFAULT_MAX_PRIMARY_RATIO",
    "DEFAULT_MAX_REFINEMENTS_CLASSICAL",
    "DEFAULT_MAX_ROBUST_MEDIAN_RATIO",
    "DEFAULT_MIN_DOF_SAVING_VS_GREEDY",
    "DEFAULT_MIN_PRIMARY_DIVERGENCES",
    "DEFAULT_MIN_ROBUST_WINS",
    "DEFAULT_ROBUST_N_SEEDS",
    "DEFAULT_ROBUST_WIN_RATIO",
    "SCENARIO_NAME",
    "TESTBED_LABELS",
    "LookaheadVsGreedyConfig",
]
