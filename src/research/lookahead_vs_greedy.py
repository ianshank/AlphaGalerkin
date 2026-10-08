"""Gate 1: MCTS look-ahead against the best classical marking policy.

Pre-registered in ``specs/lookahead_vs_greedy.spec.md`` -- question, testbeds,
arms, budgets, primary metric, break-even formula and GO criteria were committed
before any comparison run. This module runs the arms;
``src.research.lookahead_vs_greedy_metrics`` measures them and
``src.research.lookahead_vs_greedy_verdict`` scores them.

It composes the arena's Phase A primitives rather than copying them:

* greedy is ``greedy_control.run_greedy_arm_with_cache`` -- the one greedy rule;
* both MCTS arms are a ``DecisionRule`` over ``MCTS.get_action``, driven by
  ``amr_arena_episode.run_arena_episode`` on the game ``build_arena_episode``
  builds -- the game, stopping rules and point recording greedy uses -- each run
  through its own solve cache;
* Dörfler at each θ and uniform are ``run_classical_arm`` on fresh substrates;
* matched-solves / wall-clock readings are ``compare_trajectories``.

Root noise belongs to the decision rule, not the game. The game is built from
an ``MCTSClassicalAMRArenaConfig`` that satisfies the arena's scored locks
(``add_noise=False``); only the robustness arm's rule passes ``add_noise=True``
to ``MCTS.get_action``. Each committed MCTS step records its realized tree depth
(``src.mcts.node.subtree_depth`` of the root, after the search and before the
action) and whether it departed from greedy.

The adequacy precondition runs first: a testbed whose substrate fails
``gate_violations`` is aborted (:class:`AdequacyAbortedError`) before any arm runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog

from src.mcts.node import subtree_depth
from src.mcts.search import MCTS
from src.research.amr_arena_episode import build_arena_episode, run_arena_episode
from src.research.greedy_control import (
    GreedyDivergence,
    greedy_action,
    run_greedy_arm_with_cache,
)
from src.research.lookahead_vs_greedy_metrics import (
    GREEDY_LABEL,
    PRIMARY_LABEL,
    ROBUST_LABEL,
    UNIFORM_LABEL,
    SearchStep,
    assert_classical_span,
    break_even,
    dorfler_label,
    matched_reading,
)
from src.research.lookahead_vs_greedy_verdict import (
    Gate1Verdict,
    ScoredRun,
    collect_testbed_metrics,
    evaluate_verdict,
    require_finite_metrics,
)
from src.research.mcts_classical_amr_arena import (
    AdequacyPreconditionError,
    _assert_private_caches,
    compare_trajectories,
    run_classical_arm,
)
from src.research.substrates.factory import (
    adequacy_gate_for_operator,
    build_substrate_from_config,
)
from src.research.substrates.residual_evaluator import ResidualPriorErrorValueEvaluator
from src.research.substrates.sweep import gate_violations, measure_adequacy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.pde.games.substrate_refinement import SubstrateEpisodeState
    from src.poc.scenarios.lookahead_vs_greedy_config import LookaheadVsGreedyConfig
    from src.refinement.adapter import RefinementGameAdapter
    from src.refinement.substrate import RefinementSubstrate
    from src.research.amr_arena_episode import DecisionRule
    from src.research.amr_arena_types import ArenaTrajectory
    from src.research.substrates.solve_cache import FingerprintSolveCache
    from src.research.substrates.sweep import RateSeparation

logger = structlog.get_logger(__name__)

#: Metric recorded on a testbed that aborted on its adequacy precondition.
ADEQUACY_ABORTED_METRIC: Final[str] = "adequacy_aborted"
#: The value it is recorded with.
ABORTED: Final[float] = 1.0
#: Wall-clock of the whole testbed run (ungated; for the run report).
WALL_SECONDS_TOTAL_METRIC: Final[str] = "wall_seconds_total"


@dataclass(frozen=True)
class SearchArmSpec:
    """What separates the two MCTS arms: their label and whether root noise is on."""

    label: str
    add_noise: bool


#: Deterministic search (no root noise, temperature 0): one run.
PRIMARY_ARM: Final[SearchArmSpec] = SearchArmSpec(label=PRIMARY_LABEL, add_noise=False)
#: Identical, with Dirichlet root noise on: one run per robustness seed.
ROBUST_ARM: Final[SearchArmSpec] = SearchArmSpec(label=ROBUST_LABEL, add_noise=True)


class AdequacyAbortedError(AdequacyPreconditionError):
    """The testbed's substrate failed its adequacy gate: aborted, not scored."""

    def __init__(self, violations: list[str], separation: RateSeparation) -> None:
        super().__init__(violations)
        self.separation = separation


def adequacy_metrics(separation: RateSeparation) -> dict[str, float]:
    """The adequacy measurement as recordable numbers (the arena's keys)."""
    return {
        "adequacy_adaptive_rate": float(separation.adaptive_rate),
        "adequacy_uniform_rate": float(separation.uniform_rate),
        "adequacy_error_ratio": float(separation.error_ratio_at_matched_dof),
        "adequacy_matched_dof": float(separation.matched_dof),
    }


def _fresh_substrate(config: LookaheadVsGreedyConfig) -> RefinementSubstrate[Any]:
    """A new substrate instance: arms never share meshes."""
    substrate: RefinementSubstrate[Any] = build_substrate_from_config(
        config.substrate,
        operator_name=config.operator_name,
        scale=config.lshape_scale,
    )
    return substrate


def check_adequacy(config: LookaheadVsGreedyConfig) -> RateSeparation | None:
    """Measure the testbed's adequacy gate; ``None`` when the precondition is off.

    Raises:
        AdequacyAbortedError: ``gate_violations`` is non-empty. Thresholds are never
            retuned here: the gate is ``adequacy_gate_for_operator``'s.

    """
    if not config.require_adequacy_precondition:
        logger.info("lookahead_adequacy_skipped", testbed=config.testbed_label())
        return None
    gate = adequacy_gate_for_operator(config.operator_name)
    separation = measure_adequacy(_fresh_substrate(config), theta=config.adequacy_theta, gate=gate)
    violations = gate_violations(separation, gate)
    logger.info(
        "lookahead_adequacy_measured",
        testbed=config.testbed_label(),
        theta=config.adequacy_theta,
        rate_fit_dof_range=gate.rate_fit_dof_range,
        passed=not violations,
        violations=violations,
        **adequacy_metrics(separation),
    )
    if violations:
        raise AdequacyAbortedError(violations, separation)
    return separation


@dataclass
class SearchLog:
    """What an MCTS arm records as it commits actions."""

    divergence: GreedyDivergence
    steps: list[SearchStep] = field(default_factory=list)


def search_decision_rule(
    mcts: MCTS,
    game: RefinementGameAdapter,
    *,
    add_noise: bool,
    temperature: float,
    log: SearchLog,
) -> DecisionRule:
    """A ``DecisionRule`` that commits ``mcts``'s choice and records the step.

    Per step: search, read the realized tree depth before the action, compare
    the choice with greedy on the same state, then advance the tree (reuse).
    """

    def choose(state: SubstrateEpisodeState, legal: list[int]) -> int:
        action = mcts.get_action(game, temperature=temperature, add_noise=add_noise)
        root = mcts.root
        depth = 0 if root is None else subtree_depth(root)
        reference = greedy_action(state.indicators, legal)
        log.divergence.record(state, legal, action)
        log.steps.append(
            SearchStep(
                step=int(state.step),
                action=action,
                greedy_action=reference,
                tree_depth=depth,
                n_legal=len(legal),
                dof=int(state.dof),
            )
        )
        logger.debug(
            "lookahead_search_step",
            seed=log.divergence.seed,
            step=int(state.step),
            action=action,
            greedy_action=reference,
            diverged=action != reference,
            tree_depth=depth,
            n_legal=len(legal),
            dof=int(state.dof),
        )
        mcts.advance(action)
        return action

    return choose


@dataclass(frozen=True)
class SearchRun:
    """One finished MCTS episode, before it is scored."""

    arm: SearchArmSpec
    seed: int
    trajectory: ArenaTrajectory
    steps: tuple[SearchStep, ...]
    wall_seconds: float


def run_search_arm(
    config: LookaheadVsGreedyConfig,
    arm: SearchArmSpec,
    seed: int,
) -> tuple[SearchRun, FingerprintSolveCache]:
    """One MCTS episode on the arena game; also returns its cache, alive, for the fairness check.

    ``numpy``'s global stream is seeded first because the engine's root noise
    (``np.random.dirichlet``) draws from it; the deterministic primary arm draws
    nothing, so its seed is provenance only.
    """
    np.random.seed(seed)
    arena = config.arena_config()
    episode = build_arena_episode(arena, name=f"lookahead_{arm.label}_{seed}")
    adapter = episode.adapter
    mcts = MCTS(
        evaluator=ResidualPriorErrorValueEvaluator(
            n_actions=adapter.action_space_size,
            prior_temperature=config.prior_temperature,
        ),
        c_puct=config.c_puct,
        n_simulations=config.n_simulations,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        search_mode=adapter.search_mode,
        use_intermediate_rewards=False,
    )
    log = SearchLog(divergence=GreedyDivergence(seed=seed))
    choose = search_decision_rule(
        mcts, adapter, add_noise=arm.add_noise, temperature=config.temperature, log=log
    )
    started = time.perf_counter()
    trajectory = run_arena_episode(episode, method="mcts", config=arena, choose=choose)
    elapsed = time.perf_counter() - started
    trajectory.decisions_diverging_from_greedy = log.divergence.count
    depths = [step.tree_depth for step in log.steps]
    logger.info(
        "lookahead_search_arm_done",
        testbed=config.testbed_label(),
        arm=arm.label,
        seed=seed,
        add_noise=arm.add_noise,
        steps=len(log.steps),
        final_dof=trajectory.points[-1].n_dof,
        final_l2=trajectory.points[-1].l2_error,
        decisions_diverging_from_greedy=log.divergence.count,
        tree_depth_max=max(depths, default=0),
        cache_misses=episode.cache.misses,
        cache_hits=episode.cache.hits,
        wall_seconds=elapsed,
    )
    run = SearchRun(
        arm=arm, seed=seed, trajectory=trajectory, steps=tuple(log.steps), wall_seconds=elapsed
    )
    return run, episode.cache


def run_sweep_arms(config: LookaheadVsGreedyConfig) -> dict[str, ArenaTrajectory]:
    """Dörfler at each pre-registered θ, then uniform, each on a fresh substrate."""
    policies: list[tuple[str, Any, float]] = [
        (dorfler_label(theta), "adaptive", theta) for theta in config.dorfler_thetas
    ]
    # The uniform policy ignores θ; the adequacy θ is passed only to fill the argument.
    policies.append((UNIFORM_LABEL, "uniform", config.adequacy_theta))
    arms: dict[str, ArenaTrajectory] = {}
    for label, policy, theta in policies:
        started = time.perf_counter()
        arms[label] = run_classical_arm(
            _fresh_substrate(config),
            policy=policy,
            theta=theta,
            max_levels=config.max_refinements_classical,
            max_dof=config.classical_max_dof,
            error_tolerance=config.error_tolerance,
        )
        last = arms[label].points[-1] if arms[label].points else None
        logger.info(
            "lookahead_classical_arm_done",
            testbed=config.testbed_label(),
            arm=label,
            theta=theta if policy == "adaptive" else None,
            levels=len(arms[label].points),
            final_dof=None if last is None else last.n_dof,
            final_l2=None if last is None else last.l2_error,
            wall_seconds=time.perf_counter() - started,
        )
    return arms


def score_run(
    run: SearchRun,
    classical: Mapping[str, ArenaTrajectory],
    alphas: Sequence[float],
) -> ScoredRun:
    """Read one MCTS run against the classical arms, greedy and the best one."""
    reading = matched_reading(run.trajectory, classical)
    greedy = classical[GREEDY_LABEL]
    return ScoredRun(
        label=run.arm.label,
        seed=run.seed,
        add_noise=run.arm.add_noise,
        trajectory=run.trajectory,
        steps=run.steps,
        reading=reading,
        versus_greedy=compare_trajectories(reference=greedy, candidate=run.trajectory),
        versus_best=compare_trajectories(
            reference=classical[reading.best_classical], candidate=run.trajectory
        ),
        break_even=break_even(
            run.trajectory, greedy, matched_dof=reading.matched_dof, alphas=alphas
        ),
    )


@dataclass
class LookaheadVsGreedyResult:
    """One testbed: every arm, every scored MCTS run, the metrics and the verdict."""

    testbed: str
    greedy: ArenaTrajectory
    #: Dörfler at each θ, then uniform -- the sweep-driven classical arms.
    sweep_arms: dict[str, ArenaTrajectory]
    primary: ScoredRun
    robust: list[ScoredRun]
    verdict: Gate1Verdict
    metrics_by_name: dict[str, float]
    dof_convention: str
    substrate_describe: dict[str, str]
    adequacy: RateSeparation | None = None
    #: Arm label (robustness runs suffixed with their seed) -> wall-clock seconds.
    wall_seconds: dict[str, float] = field(default_factory=dict)

    def classical(self) -> dict[str, ArenaTrajectory]:
        """The five classical arms in reading order: greedy, Dörfler θs, uniform."""
        return {GREEDY_LABEL: self.greedy, **self.sweep_arms}

    def runs(self) -> list[ScoredRun]:
        """Every MCTS run: the primary first, then the robustness seeds."""
        return [self.primary, *self.robust]

    def metrics(self) -> dict[str, float]:
        """Every recorded metric (all finite), gated and ungated, with the verdict."""
        return dict(self.metrics_by_name)


def _run_search_arms(
    config: LookaheadVsGreedyConfig,
    caches: list[tuple[str, FingerprintSolveCache]],
) -> tuple[SearchRun, list[SearchRun]]:
    """The primary run, then one robustness run per seed; caches kept alive."""
    primary, cache = run_search_arm(config, PRIMARY_ARM, config.seed)
    caches.append((f"{PRIMARY_LABEL} seed {config.seed}", cache))
    robust: list[SearchRun] = []
    for seed in config.robust_seeds():
        run, cache = run_search_arm(config, ROBUST_ARM, seed)
        caches.append((f"{ROBUST_LABEL} seed {seed}", cache))
        robust.append(run)
    return primary, robust


def run_lookahead_vs_greedy(config: LookaheadVsGreedyConfig) -> LookaheadVsGreedyResult:
    """Run one testbed end to end and score it against the pre-registered criteria.

    Raises:
        AdequacyAbortedError: The testbed failed its adequacy precondition (no arm ran).
        ClassicalBudgetError: A classical arm did not span the game-driven DOF.
        BreakEvenCalibrationError: Greedy's per-solve time could not be calibrated.
        NonFiniteMetricError: A recorded metric was not finite.

    """
    testbed = config.testbed_label()
    started = time.perf_counter()
    adequacy = check_adequacy(config)
    sweep_arms = run_sweep_arms(config)
    greedy_started = time.perf_counter()
    greedy, greedy_cache = run_greedy_arm_with_cache(config.arena_config())
    wall = {GREEDY_LABEL: time.perf_counter() - greedy_started}
    caches: list[tuple[str, FingerprintSolveCache]] = [(GREEDY_LABEL, greedy_cache)]
    primary_run, robust_runs = _run_search_arms(config, caches)
    # Every cache is still alive here, so identity is meaningful (ids are reused once freed).
    _assert_private_caches(next(iter(sweep_arms.values())), caches)
    searched = [primary_run, *robust_runs]
    assert_classical_span(sweep_arms, [greedy, *(run.trajectory for run in searched)])
    wall.update({f"{run.arm.label}_seed{run.seed}": run.wall_seconds for run in searched})

    classical = {GREEDY_LABEL: greedy, **sweep_arms}
    primary = score_run(primary_run, classical, config.break_even_alphas)
    robust = [score_run(run, classical, config.break_even_alphas) for run in robust_runs]
    metrics = collect_testbed_metrics(primary, robust, robust_win_ratio=config.robust_win_ratio)
    if adequacy is not None:
        metrics.update(adequacy_metrics(adequacy))
    verdict = evaluate_verdict(testbed, metrics, config.get_default_thresholds())
    metrics.update(verdict.metrics())
    metrics[WALL_SECONDS_TOTAL_METRIC] = time.perf_counter() - started
    describe = {str(k): str(v) for k, v in _fresh_substrate(config).describe().items()}
    logger.info(
        "lookahead_verdict",
        testbed=testbed,
        verdict=verdict.label,
        best_classical=primary.reading.best_classical,
        matched_dof=primary.reading.matched_dof,
        wall_seconds_total=metrics[WALL_SECONDS_TOTAL_METRIC],
        **{item.metric: item.value for item in verdict.criteria},
    )
    return LookaheadVsGreedyResult(
        testbed=testbed,
        greedy=greedy,
        sweep_arms=sweep_arms,
        primary=primary,
        robust=robust,
        verdict=verdict,
        metrics_by_name=require_finite_metrics(metrics),
        dof_convention=describe.get("dof_convention", "unknown"),
        substrate_describe=describe,
        adequacy=adequacy,
        wall_seconds=wall,
    )


__all__ = [
    "ABORTED",
    "ADEQUACY_ABORTED_METRIC",
    "PRIMARY_ARM",
    "ROBUST_ARM",
    "WALL_SECONDS_TOTAL_METRIC",
    "AdequacyAbortedError",
    "LookaheadVsGreedyResult",
    "SearchArmSpec",
    "SearchLog",
    "SearchRun",
    "adequacy_metrics",
    "check_adequacy",
    "run_lookahead_vs_greedy",
    "run_search_arm",
    "run_sweep_arms",
    "score_run",
    "search_decision_rule",
]
