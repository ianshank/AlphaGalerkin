"""MCTS vs classical AMR on a shared ``RefinementSubstrate``.

Phase 2 of ``specs/mcts_classical_amr_arena.spec.md``. Classical arms reuse
``run_refinement_sweep`` (no solve cache). The MCTS arm and the single-element
greedy control (``src.research.greedy_control``) drive the same
``SubstrateRefinementGame`` + ``RefinementGameAdapter``, built by
``src.research.amr_arena_episode``, each with **its own**
``FingerprintSolveCache``. Sharing a cache between arms is a fairness bug.

Primary metric: matched-DOF quadrature L2 (MCTS / Dörfler, ``< 1`` means MCTS
wins). Matched-solves uses cache **misses** (unique meshes), not path-replay
``apply_action`` counts. Wall-clock is recorded ungated. The greedy control
splits the primary ratio in two: MCTS / greedy isolates what the search adds,
greedy / Dörfler isolates marking granularity. ``decisions_diverging_from_greedy``
counts the MCTS steps that chose something other than greedy.

The result records live in ``src.research.amr_arena_types``; this module
re-exports them.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import numpy as np
import structlog

from src.constants import DEFAULT_RATIO_FLOOR
from src.research.amr_arena_episode import build_arena_episode, run_arena_episode
from src.research.amr_arena_types import (
    CSV_COLUMNS,
    DETERMINISTIC_ARM_SEED,
    ArenaPoint,
    ArenaTrajectory,
    ArmName,
    MultiSeedArena,
    SeedComparison,
)
from src.research.greedy_control import (
    GREEDY_METHOD,
    GREEDY_SELECTION_RULE,
    GREEDY_TIE_BREAK,
    GreedyDivergence,
    run_greedy_arm,
    run_greedy_arm_with_cache,
)
from src.research.lshape_amr_compare import _interp_log, _step_read
from src.research.run_manifest import (
    ArmProvenance,
    GitProvenance,
    RunManifest,
    assert_proposal_grade,
    collect_git_provenance,
    collect_hardware_tag,
    collect_package_versions,
    manifest_path_for,
    write_run_manifest,
)
from src.research.substrates.config import RATIO_FLOOR, AdequacyGateConfig
from src.research.substrates.factory import build_substrate_from_config
from src.research.substrates.residual_evaluator import ResidualPriorErrorValueEvaluator
from src.research.substrates.sweep import (
    SweepPoint,
    gate_violations,
    measure_adequacy,
    run_refinement_sweep,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.pde.games.substrate_refinement import SubstrateEpisodeState
    from src.poc.scenarios.mcts_classical_amr_arena_config import (
        MCTSClassicalAMRArenaConfig,
    )
    from src.refinement.substrate import RefinementSubstrate
    from src.research.substrates.solve_cache import FingerprintSolveCache

logger = structlog.get_logger(__name__)

HARNESS_NAME: Final[str] = "scripts.run_mcts_classical_amr_arena"
#: Error per DOF divides by at least this many DOF.
MIN_DOF_DIVISOR: Final[float] = 1.0
#: Plot (marker, linestyle) per arm. The greedy control is dashed with cross
#: markers because it can coincide with the MCTS curve point for point.
_PLOT_STYLES: Final[dict[str, tuple[str, str]]] = {
    "uniform": ("o", "-"),
    "dorfler": ("o", "-"),
    "mcts": ("o", "-"),
    GREEDY_METHOD: ("x", "--"),
}


class AdequacyPreconditionError(RuntimeError):
    """The locked substrate/θ fails ``gate_violations`` — abort the comparison."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        joined = "; ".join(self.violations) if self.violations else "unknown"
        super().__init__(f"adequacy precondition failed: {joined}")


def abort_if_inadequate(
    separation: Any,
    gate: AdequacyGateConfig | None = None,
) -> None:
    """Raise if ``gate_violations`` is nonempty. Does not retune thresholds."""
    violations = gate_violations(separation, gate)
    if violations:
        raise AdequacyPreconditionError(violations)


def _sweep_to_trajectory(method: ArmName, points: list[SweepPoint]) -> ArenaTrajectory:
    """Map a classical sweep onto the shared CSV schema (no cache)."""
    traj = ArenaTrajectory(method=method, cache_id=None)
    for point in points:
        traj.points.append(
            ArenaPoint(
                level=point.level,
                n_dof=point.n_dof,
                l2_error=point.l2_error,
                wall_time_seconds=0.0,
                n_cache_misses=point.level + 1,
                n_cache_hits=0,
                n_apply_actions=point.level,
            )
        )
    return traj


def run_classical_arm(
    substrate: RefinementSubstrate[Any],
    *,
    policy: Literal["adaptive", "uniform"],
    theta: float,
    max_levels: int,
    max_dof: int,
    error_tolerance: float,
) -> ArenaTrajectory:
    """Dörfler (``adaptive``) or uniform marking via ``run_refinement_sweep``."""
    t0 = time.perf_counter()
    points = run_refinement_sweep(
        substrate,
        policy=policy,
        theta=theta,
        max_levels=max_levels,
        max_dof=max_dof,
        error_tolerance=error_tolerance,
    )
    elapsed = time.perf_counter() - t0
    method: ArmName = "dorfler" if policy == "adaptive" else "uniform"
    traj = _sweep_to_trajectory(method, points)
    if traj.points:
        n = len(traj.points)
        for idx, point in enumerate(traj.points):
            point.wall_time_seconds = elapsed * float(idx + 1) / float(n)
    return traj


def _run_mcts_episode(
    config: MCTSClassicalAMRArenaConfig,
    seed: int,
) -> tuple[ArenaTrajectory, FingerprintSolveCache]:
    """One MCTS seed; also returns its cache, alive, for the fairness check."""
    from src.mcts.search import MCTS

    np.random.seed(seed)
    episode = build_arena_episode(config, name=f"arena_mcts_{seed}")
    adapter = episode.adapter
    evaluator = ResidualPriorErrorValueEvaluator(n_actions=adapter.action_space_size)
    mcts = MCTS(
        evaluator=evaluator,
        n_simulations=config.n_simulations,
        c_puct=config.c_puct,
        search_mode=adapter.search_mode,
        use_intermediate_rewards=False,
    )
    divergence = GreedyDivergence(seed=seed)

    def choose(state: SubstrateEpisodeState, legal: list[int]) -> int:
        action = mcts.get_action(
            adapter,
            temperature=config.temperature,
            add_noise=config.add_noise,
        )
        divergence.record(state, legal, action)
        mcts.advance(action)
        return action

    traj = run_arena_episode(episode, method="mcts", config=config, choose=choose)
    traj.decisions_diverging_from_greedy = divergence.count
    logger.info(
        "arena_mcts_arm_done",
        seed=seed,
        levels=len(traj.points),
        decisions_diverging_from_greedy=divergence.count,
        cache_misses=episode.cache.misses,
        cache_hits=episode.cache.hits,
        cache_id=id(episode.cache),
    )
    return traj, episode.cache


def run_mcts_arm(config: MCTSClassicalAMRArenaConfig, seed: int) -> ArenaTrajectory:
    """One-seed MCTS arm with a private ``FingerprintSolveCache``."""
    return _run_mcts_episode(config, seed)[0]


class MatchedComparison(NamedTuple):
    """Candidate / reference ratios at the largest budget both arms reached."""

    l2_error_ratio_at_matched_dof: float
    l2_error_ratio_at_matched_solves: float
    error_per_dof_ratio_at_matched_wall_clock: float
    matched_dof: float
    matched_solves: float
    matched_wall_time_seconds: float


def compare_trajectories(
    reference: ArenaTrajectory,
    candidate: ArenaTrajectory,
) -> MatchedComparison:
    """Read ``candidate`` against ``reference`` at matched DOF, solves and wall-clock.

    Every ratio is ``candidate / reference`` (``< 1``: candidate better), each
    denominator floored. Matched solves is a step read over cache misses: an
    arm's committed error does not improve between recorded points.
    """
    floor = max(RATIO_FLOOR, DEFAULT_RATIO_FLOOR)
    matched_dof = float(min(reference.dofs().max(), candidate.dofs().max()))
    l2_ratio = _interp_log(matched_dof, candidate.dofs(), candidate.errors()) / max(
        _interp_log(matched_dof, reference.dofs(), reference.errors()),
        floor,
    )
    matched_solves = float(min(reference.solve_counts().max(), candidate.solve_counts().max()))
    solve_ratio = _step_read(matched_solves, candidate.solve_counts(), candidate.errors()) / max(
        _step_read(matched_solves, reference.solve_counts(), reference.errors()),
        floor,
    )
    matched_t = float(min(reference.wall_times().max(), candidate.wall_times().max()))
    epd_reference = reference.errors() / np.maximum(reference.dofs(), MIN_DOF_DIVISOR)
    epd_candidate = candidate.errors() / np.maximum(candidate.dofs(), MIN_DOF_DIVISOR)
    wall_ratio = _interp_log(matched_t, candidate.wall_times(), epd_candidate) / max(
        _interp_log(matched_t, reference.wall_times(), epd_reference),
        floor,
    )
    return MatchedComparison(
        l2_ratio, solve_ratio, wall_ratio, matched_dof, matched_solves, matched_t
    )


def compare_mcts_vs_dorfler(
    dorfler: ArenaTrajectory,
    mcts: ArenaTrajectory,
) -> tuple[float, float, float, float, float, float]:
    """Return matched-DOF, matched-solves, matched-wall ratios and the anchors.

    MCTS over Dörfler; :func:`compare_trajectories` with the arms named.
    """
    return compare_trajectories(reference=dorfler, candidate=mcts)


def _assert_private_caches(
    dorfler: ArenaTrajectory,
    caches: Sequence[tuple[str, FingerprintSolveCache]],
) -> None:
    """Every game-driven arm solves through its own cache, and none through Dörfler's.

    Compared by identity while every cache is alive. Trajectories only keep
    ``id(cache)``, and CPython can hand a freed cache's id to the next one, so
    comparing ids of dead caches would report sharing that never happened.
    """
    for index, (label, cache) in enumerate(caches):
        if dorfler.cache_id is not None and id(cache) == dorfler.cache_id:
            raise RuntimeError(f"{label} and Dörfler must not share a FingerprintSolveCache")
        for other_label, other in caches[index + 1 :]:
            if cache is other:
                raise RuntimeError(
                    f"{label} and {other_label} must not share a FingerprintSolveCache"
                )


def _build_substrate(config: MCTSClassicalAMRArenaConfig) -> RefinementSubstrate[Any]:
    """Fresh substrate instance (classical and MCTS must not share meshes)."""
    return build_substrate_from_config(
        config.substrate,
        operator_name=config.operator_name,
        scale=config.lshape_scale,
    )


def run_comparison(config: MCTSClassicalAMRArenaConfig) -> MultiSeedArena:
    """Adequacy abort (optional), classical arms and greedy once, MCTS per seed."""
    adequacy_metrics: dict[str, float] = {}
    if config.require_adequacy_precondition:
        gate = config.adequacy_gate()
        host = _build_substrate(config)
        separation = measure_adequacy(
            host,
            theta=config.marking_fraction,
            gate=gate,
        )
        abort_if_inadequate(separation, gate)
        adequacy_metrics = {
            "adequacy_adaptive_rate": float(separation.adaptive_rate),
            "adequacy_uniform_rate": float(separation.uniform_rate),
            "adequacy_error_ratio": float(separation.error_ratio_at_matched_dof),
            "adequacy_matched_dof": float(separation.matched_dof),
        }

    classical_host = _build_substrate(config)
    describe = classical_host.describe()
    dof_convention = str(describe.get("dof_convention", "unknown"))
    dorfler = run_classical_arm(
        classical_host,
        policy="adaptive",
        theta=config.marking_fraction,
        max_levels=config.max_refinements_classical,
        max_dof=config.max_dof,
        error_tolerance=config.error_tolerance,
    )
    uniform_host = _build_substrate(config)
    uniform = run_classical_arm(
        uniform_host,
        policy="uniform",
        theta=config.marking_fraction,
        max_levels=config.max_refinements_classical,
        max_dof=config.max_dof,
        error_tolerance=config.error_tolerance,
    )

    greedy, greedy_cache = (
        run_greedy_arm_with_cache(config) if config.include_greedy_control else (None, None)
    )
    caches: list[tuple[str, FingerprintSolveCache]] = []
    if greedy_cache is not None:
        caches.append((GREEDY_METHOD, greedy_cache))
    per_seed: list[SeedComparison] = []
    for seed in config.resolved_seeds():
        mcts, mcts_cache = _run_mcts_episode(config, seed)
        caches.append((f"mcts seed {seed}", mcts_cache))
        _assert_private_caches(dorfler, caches)
        l2, solves, wall, mdof, msolves, mt = compare_trajectories(dorfler, mcts)
        per_seed.append(
            SeedComparison(
                seed=seed,
                mcts=mcts,
                l2_error_ratio_at_matched_dof=l2,
                l2_error_ratio_at_matched_solves=solves,
                error_per_dof_ratio_at_matched_wall_clock=wall,
                matched_dof=mdof,
                matched_solves=msolves,
                matched_wall_time_seconds=mt,
                l2_error_ratio_mcts_over_greedy_at_matched_dof=(
                    None
                    if greedy is None
                    else compare_trajectories(greedy, mcts).l2_error_ratio_at_matched_dof
                ),
            )
        )
    return MultiSeedArena(
        dorfler=dorfler,
        uniform=uniform,
        per_seed=per_seed,
        seeds=list(config.resolved_seeds()),
        marking_fraction=config.marking_fraction,
        dof_convention=dof_convention,
        substrate_describe={str(k): str(v) for k, v in describe.items()},
        adequacy=adequacy_metrics,
        greedy=greedy,
        l2_error_ratio_greedy_over_dorfler_at_matched_dof=(
            None
            if greedy is None
            else compare_trajectories(dorfler, greedy).l2_error_ratio_at_matched_dof
        ),
    )


def _write_trajectory_rows(
    writer: csv.DictWriter[str],
    method: str,
    seed: int,
    traj: ArenaTrajectory,
) -> None:
    """One CSV row per committed point, in the committed artifact's number format."""
    for point in traj.points:
        writer.writerow(
            {
                "method": method,
                "seed": seed,
                "level": point.level,
                "n_dof": point.n_dof,
                "l2_error": f"{point.l2_error:.12g}",
                "wall_time_seconds": f"{point.wall_time_seconds:.6g}",
                "n_cache_misses": point.n_cache_misses,
                "n_cache_hits": point.n_cache_hits,
                "n_apply_actions": point.n_apply_actions,
            }
        )


def export_csv(result: MultiSeedArena, path: str | Path) -> Path:
    """Write per-level rows: uniform, Dörfler, greedy (if it ran), each MCTS seed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arms: list[tuple[str, int, ArenaTrajectory]] = [
        ("uniform", DETERMINISTIC_ARM_SEED, result.uniform),
        ("dorfler", DETERMINISTIC_ARM_SEED, result.dorfler),
    ]
    if result.greedy is not None:
        arms.append((GREEDY_METHOD, DETERMINISTIC_ARM_SEED, result.greedy))
    arms.extend(("mcts", item.seed, item.mcts) for item in result.per_seed)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for method, seed, traj in arms:
            _write_trajectory_rows(writer, method, seed, traj)
    return destination


def export_plot(result: MultiSeedArena, path: str | Path) -> Path | None:
    """Log-log DOF vs L2 for every arm (representative MCTS seed)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("arena_plot_skipped_no_matplotlib")
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    arms: list[tuple[str, ArenaTrajectory]] = [
        ("uniform", result.uniform),
        ("dorfler", result.dorfler),
        ("mcts", result.representative().mcts),
    ]
    if result.greedy is not None:
        arms.append((GREEDY_METHOD, result.greedy))
    for label, traj in arms:
        if traj.points:
            marker, linestyle = _PLOT_STYLES[label]
            axis.loglog(traj.dofs(), traj.errors(), marker=marker, linestyle=linestyle, label=label)
    axis.set_xlabel("DOF")
    axis.set_ylabel("quadrature L2")
    axis.legend()
    axis.set_title(
        f"θ={result.marking_fraction}, {result.dof_convention} "
        "(adequacy rates are not a look-ahead win)"
    )
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _greedy_provenance(
    result: MultiSeedArena,
    config: MCTSClassicalAMRArenaConfig,
) -> list[ArmProvenance]:
    """The greedy control's manifest row, or none when it did not run."""
    if result.greedy is None or not result.greedy.points:
        return []
    last = result.greedy.points[-1]
    return [
        ArmProvenance(
            name=GREEDY_METHOD,
            parameters={
                "selection_rule": GREEDY_SELECTION_RULE,
                "tie_break": GREEDY_TIE_BREAK,
                "top_k_actions": config.top_k_actions,
                "max_steps": config.max_steps,
                "max_action_space": config.max_action_space,
                "cache": "private_FingerprintSolveCache",
            },
            counters={
                "n_levels": float(len(result.greedy.points)),
                "final_dof": float(last.n_dof),
                "n_cache_misses": float(last.n_cache_misses),
            },
        )
    ]


def write_arena_manifest(
    result: MultiSeedArena,
    config: MCTSClassicalAMRArenaConfig,
    csv_path: Path,
    png_path: Path | None,
    *,
    proposal_grade: bool = False,
    git: GitProvenance | None = None,
) -> Path:
    """Persist ``*.run.json`` beside the CSV. Collectors never raise.

    ``git`` should be snapshotted *before* CSV/PNG write. Collecting after
    those files exist marks an otherwise clean tree dirty via untracked
    (or overwritten) artifacts, which is a chicken-egg, not a dirty source
    tree. ``None`` falls back to a live probe for callers that do not write.
    """
    metrics = result.metrics()
    git_state = git if git is not None else collect_git_provenance()
    manifest = RunManifest(
        run_id=config.compute_hash(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        harness=HARNESS_NAME,
        config_hash=config.compute_hash(),
        config=config.model_dump(mode="json"),
        git=git_state,
        packages=collect_package_versions(),
        hardware_tag=collect_hardware_tag(),
        seeds=list(result.seeds),
        arms=[
            ArmProvenance(
                name="dorfler",
                parameters={
                    "marking_fraction": config.marking_fraction,
                    "max_dof": config.max_dof,
                    "cache": "none",
                    "dof_convention": result.dof_convention,
                    "substrate_describe": result.substrate_describe,
                },
                counters={
                    "n_levels": float(len(result.dorfler.points)),
                    "final_dof": float(result.dorfler.points[-1].n_dof)
                    if result.dorfler.points
                    else 0.0,
                },
            ),
            ArmProvenance(
                name="uniform",
                parameters={"max_dof": config.max_dof, "cache": "none"},
                counters={"n_levels": float(len(result.uniform.points))},
            ),
            ArmProvenance(
                name="mcts",
                parameters={
                    "evaluator": config.evaluator_name,
                    "search_mode": config.search_mode,
                    "add_noise": config.add_noise,
                    "temperature": config.temperature,
                    "n_simulations": config.n_simulations,
                    "top_k_actions": config.top_k_actions,
                    "max_action_space": config.max_action_space,
                    "use_intermediate_rewards": config.use_intermediate_rewards,
                    "cache": "per_seed_FingerprintSolveCache",
                },
                counters={
                    "n_seeds": float(len(result.seeds)),
                    "median_l2_ratio": metrics["l2_error_ratio_at_matched_dof"],
                },
            ),
            *_greedy_provenance(result, config),
        ],
        metrics=metrics,
        artifacts={"csv": str(csv_path), **({"png": str(png_path)} if png_path else {})},
        notes=(
            "Primary metric is matched-DOF quadrature L2 (MCTS/Dörfler). "
            "Matched-solves uses cache misses, not apply_action replay. "
            f"θ={config.marking_fraction}; policy max_dof={config.max_dof}; "
            f"adequacy window={config.adequacy_gate().rate_fit_dof_range}; "
            f"dof_convention={result.dof_convention}. "
            "Adequacy rates are gate evidence, not a look-ahead win. "
            "Legacy results/lshape_mcts_vs_dorfler.csv is non-informative for "
            "element-local policy. decisions_diverging_from_greedy counts MCTS "
            "steps whose action differed from single-element greedy marking. "
            + (
                "Greedy control (own cache, same game): MCTS/greedy isolates "
                "what search adds, greedy/Dörfler isolates marking granularity."
                if result.greedy is not None
                else "The greedy control arm did not run."
            )
        ),
    )
    sidecar = write_run_manifest(manifest, manifest_path_for(csv_path))
    if proposal_grade:
        assert_proposal_grade(manifest)
    return sidecar


__all__ = [
    "CSV_COLUMNS",
    "AdequacyPreconditionError",
    "ArenaPoint",
    "ArenaTrajectory",
    "MatchedComparison",
    "MultiSeedArena",
    "SeedComparison",
    "abort_if_inadequate",
    "compare_mcts_vs_dorfler",
    "compare_trajectories",
    "export_csv",
    "export_plot",
    "run_classical_arm",
    "run_comparison",
    "run_greedy_arm",
    "run_mcts_arm",
    "write_arena_manifest",
]
