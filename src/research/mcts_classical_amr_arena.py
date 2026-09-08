"""MCTS vs classical AMR on a shared ``RefinementSubstrate``.

Phase 2 of ``specs/mcts_classical_amr_arena.spec.md``. Classical arms reuse
``run_refinement_sweep`` (no solve cache). The MCTS arm uses
``SubstrateRefinementGame`` + ``RefinementGameAdapter`` with **its own**
``FingerprintSolveCache``. Sharing that cache with Dörfler is a fairness bug.

Primary metric: matched-DOF quadrature L2 (MCTS / Dörfler, ``< 1`` means MCTS
wins). Matched-solves uses cache **misses** (unique meshes), not path-replay
``apply_action`` counts. Wall-clock is recorded ungated.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import structlog

from src.constants import DEFAULT_RATIO_FLOOR
from src.pde.games.substrate_refinement import (
    SubstrateEpisodeState,
    SubstrateRefinementGame,
)
from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.refinement.adapter import RefinementGameAdapter
from src.research.lshape_amr_compare import _interp_log, _step_read
from src.research.run_manifest import (
    ArmProvenance,
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
from src.research.substrates.solve_cache import FingerprintSolveCache
from src.research.substrates.sweep import (
    SweepPoint,
    gate_violations,
    measure_adequacy,
    run_refinement_sweep,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.poc.scenarios.mcts_classical_amr_arena_config import (
        MCTSClassicalAMRArenaConfig,
    )
    from src.refinement.substrate import RefinementSubstrate

logger = structlog.get_logger(__name__)

CSV_COLUMNS: Final[tuple[str, ...]] = (
    "method",
    "seed",
    "level",
    "n_dof",
    "l2_error",
    "wall_time_seconds",
    "n_cache_misses",
    "n_cache_hits",
    "n_apply_actions",
)
HARNESS_NAME: Final[str] = "scripts.run_mcts_classical_amr_arena"
ArmName = Literal["dorfler", "uniform", "mcts"]


class AdequacyPreconditionError(RuntimeError):
    """The locked substrate/θ fails ``gate_violations`` — abort the comparison."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        joined = "; ".join(self.violations) if self.violations else "unknown"
        super().__init__(f"adequacy precondition failed: {joined}")


@dataclass
class ArenaPoint:
    """One committed (level, DOF, error, cost) reading."""

    level: int
    n_dof: int
    l2_error: float
    wall_time_seconds: float
    n_cache_misses: int
    n_cache_hits: int
    n_apply_actions: int


@dataclass
class ArenaTrajectory:
    """One arm's refinement trajectory."""

    method: ArmName
    points: list[ArenaPoint] = field(default_factory=list)
    cache_id: int | None = None

    def dofs(self) -> NDArray[np.float64]:
        """DOF counts along the trajectory."""
        return np.array([p.n_dof for p in self.points], dtype=np.float64)

    def errors(self) -> NDArray[np.float64]:
        """Quadrature L2 along the trajectory."""
        return np.array([p.l2_error for p in self.points], dtype=np.float64)

    def wall_times(self) -> NDArray[np.float64]:
        """Cumulative wall-clock along the trajectory."""
        return np.array([p.wall_time_seconds for p in self.points], dtype=np.float64)

    def solve_counts(self) -> NDArray[np.float64]:
        """Unique-mesh solves (cache misses). Path replay is ``n_cache_hits``."""
        return np.array([p.n_cache_misses for p in self.points], dtype=np.float64)


@dataclass
class SeedComparison:
    """One seed's MCTS arm against the shared classical trajectories."""

    seed: int
    mcts: ArenaTrajectory
    l2_error_ratio_at_matched_dof: float
    l2_error_ratio_at_matched_solves: float
    error_per_dof_ratio_at_matched_wall_clock: float
    matched_dof: float
    matched_solves: float
    matched_wall_time_seconds: float


@dataclass
class MultiSeedArena:
    """Median-over-seeds headline plus per-seed spread."""

    dorfler: ArenaTrajectory
    uniform: ArenaTrajectory
    per_seed: list[SeedComparison]
    seeds: list[int]
    marking_fraction: float
    dof_convention: str
    substrate_describe: dict[str, str] = field(default_factory=dict)
    adequacy: dict[str, float] = field(default_factory=dict)

    def l2_ratios(self) -> list[float]:
        """Per-seed matched-DOF ratios."""
        return [item.l2_error_ratio_at_matched_dof for item in self.per_seed]

    def representative(self) -> SeedComparison:
        """Seed whose matched-DOF ratio is the median."""
        ratios = self.l2_ratios()
        order = sorted(range(len(ratios)), key=lambda idx: ratios[idx])
        return self.per_seed[order[len(order) // 2]]

    def metrics(self) -> dict[str, float]:
        """Headline medians plus spread. Only matched-DOF is gated upstream."""
        l2 = np.array(self.l2_ratios(), dtype=np.float64)
        solves = np.array(
            [item.l2_error_ratio_at_matched_solves for item in self.per_seed],
            dtype=np.float64,
        )
        walls = np.array(
            [item.error_per_dof_ratio_at_matched_wall_clock for item in self.per_seed],
            dtype=np.float64,
        )
        representative = self.representative()
        return {
            "l2_error_ratio_at_matched_dof": float(np.median(l2)),
            "l2_error_ratio_at_matched_solves": float(np.median(solves)),
            "error_per_dof_ratio_mcts_over_dorfler": float(np.median(walls)),
            "matched_dof": float(representative.matched_dof),
            "matched_solves": float(representative.matched_solves),
            "matched_wall_time_seconds": float(representative.matched_wall_time_seconds),
            "mcts_win_fraction": float(np.mean(l2 < 1.0)),
            "l2_ratio_seed_min": float(np.min(l2)),
            "l2_ratio_seed_max": float(np.max(l2)),
            "l2_ratio_seed_std": float(np.std(l2, ddof=0)),
            "n_seeds": float(len(self.seeds)),
            "marking_fraction": float(self.marking_fraction),
            **self.adequacy,
        }


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


def _assert_action_space_covers(game: SubstrateRefinementGame, state: Any) -> None:
    """Refuse silent truncation of high-index triangles."""
    if not isinstance(state, SubstrateEpisodeState) or state.mesh is None:
        return
    n_units = int(game.substrate.n_units(state.mesh))
    if n_units > game.action_space_size:
        raise ValueError(
            f"max_action_space={game.action_space_size} truncates n_units={n_units}; "
            "raise max_action_space above the window's element count"
        )


def _point_from_adapter(
    adapter: RefinementGameAdapter,
    cache: FingerprintSolveCache,
    t0: float,
    apply_actions: int,
) -> ArenaPoint:
    """Snapshot the committed MCTS state plus cache counters."""
    return ArenaPoint(
        level=int(adapter.state.step),
        n_dof=int(adapter.state.dof),
        l2_error=float(adapter.state.error_estimate),
        wall_time_seconds=time.perf_counter() - t0,
        n_cache_misses=int(cache.misses),
        n_cache_hits=int(cache.hits),
        n_apply_actions=apply_actions,
    )


def run_mcts_arm(config: MCTSClassicalAMRArenaConfig, seed: int) -> ArenaTrajectory:
    """One-seed MCTS arm with a private ``FingerprintSolveCache``."""
    from src.mcts.search import MCTS

    np.random.seed(seed)
    cache = FingerprintSolveCache(config.substrate.solve_cache_max_entries)
    game = SubstrateRefinementGame(
        SubstrateRefinementConfig(
            name=f"arena_mcts_{seed}",
            substrate=config.substrate,
            operator_name=config.operator_name,
            lshape_scale=config.lshape_scale,
            max_steps=config.max_steps,
            error_tolerance=config.error_tolerance,
            computational_budget=float(config.max_steps + 1),
            refine_cost=1.0,
            max_action_space=config.max_action_space,
            value_scale=config.value_scale,
            top_k_actions=config.top_k_actions,
        ),
        solve_cache=cache,
    )
    adapter = RefinementGameAdapter(game)
    _assert_action_space_covers(game, adapter.state)
    evaluator = ResidualPriorErrorValueEvaluator(n_actions=adapter.action_space_size)
    mcts = MCTS(
        evaluator=evaluator,
        n_simulations=config.n_simulations,
        c_puct=config.c_puct,
        search_mode=adapter.search_mode,
        use_intermediate_rewards=False,
    )
    t0 = time.perf_counter()
    traj = ArenaTrajectory(method="mcts", cache_id=id(cache))
    traj.points.append(_point_from_adapter(adapter, cache, t0, apply_actions=0))
    for _step in range(config.max_steps):
        if adapter.is_terminal() or not adapter.get_legal_actions():
            break
        if adapter.state.dof >= config.max_dof:
            break
        action = mcts.get_action(
            adapter,
            temperature=config.temperature,
            add_noise=config.add_noise,
        )
        adapter.apply_action(action)
        mcts.advance(action)
        _assert_action_space_covers(game, adapter.state)
        traj.points.append(
            _point_from_adapter(
                adapter,
                cache,
                t0,
                apply_actions=int(adapter.state.step),
            )
        )
    logger.info(
        "arena_mcts_arm_done",
        seed=seed,
        levels=len(traj.points),
        cache_misses=cache.misses,
        cache_hits=cache.hits,
        cache_id=id(cache),
    )
    return traj


def compare_mcts_vs_dorfler(
    dorfler: ArenaTrajectory,
    mcts: ArenaTrajectory,
) -> tuple[float, float, float, float, float, float]:
    """Return matched-DOF, matched-solves, matched-wall ratios and the anchors."""
    floor = max(RATIO_FLOOR, DEFAULT_RATIO_FLOOR)
    matched_dof = float(min(dorfler.dofs().max(), mcts.dofs().max()))
    l2_ratio = _interp_log(matched_dof, mcts.dofs(), mcts.errors()) / max(
        _interp_log(matched_dof, dorfler.dofs(), dorfler.errors()),
        floor,
    )
    matched_solves = float(min(dorfler.solve_counts().max(), mcts.solve_counts().max()))
    solve_ratio = _step_read(matched_solves, mcts.solve_counts(), mcts.errors()) / max(
        _step_read(matched_solves, dorfler.solve_counts(), dorfler.errors()),
        floor,
    )
    matched_t = float(min(dorfler.wall_times().max(), mcts.wall_times().max()))
    epd_d = dorfler.errors() / np.maximum(dorfler.dofs(), 1.0)
    epd_m = mcts.errors() / np.maximum(mcts.dofs(), 1.0)
    wall_ratio = _interp_log(matched_t, mcts.wall_times(), epd_m) / max(
        _interp_log(matched_t, dorfler.wall_times(), epd_d),
        floor,
    )
    return l2_ratio, solve_ratio, wall_ratio, matched_dof, matched_solves, matched_t


def _build_substrate(config: MCTSClassicalAMRArenaConfig) -> RefinementSubstrate[Any]:
    """Fresh substrate instance (classical and MCTS must not share meshes)."""
    return build_substrate_from_config(
        config.substrate,
        operator_name=config.operator_name,
        scale=config.lshape_scale,
    )


def run_comparison(config: MCTSClassicalAMRArenaConfig) -> MultiSeedArena:
    """Adequacy abort (optional), classical arms once, MCTS per seed."""
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

    per_seed: list[SeedComparison] = []
    for seed in config.resolved_seeds():
        mcts = run_mcts_arm(config, seed)
        if dorfler.cache_id is not None and mcts.cache_id == dorfler.cache_id:
            raise RuntimeError("MCTS and Dörfler must not share a FingerprintSolveCache")
        l2, solves, wall, mdof, msolves, mt = compare_mcts_vs_dorfler(dorfler, mcts)
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
    )


def export_csv(result: MultiSeedArena, path: str | Path) -> Path:
    """Write per-level rows for uniform, Dörfler, and each MCTS seed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for method, traj, seed in (
            ("uniform", result.uniform, -1),
            ("dorfler", result.dorfler, -1),
        ):
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
        for item in result.per_seed:
            for point in item.mcts.points:
                writer.writerow(
                    {
                        "method": "mcts",
                        "seed": item.seed,
                        "level": point.level,
                        "n_dof": point.n_dof,
                        "l2_error": f"{point.l2_error:.12g}",
                        "wall_time_seconds": f"{point.wall_time_seconds:.6g}",
                        "n_cache_misses": point.n_cache_misses,
                        "n_cache_hits": point.n_cache_hits,
                        "n_apply_actions": point.n_apply_actions,
                    }
                )
    return destination


def export_plot(result: MultiSeedArena, path: str | Path) -> Path | None:
    """Log-log DOF vs L2 for the three arms (representative MCTS seed)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("arena_plot_skipped_no_matplotlib")
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    for label, traj in (
        ("uniform", result.uniform),
        ("dorfler", result.dorfler),
        ("mcts", result.representative().mcts),
    ):
        if traj.points:
            axis.loglog(traj.dofs(), traj.errors(), marker="o", label=label)
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


def write_arena_manifest(
    result: MultiSeedArena,
    config: MCTSClassicalAMRArenaConfig,
    csv_path: Path,
    png_path: Path | None,
    *,
    proposal_grade: bool = False,
) -> Path:
    """Persist ``*.run.json`` beside the CSV. Collectors never raise."""
    metrics = result.metrics()
    manifest = RunManifest(
        run_id=config.compute_hash(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        harness=HARNESS_NAME,
        config_hash=config.compute_hash(),
        config=config.model_dump(mode="json"),
        git=collect_git_provenance(),
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
            "element-local policy."
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
    "MultiSeedArena",
    "SeedComparison",
    "abort_if_inadequate",
    "compare_mcts_vs_dorfler",
    "export_csv",
    "export_plot",
    "run_classical_arm",
    "run_comparison",
    "run_mcts_arm",
    "write_arena_manifest",
]
