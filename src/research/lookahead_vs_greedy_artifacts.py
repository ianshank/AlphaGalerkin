"""Artifacts of one Gate 1 testbed: the CSV, the plot and the provenance sidecar.

* **CSV** -- the arena's column schema (``amr_arena_types.CSV_COLUMNS``) and
  number format (``mcts_classical_amr_arena._write_trajectory_rows``), one row
  per committed point, with ``method`` = ``uniform``, ``dorfler_theta<θ>``,
  ``greedy``, ``mcts_primary`` or ``mcts_robust``. Classical arms and greedy run
  once and carry ``seed = -1``; each MCTS row carries the seed its run used.
* **PNG** -- log-log DOF vs quadrature L2 for every arm, the primary run's
  matched DOF marked.
* **``.run.json``** -- written through :class:`~src.research.run_manifest.RunRecorder`,
  whose git snapshot was taken before the run computed or wrote anything, and
  which re-verifies the sidecar from disk under proposal grade. It carries the
  verdict and every criterion's value (``metrics`` and ``notes``), each arm's
  parameters and counters, and each MCTS run's per-step realized tree depth --
  the file ``results/MANIFEST.sha256`` freezes.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import structlog

from src.research.amr_arena_types import CSV_COLUMNS, DETERMINISTIC_ARM_SEED
from src.research.greedy_control import GREEDY_SELECTION_RULE, GREEDY_TIE_BREAK
from src.research.lookahead_vs_greedy_metrics import (
    GREEDY_LABEL,
    PRIMARY_LABEL,
    ROBUST_LABEL,
    UNIFORM_LABEL,
    alpha_suffix,
    dorfler_label,
)
from src.research.mcts_classical_amr_arena import _write_trajectory_rows
from src.research.run_manifest import (
    ArmProvenance,
    RunManifest,
    collect_hardware_tag,
    collect_package_versions,
)

if TYPE_CHECKING:
    from src.poc.scenarios.lookahead_vs_greedy_config import LookaheadVsGreedyConfig
    from src.research.amr_arena_types import ArenaTrajectory
    from src.research.lookahead_vs_greedy import LookaheadVsGreedyResult
    from src.research.lookahead_vs_greedy_verdict import ScoredRun
    from src.research.run_manifest import RunRecorder

logger = structlog.get_logger(__name__)

#: ``harness`` recorded in the sidecar: the entry point that produces the artifacts.
HARNESS_NAME: Final[str] = "scripts.run_lookahead_vs_greedy"
#: Cache description recorded per arm.
PRIVATE_CACHE: Final[str] = "private_FingerprintSolveCache"
NO_CACHE: Final[str] = "none"
#: Plot (marker, linestyle) per arm family; Dörfler arms share one style.
_PLOT_STYLES: Final[dict[str, tuple[str, str]]] = {
    UNIFORM_LABEL: ("s", "-"),
    "dorfler": ("o", "-"),
    GREEDY_LABEL: ("x", "--"),
    PRIMARY_LABEL: ("D", "-"),
    ROBUST_LABEL: ("", ":"),
}
#: Opacity of the per-seed robustness curves (drawn behind the primary run).
_ROBUST_ALPHA: Final[float] = 0.45
_FIGURE_SIZE: Final[tuple[float, float]] = (7.5, 5.5)


def _csv_arms(result: LookaheadVsGreedyResult) -> list[tuple[str, int, ArenaTrajectory]]:
    """``(method, seed, trajectory)`` in CSV order: uniform, Dörfler, greedy, MCTS."""
    uniform = result.sweep_arms[UNIFORM_LABEL]
    dorfler = [(label, t) for label, t in result.sweep_arms.items() if label != UNIFORM_LABEL]
    return [
        (UNIFORM_LABEL, DETERMINISTIC_ARM_SEED, uniform),
        *((label, DETERMINISTIC_ARM_SEED, t) for label, t in dorfler),
        (GREEDY_LABEL, DETERMINISTIC_ARM_SEED, result.greedy),
        *((run.label, run.seed, run.trajectory) for run in result.runs()),
    ]


def export_csv(result: LookaheadVsGreedyResult, path: str | Path) -> Path:
    """Write every arm's committed points in the arena's schema (LF line endings)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for method, seed, trajectory in _csv_arms(result):
            _write_trajectory_rows(writer, method, seed, trajectory)
    return destination


def _plot_style(label: str) -> tuple[str, str]:
    """Marker and linestyle of an arm; every Dörfler θ shares the Dörfler style."""
    return _PLOT_STYLES.get(label, _PLOT_STYLES["dorfler"])


def export_plot(result: LookaheadVsGreedyResult, path: str | Path) -> Path | None:
    """Log-log DOF vs L2 for every arm; ``None`` when matplotlib is not installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("lookahead_plot_skipped_no_matplotlib")
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=_FIGURE_SIZE)
    for label, trajectory in result.classical().items():
        marker, linestyle = _plot_style(label)
        axis.loglog(
            trajectory.dofs(), trajectory.errors(), marker=marker, linestyle=linestyle, label=label
        )
    for index, run in enumerate(result.robust):
        marker, linestyle = _plot_style(ROBUST_LABEL)
        axis.loglog(
            run.trajectory.dofs(),
            run.trajectory.errors(),
            marker=marker,
            linestyle=linestyle,
            alpha=_ROBUST_ALPHA,
            label=ROBUST_LABEL if index == 0 else None,
        )
    marker, linestyle = _plot_style(PRIMARY_LABEL)
    axis.loglog(
        result.primary.trajectory.dofs(),
        result.primary.trajectory.errors(),
        marker=marker,
        linestyle=linestyle,
        label=PRIMARY_LABEL,
    )
    axis.axvline(result.primary.reading.matched_dof, linestyle="--", color="grey", linewidth=1.0)
    axis.set_xlabel(f"DOF ({result.dof_convention})")
    axis.set_ylabel("quadrature L2")
    axis.legend(fontsize="small")
    axis.set_title(
        f"Gate 1 {result.testbed}: {result.verdict.label} "
        f"(MCTS/best classical {result.primary.reading.ratio:.4f} at DOF "
        f"{result.primary.reading.matched_dof:g})"
    )
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _final_counters(trajectory: ArenaTrajectory) -> dict[str, float]:
    """Levels and the last committed point of a trajectory."""
    last = trajectory.points[-1]
    return {
        "n_levels": float(len(trajectory.points)),
        "final_dof": float(last.n_dof),
        "final_l2": float(last.l2_error),
        "n_cache_misses": float(last.n_cache_misses),
        "wall_seconds": float(last.wall_time_seconds),
    }


def _classical_provenance(
    result: LookaheadVsGreedyResult, config: LookaheadVsGreedyConfig
) -> list[ArmProvenance]:
    """Greedy, each Dörfler θ and uniform."""
    arms = [
        ArmProvenance(
            name=GREEDY_LABEL,
            parameters={
                "selection_rule": GREEDY_SELECTION_RULE,
                "tie_break": GREEDY_TIE_BREAK,
                "top_k_actions": config.top_k_actions,
                "max_steps": config.max_steps,
                "max_action_space": config.max_action_space,
                "cache": PRIVATE_CACHE,
            },
            counters=_final_counters(result.greedy),
        )
    ]
    for theta in config.dorfler_thetas:
        label = dorfler_label(theta)
        arms.append(
            ArmProvenance(
                name=label,
                parameters={
                    "policy": "adaptive",
                    "marking_fraction": theta,
                    "max_dof": config.classical_max_dof,
                    "max_levels": config.max_refinements_classical,
                    "cache": NO_CACHE,
                },
                counters=_final_counters(result.sweep_arms[label]),
            )
        )
    arms.append(
        ArmProvenance(
            name=UNIFORM_LABEL,
            parameters={
                "policy": "uniform",
                "max_dof": config.classical_max_dof,
                "max_levels": config.max_refinements_classical,
                "cache": NO_CACHE,
            },
            counters=_final_counters(result.sweep_arms[UNIFORM_LABEL]),
        )
    )
    return arms


def _search_provenance(run: ScoredRun, config: LookaheadVsGreedyConfig) -> ArmProvenance:
    """One MCTS run: its search settings, per-step record, and every reading."""
    counters = {
        **_final_counters(run.trajectory),
        "n_cache_hits": float(run.trajectory.points[-1].n_cache_hits),
        "matched_dof": run.reading.matched_dof,
        "l2_ratio_vs_best_classical": run.reading.ratio,
        "l2_ratio_vs_greedy": run.reading.ratio_vs(GREEDY_LABEL),
        "decisions_diverging_from_greedy": float(run.divergences),
        "tree_depth_max": float(max((step.tree_depth for step in run.steps), default=0)),
        "dof_saving_vs_greedy": run.break_even.dof_saving,
    }
    for alpha, reuses in run.break_even.reuses.items():
        if reuses is not None:
            counters[f"break_even_reuses_{alpha_suffix(alpha)}"] = reuses
    parameters: dict[str, Any] = {
        "seed": run.seed,
        "add_noise": run.add_noise,
        "temperature": config.temperature,
        "n_simulations": config.n_simulations,
        "top_k_actions": config.top_k_actions,
        "c_puct": config.c_puct,
        "value_scale": config.value_scale,
        "prior_temperature": config.prior_temperature,
        "dirichlet_alpha": config.dirichlet_alpha,
        "dirichlet_epsilon": config.dirichlet_epsilon,
        "evaluator": config.evaluator_name,
        "search_mode": config.search_mode,
        "use_intermediate_rewards": False,
        "cache": PRIVATE_CACHE,
        "best_classical": run.reading.best_classical,
        "action_per_step": [step.action for step in run.steps],
        "greedy_action_per_step": [step.greedy_action for step in run.steps],
        "tree_depth_per_step": [step.tree_depth for step in run.steps],
    }
    return ArmProvenance(
        name=f"{run.label}_seed{run.seed}", parameters=parameters, counters=counters
    )


def _notes(result: LookaheadVsGreedyResult, config: LookaheadVsGreedyConfig) -> str:
    """What these numbers do and do not establish."""
    thetas = ", ".join(f"{theta:g}" for theta in config.dorfler_thetas)
    return (
        f"Gate 1, pre-registered in specs/lookahead_vs_greedy.spec.md. {result.verdict.summary()}\n"
        "Primary metric: MCTS quadrature L2 over the best classical arm (greedy, Dörfler "
        f"θ ∈ {{{thetas}}}, uniform) at the largest DOF every arm reached (primary run: "
        f"N* = {result.primary.reading.matched_dof:g}, best = "
        f"{result.primary.reading.best_classical}). K*_alpha is the number of downstream "
        "solves on the equal-accuracy mesh that repays MCTS's extra construction wall-clock "
        "against greedy (solve cost kappa*DOF^alpha, kappa calibrated from greedy's measured "
        "per-step time); it is finite iff the DOF saving is positive, and an infinite K* is "
        "recorded as primary_break_even_finite = 0. Realized tree depth counts visited nodes "
        "only. Matched-solves and matched-wall-clock ratios are recorded ungated. Adequacy "
        "rates are a precondition, not a look-ahead result. A NO-GO on T1/T2 does not close "
        "the thesis (T3, the moving front, is deferred and decisive); a GO needs replication "
        "before any claim. Untrained evaluator."
    )


class SidecarHashMismatchError(ValueError):
    """The config about to be recorded does not reproduce the hash the run started with."""


def require_reproducible_hash(config: LookaheadVsGreedyConfig, recorder: RunRecorder) -> None:
    """Refuse a config whose hash is not the one the run started with.

    The sidecar records this config's JSON dump beside the recorder's pre-run
    hash, so ``load_config_from_dict(sidecar["config"]).compute_hash()`` must
    reproduce ``sidecar["config_hash"]``. A config that changed after the
    snapshot would break that silently; it is refused instead.

    Raises:
        SidecarHashMismatchError: ``config.compute_hash()`` is not ``recorder.config_hash``.

    """
    current = config.compute_hash()
    if current != recorder.config_hash:
        raise SidecarHashMismatchError(
            f"config hashes to {current!r}, but the run started with "
            f"{recorder.config_hash!r}: the recorded config would not reproduce its hash"
        )


def build_run_manifest(
    result: LookaheadVsGreedyResult,
    config: LookaheadVsGreedyConfig,
    recorder: RunRecorder,
    artifacts: dict[str, str],
) -> RunManifest:
    """The run's sidecar, carrying the recorder's pre-write snapshot and hash.

    Raises:
        SidecarHashMismatchError: ``config`` no longer hashes to ``recorder.config_hash``.

    """
    require_reproducible_hash(config, recorder)
    return RunManifest(
        run_id=f"lookahead-vs-greedy-{result.testbed}-{recorder.config_hash}",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        harness=HARNESS_NAME,
        config_hash=recorder.config_hash,
        config=config.model_dump(mode="json"),
        git=recorder.git,
        packages=collect_package_versions(),
        hardware_tag=collect_hardware_tag(),
        seeds=[run.seed for run in result.runs()],
        arms=[
            *_classical_provenance(result, config),
            *(_search_provenance(run, config) for run in result.runs()),
        ],
        metrics=result.metrics(),
        artifacts=artifacts,
        notes=_notes(result, config),
    )


def write_artifacts(
    result: LookaheadVsGreedyResult,
    config: LookaheadVsGreedyConfig,
    recorder: RunRecorder,
) -> dict[str, Path]:
    """Write the CSV and PNG, then the sidecar through ``recorder``; return every path.

    Raises:
        SidecarHashMismatchError: Before anything is written, when ``config`` no
            longer hashes to ``recorder.config_hash``.

    """
    require_reproducible_hash(config, recorder)
    base = Path(config.output_dir) / config.artifact_basename
    csv_path = export_csv(result, Path(f"{base}.csv"))
    png_path = export_plot(result, Path(f"{base}.png"))
    written: dict[str, Path] = {"csv": csv_path}
    if png_path is not None:
        written["png"] = png_path
    manifest = build_run_manifest(
        result, config, recorder, {name: str(path) for name, path in written.items()}
    )
    written["run_json"] = recorder.write_sidecar(csv_path, manifest)
    logger.info(
        "lookahead_artifacts_written",
        testbed=result.testbed,
        verdict=result.verdict.label,
        **{name: str(path) for name, path in written.items()},
    )
    return written


__all__ = [
    "HARNESS_NAME",
    "SidecarHashMismatchError",
    "build_run_manifest",
    "export_csv",
    "export_plot",
    "require_reproducible_hash",
    "write_artifacts",
]
