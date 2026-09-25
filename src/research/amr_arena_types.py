"""Result records of the MCTS-vs-classical AMR arena.

These types used to live in ``src.research.mcts_classical_amr_arena``, which
still re-exports every one of them. They moved so the shared episode driver
(``src.research.amr_arena_episode``) and the greedy control
(``src.research.greedy_control``) can build trajectories without importing the
harness that drives them.

A trajectory is one arm's committed (level, DOF, error, cost) readings. The
per-seed and multi-seed records hold the matched-budget ratios the harness
computes; :meth:`MultiSeedArena.metrics` only aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
#: CSV ``seed`` of an arm that runs once: the classical arms and the greedy control.
DETERMINISTIC_ARM_SEED: Final[int] = -1

#: Median over seeds of MCTS decisions that differed from the greedy choice.
DIVERGENCE_METRIC: Final[str] = "decisions_diverging_from_greedy"
#: Largest per-seed count of those decisions.
DIVERGENCE_MAX_METRIC: Final[str] = "decisions_diverging_from_greedy_max"
#: Median over seeds of MCTS / greedy quadrature L2 at matched DOF (isolates look-ahead).
MCTS_OVER_GREEDY_METRIC: Final[str] = "l2_error_ratio_mcts_over_greedy_at_matched_dof"
#: Greedy / Dörfler quadrature L2 at matched DOF (isolates marking granularity).
GREEDY_OVER_DORFLER_METRIC: Final[str] = "l2_error_ratio_greedy_over_dorfler_at_matched_dof"

ArmName = Literal["dorfler", "uniform", "mcts", "greedy"]


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
    #: How many of this arm's decisions differed from the greedy choice on the
    #: same state. Only the MCTS arm measures it; ``None`` means "not measured",
    #: never zero.
    decisions_diverging_from_greedy: int | None = None

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
    #: MCTS / greedy quadrature L2 at matched DOF; ``None`` when the greedy
    #: control did not run.
    l2_error_ratio_mcts_over_greedy_at_matched_dof: float | None = None


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
    #: The single-element greedy control; ``None`` when it did not run.
    greedy: ArenaTrajectory | None = None
    #: Greedy / Dörfler quadrature L2 at matched DOF; set whenever ``greedy`` is.
    l2_error_ratio_greedy_over_dorfler_at_matched_dof: float | None = None

    def l2_ratios(self) -> list[float]:
        """Per-seed matched-DOF ratios."""
        return [item.l2_error_ratio_at_matched_dof for item in self.per_seed]

    def representative(self) -> SeedComparison:
        """Seed whose matched-DOF ratio is the median."""
        ratios = self.l2_ratios()
        order = sorted(range(len(ratios)), key=lambda idx: ratios[idx])
        return self.per_seed[order[len(order) // 2]]

    def metrics(self) -> dict[str, float]:
        """Headline medians plus spread. Only matched-DOF is gated upstream.

        The divergence keys appear whenever every seed's MCTS trajectory
        carries a count -- on every :func:`run_comparison` result, whether or
        not the greedy control ran. The two greedy-ratio keys appear only when
        the greedy control ran.
        """
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
            **self._divergence_metrics(),
            **self._greedy_metrics(),
            **self.adequacy,
        }

    def _divergence_metrics(self) -> dict[str, float]:
        """Median and max divergent decisions, or nothing if any seed is unmeasured."""
        counts = [item.mcts.decisions_diverging_from_greedy for item in self.per_seed]
        measured = [count for count in counts if count is not None]
        if not measured or len(measured) != len(counts):
            return {}
        values = np.array(measured, dtype=np.float64)
        return {
            DIVERGENCE_METRIC: float(np.median(values)),
            DIVERGENCE_MAX_METRIC: float(np.max(values)),
        }

    def _greedy_metrics(self) -> dict[str, float]:
        """The greedy-control ratios, or nothing when the control did not run.

        Raises:
            ValueError: If the greedy trajectory is present but a ratio derived
                from it is missing -- a half-built arena, which would otherwise
                publish a metric set that looks complete and is not.

        """
        if self.greedy is None:
            return {}
        per_seed = [item.l2_error_ratio_mcts_over_greedy_at_matched_dof for item in self.per_seed]
        mcts_over_greedy = [ratio for ratio in per_seed if ratio is not None]
        greedy_over_dorfler = self.l2_error_ratio_greedy_over_dorfler_at_matched_dof
        if greedy_over_dorfler is None or len(mcts_over_greedy) != len(per_seed):
            raise ValueError(
                "the greedy control ran but its matched-DOF ratios are missing; "
                "build the arena with run_comparison or set every greedy ratio"
            )
        return {
            MCTS_OVER_GREEDY_METRIC: float(np.median(np.array(mcts_over_greedy, dtype=np.float64))),
            GREEDY_OVER_DORFLER_METRIC: float(greedy_over_dorfler),
        }


__all__ = [
    "CSV_COLUMNS",
    "DETERMINISTIC_ARM_SEED",
    "DIVERGENCE_MAX_METRIC",
    "DIVERGENCE_METRIC",
    "GREEDY_OVER_DORFLER_METRIC",
    "MCTS_OVER_GREEDY_METRIC",
    "ArenaPoint",
    "ArenaTrajectory",
    "ArmName",
    "MultiSeedArena",
    "SeedComparison",
]
