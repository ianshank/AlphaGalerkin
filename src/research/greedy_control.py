"""Single-element greedy control arm for the MCTS-vs-classical AMR arena.

The arena's MCTS arm refines one element per step. Its legal set is pre-ranked
by residual indicator and its prior is a softmax over those same indicators.
Dörfler bulk-marks a fraction of elements per step. A matched-DOF ratio between
those two arms therefore mixes two effects -- *marking granularity* (one
element at a time versus a bulk fraction) and *look-ahead* (the tree search) --
and cannot tell them apart.

This control removes the search and keeps everything else. Every step it
refines the legal element with the largest residual indicator, ties to the
lowest index, on the game, adapter and stopping rules the MCTS arm uses
(``src.research.amr_arena_episode``), through its own solve cache. MCTS over
greedy isolates look-ahead; greedy over Dörfler isolates marking granularity.

:func:`greedy_action` is the one definition of the greedy choice. It ranks with
:func:`src.pde.games.substrate_refinement.rank_by_indicator`, the rule behind
the game's top-k legal set, so greedy here and the legal set the search sees
cannot drift apart. :class:`GreedyDivergence` counts how many of another arm's
decisions differed from it on the same state.

This is a reference baseline, so it must not import the search engine
(``src.mcts``): a control that shares the candidate's implementation moves with
the candidate's defects, and that is invisible in a ratio. It is in the scope
of the ``reference-baselines-do-not-import-the-candidate`` import contract, and
reaches the refinement layer only through the shared episode builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import structlog

from src.pde.games.substrate_refinement import indicator_score, rank_by_indicator
from src.research.amr_arena_episode import build_arena_episode, run_arena_episode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from src.pde.games.substrate_refinement import SubstrateEpisodeState
    from src.poc.scenarios.mcts_classical_amr_arena_config import (
        MCTSClassicalAMRArenaConfig,
    )
    from src.research.amr_arena_types import ArenaTrajectory
    from src.research.substrates.solve_cache import FingerprintSolveCache

logger = structlog.get_logger(__name__)

#: ``method`` of the greedy control in the arena CSV and manifest.
GREEDY_METHOD: Final[Literal["greedy"]] = "greedy"
#: Config name of the greedy control's game (for logs; no numeric effect).
GREEDY_GAME_NAME: Final[str] = "arena_greedy"
#: Manifest description of the decision rule.
GREEDY_SELECTION_RULE: Final[str] = "single_element_max_residual_indicator"
#: Manifest description of the tie-break.
GREEDY_TIE_BREAK: Final[str] = "lowest_action_index"
#: How many offending actions a non-finite-indicator error names.
MAX_ACTIONS_NAMED_IN_ERRORS: Final[int] = 8


def greedy_action(indicators: ArrayLike, legal_actions: Sequence[int]) -> int:
    """The legal action with the largest residual indicator; ties to the lowest index.

    Scores come from :func:`~src.pde.games.substrate_refinement.rank_by_indicator`,
    the ranking behind ``SubstrateRefinementGame.get_valid_actions``'s top-k set,
    including its rule that an index outside ``indicators`` scores ``0.0``. The
    candidates are ranked in ascending index order, so the stable sort resolves
    ties to the lowest index whatever order ``legal_actions`` arrives in.
    ``legal_actions[0]`` is *not* the greedy choice in general: without an
    active top-k cap the game returns its refinable units in index order.

    Args:
        indicators: Per-unit residual indicators (any array-like; flattened).
        legal_actions: The actions to choose from.

    Returns:
        The greedy action.

    Raises:
        ValueError: If ``legal_actions`` is empty, or a legal action scores a
            non-finite value -- a NaN compares false both ways, so "largest" is
            undefined and a sort would return an arbitrary element.

    """
    if len(legal_actions) == 0:
        raise ValueError("greedy_action needs at least one legal action; the legal set is empty")
    ranked = rank_by_indicator(indicators, sorted(legal_actions))
    non_finite = [action for score, action in ranked if not math.isfinite(score)]
    if non_finite:
        raise ValueError(
            f"greedy_action cannot rank {len(non_finite)} non-finite residual "
            f"indicator(s), at actions {non_finite[:MAX_ACTIONS_NAMED_IN_ERRORS]}"
        )
    return ranked[0][1]


@dataclass
class GreedyDivergence:
    """Counts decisions that differ from :func:`greedy_action` on the same state."""

    seed: int
    count: int = 0

    def record(self, state: SubstrateEpisodeState, legal: Sequence[int], action: int) -> None:
        """Compare ``action`` with the greedy choice on ``state``, before it is applied.

        Args:
            state: The state the decision was made in.
            legal: That state's legal set.
            action: The action the arm is about to commit.

        """
        greedy = greedy_action(state.indicators, legal)
        if action == greedy:
            return
        self.count += 1
        flat = np.asarray(state.indicators, dtype=np.float64).reshape(-1)
        logger.debug(
            "arena_decision_diverged_from_greedy",
            seed=self.seed,
            step=int(state.step),
            action=int(action),
            greedy_action=greedy,
            action_indicator=indicator_score(flat, int(action)),
            greedy_indicator=indicator_score(flat, greedy),
            dof=int(state.dof),
            n_legal=len(legal),
        )


def _choose_greedy(state: SubstrateEpisodeState, legal: list[int]) -> int:
    """Decision rule of the greedy control arm."""
    action = greedy_action(state.indicators, legal)
    flat = np.asarray(state.indicators, dtype=np.float64).reshape(-1)
    logger.debug(
        "arena_greedy_step",
        step=int(state.step),
        action=action,
        indicator=indicator_score(flat, action),
        dof=int(state.dof),
        n_legal=len(legal),
    )
    return action


def run_greedy_arm_with_cache(
    config: MCTSClassicalAMRArenaConfig,
) -> tuple[ArenaTrajectory, FingerprintSolveCache]:
    """Run the greedy control; also return its cache for the harness's fairness check.

    The cache is returned rather than only its ``id``: once a cache is freed,
    CPython may give the next one the same ``id``, so comparing ids of caches
    that are no longer alive would report sharing that never happened.
    """
    episode = build_arena_episode(config, name=GREEDY_GAME_NAME)
    traj = run_arena_episode(episode, method=GREEDY_METHOD, config=config, choose=_choose_greedy)
    logger.info(
        "arena_greedy_arm_done",
        levels=len(traj.points),
        final_dof=traj.points[-1].n_dof,
        cache_misses=episode.cache.misses,
        cache_hits=episode.cache.hits,
        cache_id=id(episode.cache),
    )
    return traj, episode.cache


def run_greedy_arm(config: MCTSClassicalAMRArenaConfig) -> ArenaTrajectory:
    """Single-element greedy marking on the MCTS arm's game, with its own cache.

    Deterministic -- no search, no noise, no seed -- so it runs once and is
    written to the CSV with ``seed = -1``, like the classical arms.
    """
    return run_greedy_arm_with_cache(config)[0]


__all__ = [
    "GREEDY_GAME_NAME",
    "GREEDY_METHOD",
    "GREEDY_SELECTION_RULE",
    "GREEDY_TIE_BREAK",
    "MAX_ACTIONS_NAMED_IN_ERRORS",
    "GreedyDivergence",
    "greedy_action",
    "run_greedy_arm",
    "run_greedy_arm_with_cache",
]
