"""Build the arena's refinement game and drive one episode of it.

The MCTS arm and the single-element greedy control must differ in exactly one
thing: how the next action is chosen. Everything else comes from this module,
so the two cannot drift apart in construction -- the ``SubstrateRefinementGame``
(substrate config, operator, ``max_steps``, ``top_k_actions``, budget, action
space), its ``RefinementGameAdapter``, the stopping rules, and the
``ArenaPoint`` recorded after every committed action. Each episode gets a fresh
``FingerprintSolveCache``; two arms sharing one is a fairness bug the harness
checks for.

This is the only arena module that touches the refinement layer. That is what
lets ``src.research.greedy_control`` sit inside the reference-baseline import
contract: it asks this module for an episode and imports neither the search
engine nor the refinement layer itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.pde.games.substrate_refinement import SubstrateEpisodeState, SubstrateRefinementGame
from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.refinement.adapter import RefinementGameAdapter
from src.research.amr_arena_types import ArenaPoint, ArenaTrajectory
from src.research.substrates.solve_cache import FingerprintSolveCache

if TYPE_CHECKING:
    from src.poc.scenarios.mcts_classical_amr_arena_config import (
        MCTSClassicalAMRArenaConfig,
    )
    from src.refinement.state import RefinementState
    from src.research.amr_arena_types import ArmName

#: Budget charged for one single-element refinement.
REFINE_COST: Final[float] = 1.0
#: Refinements of budget above ``max_steps``, so the computational budget never
#: ends an episode before ``max_steps`` does.
BUDGET_HEADROOM_STEPS: Final[int] = 1

#: Picks the action to commit from the pre-action state and its legal set.
DecisionRule = Callable[[SubstrateEpisodeState, list[int]], int]


@dataclass(frozen=True)
class ArenaEpisode:
    """A freshly built arena game, its adapter, and the cache it solves through."""

    game: SubstrateRefinementGame
    adapter: RefinementGameAdapter
    cache: FingerprintSolveCache


def assert_action_space_covers(game: SubstrateRefinementGame, state: RefinementState) -> None:
    """Refuse silent truncation of high-index triangles."""
    if not isinstance(state, SubstrateEpisodeState) or state.mesh is None:
        return
    n_units = int(game.substrate.n_units(state.mesh))
    if n_units > game.action_space_size:
        raise ValueError(
            f"max_action_space={game.action_space_size} truncates n_units={n_units}; "
            "raise max_action_space above the window's element count"
        )


def build_arena_episode(config: MCTSClassicalAMRArenaConfig, *, name: str) -> ArenaEpisode:
    """Build the arena game the way every game-driven arm must, with a fresh cache.

    Args:
        config: The arena config; every game knob is read from it.
        name: Config name of this arm's game (for logs; no numeric effect).

    Raises:
        ValueError: If ``max_action_space`` would truncate the initial mesh.

    """
    cache = FingerprintSolveCache(config.substrate.solve_cache_max_entries)
    game = SubstrateRefinementGame(
        SubstrateRefinementConfig(
            name=name,
            substrate=config.substrate,
            operator_name=config.operator_name,
            lshape_scale=config.lshape_scale,
            max_steps=config.max_steps,
            error_tolerance=config.error_tolerance,
            computational_budget=float(config.max_steps + BUDGET_HEADROOM_STEPS),
            refine_cost=REFINE_COST,
            max_action_space=config.max_action_space,
            value_scale=config.value_scale,
            top_k_actions=config.top_k_actions,
        ),
        solve_cache=cache,
    )
    adapter = RefinementGameAdapter(game)
    assert_action_space_covers(game, adapter.state)
    return ArenaEpisode(game=game, adapter=adapter, cache=cache)


def point_from_adapter(
    adapter: RefinementGameAdapter,
    cache: FingerprintSolveCache,
    t0: float,
    apply_actions: int,
) -> ArenaPoint:
    """Snapshot the committed state plus cache counters."""
    return ArenaPoint(
        level=int(adapter.state.step),
        n_dof=int(adapter.state.dof),
        l2_error=float(adapter.state.error_estimate),
        wall_time_seconds=time.perf_counter() - t0,
        n_cache_misses=int(cache.misses),
        n_cache_hits=int(cache.hits),
        n_apply_actions=apply_actions,
    )


def _episode_state(adapter: RefinementGameAdapter) -> SubstrateEpisodeState:
    """The adapter's committed state, which a substrate game always types this way."""
    state = adapter.state
    if not isinstance(state, SubstrateEpisodeState):
        raise TypeError(f"arena episodes run on SubstrateEpisodeState, got {type(state).__name__}")
    return state


def run_arena_episode(
    episode: ArenaEpisode,
    *,
    method: ArmName,
    config: MCTSClassicalAMRArenaConfig,
    choose: DecisionRule,
) -> ArenaTrajectory:
    """Commit ``choose``'s action each step until the episode stops.

    The first point is the initial solve. The episode stops when the game is
    terminal, when no action is legal, when the committed DOF reaches
    ``config.max_dof``, or after ``config.max_steps`` actions -- checked in that
    order before every decision. ``choose`` sees the state *before* the action.
    """
    adapter, cache = episode.adapter, episode.cache
    t0 = time.perf_counter()
    traj = ArenaTrajectory(method=method, cache_id=id(cache))
    traj.points.append(point_from_adapter(adapter, cache, t0, apply_actions=0))
    for _step in range(config.max_steps):
        if adapter.is_terminal():
            break
        legal = adapter.get_legal_actions()
        if not legal or adapter.state.dof >= config.max_dof:
            break
        action = choose(_episode_state(adapter), legal)
        adapter.apply_action(action)
        assert_action_space_covers(episode.game, adapter.state)
        traj.points.append(
            point_from_adapter(adapter, cache, t0, apply_actions=int(adapter.state.step))
        )
    return traj


__all__ = [
    "BUDGET_HEADROOM_STEPS",
    "REFINE_COST",
    "ArenaEpisode",
    "DecisionRule",
    "assert_action_space_covers",
    "build_arena_episode",
    "point_from_adapter",
    "run_arena_episode",
]
