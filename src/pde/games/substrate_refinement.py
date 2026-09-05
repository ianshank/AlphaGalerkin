"""``RefinementGame`` over a ``RefinementSubstrate`` (Slice E registrant).

Pure ``apply_action``: the mesh lives on the episode state (shared by reference
when immutable), never as mutable instance fields. Solves go through
:class:`~src.research.substrates.solve_cache.FingerprintSolveCache` so
``fingerprint`` has a production reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.refinement.game import RefinementGame
from src.refinement.registry import register_refinement_game
from src.refinement.state import RefinementState
from src.research.substrates.factory import build_substrate_from_config
from src.research.substrates.solve_cache import FingerprintSolveCache

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.refinement.substrate import RefinementSubstrate, SubstrateSolveResult

logger = structlog.get_logger(__name__)

GAME_REGISTRY_NAME = "substrate_refinement"


@dataclass
class SubstrateEpisodeState(RefinementState):
    """``RefinementState`` plus an opaque mesh handle for the substrate.

    ``mesh`` is shared by reference across ``clone()`` when the substrate
    enforces immutability; MCTS sibling branches must not mutate it in place.
    """

    mesh: Any = field(default=None, repr=False, compare=False)

    def clone(self) -> SubstrateEpisodeState:
        """Deep-copy arrays/history; share the immutable mesh handle."""
        return SubstrateEpisodeState(
            values=self.values.copy(),
            indicators=self.indicators.copy(),
            error_estimate=self.error_estimate,
            dof=self.dof,
            step=self.step,
            budget_remaining=self.budget_remaining,
            history=list(self.history),
            mesh=self.mesh,
        )


def _state_from_solve(
    *,
    mesh: Any,
    result: SubstrateSolveResult,
    step: int,
    budget_remaining: float,
    history: list[int],
) -> SubstrateEpisodeState:
    """Map a ``SubstrateSolveResult`` onto a ``SubstrateEpisodeState``."""
    values = np.asarray(result.values, dtype=np.float32).reshape(-1)
    indicators = np.asarray(result.indicators, dtype=np.float32).reshape(-1)
    return SubstrateEpisodeState(
        values=values,
        indicators=indicators,
        error_estimate=float(result.l2_error),
        dof=int(result.n_dof),
        step=step,
        budget_remaining=budget_remaining,
        history=list(history),
        mesh=mesh,
    )


@register_refinement_game(GAME_REGISTRY_NAME)
class SubstrateRefinementGame(RefinementGame):
    """Single-element refine game driven by a registry-resolved substrate."""

    def __init__(
        self,
        config: SubstrateRefinementConfig | None = None,
        *,
        substrate: RefinementSubstrate[Any] | None = None,
        solve_cache: FingerprintSolveCache | None = None,
    ) -> None:
        """Construct a pure substrate-backed refinement game.

        Args:
            config: Typed knobs; defaults to a CPU-safe tensor-grid Poisson setup.
            substrate: Optional pre-built substrate (tests). When omitted, built
                via :func:`build_substrate_from_config` (registry lookup).
            solve_cache: Optional shared cache; when omitted, sized from
                ``config.substrate.solve_cache_max_entries``.

        """
        self._config = config or SubstrateRefinementConfig(name="substrate_refinement")
        self._substrate: RefinementSubstrate[Any] = substrate or build_substrate_from_config(
            self._config.substrate,
            operator_name=self._config.operator_name,
            scale=self._config.lshape_scale,
        )
        self._cache = solve_cache or FingerprintSolveCache(
            self._config.substrate.solve_cache_max_entries
        )
        # Stateless w.r.t. episode: all per-episode data lives on SubstrateEpisodeState.
        self._log = logger.bind(
            game=GAME_REGISTRY_NAME,
            substrate_kind=self._config.substrate.kind,
            max_steps=self._config.max_steps,
        )
        self._log.info("substrate_refinement_game_initialised")

    @property
    def config(self) -> SubstrateRefinementConfig:
        """Typed configuration for this game."""
        return self._config

    @property
    def substrate(self) -> RefinementSubstrate[Any]:
        """The underlying substrate (for tests / arena wiring)."""
        return self._substrate

    @property
    def solve_cache(self) -> FingerprintSolveCache:
        """Fingerprint-keyed solve cache (production ``fingerprint`` reader)."""
        return self._cache

    @property
    def action_space_size(self) -> int:
        """Fixed action-index range (valid actions are a dynamic subset)."""
        return int(self._config.max_action_space)

    def get_initial_state(self) -> SubstrateEpisodeState:
        """Solve on the coarse mesh and return the starting episode state."""
        mesh = self._substrate.initial_mesh()
        result = self._cache.get_or_solve(self._substrate, mesh)
        state = _state_from_solve(
            mesh=mesh,
            result=result,
            step=0,
            budget_remaining=float(self._config.computational_budget),
            history=[],
        )
        self._log.debug(
            "substrate_game_initial_state",
            error=state.error_estimate,
            dof=state.dof,
            n_units=int(self._substrate.n_units(mesh)),
        )
        return state

    def get_valid_actions(self, state: RefinementState) -> list[int]:
        """Refinable unit indices in the current mesh, capped by action space."""
        if self.is_terminal(state):
            return []
        if not isinstance(state, SubstrateEpisodeState) or state.mesh is None:
            raise TypeError(
                "SubstrateRefinementGame requiress SubstrateEpisodeState with a mesh"
            )
        mask = self._substrate.refinable_mask(state.mesh)
        n_units = int(self._substrate.n_units(state.mesh))
        limit = min(n_units, self.action_space_size, len(mask))
        return sorted(int(i) for i in range(limit) if bool(mask[i]))

    def apply_action(self, state: RefinementState, action: int) -> SubstrateEpisodeState:
        """Refine a single unit; pure in ``(state, action)`` — no instance mutation."""
        if not isinstance(state, SubstrateEpisodeState) or state.mesh is None:
            raise TypeError(
                "SubstrateRefinementGame requires SubstrateEpisodeState with a mesh"
            )
        valid = self.get_valid_actions(state)
        if action not in valid:
            raise ValueError(
                f"action {action} is not valid in this state; valid={valid[:16]}…"
                if len(valid) > 16
                else f"action {action} is not valid in this state; valid={valid}"
            )
        n_units = int(self._substrate.n_units(state.mesh))
        marked = np.zeros(n_units, dtype=np.bool_)
        marked[action] = True
        new_mesh = self._substrate.refine(state.mesh, marked)
        result = self._cache.get_or_solve(self._substrate, new_mesh)
        new_state = _state_from_solve(
            mesh=new_mesh,
            result=result,
            step=state.step + 1,
            budget_remaining=float(state.budget_remaining) - float(self._config.refine_cost),
            history=[*state.history, int(action)],
        )
        self._log.debug(
            "substrate_game_action_applied",
            action=action,
            error_before=state.error_estimate,
            error_after=new_state.error_estimate,
            step=new_state.step,
            cache_hits=self._cache.hits,
            cache_misses=self._cache.misses,
        )
        return new_state

    def is_terminal(self, state: RefinementState) -> bool:
        """True when steps, error tolerance, or budget are exhausted."""
        return (
            state.step >= self._config.max_steps
            or state.error_estimate <= self._config.error_tolerance
            or state.budget_remaining <= 0.0
        )

    def get_reward(self, state: RefinementState, prev_state: RefinementState) -> float:
        """Error reduction minus optional cost weight."""
        reduction = float(prev_state.error_estimate) - float(state.error_estimate)
        return reduction - float(self._config.reward_cost_weight) * float(
            self._config.refine_cost
        )

    def get_winner(self, state: RefinementState) -> int:
        """Map terminal error to ``{-1, 1}`` against ``winner_error_threshold``."""
        return 1 if state.error_estimate < self._config.winner_error_threshold else -1

    def to_tensor(self, state: RefinementState) -> NDArray[np.float32]:
        """Pad/truncate indicators into a fixed-width float32 vector for evaluators."""
        width = self.action_space_size
        out = np.zeros(width, dtype=np.float32)
        indicators = np.asarray(state.indicators, dtype=np.float32).reshape(-1)
        n = min(width, indicators.shape[0])
        out[:n] = indicators[:n]
        return out


__all__ = [
    "GAME_REGISTRY_NAME",
    "SubstrateEpisodeState",
    "SubstrateRefinementGame",
]
