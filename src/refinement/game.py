"""Domain-free abstract base for sequential refinement games.

``RefinementGame`` is the engine-level abstraction that ``src.pde`` implements. It
knows nothing about PDEs — a refinement game is any sequential decision problem where
each action refines a discretisation (add a basis function, split a mesh element) and
the objective is to reduce an error estimate under a budget.

It mirrors the shape of ``src.pde.game.PDEGame`` but over ``RefinementState`` and
with no PDE-specific fields, so the same MCTS engine (via
``RefinementGameAdapter``, single-agent) drives every domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from src.refinement.state import RefinementState


class RefinementGame(ABC):
    """Abstract sequential-refinement game over :class:`RefinementState`."""

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of distinct actions (the fixed action-index range)."""

    @abstractmethod
    def get_initial_state(self) -> RefinementState:
        """Return the state before any refinement action."""

    @abstractmethod
    def get_valid_actions(self, state: RefinementState) -> list[int]:
        """Return the legal action indices in ``state`` (sorted)."""

    @abstractmethod
    def apply_action(self, state: RefinementState, action: int) -> RefinementState:
        """Apply ``action`` and return the resulting (new) state.

        Must be a pure function of ``(state, action)`` and must not mutate
        ``state``. Determinism (same state + action → identical result) is what
        lets MCTS identify a node by its action sequence.
        """

    @abstractmethod
    def is_terminal(self, state: RefinementState) -> bool:
        """Return True when the episode has ended (converged or budget spent)."""

    @abstractmethod
    def get_reward(self, state: RefinementState, prev_state: RefinementState) -> float:
        """Immediate reward for the transition ``prev_state -> state``.

        Typically ``error_reduction - cost``. Read by MCTS only when
        intermediate rewards are enabled; the terminal bonus (if any) is applied
        by the adapter, outside this per-edge reward, so the shaped return
        telescopes.
        """

    @abstractmethod
    def get_winner(self, state: RefinementState) -> int:
        """Map a terminal ``state`` to an outcome in ``{-1, 0, 1}``."""

    @abstractmethod
    def to_tensor(self, state: RefinementState) -> NDArray[np.float32]:
        """Encode ``state`` as a tensor for the evaluator."""

    def clone(self) -> RefinementGame:
        """Return a copy safe to mutate independently of ``self``.

        Default returns ``self`` because many refinement games are stateless
        (all per-episode state lives on ``RefinementState``). Games that hold
        mutable per-episode state on the instance **must** override this — see
        the F3 clone-isolation contract.
        """
        return self
