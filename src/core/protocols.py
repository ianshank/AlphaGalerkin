"""Structural Protocol definitions for AlphaGalerkin core abstractions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

    from src.mcts.evaluator import EvaluationResult
    from src.research.baselines import SolverResult


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for MCTS neural and heuristic evaluators."""

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Evaluate a single state returning policy prior and value estimate."""
        ...

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of states for vectorized tree search."""
        ...


@runtime_checkable
class GameProtocol(Protocol):
    """Protocol for discrete games and continuous domain refinement games."""

    def get_state(self) -> NDArray[np.float32]:
        """Return the current game state representation."""
        ...

    def get_legal_actions(self) -> list[int]:
        """Return legal action indices for the current state."""
        ...

    def apply_action(self, action: int) -> None:
        """Apply an action in-place to advance the state."""
        ...

    def is_terminal(self) -> bool:
        """Return True if the game/episode has reached a terminal state."""
        ...

    def get_winner(self) -> float:
        """Return the scalar outcome or terminal reward."""
        ...

    def clone(self) -> GameProtocol:
        """Return an independent deep copy of the game state."""
        ...


@runtime_checkable
class OperatorProtocol(Protocol):
    """Protocol for partial differential equation operators."""

    def residual(
        self,
        u: torch.Tensor,
        coords: torch.Tensor,
    ) -> Any:
        """Compute the PDE residual given a candidate field and coordinates."""
        ...

    def exact_solution(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the manufactured or analytical exact solution at coords."""
        ...


@runtime_checkable
class SolverProtocol(Protocol):
    """Protocol for PDE and baseline solvers."""

    def solve(
        self,
        operator: Any,
    ) -> SolverResult:
        """Solve the given operator problem returning a SolverResult."""
        ...


__all__ = [
    "EvaluatorProtocol",
    "GameProtocol",
    "OperatorProtocol",
    "SolverProtocol",
]
