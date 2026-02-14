"""Shared MCTS protocol for interoperability between implementations.

AlphaGalerkin has two MCTS implementations:

1. ``src.alphagalerkin.mcts`` -- discretization-specific MCTS using
   :class:`TreeManager`, operating on :class:`DiscretizationState`
   and :class:`Action`.

2. ``src.mcts`` -- general-purpose MCTS operating on integer actions
   and numpy state arrays (for board games like Go/Chess).

This module defines runtime-checkable protocols that both
implementations can satisfy, enabling code that is generic over
the MCTS backend.

All protocols use :func:`typing.runtime_checkable` so that
``isinstance()`` checks work at runtime.
"""
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type


@runtime_checkable
class MCTSSearchable(Protocol[S, A]):
    """Protocol that any MCTS implementation must satisfy.

    An MCTSSearchable runs a search from a root state and returns
    the best action along with the full policy (action -> visit
    probability mapping).

    Type Parameters
    ---------------
    S : State type (e.g. :class:`DiscretizationState` or
        :class:`numpy.ndarray`).
    A : Action type (e.g. :class:`Action` or ``int``).

    """

    def search(
        self,
        root_state: S,
        step: int = 0,
    ) -> tuple[A, dict[A, float]]:
        """Run MCTS search and return (best_action, policy).

        Parameters
        ----------
        root_state:
            The state from which to start the search.
        step:
            Episode step index (used for temperature annealing).

        Returns
        -------
        tuple[A, dict[A, float]]
            A tuple of ``(selected_action, policy_distribution)``
            where the policy maps every expanded action to its
            normalised visit-count share.

        """
        ...


@runtime_checkable
class MCTSEvaluable(Protocol[S, A]):
    """Protocol for neural network evaluation functions.

    An MCTSEvaluable is a callable that takes a game state and
    returns action priors and a scalar value estimate.

    Type Parameters
    ---------------
    S : State type.
    A : Action type.

    """

    def __call__(
        self,
        state: S,
    ) -> tuple[dict[A, float], float]:
        """Evaluate state, return (action_priors, value).

        Parameters
        ----------
        state:
            The game state to evaluate.

        Returns
        -------
        tuple[dict[A, float], float]
            A tuple of ``(action_priors, value_estimate)``
            where ``action_priors`` maps actions to prior
            probabilities and ``value_estimate`` is a scalar
            in ``[-1, 1]``.

        """
        ...


@runtime_checkable
class GameInterface(Protocol[S, A]):
    """Protocol for games playable by MCTS.

    This bridges the ``alphagalerkin.mcts`` and the top-level
    ``src.mcts`` implementations by defining a common game
    interface.  Any game that implements these five methods can
    be played by either MCTS backend.

    Type Parameters
    ---------------
    S : State type.
    A : Action type.

    """

    def get_initial_state(self) -> S:
        """Return the starting state of the game.

        Returns
        -------
        S
            A fresh initial state.

        """
        ...

    def get_valid_actions(self, state: S) -> list[A]:
        """Return all legal actions from *state*.

        Parameters
        ----------
        state:
            The current game state.

        Returns
        -------
        list[A]
            List of valid actions.

        """
        ...

    def apply_action(self, state: S, action: A) -> S:
        """Apply *action* to *state* and return the new state.

        The original *state* must not be mutated.

        Parameters
        ----------
        state:
            The current game state.
        action:
            The action to apply.

        Returns
        -------
        S
            The resulting game state.

        """
        ...

    def is_terminal(self, state: S) -> bool:
        """Check whether *state* is a terminal (game-over) state.

        Parameters
        ----------
        state:
            The game state to check.

        Returns
        -------
        bool
            ``True`` if the game is over.

        """
        ...

    def get_reward(self, state: S) -> float:
        """Compute the reward for a terminal *state*.

        Parameters
        ----------
        state:
            A terminal game state.

        Returns
        -------
        float
            Scalar reward value.

        """
        ...
