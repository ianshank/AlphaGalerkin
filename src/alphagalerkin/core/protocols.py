"""Protocol definitions for AlphaGalerkin's dependency-inversion boundaries.

This module uses :pep:`544` structural sub-typing so that concrete
implementations never need to import the interface -- they only need
to expose the right methods and attributes.

All forward references to concrete types (``MCTSNode``,
``DiscretizationState``, ``Action``) live behind
``TYPE_CHECKING`` guards and are quoted in signatures to avoid
circular imports at runtime.

Protocol classes
----------------
SelectionStrategy
    MCTS child-selection (e.g. PUCT, UCB1).
EvaluationStrategy
    State evaluation for leaf expansion.
ActionValidator
    Per-action legality check.
PhysicsModule
    Full PDE-problem specification used by the environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from src.alphagalerkin.core.types import PDEType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.alphagalerkin.core.config import (
        PhysicsConfig,
    )
    from src.alphagalerkin.env.actions import Action
    from src.alphagalerkin.env.state import DiscretizationState
    from src.alphagalerkin.mcts.node import MCTSNode
    from src.alphagalerkin.physics.base import (
        BoundaryCondition,
        ManufacturedSolution,
        SolveResult,
    )


# -------------------------------------------------------------------
# MCTS protocols
# -------------------------------------------------------------------


@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol for MCTS child-selection strategies.

    Implementations must expose a single ``select_child`` method
    that, given a fully-expanded node, returns the child to
    traverse next.

    Typical implementations: PUCT, UCB1, RAVE.
    """

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the most promising child of *node*.

        Parameters
        ----------
        node:
            An expanded (non-leaf) MCTS node whose children
            dict is non-empty.

        Returns
        -------
        MCTSNode
            The chosen child node.

        """
        ...


@runtime_checkable
class EvaluationStrategy(Protocol):
    """Protocol for MCTS leaf-evaluation strategies.

    The evaluation strategy assigns a scalar value to a
    discretization state.  This value is then backed up through
    the search tree.

    Typical implementations: neural-network value head, rollout,
    heuristic (e.g. residual-based).
    """

    def evaluate(
        self,
        state: DiscretizationState,
    ) -> float:
        """Evaluate a discretization state.

        Parameters
        ----------
        state:
            The discretization state at the leaf node.

        Returns
        -------
        float
            Scalar value estimate in ``[0, 1]`` (higher is better).

        """
        ...


@runtime_checkable
class BackupStrategyProtocol(Protocol):
    """Protocol for MCTS value-backup strategies.

    Determines how leaf values are propagated up the tree to
    ancestor nodes.
    """

    def backup(
        self,
        leaf_value: float,
        path: Sequence[MCTSNode],
    ) -> None:
        """Propagate *leaf_value* up through *path*.

        Parameters
        ----------
        leaf_value:
            The evaluation result at the leaf.
        path:
            Sequence of nodes from root to leaf (inclusive).

        """
        ...


# -------------------------------------------------------------------
# Action validation
# -------------------------------------------------------------------


@runtime_checkable
class ActionValidator(Protocol):
    """Protocol for checking whether an action is legal.

    Validators are called before an action is applied to the
    environment state.  Multiple validators may be composed
    (e.g. DOF budget check + mesh integrity check).
    """

    def validate(
        self,
        action: Action,
        state: DiscretizationState,
    ) -> bool:
        """Return ``True`` if *action* is legal in *state*.

        Parameters
        ----------
        action:
            The candidate action.
        state:
            The current discretization state.

        Returns
        -------
        bool
            ``True`` if the action may be applied.

        """
        ...


# -------------------------------------------------------------------
# Physics module
# -------------------------------------------------------------------


@runtime_checkable
class PhysicsModule(Protocol):
    """Protocol for a complete PDE problem specification.

    A ``PhysicsModule`` encapsulates everything the discretization
    environment needs to know about the PDE it is solving:

    * weak-form assembly,
    * boundary conditions,
    * (optional) manufactured solution for error measurement,
    * reward computation,
    * state-feature extraction for the GNN,
    * legality validators for each action type, and
    * a default configuration.

    Concrete implementations live in
    ``alphagalerkin.physics`` (e.g. ``PoissonModule``).
    """

    # ---- Identity ----

    @property
    def name(self) -> str:
        """Human-readable name (e.g. ``"poisson_2d"``)."""
        ...

    @property
    def pde_type(self) -> PDEType:
        """Mathematical classification of this PDE."""
        ...

    # ---- Core PDE interface ----

    def weak_form(
        self,
        state: DiscretizationState,
    ) -> SolveResult:
        """Assemble and solve the weak-form system.

        Parameters
        ----------
        state:
            Current discretization (mesh + basis assignments).

        Returns
        -------
        SolveResult
            Solution vector, residual norm, condition number, etc.

        """
        ...

    def boundary_conditions(
        self,
    ) -> Sequence[BoundaryCondition]:
        """Return the boundary conditions for this problem.

        Returns
        -------
        Sequence[BoundaryCondition]
            One ``BoundaryCondition`` per boundary segment.

        """
        ...

    def manufactured_solution(
        self,
    ) -> ManufacturedSolution | None:
        """Return a manufactured solution, if available.

        Returns
        -------
        ManufacturedSolution | None
            ``None`` when no analytic solution is known.

        """
        ...

    # ---- Environment hooks ----

    def reward_function(
        self,
        prev_state: DiscretizationState,
        action: Action,
        next_state: DiscretizationState,
        solve_result: SolveResult,
    ) -> dict[str, float]:
        """Compute per-component rewards for a transition.

        Parameters
        ----------
        prev_state:
            State before the action.
        action:
            The applied action.
        next_state:
            State after the action.
        solve_result:
            Result of solving on *next_state*.

        Returns
        -------
        dict[str, float]
            Mapping from reward-component name (e.g.
            ``"accuracy"``, ``"efficiency"``) to scalar reward.

        """
        ...

    def state_features(
        self,
        state: DiscretizationState,
    ) -> dict[str, Any]:
        """Extract GNN-ready features from a state.

        Parameters
        ----------
        state:
            Current discretization state.

        Returns
        -------
        dict[str, Any]
            Must contain at least ``"node_features"``
            (``numpy.ndarray`` of shape ``(n_elements, d)``).
            May also contain ``"edge_features"``,
            ``"global_features"``, etc.

        """
        ...

    def action_validators(
        self,
    ) -> Sequence[ActionValidator]:
        """Return action validators specific to this PDE.

        Returns
        -------
        Sequence[ActionValidator]
            Validators that restrict the action space based on
            PDE-specific constraints (e.g. minimum element size
            for resolving boundary layers).

        """
        ...

    def default_config(self) -> PhysicsConfig:
        """Return a sensible default ``PhysicsConfig``.

        Returns
        -------
        PhysicsConfig
            Configuration pre-filled with values appropriate
            for this PDE.

        """
        ...


# -------------------------------------------------------------------
# Solver protocol
# -------------------------------------------------------------------


@runtime_checkable
class Solver(Protocol):
    """Protocol for a linear / nonlinear system solver.

    Abstracts away the choice between direct and iterative
    solvers so that the environment and physics module stay
    solver-agnostic.
    """

    def solve(
        self,
        stiffness: np.ndarray,
        rhs: np.ndarray,
    ) -> SolveResult:
        """Solve the system ``stiffness @ u = rhs``.

        Parameters
        ----------
        stiffness:
            System matrix of shape ``(n_dof, n_dof)``.
        rhs:
            Right-hand-side vector of shape ``(n_dof,)``.

        Returns
        -------
        SolveResult
            Solution vector and solver diagnostics.

        """
        ...


# -------------------------------------------------------------------
# Feature extractor
# -------------------------------------------------------------------


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for extracting tensor features from a state.

    Decouples the GNN input pipeline from the state
    representation so that different physics modules can define
    their own feature engineering.
    """

    @property
    def node_feature_dim(self) -> int:
        """Dimension of per-element node features."""
        ...

    @property
    def edge_feature_dim(self) -> int:
        """Dimension of per-edge features (0 = none)."""
        ...

    @property
    def global_feature_dim(self) -> int:
        """Dimension of global features (0 = none)."""
        ...

    def extract(
        self,
        state: DiscretizationState,
    ) -> dict[str, Any]:
        """Convert a state into tensor features.

        Parameters
        ----------
        state:
            The discretization state to featurise.

        Returns
        -------
        dict[str, Any]
            At minimum ``{"node_features": ndarray}``.
            May also include ``"edge_index"``,
            ``"edge_features"``, ``"global_features"``.

        """
        ...
