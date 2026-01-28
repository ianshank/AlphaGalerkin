"""PDE Game abstraction layer for AlphaGalerkin.

This module adapts AlphaZero's game interface for PDE solving:
- PDEState: Represents current approximation/mesh state
- PDEResult: Final outcome with error metrics
- PDEGame: Abstract base class for PDE-based games

The key insight is that PDE solving can be framed as a sequential
decision-making problem where:
- Actions: Add basis functions, refine mesh elements, place collocation points
- State: Current approximation quality and computational budget
- Reward: Error reduction per computational cost
- Terminal: Error < tolerance or budget exhausted

This enables MCTS to plan ahead multiple refinement steps, something
classical error indicators cannot do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

if TYPE_CHECKING:
    from src.pde.config import PDEGameConfig
    from src.pde.operators import PDEOperator


class GamePhase(str, Enum):
    """Phases of a PDE game."""

    INITIAL = "initial"  # Starting state
    EXPLORING = "exploring"  # Building approximation
    REFINING = "refining"  # Fine-tuning
    CONVERGED = "converged"  # Error < tolerance
    BUDGET_EXHAUSTED = "budget_exhausted"  # Out of resources


@dataclass
class PDEState:
    """State representation for PDE games.

    Contains all information needed to:
    - Evaluate the current approximation
    - Determine legal actions
    - Compute rewards
    - Convert to neural network input

    Attributes:
        coords: Collocation/mesh coordinates (N, dim).
        solution: Current solution values (N,).
        residuals: PDE residuals at each point (N,).
        basis_coefficients: Coefficients for basis functions.
        error_estimate: Current error estimate.
        dof: Current degrees of freedom.
        step: Current game step.
        budget_remaining: Remaining computational budget.
        mesh_levels: Refinement level per element (for mesh refinement).
        polynomial_degrees: Polynomial degree per element (for p-refinement).
        history: Action history for this state.
    """

    # Core solution data
    coords: NDArray[np.float32]  # (N, dim)
    solution: NDArray[np.float32]  # (N,)
    residuals: NDArray[np.float32]  # (N,)

    # Basis/mesh representation
    basis_coefficients: NDArray[np.float32] | None = None
    mesh_levels: NDArray[np.int32] | None = None
    polynomial_degrees: NDArray[np.int32] | None = None

    # Game state
    error_estimate: float = 1.0
    dof: int = 0
    step: int = 0
    budget_remaining: float = 1e6
    phase: GamePhase = GamePhase.INITIAL

    # History (for symmetry/augmentation)
    history: list[int] = field(default_factory=list)

    # Cached data
    _tensor_cache: Tensor | None = field(default=None, repr=False, compare=False)

    @property
    def n_points(self) -> int:
        """Number of collocation/mesh points."""
        return len(self.coords)

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return self.coords.shape[1] if self.coords.ndim > 1 else 1

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        if self.basis_coefficients is not None:
            return len(self.basis_coefficients)
        return 0

    def clone(self) -> PDEState:
        """Create a deep copy of the state."""
        return PDEState(
            coords=self.coords.copy(),
            solution=self.solution.copy(),
            residuals=self.residuals.copy(),
            basis_coefficients=(
                self.basis_coefficients.copy()
                if self.basis_coefficients is not None
                else None
            ),
            mesh_levels=(
                self.mesh_levels.copy() if self.mesh_levels is not None else None
            ),
            polynomial_degrees=(
                self.polynomial_degrees.copy()
                if self.polynomial_degrees is not None
                else None
            ),
            error_estimate=self.error_estimate,
            dof=self.dof,
            step=self.step,
            budget_remaining=self.budget_remaining,
            phase=self.phase,
            history=list(self.history),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "coords": self.coords.tolist(),
            "solution": self.solution.tolist(),
            "residuals": self.residuals.tolist(),
            "basis_coefficients": (
                self.basis_coefficients.tolist()
                if self.basis_coefficients is not None
                else None
            ),
            "mesh_levels": (
                self.mesh_levels.tolist() if self.mesh_levels is not None else None
            ),
            "polynomial_degrees": (
                self.polynomial_degrees.tolist()
                if self.polynomial_degrees is not None
                else None
            ),
            "error_estimate": self.error_estimate,
            "dof": self.dof,
            "step": self.step,
            "budget_remaining": self.budget_remaining,
            "phase": self.phase.value,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PDEState:
        """Create from dictionary."""
        return cls(
            coords=np.array(data["coords"], dtype=np.float32),
            solution=np.array(data["solution"], dtype=np.float32),
            residuals=np.array(data["residuals"], dtype=np.float32),
            basis_coefficients=(
                np.array(data["basis_coefficients"], dtype=np.float32)
                if data.get("basis_coefficients") is not None
                else None
            ),
            mesh_levels=(
                np.array(data["mesh_levels"], dtype=np.int32)
                if data.get("mesh_levels") is not None
                else None
            ),
            polynomial_degrees=(
                np.array(data["polynomial_degrees"], dtype=np.int32)
                if data.get("polynomial_degrees") is not None
                else None
            ),
            error_estimate=data["error_estimate"],
            dof=data["dof"],
            step=data["step"],
            budget_remaining=data["budget_remaining"],
            phase=GamePhase(data["phase"]),
            history=data.get("history", []),
        )


@dataclass
class PDEResult:
    """Result of a completed PDE game.

    Contains final metrics and trajectory information for:
    - Evaluating solution quality
    - Computing rewards for training
    - Analyzing algorithm performance
    """

    # Final state metrics
    final_error: float
    final_dof: int
    n_steps: int
    converged: bool

    # Error components
    l2_error: float
    h1_error: float
    linf_error: float
    residual_norm: float

    # Efficiency metrics
    error_reduction_rate: float  # Error per step
    dof_efficiency: float  # Error reduction per DOF
    compute_efficiency: float  # Error reduction per FLOP

    # Trajectory statistics
    initial_error: float
    best_error: float
    average_error: float
    error_history: list[float]

    # Termination info
    termination_reason: str
    budget_used: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_error": self.final_error,
            "final_dof": self.final_dof,
            "n_steps": self.n_steps,
            "converged": self.converged,
            "l2_error": self.l2_error,
            "h1_error": self.h1_error,
            "linf_error": self.linf_error,
            "residual_norm": self.residual_norm,
            "error_reduction_rate": self.error_reduction_rate,
            "dof_efficiency": self.dof_efficiency,
            "compute_efficiency": self.compute_efficiency,
            "initial_error": self.initial_error,
            "best_error": self.best_error,
            "average_error": self.average_error,
            "error_history": self.error_history,
            "termination_reason": self.termination_reason,
            "budget_used": self.budget_used,
        }


class PDEGame(ABC):
    """Abstract base class for PDE-based games.

    This interface defines the contract for all PDE game implementations,
    enabling the AlphaGalerkin MCTS to work with any PDE without
    modification to the core search algorithm.

    Implementing a new PDE game requires:
    1. Implementing all abstract methods
    2. Registering with PDEGameRegistry
    3. Providing tensor encoding for neural network input

    The game loop follows:
    1. Initialize with PDE operator and config
    2. get_initial_state() -> starting approximation
    3. Loop until is_terminal():
       - get_valid_actions() -> legal moves
       - apply_action() -> new state
       - get_reward() -> immediate reward
    4. get_result() -> final metrics
    """

    # Class-level attributes (override in subclasses)
    name: str = "pde_game"
    description: str = "Abstract PDE game"

    def __init__(
        self,
        pde_operator: PDEOperator,
        config: PDEGameConfig,
    ) -> None:
        """Initialize PDE game.

        Args:
            pde_operator: The PDE to solve.
            config: Game configuration.
        """
        self.pde_operator = pde_operator
        self.config = config

        # Derived properties
        self._action_space_size: int | None = None
        self._state_channels: int | None = None

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Get total size of action space.

        Returns:
            Number of possible actions at any state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_channels(self) -> int:
        """Get number of input channels for neural network.

        Returns:
            Number of feature planes in tensor encoding.
        """
        raise NotImplementedError

    @abstractmethod
    def get_initial_state(self) -> PDEState:
        """Create initial game state.

        Returns:
            Initial PDEState (typically zero solution or coarse mesh).
        """
        raise NotImplementedError

    @abstractmethod
    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Get list of legal actions from current state.

        Args:
            state: Current PDE state.

        Returns:
            List of legal action indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        """Get action mask indicating legal actions.

        Args:
            state: Current PDE state.

        Returns:
            Boolean mask with True for legal actions.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Apply action and return new state.

        This should:
        1. Modify approximation (add basis, refine element, etc.)
        2. Recompute solution
        3. Update error estimate
        4. Decrease budget

        Args:
            state: Current PDE state.
            action: Action index to apply.

        Returns:
            New PDEState after applying action.

        Raises:
            ValueError: If action is illegal.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        """Compute immediate reward for state transition.

        Typically: reward = error_reduction - cost

        Args:
            state: New state after action.
            prev_state: Previous state before action.

        Returns:
            Immediate reward value.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: PDEState) -> bool:
        """Check if game has ended.

        Terminal conditions:
        - Error < tolerance (converged)
        - Budget exhausted
        - Max steps reached
        - Max DOF reached

        Args:
            state: Current PDE state.

        Returns:
            True if game is over.
        """
        raise NotImplementedError

    @abstractmethod
    def get_result(self, state: PDEState, error_history: list[float]) -> PDEResult:
        """Get game result from terminal state.

        Args:
            state: Terminal PDE state.
            error_history: Error values throughout the game.

        Returns:
            PDEResult with final metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_exact_error(self, state: PDEState) -> dict[str, float]:
        """Compute exact error metrics against ground truth.

        Args:
            state: Current PDE state.

        Returns:
            Dictionary with error metrics (l2, h1, linf, residual).
        """
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self, state: PDEState) -> Float[Tensor, "channels height width"]:
        """Convert state to neural network input tensor.

        The encoding should be resolution-independent using:
        - Fourier features for position encoding
        - Normalized coordinates in [0, 1]
        - Per-point features (solution, residual, error indicator)

        Args:
            state: PDE state to encode.

        Returns:
            Tensor of shape (channels, height, width) or (channels, n_points).
        """
        raise NotImplementedError

    def batch_to_tensor(
        self,
        states: list[PDEState],
        device: torch.device | str = "cpu",
    ) -> Float[Tensor, "batch channels height width"]:
        """Convert batch of states to tensor.

        Args:
            states: Sequence of PDE states.
            device: Target device for tensor.

        Returns:
            Batched tensor.
        """
        tensors = [self.to_tensor(state) for state in states]
        return torch.stack(tensors).to(device)

    def get_symmetries(
        self,
        state: PDEState,
        policy: NDArray[np.float32],
    ) -> list[tuple[PDEState, NDArray[np.float32]]]:
        """Get symmetric transformations of state and policy.

        Used for data augmentation during training.
        Default implementation returns identity only.

        Args:
            state: PDE state.
            policy: Policy distribution over actions.

        Returns:
            List of (transformed_state, transformed_policy) tuples.
        """
        return [(state, policy)]

    def get_phase(self, state: PDEState) -> GamePhase:
        """Get current game phase.

        Args:
            state: Current PDE state.

        Returns:
            Current GamePhase.
        """
        if self.is_terminal(state):
            if state.error_estimate < self.config.error_tolerance:
                return GamePhase.CONVERGED
            return GamePhase.BUDGET_EXHAUSTED

        # Heuristic based on progress
        error_progress = state.error_estimate / 1.0  # Initial error assumed 1.0
        budget_progress = state.budget_remaining / self.config.computational_budget

        if state.step < 5:
            return GamePhase.INITIAL
        elif error_progress > 0.1:
            return GamePhase.EXPLORING
        else:
            return GamePhase.REFINING

    def action_to_string(self, action: int) -> str:
        """Convert action index to human-readable string.

        Args:
            action: Action index.

        Returns:
            Human-readable action description.
        """
        return f"action_{action}"

    def validate_action(self, state: PDEState, action: int) -> bool:
        """Check if action is valid.

        Args:
            state: Current PDE state.
            action: Action to validate.

        Returns:
            True if action is valid.
        """
        if action < 0 or action >= self.action_space_size:
            return False
        return action in self.get_valid_actions(state)

    def clone(self) -> PDEGame:
        """Create a copy of the game interface.

        Returns:
            New instance of the game.
        """
        return type(self)(self.pde_operator, self.config)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
