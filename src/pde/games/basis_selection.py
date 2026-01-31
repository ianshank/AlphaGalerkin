"""Basis Selection Game for Galerkin methods.

This module implements a PDEGame where:
- State: Current Galerkin approximation space
- Actions: Add a new basis function from candidate set
- Reward: Error reduction per basis function added
- Terminal: Error < tolerance or max basis reached

The game enables MCTS to plan ahead multiple basis additions,
selecting the optimal sequence of basis functions to minimize
error with fewest degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import BasisSelectionConfig, PDEGameConfig
from src.pde.game import GamePhase, PDEGame, PDEResult, PDEState

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator


@dataclass
class BasisFunction:
    """Representation of a basis function.

    Attributes:
        type: Basis type ('fourier', 'polynomial', 'rbf').
        params: Parameters defining the basis function.
        index: Index in the candidate set.

    """

    type: str
    params: dict[str, float | int]
    index: int

    def evaluate(
        self,
        coords: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Evaluate basis function at given coordinates.

        Args:
            coords: Points to evaluate at (N, dim).

        Returns:
            Basis function values (N,).

        """
        if self.type == "fourier":
            k_x = self.params.get("k_x", 1)
            k_y = self.params.get("k_y", 0)
            phase = self.params.get("phase", 0.0)

            x = coords[:, 0]
            y = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(x)

            return np.sin(2 * np.pi * (k_x * x + k_y * y) + phase).astype(np.float32)

        elif self.type == "polynomial":
            degree_x = int(self.params.get("degree_x", 0))
            degree_y = int(self.params.get("degree_y", 0))

            x = coords[:, 0]
            y = coords[:, 1] if coords.shape[1] > 1 else np.ones_like(x)

            return (x ** degree_x * y ** degree_y).astype(np.float32)

        elif self.type == "rbf":
            center_x = self.params.get("center_x", 0.5)
            center_y = self.params.get("center_y", 0.5)
            sigma = self.params.get("sigma", 0.1)

            x = coords[:, 0]
            y = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(x)

            r_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            return np.exp(-r_sq / (2 * sigma ** 2)).astype(np.float32)

        else:
            raise ValueError(f"Unknown basis type: {self.type}")


class BasisSelectionGame(PDEGame):
    """Basis selection game for Galerkin methods.

    In this game, the agent builds a Galerkin approximation by
    selecting basis functions from a candidate set. Each action
    adds one basis function, and the game ends when either:
    - Error falls below tolerance
    - Maximum number of basis functions reached
    - Computational budget exhausted

    The reward is shaped to encourage efficient basis selection:
        reward = error_reduction - cost_per_dof

    MCTS can look ahead to find sequences of basis functions
    that achieve the best error-to-DOF trade-off.
    """

    name = "basis_selection"
    description = "Galerkin basis function selection game"

    def __init__(
        self,
        pde_operator: PDEOperator,
        config: PDEGameConfig,
    ) -> None:
        """Initialize basis selection game.

        Args:
            pde_operator: PDE operator to solve.
            config: Game configuration.

        """
        super().__init__(pde_operator, config)

        self.basis_config = config.basis_config or BasisSelectionConfig(
            name="default_basis"
        )

        # Generate candidate basis functions
        self._candidate_bases = self._generate_candidates()

        # Cache collocation points (use configurable count)
        n_collocation = self.basis_config.n_collocation_points
        self._collocation_points = pde_operator.generate_collocation_points(
            n_points=n_collocation,
            method="lhs",
        )

        # Cache boundary points (use configurable count)
        n_boundary_per_face = self.basis_config.n_boundary_points_per_face
        self._boundary_points = pde_operator.generate_boundary_points(
            n_points_per_face=n_boundary_per_face,
        )

        # Pre-compute exact solution if available
        self._exact_solution = pde_operator.exact_solution(self._collocation_points)

    def _generate_candidates(self) -> list[BasisFunction]:
        """Generate candidate basis functions.

        Returns:
            List of candidate BasisFunction objects.

        """
        candidates = []
        basis_type = self.basis_config.basis_type
        n_candidates = self.basis_config.n_candidate_bases

        if basis_type == "fourier":
            # Generate Fourier basis at various frequencies
            max_freq = self.basis_config.max_frequency
            idx = 0
            for k_x in range(-max_freq, max_freq + 1):
                for k_y in range(-max_freq, max_freq + 1):
                    if k_x == 0 and k_y == 0:
                        continue  # Skip DC component (or add separately)

                    for phase in [0.0, np.pi / 2]:  # sin and cos
                        candidates.append(BasisFunction(
                            type="fourier",
                            params={"k_x": k_x, "k_y": k_y, "phase": phase},
                            index=idx,
                        ))
                        idx += 1

                        if len(candidates) >= n_candidates:
                            break
                    if len(candidates) >= n_candidates:
                        break
                if len(candidates) >= n_candidates:
                    break

            # Add DC component if configured
            if self.basis_config.include_dc_component:
                candidates.insert(0, BasisFunction(
                    type="fourier",
                    params={"k_x": 0, "k_y": 0, "phase": 0.0},
                    index=len(candidates),
                ))

        elif basis_type == "polynomial":
            # Generate polynomial basis up to certain degree
            max_deg = self.basis_config.max_frequency  # Reuse as max degree
            idx = 0
            for deg in range(max_deg + 1):
                for dx in range(deg + 1):
                    dy = deg - dx
                    candidates.append(BasisFunction(
                        type="polynomial",
                        params={"degree_x": dx, "degree_y": dy},
                        index=idx,
                    ))
                    idx += 1

                    if len(candidates) >= n_candidates:
                        break
                if len(candidates) >= n_candidates:
                    break

        elif basis_type == "rbf":
            # Generate RBF basis with various centers
            rng = np.random.default_rng(self.basis_config.seed)
            scale_lo, scale_hi = self.basis_config.basis_scale_range

            for idx in range(n_candidates):
                candidates.append(BasisFunction(
                    type="rbf",
                    params={
                        "center_x": rng.uniform(0, 1),
                        "center_y": rng.uniform(0, 1),
                        "sigma": rng.uniform(scale_lo, scale_hi),
                    },
                    index=idx,
                ))

        return candidates[:n_candidates]

    @property
    def action_space_size(self) -> int:
        """Number of candidate basis functions."""
        return len(self._candidate_bases)

    @property
    def state_channels(self) -> int:
        """Neural network input channels.

        Channels:
        - 1: Current solution
        - 1: Residual
        - 1: Error indicator
        - N: Selected basis indicators
        """
        return 3 + self.basis_config.max_basis_functions

    def get_initial_state(self) -> PDEState:
        """Create initial state with empty basis set.

        Returns:
            Initial PDEState.

        """
        n_points = len(self._collocation_points)
        dim = self._collocation_points.shape[1]

        # Start with zero solution
        solution = np.zeros(n_points, dtype=np.float32)

        # Compute initial residual (= -source term for zero solution)
        source = self.pde_operator.source_term(self._collocation_points)
        if isinstance(source, Tensor):
            source = source.numpy()
        residuals = -source.astype(np.float32)

        # Compute initial error
        if self._exact_solution is not None:
            if isinstance(self._exact_solution, Tensor):
                exact = self._exact_solution.numpy()
            else:
                exact = self._exact_solution
            error = float(np.sqrt(np.mean((solution - exact) ** 2)))
        else:
            error = float(np.sqrt(np.mean(residuals ** 2)))

        return PDEState(
            coords=self._collocation_points.copy(),
            solution=solution,
            residuals=residuals,
            basis_coefficients=np.array([], dtype=np.float32),
            error_estimate=error,
            dof=0,
            step=0,
            budget_remaining=self.config.computational_budget,
            phase=GamePhase.INITIAL,
            history=[],
        )

    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Get list of valid actions (unselected basis functions).

        Args:
            state: Current game state.

        Returns:
            List of valid action indices.

        """
        # All bases not yet selected
        selected = set(state.history)
        valid = [i for i in range(self.action_space_size) if i not in selected]

        # Limit by max basis functions
        if state.n_basis >= self.basis_config.max_basis_functions:
            return []

        return valid

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        """Get boolean mask for valid actions.

        Args:
            state: Current game state.

        Returns:
            Boolean mask array.

        """
        mask = np.ones(self.action_space_size, dtype=bool)

        # Mask out already selected bases
        for idx in state.history:
            mask[idx] = False

        # Mask all if at max
        if state.n_basis >= self.basis_config.max_basis_functions:
            mask[:] = False

        return mask

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Apply action (add basis function) to get new state.

        Args:
            state: Current game state.
            action: Basis function index to add.

        Returns:
            New PDEState after adding basis.

        Raises:
            ValueError: If action is invalid.

        """
        if action in state.history:
            raise ValueError(f"Basis {action} already selected")
        if action < 0 or action >= self.action_space_size:
            raise ValueError(f"Invalid action: {action}")

        new_state = state.clone()
        new_state.history.append(action)
        new_state.step += 1

        # Build basis matrix with new function
        basis_funcs = [self._candidate_bases[i] for i in new_state.history]
        Phi = self._build_basis_matrix(basis_funcs, self._collocation_points)

        # Solve for coefficients: Phi @ c = target
        # Use least squares for overdetermined system
        if self._exact_solution is not None:
            if isinstance(self._exact_solution, Tensor):
                target = self._exact_solution.numpy()
            else:
                target = self._exact_solution
        else:
            # Use residual minimization
            source = self.pde_operator.source_term(self._collocation_points)
            if isinstance(source, Tensor):
                source = source.numpy()
            target = source.astype(np.float32)

        try:
            coeffs, residual_norm, _, _ = np.linalg.lstsq(Phi, target, rcond=None)
            new_state.basis_coefficients = coeffs.astype(np.float32)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            coeffs = np.linalg.pinv(Phi) @ target
            new_state.basis_coefficients = coeffs.astype(np.float32)

        # Compute new solution
        new_state.solution = (Phi @ new_state.basis_coefficients).astype(np.float32)

        # Compute residual and error
        residual_result = self.pde_operator.residual(
            torch.from_numpy(new_state.solution),
            torch.from_numpy(new_state.coords),
            compute_derivatives=False,
        )
        if isinstance(residual_result.values, Tensor):
            new_state.residuals = residual_result.values.numpy().astype(np.float32)
        else:
            new_state.residuals = residual_result.values.astype(np.float32)

        # Compute error estimate
        errors = self.compute_exact_error(new_state)
        new_state.error_estimate = errors["l2"]

        # Update DOF and budget
        new_state.dof = len(new_state.history)
        cost = 1.0  # Unit cost per basis function
        new_state.budget_remaining -= cost

        # Update phase
        if new_state.error_estimate < self.config.error_tolerance:
            new_state.phase = GamePhase.CONVERGED
        elif new_state.budget_remaining <= 0:
            new_state.phase = GamePhase.BUDGET_EXHAUSTED
        elif new_state.error_estimate > 0.1:
            new_state.phase = GamePhase.EXPLORING
        else:
            new_state.phase = GamePhase.REFINING

        return new_state

    def _build_basis_matrix(
        self,
        basis_funcs: list[BasisFunction],
        coords: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Build basis function matrix.

        Args:
            basis_funcs: List of basis functions.
            coords: Evaluation points.

        Returns:
            Matrix Phi where Phi[i,j] = phi_j(x_i).

        """
        n_points = len(coords)
        n_basis = len(basis_funcs)

        Phi = np.zeros((n_points, n_basis), dtype=np.float32)
        for j, basis in enumerate(basis_funcs):
            Phi[:, j] = basis.evaluate(coords)

        return Phi

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        """Compute reward for state transition.

        Args:
            state: New state after action.
            prev_state: Previous state.

        Returns:
            Reward value.

        """
        # Error reduction
        error_reduction = prev_state.error_estimate - state.error_estimate

        # Cost penalty
        dof_added = state.dof - prev_state.dof
        cost = self.config.cost_per_dof * dof_added

        # Base reward
        reward = self.config.reward_per_error_reduction * error_reduction - cost

        # Bonus for reaching tolerance
        if state.error_estimate < self.config.error_tolerance:
            reward += self.config.terminal_bonus

        return reward

    def is_terminal(self, state: PDEState) -> bool:
        """Check if game has ended.

        Args:
            state: Current state.

        Returns:
            True if terminal.

        """
        # Check error tolerance
        if state.error_estimate < self.config.error_tolerance:
            return True

        # Check max basis functions
        if state.n_basis >= self.basis_config.max_basis_functions:
            return True

        # Check budget
        if state.budget_remaining <= 0:
            return True

        # Check max steps
        if state.step >= self.config.max_steps:
            return True

        # Check no valid actions
        if len(self.get_valid_actions(state)) == 0:
            return True

        return False

    def get_result(self, state: PDEState, error_history: list[float]) -> PDEResult:
        """Get final game result.

        Args:
            state: Terminal state.
            error_history: Error values throughout game.

        Returns:
            PDEResult with metrics.

        """
        errors = self.compute_exact_error(state)

        converged = state.error_estimate < self.config.error_tolerance

        # Compute efficiency metrics
        if len(error_history) > 1:
            error_reduction_rate = (error_history[0] - error_history[-1]) / len(error_history)
            dof_efficiency = (error_history[0] - error_history[-1]) / max(1, state.dof)
        else:
            error_reduction_rate = 0.0
            dof_efficiency = 0.0

        budget_used = self.config.computational_budget - state.budget_remaining
        compute_efficiency = (
            (error_history[0] - error_history[-1]) / max(1, budget_used)
            if len(error_history) > 1 else 0.0
        )

        termination_reason = (
            "converged" if converged
            else "max_basis" if state.n_basis >= self.basis_config.max_basis_functions
            else "budget_exhausted" if state.budget_remaining <= 0
            else "max_steps"
        )

        return PDEResult(
            final_error=state.error_estimate,
            final_dof=state.dof,
            n_steps=state.step,
            converged=converged,
            l2_error=errors["l2"],
            h1_error=errors["h1"],
            linf_error=errors["linf"],
            residual_norm=errors["residual"],
            error_reduction_rate=error_reduction_rate,
            dof_efficiency=dof_efficiency,
            compute_efficiency=compute_efficiency,
            initial_error=error_history[0] if error_history else state.error_estimate,
            best_error=min(error_history) if error_history else state.error_estimate,
            average_error=float(np.mean(error_history)) if error_history else state.error_estimate,
            error_history=error_history,
            termination_reason=termination_reason,
            budget_used=budget_used,
        )

    def compute_exact_error(self, state: PDEState) -> dict[str, float]:
        """Compute error metrics against exact solution.

        Args:
            state: Current state.

        Returns:
            Dictionary with error metrics.

        """
        # L2 error
        if self._exact_solution is not None:
            if isinstance(self._exact_solution, Tensor):
                exact = self._exact_solution.numpy()
            else:
                exact = self._exact_solution
            l2_error = float(np.sqrt(np.mean((state.solution - exact) ** 2)))
            linf_error = float(np.max(np.abs(state.solution - exact)))
        else:
            l2_error = float(np.sqrt(np.mean(state.residuals ** 2)))
            linf_error = float(np.max(np.abs(state.residuals)))

        # H1 error (would need gradient computation)
        h1_error = l2_error  # Approximation

        # Residual norm
        residual_norm = float(np.sqrt(np.mean(state.residuals ** 2)))

        return {
            "l2": l2_error,
            "h1": h1_error,
            "linf": linf_error,
            "residual": residual_norm,
        }

    def to_tensor(self, state: PDEState) -> Float[Tensor, "channels height width"]:
        """Convert state to neural network input.

        Args:
            state: PDE state.

        Returns:
            Tensor encoding of state.

        """
        n_points = state.n_points

        # Assume square grid (simplification)
        grid_size = int(np.sqrt(n_points))
        if grid_size ** 2 != n_points:
            # Non-square: use 1D representation
            grid_size = n_points
            n_channels = self.state_channels

            tensor = torch.zeros(n_channels, 1, n_points)
            tensor[0, 0] = torch.from_numpy(state.solution)
            tensor[1, 0] = torch.from_numpy(state.residuals)
            tensor[2, 0] = torch.from_numpy(np.abs(state.residuals))  # Error indicator

            # Basis selection indicators
            for i, idx in enumerate(state.history):
                if i + 3 < n_channels:
                    tensor[i + 3, 0] = 1.0

            return tensor

        # Square grid: reshape to 2D
        n_channels = self.state_channels
        tensor = torch.zeros(n_channels, grid_size, grid_size)

        # Reshape solution and residuals to grid
        tensor[0] = torch.from_numpy(state.solution.reshape(grid_size, grid_size))
        tensor[1] = torch.from_numpy(state.residuals.reshape(grid_size, grid_size))
        tensor[2] = torch.from_numpy(np.abs(state.residuals).reshape(grid_size, grid_size))

        # Basis selection indicators (one channel per selected basis)
        for i, idx in enumerate(state.history):
            if i + 3 < n_channels:
                # Mark this basis as selected (uniform indicator)
                tensor[i + 3] = 1.0

        return tensor

    def action_to_string(self, action: int) -> str:
        """Convert action to human-readable string.

        Args:
            action: Action index.

        Returns:
            Description of basis function.

        """
        if action < 0 or action >= len(self._candidate_bases):
            return f"invalid_action_{action}"

        basis = self._candidate_bases[action]
        if basis.type == "fourier":
            return f"fourier(k={basis.params['k_x']},{basis.params['k_y']})"
        elif basis.type == "polynomial":
            return f"poly(x^{basis.params['degree_x']}*y^{basis.params['degree_y']})"
        elif basis.type == "rbf":
            return f"rbf(c={basis.params['center_x']:.2f},{basis.params['center_y']:.2f})"
        return f"{basis.type}_{action}"
