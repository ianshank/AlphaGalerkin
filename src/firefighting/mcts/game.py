"""Fire spread mesh refinement game for MCTS.

Extends the PDE game framework to automatically concentrate
mesh resolution near the fire front and coarsen in far-field
regions, optimizing prediction accuracy within compute budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.pde.game import PDEGame, PDEResult, PDEState

if TYPE_CHECKING:
    from src.pde.config import PDEGameConfig
    from src.pde.operators import PDEOperator


class FireSpreadGame(PDEGame):
    """MCTS game for fire spread mesh adaptation.

    Actions: refine near fire front, coarsen far field, or hold.
    State: temperature field + fuel state on adaptive mesh.
    Reward: prediction accuracy improvement per DOF.
    """

    name = "fire_spread_refinement"
    description = "MCTS-guided mesh adaptation for wildfire prediction"

    def __init__(
        self,
        operator: PDEOperator,
        config: PDEGameConfig,
        n_regions: int = 16,
    ) -> None:
        self.operator = operator
        self.config = config
        self.n_regions = n_regions
        self._action_size = n_regions + 1  # +1 for hold

    @property
    def action_space_size(self) -> int:
        return self._action_size

    @property
    def state_channels(self) -> int:
        return 3  # temperature, fuel, error

    def get_initial_state(self) -> PDEState:
        n_points = 64
        dim = self.operator.dim
        coords = np.random.rand(n_points, dim).astype(np.float32)
        coords = (
            coords * (self.operator.domain_max - self.operator.domain_min)
            + self.operator.domain_min
        )
        return PDEState(
            coords=coords,
            solution=np.zeros(n_points, dtype=np.float32),
            residuals=np.ones(n_points, dtype=np.float32),
            error_estimate=1.0,
            dof=n_points,
            step=0,
            budget_remaining=self.config.max_budget - n_points,
        )

    def get_valid_actions(self, state: PDEState) -> list[int]:
        if state.budget_remaining <= 0:
            return [self._action_size - 1]
        return list(range(self._action_size))

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        if action == self._action_size - 1:
            return PDEState(
                coords=state.coords,
                solution=state.solution,
                residuals=state.residuals,
                error_estimate=state.error_estimate,
                dof=state.dof,
                step=state.step + 1,
                budget_remaining=state.budget_remaining,
            )

        n_new = min(8, state.budget_remaining)
        if n_new <= 0:
            return state

        # Add points in target region
        n_per_dim = int(np.sqrt(self.n_regions))
        rx = action % n_per_dim
        ry = action // n_per_dim
        domain_size = self.operator.domain_max - self.operator.domain_min
        region_size = domain_size / n_per_dim
        region_min = (
            self.operator.domain_min + np.array([rx, ry][: self.operator.dim]) * region_size
        )

        new_coords = np.random.rand(n_new, self.operator.dim).astype(np.float32)
        new_coords = new_coords * region_size + region_min

        all_coords = np.vstack([state.coords, new_coords])
        new_dof = state.dof + n_new
        error_reduction = 0.92 ** (n_new / 8.0)

        return PDEState(
            coords=all_coords,
            solution=np.zeros(new_dof, dtype=np.float32),
            residuals=np.ones(new_dof, dtype=np.float32),
            error_estimate=state.error_estimate * error_reduction,
            dof=new_dof,
            step=state.step + 1,
            budget_remaining=state.budget_remaining - n_new,
        )

    def is_terminal(self, state: PDEState) -> bool:
        return (
            state.error_estimate < self.config.convergence_tolerance
            or state.budget_remaining <= 0
            or state.step >= self.config.max_steps
        )

    def get_result(self, state: PDEState) -> PDEResult:
        return PDEResult(
            final_error=state.error_estimate,
            final_dof=state.dof,
            n_steps=state.step,
            converged=state.error_estimate < self.config.convergence_tolerance,
            error_history=[state.error_estimate],
        )

    def to_tensor(self, state: PDEState) -> NDArray[np.float32]:
        grid_size = 32
        tensor = np.zeros((self.state_channels, grid_size, grid_size), dtype=np.float32)
        if len(state.coords) > 0 and self.operator.dim >= 2:
            xs = np.clip(
                (
                    (state.coords[:, 0] - self.operator.domain_min[0])
                    / max(self.operator.domain_max[0] - self.operator.domain_min[0], 1e-10)
                    * (grid_size - 1)
                ).astype(int),
                0,
                grid_size - 1,
            )
            ys = np.clip(
                (
                    (state.coords[:, 1] - self.operator.domain_min[1])
                    / max(self.operator.domain_max[1] - self.operator.domain_min[1], 1e-10)
                    * (grid_size - 1)
                ).astype(int),
                0,
                grid_size - 1,
            )
            np.add.at(tensor[0], (ys, xs), 1.0)
            max_val = tensor[0].max()
            if max_val > 0:
                tensor[0] /= max_val
        return tensor
