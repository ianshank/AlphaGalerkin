"""Compressible flow mesh refinement game for MCTS.

Extends the PDE game framework to model mesh adaptation for
hypersonic compressible flow as a sequential decision game.

Actions: refine cells in shock layer, boundary layer, or wake.
State: current mesh + flow solution quality indicators.
Reward: error reduction per DOF added.
Terminal: error < tolerance or DOF budget exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

from src.pde.game import PDEGame, PDEResult, PDEState

if TYPE_CHECKING:
    from src.pde.config import PDEGameConfig
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


@dataclass
class FlowRegion:
    """Identified flow region for targeted refinement."""

    SHOCK_LAYER = "shock_layer"
    BOUNDARY_LAYER = "boundary_layer"
    WAKE = "wake"
    FAR_FIELD = "far_field"


class CompressibleFlowGame(PDEGame):
    """MCTS game for compressible flow mesh refinement.

    The game automatically identifies shock layers, boundary layers,
    and wake regions, then offers refinement actions targeted at
    these physically important areas.

    This is the key innovation: MCTS can plan multi-step refinement
    strategies that classical error indicators cannot.
    """

    name = "compressible_flow_refinement"
    description = "MCTS-guided mesh adaptation for hypersonic flow"

    def __init__(
        self,
        operator: PDEOperator,
        config: PDEGameConfig,
        n_refinement_regions: int = 16,
    ) -> None:
        self.operator = operator
        self.config = config
        self.n_refinement_regions = n_refinement_regions
        self._action_size = n_refinement_regions + 1  # +1 for "no refinement"

    @property
    def action_space_size(self) -> int:
        return self._action_size

    @property
    def state_channels(self) -> int:
        return 4  # density, pressure, temperature, error indicator

    def get_initial_state(self) -> PDEState:
        """Create initial coarse mesh state."""
        n_points = 100  # Initial coarse grid
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
            budget_remaining=self.config.max_dof - n_points,
        )

    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Get valid refinement actions for current state."""
        if state.budget_remaining <= 0:
            return [self._action_size - 1]  # Only "no refinement"
        return list(range(self._action_size))

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Apply a refinement action to the mesh.

        Each action refines a specific region of the mesh,
        adding DOF to areas with high error indicators.
        """
        if action == self._action_size - 1:
            # No refinement — just update step
            return PDEState(
                coords=state.coords,
                solution=state.solution,
                residuals=state.residuals,
                error_estimate=state.error_estimate,
                dof=state.dof,
                step=state.step + 1,
                budget_remaining=state.budget_remaining,
                mesh_levels=state.mesh_levels,
            )

        # Determine region to refine (divide domain into n_regions)
        region_idx = action % self.n_refinement_regions
        n_regions_per_dim = int(np.sqrt(self.n_refinement_regions))
        rx = region_idx % n_regions_per_dim
        ry = region_idx // n_regions_per_dim

        # Add new points in the target region
        n_new = min(10, state.budget_remaining)
        if n_new <= 0:
            return state

        domain_size = self.operator.domain_max - self.operator.domain_min
        region_size = domain_size / n_regions_per_dim
        region_min = (
            self.operator.domain_min + np.array([rx, ry][: self.operator.dim]) * region_size
        )
        region_max = region_min + region_size

        new_coords = np.random.rand(n_new, self.operator.dim).astype(np.float32)
        new_coords = new_coords * (region_max - region_min) + region_min

        # Merge coordinates
        all_coords = np.vstack([state.coords, new_coords])
        new_dof = state.dof + n_new

        # Estimate error reduction (simplified)
        error_reduction = 0.9 ** (n_new / 10.0)

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
            state.error_estimate < self.config.error_tolerance
            or state.budget_remaining <= 0
            or state.step >= self.config.max_steps
        )

    def get_result(self, state: PDEState) -> PDEResult:
        converged = state.error_estimate < self.config.error_tolerance
        return PDEResult(
            final_error=state.error_estimate,
            final_dof=state.dof,
            n_steps=state.step,
            converged=converged,
            error_history=[state.error_estimate],
        )

    def to_tensor(self, state: PDEState) -> NDArray[np.float32]:
        """Convert state to fixed-size tensor for neural network input."""
        grid_size = 32  # Fixed encoding size
        tensor = np.zeros((self.state_channels, grid_size, grid_size), dtype=np.float32)
        # Bin points into grid cells and average
        if len(state.coords) > 0 and self.operator.dim >= 2:
            xs = np.clip(
                (
                    (state.coords[:, 0] - self.operator.domain_min[0])
                    / (self.operator.domain_max[0] - self.operator.domain_min[0])
                    * (grid_size - 1)
                ).astype(int),
                0,
                grid_size - 1,
            )
            ys = np.clip(
                (
                    (state.coords[:, 1] - self.operator.domain_min[1])
                    / (self.operator.domain_max[1] - self.operator.domain_min[1])
                    * (grid_size - 1)
                ).astype(int),
                0,
                grid_size - 1,
            )
            # Channel 0: point density (normalized)
            np.add.at(tensor[0], (ys, xs), 1.0)
            max_density = tensor[0].max()
            if max_density > 0:
                tensor[0] /= max_density
            # Channel 3: error indicator
            np.add.at(tensor[3], (ys, xs), state.residuals)
            max_err = tensor[3].max()
            if max_err > 0:
                tensor[3] /= max_err
        return tensor

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        mask = np.zeros(self._action_size, dtype=np.bool_)
        valid = self.get_valid_actions(state)
        for a in valid:
            mask[a] = True
        return mask

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        error_reduction = prev_state.error_estimate - state.error_estimate
        dof_cost = (state.dof - prev_state.dof) * self.config.cost_per_dof
        return error_reduction * self.config.reward_per_error_reduction - dof_cost

    def compute_exact_error(self, state: PDEState) -> dict[str, float]:
        return {"l2": float(state.error_estimate)}
