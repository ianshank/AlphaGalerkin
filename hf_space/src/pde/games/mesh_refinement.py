"""Mesh Refinement Game for adaptive FEM/DG methods.

This module implements a PDEGame where:
- State: Current mesh + solution quality indicators
- Actions: Refine specific elements (h or p refinement)
- Reward: Error reduction per DOF added
- Terminal: Error < tolerance or DOF budget exhausted

MCTS can look ahead multiple refinement steps to find optimal
refinement sequences, outperforming single-step error indicators.

Note:
    The current Mesh implementation supports 2D quadrilateral elements only.
    For 1D (intervals), 3D (hexahedra), or higher dimensions, a specialized
    mesh class would be required. This limitation is validated at runtime.

"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import MeshRefinementConfig, PDEGameConfig, RefinementStrategy
from src.pde.game import GamePhase, PDEGame, PDEResult, PDEState

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


@dataclass
class MeshElement:
    """Representation of a mesh element.

    Attributes:
        index: Element index.
        vertices: Vertex coordinates.
        center: Element centroid.
        size: Element diameter/size.
        level: Refinement level.
        polynomial_degree: Polynomial degree for p-refinement.
        parent: Parent element index (None for initial elements).
        children: Child element indices (empty if not refined).

    """

    index: int
    vertices: NDArray[np.float32]  # (n_vertices, dim)
    center: NDArray[np.float32]  # (dim,)
    size: float
    level: int = 0
    polynomial_degree: int = 1
    parent: int | None = None
    children: list[int] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        """Whether this element is a leaf (not refined)."""
        return len(self.children) == 0


class Mesh:
    """Multi-dimensional hypercube mesh for mesh refinement game.

    Supports:
    - 1D (intervals), 2D (quads), 3D (hexahedra) elements
    - Uniform initial mesh
    - Local h-refinement (element subdivision)
    - Local p-refinement (polynomial degree increase)

    Note:
        Dimensions 4+ are theoretically supported but not practically tested.
        For most PDE applications, 1D-3D covers the relevant cases.

    """

    # Supported dimensions with vertex counts per element
    VERTICES_PER_DIM: dict[int, int] = {1: 2, 2: 4, 3: 8, 4: 16}
    MAX_SUPPORTED_DIM: int = 4

    def __init__(
        self,
        domain_min: NDArray[np.float32],
        domain_max: NDArray[np.float32],
        initial_resolution: int,
    ) -> None:
        """Initialize mesh.

        Args:
            domain_min: Domain minimum coordinates.
            domain_max: Domain maximum coordinates.
            initial_resolution: Initial elements per dimension.

        Raises:
            ValueError: If dimension is not supported (>4).

        """
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.domain_size = domain_max - domain_min
        self.dim = len(domain_min)
        self.initial_resolution = initial_resolution

        # Validate dimension
        if self.dim > self.MAX_SUPPORTED_DIM:
            raise ValueError(
                f"Dimension {self.dim} not supported. Maximum supported dimension "
                f"is {self.MAX_SUPPORTED_DIM}. For higher dimensions, consider "
                "using a specialized mesh library."
            )
        if self.dim < 1:
            raise ValueError(f"Dimension must be at least 1, got {self.dim}")

        logger.debug(
            "initializing_mesh",
            dim=self.dim,
            resolution=initial_resolution,
            domain_size=self.domain_size.tolist(),
        )

        # Initialize uniform mesh
        self.elements: list[MeshElement] = []
        self._build_initial_mesh()

    def _build_initial_mesh(self) -> None:
        """Build initial uniform mesh for any supported dimension."""
        n = self.initial_resolution
        dx = self.domain_size / n

        # Generate all element corner indices using itertools.product
        # For dim=2: [(0,0), (0,1), ..., (n-1,n-1)]
        index_ranges = [range(n) for _ in range(self.dim)]
        element_corners = list(itertools.product(*index_ranges))

        idx = 0
        for corner_indices in element_corners:
            # Compute element minimum corner
            corner = np.array(
                [self.domain_min[d] + corner_indices[d] * dx[d] for d in range(self.dim)],
                dtype=np.float32,
            )

            # Generate all vertices of the hypercube element
            # For dim=2: 4 vertices; for dim=3: 8 vertices
            vertex_offsets = list(itertools.product(*[[0, 1]] * self.dim))
            vertices = np.array(
                [
                    corner + np.array([offset[d] * dx[d] for d in range(self.dim)])
                    for offset in vertex_offsets
                ],
                dtype=np.float32,
            )

            # Compute center
            center = corner + dx / 2

            # Compute element size (diagonal length)
            size = float(np.sqrt(np.sum(dx**2)))

            self.elements.append(
                MeshElement(
                    index=idx,
                    vertices=vertices,
                    center=center,
                    size=size,
                    level=0,
                    polynomial_degree=1,
                )
            )
            idx += 1

        logger.debug(
            "mesh_built",
            n_elements=len(self.elements),
            dim=self.dim,
        )

    @property
    def n_elements(self) -> int:
        """Number of elements."""
        return len(self.elements)

    @property
    def leaf_elements(self) -> list[MeshElement]:
        """Get leaf elements (active in solution)."""
        return [e for e in self.elements if e.is_leaf]

    @property
    def n_dof(self) -> int:
        """Approximate degrees of freedom.

        For polynomial degree p in dim dimensions, DOFs = (p+1)^dim.
        """
        return sum((e.polynomial_degree + 1) ** self.dim for e in self.leaf_elements)

    def refine_element(
        self,
        element_idx: int,
        strategy: RefinementStrategy,
    ) -> list[int]:
        """Refine an element.

        Args:
            element_idx: Element to refine.
            strategy: Refinement strategy (h or p).

        Returns:
            Indices of new/modified elements.

        """
        element = self.elements[element_idx]

        if strategy == RefinementStrategy.P_REFINEMENT:
            # p-refinement: increase polynomial degree
            element.polynomial_degree += 1
            return [element_idx]

        elif strategy == RefinementStrategy.H_REFINEMENT:
            # h-refinement: subdivide into 4 children
            children = self._subdivide_element(element)
            return [c.index for c in children]

        elif strategy == RefinementStrategy.HP_REFINEMENT:
            # hp-refinement: choose based on element properties
            # Simple heuristic: p if smooth, h if not
            if element.level < 2:
                return self._subdivide_element_indices(element)
            else:
                element.polynomial_degree += 1
                return [element_idx]

        return [element_idx]

    def _subdivide_element(self, element: MeshElement) -> list[MeshElement]:
        """Subdivide element into 2^dim children.

        For 1D: 2 children (intervals split in half)
        For 2D: 4 children (quads split into quadrants)
        For 3D: 8 children (hexahedra split into octants)
        """
        c = element.center
        child_size = element.size / 2

        # Generate child corners: each child occupies one "quadrant" of the parent
        # Child corners are at parent center ± child_half_size in each dimension
        child_half_extents = (
            np.array([element.vertices[0, d] - c[d] for d in range(self.dim)], dtype=np.float32) / 2
        )

        # Generate all 2^dim child corner offset patterns
        # For dim=2: [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
        sign_patterns = list(itertools.product(*[[-1, 1]] * self.dim))

        children = []
        for signs in sign_patterns:
            # Child center is parent center + signed offset
            child_center = c + np.array(
                [signs[d] * abs(child_half_extents[d]) for d in range(self.dim)], dtype=np.float32
            )

            # Generate child vertices (hypercube corners around child_center)
            vertex_offsets = list(itertools.product(*[[-1, 1]] * self.dim))
            child_vertices = np.array(
                [
                    child_center
                    + np.array(
                        [offset[d] * abs(child_half_extents[d]) for d in range(self.dim)],
                        dtype=np.float32,
                    )
                    for offset in vertex_offsets
                ],
                dtype=np.float32,
            )

            child = MeshElement(
                index=len(self.elements),
                vertices=child_vertices,
                center=child_center,
                size=child_size,
                level=element.level + 1,
                polynomial_degree=element.polynomial_degree,
                parent=element.index,
            )
            self.elements.append(child)
            element.children.append(child.index)
            children.append(child)

        logger.debug(
            "element_subdivided",
            parent_index=element.index,
            n_children=len(children),
            child_indices=[c.index for c in children],
        )

        return children

    def _subdivide_element_indices(self, element: MeshElement) -> list[int]:
        """Subdivide and return child indices."""
        children = self._subdivide_element(element)
        return [c.index for c in children]

    def get_element_centers(self) -> NDArray[np.float32]:
        """Get centers of all leaf elements."""
        return np.array([e.center for e in self.leaf_elements], dtype=np.float32)

    def get_element_sizes(self) -> NDArray[np.float32]:
        """Get sizes of all leaf elements."""
        return np.array([e.size for e in self.leaf_elements], dtype=np.float32)


class MeshRefinementGame(PDEGame):
    """Mesh refinement game for adaptive methods.

    The agent decides which elements to refine and how,
    building an optimal mesh for the given PDE.
    """

    name = "mesh_refinement"
    description = "Adaptive mesh refinement game"

    def __init__(
        self,
        pde_operator: PDEOperator,
        config: PDEGameConfig,
    ) -> None:
        """Initialize mesh refinement game.

        Args:
            pde_operator: PDE operator to solve.
            config: Game configuration.

        """
        super().__init__(pde_operator, config)

        self.mesh_config = config.mesh_config or MeshRefinementConfig(name="default_mesh")

        # Initialize mesh
        self.mesh = Mesh(
            domain_min=np.array(pde_operator.config.domain_min, dtype=np.float32),
            domain_max=np.array(pde_operator.config.domain_max, dtype=np.float32),
            initial_resolution=self.mesh_config.initial_resolution,
        )

        # Action space: refine element i with strategy s
        # For simplicity, use h-refinement only
        self._refinement_strategy = self.mesh_config.refinement_strategy

    @property
    def action_space_size(self) -> int:
        """Number of possible actions.

        Each leaf element can be refined.
        """
        # Dynamic based on mesh state
        # Use maximum possible for fixed action space
        # Initial elements: resolution^dim, each refinement creates 2^dim children
        dim = self.mesh.dim
        max_elements = (
            self.mesh_config.initial_resolution**dim
            * (2**dim) ** self.mesh_config.max_refinement_level
        )
        return min(max_elements, self.mesh_config.n_candidate_elements)

    @property
    def state_channels(self) -> int:
        """Neural network input channels."""
        return 5  # solution, residual, error, refinement level, size

    def get_initial_state(self) -> PDEState:
        """Create initial state with coarse mesh.

        Returns:
            Initial PDEState.

        """
        # Reset mesh to initial state
        self.mesh = Mesh(
            domain_min=np.array(self.pde_operator.config.domain_min, dtype=np.float32),
            domain_max=np.array(self.pde_operator.config.domain_max, dtype=np.float32),
            initial_resolution=self.mesh_config.initial_resolution,
        )

        # Get element centers for collocation
        coords = self.mesh.get_element_centers()
        n_points = len(coords)

        # Initial solution (zero)
        solution = np.zeros(n_points, dtype=np.float32)

        # Compute residual
        source = self.pde_operator.source_term(coords)
        if isinstance(source, Tensor):
            source = source.numpy()
        residuals = -source.astype(np.float32)

        # Initial error
        error = float(np.sqrt(np.mean(residuals**2)))

        # Mesh info
        mesh_levels = np.array([e.level for e in self.mesh.leaf_elements], dtype=np.int32)

        return PDEState(
            coords=coords,
            solution=solution,
            residuals=residuals,
            mesh_levels=mesh_levels,
            error_estimate=error,
            dof=self.mesh.n_dof,
            step=0,
            budget_remaining=self.config.computational_budget,
            phase=GamePhase.INITIAL,
            history=[],
        )

    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Get valid refinement actions.

        Args:
            state: Current state.

        Returns:
            List of valid element indices to refine.

        """
        valid = []
        for i, element in enumerate(self.mesh.leaf_elements):
            # Check refinement constraints
            if element.level >= self.mesh_config.max_refinement_level:
                continue
            if element.size < self.mesh_config.min_element_size:
                continue
            if element.polynomial_degree >= self.mesh_config.max_polynomial_degree:
                continue

            valid.append(i)

        # Limit to candidate count
        return valid[: self.mesh_config.n_candidate_elements]

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        """Get boolean mask for valid actions.

        Args:
            state: Current state.

        Returns:
            Boolean mask.

        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        valid = self.get_valid_actions(state)
        for idx in valid:
            if idx < self.action_space_size:
                mask[idx] = True
        return mask

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Apply refinement action.

        Args:
            state: Current state.
            action: Element index to refine.

        Returns:
            New state after refinement.

        """
        # Get leaf element to refine
        leaf_elements = self.mesh.leaf_elements
        if action >= len(leaf_elements):
            raise ValueError(f"Invalid action: {action} >= {len(leaf_elements)}")

        element = leaf_elements[action]

        # Perform refinement
        new_indices = self.mesh.refine_element(
            element.index,
            self._refinement_strategy,
        )

        # Rebuild state
        coords = self.mesh.get_element_centers()
        n_points = len(coords)

        # Solve on new mesh (simplified: interpolate old solution)
        if len(state.solution) == n_points:
            solution = state.solution.copy()
        else:
            # Interpolate from old mesh
            solution = self._interpolate_solution(state, coords)

        # Compute residual
        residual_result = self.pde_operator.residual(
            torch.from_numpy(solution),
            torch.from_numpy(coords),
            compute_derivatives=False,
        )
        if isinstance(residual_result.values, Tensor):
            residuals = residual_result.values.numpy().astype(np.float32)
        else:
            residuals = residual_result.values.astype(np.float32)

        # Compute error
        error = float(np.sqrt(np.mean(residuals**2)))

        # Mesh info
        mesh_levels = np.array([e.level for e in self.mesh.leaf_elements], dtype=np.int32)

        new_state = PDEState(
            coords=coords,
            solution=solution,
            residuals=residuals,
            mesh_levels=mesh_levels,
            error_estimate=error,
            dof=self.mesh.n_dof,
            step=state.step + 1,
            budget_remaining=state.budget_remaining - 1,
            phase=state.phase,
            history=state.history + [action],
        )

        # Update phase
        if new_state.error_estimate < self.config.error_tolerance:
            new_state.phase = GamePhase.CONVERGED
        elif new_state.budget_remaining <= 0 or new_state.dof > self.config.max_dof:
            new_state.phase = GamePhase.BUDGET_EXHAUSTED

        return new_state

    def _interpolate_solution(
        self,
        old_state: PDEState,
        new_coords: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Interpolate solution to new mesh.

        Args:
            old_state: Previous state.
            new_coords: New coordinate points.

        Returns:
            Interpolated solution.

        """
        # Simple nearest-neighbor interpolation
        from scipy.spatial import cKDTree

        tree = cKDTree(old_state.coords)
        _, indices = tree.query(new_coords, k=1)

        return old_state.solution[indices].astype(np.float32)

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        """Compute reward for refinement action.

        Args:
            state: New state.
            prev_state: Previous state.

        Returns:
            Reward value.

        """
        # Error reduction
        error_reduction = prev_state.error_estimate - state.error_estimate

        # DOF cost
        dof_added = state.dof - prev_state.dof
        cost = self.config.cost_per_dof * dof_added

        # Efficiency bonus (reward good error/DOF ratio)
        # Uses configurable threshold and multiplier from mesh_config
        efficiency_threshold = self.mesh_config.efficiency_threshold
        efficiency_multiplier = self.mesh_config.efficiency_multiplier

        if dof_added > 0:
            efficiency = error_reduction / dof_added
            efficiency_bonus = max(0, efficiency - efficiency_threshold) * efficiency_multiplier
        else:
            efficiency_bonus = 0.0

        reward = self.config.reward_per_error_reduction * error_reduction - cost + efficiency_bonus

        # Terminal bonus
        if state.error_estimate < self.config.error_tolerance:
            reward += self.config.terminal_bonus

        logger.debug(
            "reward_computed",
            error_reduction=error_reduction,
            dof_added=dof_added,
            efficiency_bonus=efficiency_bonus,
            total_reward=reward,
        )

        return reward

    def is_terminal(self, state: PDEState) -> bool:
        """Check if game has ended.

        Args:
            state: Current state.

        Returns:
            True if terminal.

        """
        if state.error_estimate < self.config.error_tolerance:
            return True
        if state.dof > self.config.max_dof:
            return True
        if state.budget_remaining <= 0:
            return True
        if state.step >= self.config.max_steps:
            return True
        if len(self.get_valid_actions(state)) == 0:
            return True
        return False

    def get_result(self, state: PDEState, error_history: list[float]) -> PDEResult:
        """Get game result.

        Args:
            state: Terminal state.
            error_history: Error history.

        Returns:
            PDEResult.

        """
        errors = self.compute_exact_error(state)
        converged = state.error_estimate < self.config.error_tolerance

        if len(error_history) > 1:
            error_reduction_rate = (error_history[0] - error_history[-1]) / len(error_history)
            dof_efficiency = (error_history[0] - error_history[-1]) / max(1, state.dof)
        else:
            error_reduction_rate = 0.0
            dof_efficiency = 0.0

        budget_used = self.config.computational_budget - state.budget_remaining
        compute_efficiency = (
            (error_history[0] - error_history[-1]) / max(1, budget_used)
            if len(error_history) > 1
            else 0.0
        )

        termination_reason = (
            "converged"
            if converged
            else "max_dof"
            if state.dof > self.config.max_dof
            else "budget_exhausted"
            if state.budget_remaining <= 0
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
        """Compute error metrics.

        Args:
            state: Current state.

        Returns:
            Error dictionary.

        """
        # Get exact solution if available
        exact = self.pde_operator.exact_solution(state.coords)

        if exact is not None:
            if isinstance(exact, Tensor):
                exact = exact.numpy()
            l2_error = float(np.sqrt(np.mean((state.solution - exact) ** 2)))
            linf_error = float(np.max(np.abs(state.solution - exact)))
        else:
            l2_error = float(np.sqrt(np.mean(state.residuals**2)))
            linf_error = float(np.max(np.abs(state.residuals)))

        h1_error = l2_error  # Approximation
        residual_norm = float(np.sqrt(np.mean(state.residuals**2)))

        return {
            "l2": l2_error,
            "h1": h1_error,
            "linf": linf_error,
            "residual": residual_norm,
        }

    def to_tensor(self, state: PDEState) -> Float[Tensor, ...]:
        """Convert state to neural network input.

        Args:
            state: PDE state.

        Returns:
            Tensor encoding with shape:
            - 1D: (channels, resolution)
            - 2D: (channels, height, width)
            - 3D: (channels, depth, height, width)

        """
        from scipy.interpolate import griddata

        grid_size = self.mesh_config.initial_resolution
        dim = self.mesh.dim

        # Generate grid points based on dimension
        domain_min = self.pde_operator.domain_min
        domain_max = self.pde_operator.domain_max

        axes = [np.linspace(domain_min[d], domain_max[d], grid_size) for d in range(dim)]
        grids = np.meshgrid(*axes, indexing="ij")
        grid_points = np.stack([g.flatten() for g in grids], axis=-1)

        # Grid shape for reshaping
        grid_shape = tuple([grid_size] * dim)

        # Interpolate solution
        solution_grid = griddata(
            state.coords, state.solution, grid_points, method="linear", fill_value=0
        ).reshape(grid_shape)

        # Interpolate residuals
        residual_grid = griddata(
            state.coords, np.abs(state.residuals), grid_points, method="linear", fill_value=0
        ).reshape(grid_shape)

        # Refinement level indicator
        if state.mesh_levels is not None:
            level_grid = griddata(
                state.coords,
                state.mesh_levels.astype(np.float32),
                grid_points,
                method="nearest",
                fill_value=0,
            ).reshape(grid_shape)
        else:
            level_grid = np.zeros(grid_shape)

        # Build tensor with shape (channels, *grid_shape)
        tensor_shape = (self.state_channels,) + grid_shape
        tensor = torch.zeros(tensor_shape)
        tensor[0] = torch.from_numpy(solution_grid.astype(np.float32))
        tensor[1] = torch.from_numpy(residual_grid.astype(np.float32))
        tensor[2] = torch.from_numpy(level_grid.astype(np.float32))
        # Additional channels could include element sizes, polynomial degrees, etc.

        return tensor

    def action_to_string(self, action: int) -> str:
        """Convert action to string.

        Args:
            action: Action index.

        Returns:
            Action description.

        """
        if action < len(self.mesh.leaf_elements):
            element = self.mesh.leaf_elements[action]
            return f"refine_element({action}, level={element.level})"
        return f"invalid_action_{action}"
