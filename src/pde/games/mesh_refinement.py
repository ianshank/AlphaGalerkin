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

import copy
import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import MeshRefinementConfig, PDEGameConfig, RefinementStrategy
from src.pde.game import GamePhase, PDEGame, PDEResult, PDEState
from src.pde.reward import log_reward

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


class ActionKind(IntEnum):
    """Action type for the mesh-refinement game.

    When ``MeshRefinementConfig.allow_coarsening`` is true the action space
    is partitioned into ``REFINE`` (low half) and ``COARSEN`` (high half)
    slots of width ``n_candidate_elements``.
    """

    REFINE = 0
    COARSEN = 1


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
        active: False once the element has been merged back into its parent
            by coarsening; such an element is kept in ``Mesh.elements`` for
            stable global indexing but is excluded from ``leaf_elements``.

    """

    index: int
    vertices: NDArray[np.float32]  # (n_vertices, dim)
    center: NDArray[np.float32]  # (dim,)
    size: float
    level: int = 0
    polynomial_degree: int = 1
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    active: bool = True

    @property
    def is_leaf(self) -> bool:
        """Whether this element is a live leaf (active and not refined)."""
        return self.active and len(self.children) == 0


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

    def can_coarsen_element(self, element_idx: int) -> bool:
        """Whether the leaf ``element_idx`` can be merged back into its parent.

        Coarsening is valid only when every sibling (i.e. each child of the
        common parent) is itself an active leaf — otherwise merging would
        discard refinement work done further down the tree.

        Args:
            element_idx: Global element index.

        Returns:
            True if the element participates in a fully-leaf sibling group
            that can be collapsed back to its parent.

        """
        element = self.elements[element_idx]
        if not element.is_leaf or element.parent is None:
            return False
        parent = self.elements[element.parent]
        if not parent.children:
            return False
        return all(self.elements[c].is_leaf for c in parent.children)

    def coarsen_element(self, element_idx: int) -> int:
        """Merge the sibling group containing ``element_idx`` back to the parent.

        Undoes a previous ``H_REFINEMENT`` on the parent: all children are
        marked inactive (so they drop out of ``leaf_elements`` while keeping
        their global indices stable for history replay), and the parent
        becomes a leaf again. The parent's polynomial degree is left intact
        so that any p-refinement that happened before the subdivision is
        preserved.

        Args:
            element_idx: Global element index of a leaf in a coarsenable group.

        Returns:
            Global index of the parent element (now a leaf again).

        Raises:
            ValueError: If the element cannot be coarsened (no parent, or
                at least one sibling has been refined further).

        """
        if not self.can_coarsen_element(element_idx):
            raise ValueError(
                f"Element {element_idx} is not coarsenable: it must be a leaf "
                "with a parent whose every child is also an active leaf."
            )
        element = self.elements[element_idx]
        parent_idx = element.parent
        assert parent_idx is not None  # guaranteed by can_coarsen_element
        parent = self.elements[parent_idx]

        sibling_indices = list(parent.children)
        for sibling_idx in sibling_indices:
            self.elements[sibling_idx].active = False
        parent.children = []

        logger.debug(
            "element_coarsened",
            parent_index=parent_idx,
            merged_indices=sibling_indices,
        )
        return parent_idx

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

    def clone(self) -> PDEGame:
        """MCTS-safe clone with an independent mesh.

        ``apply_action`` mutates ``self.mesh`` in-place (both refine and
        coarsen edit the element tree), so sibling MCTS simulations must
        not share it. The expensive immutables — ``pde_operator``,
        ``config``, ``mesh_config`` — are shared by reference; only the
        mutable mesh tree is deep-copied.
        """
        cls = type(self)
        cloned = cls.__new__(cls)
        cloned.pde_operator = self.pde_operator
        cloned.config = self.config
        cloned._action_space_size = self._action_space_size
        cloned._state_channels = self._state_channels
        cloned.mesh_config = self.mesh_config
        cloned._refinement_strategy = self._refinement_strategy
        cloned.mesh = copy.deepcopy(self.mesh)
        logger.debug(
            "mesh_game_cloned",
            n_elements=cloned.mesh.n_elements,
            n_leaves=len(cloned.mesh.leaf_elements),
        )
        return cloned

    @property
    def _coarsen_enabled(self) -> bool:
        """Whether the coarsen half of the action space is active."""
        return self.mesh_config.allow_coarsening

    @property
    def _refine_slot_count(self) -> int:
        """Effective width of the refine half of the action space.

        This is the *single source of truth* for the refine/coarsen
        partition point. ``action_space_size``, ``_decode_action``,
        ``get_valid_actions``, and ``get_action_mask`` all derive their
        slot count from this property so the partition stays consistent
        when ``max_elements`` is smaller than ``n_candidate_elements``
        (e.g. low-dim / low-level configs).
        """
        dim = self.mesh.dim
        max_elements = (
            self.mesh_config.initial_resolution**dim
            * (2**dim) ** self.mesh_config.max_refinement_level
        )
        return min(max_elements, self.mesh_config.n_candidate_elements)

    @property
    def action_space_size(self) -> int:
        """Number of possible actions.

        When ``allow_coarsening`` is set the action space is partitioned
        into two equal halves of width :attr:`_refine_slot_count`: the
        low half refines, the high half coarsens.
        """
        n = self._refine_slot_count
        return n * 2 if self._coarsen_enabled else n

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

    def _decode_action(self, action: int) -> tuple[ActionKind, int]:
        """Split a flat action into (kind, leaf_index).

        When coarsening is enabled the action space is::

            [0, n)          -> REFINE leaf_elements[action]
            [n, 2n)         -> COARSEN leaf_elements[action - n]

        where ``n = _refine_slot_count`` (the effective slot width —
        matches the bound used by ``action_space_size``, never the raw
        ``n_candidate_elements`` config value). When coarsening is
        disabled the upper half is absent and every action decodes as a
        refinement.

        Args:
            action: Flat action index.

        Returns:
            Pair of the decoded action kind and the leaf-order index.

        Raises:
            ValueError: If the action is out of range for the current
                action space.

        """
        slots = self._refine_slot_count
        if action < 0 or action >= self.action_space_size:
            raise ValueError(f"Invalid action: {action} not in [0, {self.action_space_size})")
        if not self._coarsen_enabled or action < slots:
            return ActionKind.REFINE, action
        return ActionKind.COARSEN, action - slots

    def _refine_eligible(self, element: MeshElement) -> bool:
        """Whether a leaf element can be refined under current config limits."""
        if element.level >= self.mesh_config.max_refinement_level:
            return False
        if element.size < self.mesh_config.min_element_size:
            return False
        if element.polynomial_degree >= self.mesh_config.max_polynomial_degree:
            return False
        return True

    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Get valid refinement (and optionally coarsening) actions.

        The partition point is driven by :attr:`_refine_slot_count` so
        emitted indices are always within ``action_space_size``.

        Coarsen actions are deduplicated by parent: every child in a
        coarsenable sibling group triggers the same parent collapse, so
        exposing one action per parent (rather than one per child)
        keeps the MCTS branching factor minimal without losing
        expressivity.

        Args:
            state: Current state.

        Returns:
            List of valid flat action indices. When ``allow_coarsening``
            is enabled, indices below :attr:`_refine_slot_count` are
            refine actions and indices above are coarsen actions.

        """
        slots = self._refine_slot_count
        leaves = self.mesh.leaf_elements

        refine_actions: list[int] = []
        for i, element in enumerate(leaves):
            if i >= slots:
                break
            if self._refine_eligible(element):
                refine_actions.append(i)

        if not self._coarsen_enabled:
            return refine_actions

        coarsen_actions: list[int] = []
        seen_parents: set[int] = set()
        for i, element in enumerate(leaves):
            if i >= slots:
                break
            if element.parent is None or element.parent in seen_parents:
                continue
            if self.mesh.can_coarsen_element(element.index):
                coarsen_actions.append(slots + i)
                seen_parents.add(element.parent)

        return refine_actions + coarsen_actions

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        """Get boolean mask for valid actions.

        Args:
            state: Current state.

        Returns:
            Boolean mask of length ``action_space_size``.

        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        valid = self.get_valid_actions(state)
        for idx in valid:
            if idx < self.action_space_size:
                mask[idx] = True
        return mask

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Apply a refinement or coarsening action.

        Args:
            state: Current state.
            action: Flat action index; see :meth:`_decode_action`.

        Returns:
            New state after the action.

        """
        kind, leaf_idx = self._decode_action(action)

        leaf_elements = self.mesh.leaf_elements
        if leaf_idx >= len(leaf_elements):
            raise ValueError(f"Invalid action: {action} (leaf {leaf_idx} missing)")

        element = leaf_elements[leaf_idx]

        if kind is ActionKind.REFINE:
            self.mesh.refine_element(
                element.index,
                self._refinement_strategy,
            )
        else:
            # ActionKind.COARSEN: raise a clean error if the action space
            # happened to expose a slot whose leaf is no longer coarsenable
            # (e.g. a sibling was refined further between mask evaluation
            # and dispatch). This preserves the "invalid action" contract.
            if not self.mesh.can_coarsen_element(element.index):
                raise ValueError(
                    f"Invalid coarsen action {action}: element {element.index} "
                    "is not in a fully-leaf sibling group."
                )
            self.mesh.coarsen_element(element.index)

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
        """Interpolate the old solution onto the refined mesh.

        Uses piecewise-linear interpolation (barycentric in 2D+, linear in
        1D) over the previous mesh's collocation points. Points that fall
        outside the convex hull of the old coords fall back to nearest-
        neighbor so the returned array is always well-defined.

        A proper Galerkin :math:`L^2`-projection onto the refined trial
        space would require assembling the refined mass matrix; that
        refinement is tracked as a future deliverable alongside the FEM
        baseline (see ``docs/doe_genesis/mdp_specification.md § 4``).

        Args:
            old_state: Previous state.
            new_coords: New coordinate points; shape ``(n_new, dim)``.

        Returns:
            Interpolated solution at ``new_coords`` with ``np.float32`` dtype.

        """
        old_coords = np.asarray(old_state.coords, dtype=np.float64)
        new_coords_f64 = np.asarray(new_coords, dtype=np.float64)
        old_solution = np.asarray(old_state.solution, dtype=np.float64)

        # Degenerate case: nothing to interpolate from.
        if old_coords.size == 0:
            return np.zeros(len(new_coords_f64), dtype=np.float32)

        if old_coords.ndim == 1 or old_coords.shape[1] == 1:
            # 1-D: sort by x and use numpy linear interpolation; out-of-range
            # points are clamped to the edge values (extrapolation would be
            # worse than a constant for a collocation mesh).
            xs = old_coords.reshape(-1)
            order = np.argsort(xs)
            xs_sorted = xs[order]
            ys_sorted = old_solution[order]
            queries = new_coords_f64.reshape(-1)
            interpolated = np.interp(queries, xs_sorted, ys_sorted)
            return interpolated.astype(np.float32)

        # 2-D and higher: linear interpolation with nearest-neighbor fallback.
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        nearest = NearestNDInterpolator(old_coords, old_solution)
        try:
            linear = LinearNDInterpolator(old_coords, old_solution, fill_value=np.nan)
            values = linear(new_coords_f64)
        except Exception:  # pragma: no cover - degenerate triangulation
            values = np.full(len(new_coords_f64), np.nan)

        missing = np.isnan(values)
        if missing.any():
            values[missing] = nearest(new_coords_f64[missing])

        return values.astype(np.float32)

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        """Compute reward for refinement action.

        Two forms are supported, selected by ``PDEGameConfig.reward_form``:

        * ``"linear"`` (default): error-reduction reward minus DOF cost
          plus an efficiency bonus plus terminal bonus.
        * ``"log"``: the DOE Genesis proposal reward
          ``-alpha * log(error) - beta * log(cost)`` with ``cost = state.dof``,
          plus the terminal bonus.

        Args:
            state: New state.
            prev_state: Previous state.

        Returns:
            Reward value.

        """
        if self.config.reward_form == "log":
            reward = log_reward(
                error=state.error_estimate,
                cost=float(state.dof),
                alpha=self.config.log_reward_alpha,
                beta=self.config.log_reward_beta,
                epsilon=self.config.log_reward_epsilon,
            )
            if state.error_estimate < self.config.error_tolerance:
                reward += self.config.terminal_bonus

            logger.debug(
                "reward_computed",
                form="log",
                error=state.error_estimate,
                dof=state.dof,
                total_reward=reward,
            )
            return reward

        # Linear form (historical default).
        error_reduction = prev_state.error_estimate - state.error_estimate

        dof_added = state.dof - prev_state.dof
        cost = self.config.cost_per_dof * dof_added

        efficiency_threshold = self.mesh_config.efficiency_threshold
        efficiency_multiplier = self.mesh_config.efficiency_multiplier

        if dof_added > 0:
            efficiency = error_reduction / dof_added
            efficiency_bonus = max(0, efficiency - efficiency_threshold) * efficiency_multiplier
        else:
            efficiency_bonus = 0.0

        reward = self.config.reward_per_error_reduction * error_reduction - cost + efficiency_bonus

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
        return len(self.get_valid_actions(state)) == 0

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
        if action < 0 or action >= self.action_space_size:
            return f"invalid_action_{action}"
        kind, leaf_idx = self._decode_action(action)
        if leaf_idx >= len(self.mesh.leaf_elements):
            return f"invalid_action_{action}"
        element = self.mesh.leaf_elements[leaf_idx]
        verb = "refine_element" if kind is ActionKind.REFINE else "coarsen_element"
        return f"{verb}({leaf_idx}, level={element.level})"
