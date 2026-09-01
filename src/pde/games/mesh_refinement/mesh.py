"""Pure quadtree/hypercube mesh data structure for adaptive refinement.

This module holds the domain-free mesh representation consumed by
:class:`~src.pde.games.mesh_refinement.game.MeshRefinementGame`: ``ActionKind``,
``MeshElement``, and ``Mesh`` have **no** dependency on MCTS, the PDE game
framework, or ``torch`` -- they operate purely on ``numpy`` arrays and a
config enum (``RefinementStrategy``). This is what makes the module a clean
half of the ``mesh_refinement.py`` -> ``mesh_refinement/`` package split
(docs/CODE_HYGIENE_AUDIT.md B4/§3.1: "a pure quadtree data structure
(``Mesh``, ~313 lines) and the MCTS game (~700 lines) in one file").

Note:
    The current Mesh implementation supports 2D quadrilateral elements only.
    For 1D (intervals), 3D (hexahedra), or higher dimensions, a specialized
    mesh class would be required. This limitation is validated at runtime.

"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import structlog
from numpy.typing import NDArray

from src.pde.config import RefinementStrategy

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
        hp_switchover_level: int = 2,
    ) -> None:
        """Initialize mesh.

        Args:
            domain_min: Domain minimum coordinates.
            domain_max: Domain maximum coordinates.
            initial_resolution: Initial elements per dimension.
            hp_switchover_level: Refinement level threshold used by
                :meth:`refine_element`'s ``HP_REFINEMENT`` branch to choose
                h- vs p-refinement. Defaults to 2, matching
                ``MeshRefinementConfig.hp_switchover_level``'s default;
                callers driven by that config should pass it explicitly.

        Raises:
            ValueError: If dimension is not supported (>4).

        """
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.domain_size = domain_max - domain_min
        self.dim = len(domain_min)
        self.initial_resolution = initial_resolution
        self.hp_switchover_level = hp_switchover_level

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
            if element.level < self.hp_switchover_level:
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
