"""Mesh topology and element operations.

Provides :class:`Element` (a single mesh cell with vertices, adjacency
and refinement metadata) and :class:`MeshGraph` (the topology manager
that supports uniform mesh generation, h-refinement, adjacency
tracking, and deep copies).

All geometric computations use NumPy.  No hardcoded values -- mesh
bounds, element counts, and tolerances are always passed as arguments.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import structlog

from src.alphagalerkin.core.types import ElementID

logger = structlog.get_logger("env.mesh_graph")


# -------------------------------------------------------------------
# Element
# -------------------------------------------------------------------

@dataclass
class Element:
    """Single mesh element (triangle or quadrilateral in 2-D).

    Attributes
    ----------
    id:
        Unique element identifier.
    vertices:
        Vertex coordinates, shape ``(n_vertices, 2)`` for 2-D.
    neighbors:
        IDs of face-adjacent elements.
    parent_id:
        Parent element before refinement (``None`` for root elements).
    children_ids:
        Children created by h-refinement.
    level:
        Refinement level (0 = coarsest).

    """

    id: ElementID
    vertices: np.ndarray  # shape (n_vertices, 2) for 2D
    neighbors: list[ElementID] = field(default_factory=list)
    parent_id: ElementID | None = None
    children_ids: list[ElementID] = field(default_factory=list)
    level: int = 0  # refinement level

    @property
    def centroid(self) -> np.ndarray:
        """Arithmetic mean of the vertex coordinates."""
        result: np.ndarray = np.mean(self.vertices, axis=0)
        return result

    @property
    def size(self) -> float:
        """Element diameter (maximum pairwise vertex distance)."""
        n = len(self.vertices)
        max_dist = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(
                    np.linalg.norm(
                        self.vertices[i] - self.vertices[j]
                    )
                )
                max_dist = max(max_dist, dist)
        return max_dist


# -------------------------------------------------------------------
# MeshGraph
# -------------------------------------------------------------------

class MeshGraph:
    """Mesh topology manager with element operations.

    Supports h-refinement (element splitting) and automatic adjacency
    rebuilding.  Elements are stored in a dict keyed by
    :class:`ElementID`.

    Parameters
    ----------
    elements:
        Pre-built element dict.  When ``None`` an empty mesh is
        created; use one of the ``create_*`` class methods instead.

    """

    def __init__(
        self,
        elements: dict[ElementID, Element] | None = None,
    ) -> None:
        self._elements: dict[ElementID, Element] = (
            elements or {}
        )
        self._retired_elements: dict[ElementID, Element] = {}
        self._next_id: int = len(self._elements)

    # -- properties --------------------------------------------------

    @property
    def num_elements(self) -> int:
        """Number of active (leaf) elements."""
        return len(self._elements)

    @property
    def element_ids(self) -> list[ElementID]:
        """Sorted list of active element IDs."""
        return sorted(self._elements.keys())

    # -- element access ----------------------------------------------

    def get_element(self, eid: ElementID) -> Element:
        """Return element by ID.  Raises ``KeyError`` if missing."""
        if eid not in self._elements:
            msg = f"Element {eid} not found in mesh"
            raise KeyError(msg)
        return self._elements[eid]

    # -- ID generation -----------------------------------------------

    def _generate_id(self) -> ElementID:
        eid = ElementID(f"e{self._next_id}")
        self._next_id += 1
        return eid

    # -- factory class methods ---------------------------------------

    @classmethod
    def create_uniform_quad(
        cls,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        num_elements: tuple[int, int],
    ) -> MeshGraph:
        """Create a uniform quadrilateral mesh.

        Parameters
        ----------
        bounds:
            ``((x_min, x_max), (y_min, y_max))`` domain limits.
        num_elements:
            ``(nx, ny)`` element counts in each direction.

        """
        (x_min, x_max), (y_min, y_max) = bounds
        nx, ny = num_elements
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny

        mesh = cls()
        for i in range(nx):
            for j in range(ny):
                x0 = x_min + i * dx
                y0 = y_min + j * dy
                vertices = np.array([
                    [x0, y0],
                    [x0 + dx, y0],
                    [x0 + dx, y0 + dy],
                    [x0, y0 + dy],
                ])
                eid = mesh._generate_id()
                mesh._elements[eid] = Element(
                    id=eid, vertices=vertices
                )

        mesh._build_adjacency()
        return mesh

    @classmethod
    def create_uniform_tri(
        cls,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        num_elements: tuple[int, int],
    ) -> MeshGraph:
        """Create a uniform triangular mesh.

        Each logical quad cell is split into two right triangles
        along the diagonal from bottom-right to top-left.

        Parameters
        ----------
        bounds:
            ``((x_min, x_max), (y_min, y_max))`` domain limits.
        num_elements:
            ``(nx, ny)`` cell counts in each direction (each cell
            produces 2 triangles).

        """
        (x_min, x_max), (y_min, y_max) = bounds
        nx, ny = num_elements
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny

        mesh = cls()
        for i in range(nx):
            for j in range(ny):
                x0 = x_min + i * dx
                y0 = y_min + j * dy
                # Lower-right triangle
                v1 = np.array([
                    [x0, y0],
                    [x0 + dx, y0],
                    [x0 + dx, y0 + dy],
                ])
                eid1 = mesh._generate_id()
                mesh._elements[eid1] = Element(
                    id=eid1, vertices=v1
                )
                # Upper-left triangle
                v2 = np.array([
                    [x0, y0],
                    [x0 + dx, y0 + dy],
                    [x0, y0 + dy],
                ])
                eid2 = mesh._generate_id()
                mesh._elements[eid2] = Element(
                    id=eid2, vertices=v2
                )

        mesh._build_adjacency()
        return mesh

    # -- h-refinement ------------------------------------------------

    def h_refine(self, eid: ElementID) -> list[ElementID]:
        """Split an element into children by midpoint subdivision.

        Quadrilaterals are split into 4 quads; triangles into 4
        triangles.  The parent is removed from the active element
        set but its ``children_ids`` are populated for tree
        traversal.

        Returns the IDs of the newly created child elements.
        """
        element = self.get_element(eid)
        n_vertices = len(element.vertices)

        if n_vertices == 4:
            return self._refine_quad(element)
        if n_vertices == 3:
            return self._refine_triangle(element)

        msg = f"Cannot refine element with {n_vertices} vertices"
        raise ValueError(msg)

    def _refine_quad(self, element: Element) -> list[ElementID]:
        """Refine a quadrilateral into 4 child quads."""
        v = element.vertices
        center = np.mean(v, axis=0)
        mid01 = (v[0] + v[1]) / 2
        mid12 = (v[1] + v[2]) / 2
        mid23 = (v[2] + v[3]) / 2
        mid30 = (v[3] + v[0]) / 2

        child_verts = [
            np.array([v[0], mid01, center, mid30]),
            np.array([mid01, v[1], mid12, center]),
            np.array([center, mid12, v[2], mid23]),
            np.array([mid30, center, mid23, v[3]]),
        ]

        new_ids: list[ElementID] = []
        for cv in child_verts:
            cid = self._generate_id()
            child = Element(
                id=cid,
                vertices=cv,
                parent_id=element.id,
                level=element.level + 1,
            )
            self._elements[cid] = child
            new_ids.append(cid)
            element.children_ids.append(cid)

        # Save parent in retired elements before removal
        self._retired_elements[element.id] = element
        del self._elements[element.id]
        self._build_adjacency()

        logger.debug(
            "mesh.h_refine",
            parent=str(element.id),
            children=[str(c) for c in new_ids],
        )
        return new_ids

    def _refine_triangle(
        self, element: Element
    ) -> list[ElementID]:
        """Refine a triangle into 4 children by midpoint subdivision.

        The three edge midpoints create four congruent sub-triangles:
        three at the corners and one central (inverted).
        """
        v = element.vertices
        mid01 = (v[0] + v[1]) / 2
        mid12 = (v[1] + v[2]) / 2
        mid20 = (v[2] + v[0]) / 2

        child_verts = [
            np.array([v[0], mid01, mid20]),
            np.array([mid01, v[1], mid12]),
            np.array([mid20, mid12, v[2]]),
            np.array([mid01, mid12, mid20]),  # central
        ]

        new_ids: list[ElementID] = []
        for cv in child_verts:
            cid = self._generate_id()
            child = Element(
                id=cid,
                vertices=cv,
                parent_id=element.id,
                level=element.level + 1,
            )
            self._elements[cid] = child
            new_ids.append(cid)
            element.children_ids.append(cid)

        # Save parent in retired elements before removal
        self._retired_elements[element.id] = element
        del self._elements[element.id]
        self._build_adjacency()
        return new_ids

    # -- h-coarsening ------------------------------------------------

    def h_coarsen(self, eid: ElementID) -> ElementID | None:
        """Merge sibling elements back into their parent.

        Looks up the element's parent, finds all siblings (children
        of that parent), verifies they are all leaf elements, removes
        them, and restores the parent element from the retired store.

        Parameters
        ----------
        eid:
            ID of any child element whose sibling group should be
            merged.

        Returns
        -------
        ElementID | None
            The restored parent element ID, or ``None`` if coarsening
            is not possible (e.g. element is a root element, parent
            has no retired data, or siblings have children).

        """
        element = self.get_element(eid)

        if element.parent_id is None:
            logger.debug(
                "mesh.h_coarsen.skip_root",
                element=str(eid),
            )
            return None

        parent_id = element.parent_id

        # Parent must be in retired elements store
        if parent_id not in self._retired_elements:
            logger.debug(
                "mesh.h_coarsen.no_retired_parent",
                element=str(eid),
                parent=str(parent_id),
            )
            return None

        parent = self._retired_elements[parent_id]

        # Find all sibling elements (children of same parent)
        sibling_ids = [
            sid
            for sid, sel in self._elements.items()
            if sel.parent_id == parent_id
        ]

        if not sibling_ids:
            return None

        # Verify all siblings are leaf elements (no children of their own)
        for sid in sibling_ids:
            sibling = self._elements[sid]
            # Check if any active element lists this sibling as parent
            has_children = any(
                el.parent_id == sid
                for el in self._elements.values()
                if el.id != sid
            )
            # Also check retired elements for children
            if not has_children:
                has_children = any(
                    el.parent_id == sid
                    for el in self._retired_elements.values()
                    if el.id != sid
                )
            if has_children:
                logger.debug(
                    "mesh.h_coarsen.non_leaf_sibling",
                    element=str(eid),
                    sibling=str(sid),
                )
                return None

        # Remove all siblings from active elements
        for sid in sibling_ids:
            del self._elements[sid]

        # Restore parent: reset children_ids and add to active elements
        restored_parent = Element(
            id=parent.id,
            vertices=parent.vertices.copy(),
            parent_id=parent.parent_id,
            children_ids=[],
            level=parent.level,
        )
        self._elements[parent_id] = restored_parent

        # Remove parent from retired elements
        del self._retired_elements[parent_id]

        self._build_adjacency()

        logger.debug(
            "mesh.h_coarsen",
            parent=str(parent_id),
            removed_children=[str(s) for s in sibling_ids],
        )
        return parent_id

    # -- adjacency ---------------------------------------------------

    def _build_adjacency(self) -> None:
        """Rebuild adjacency lists by shared-edge detection.

        Two elements are neighbours iff they share at least two
        vertices (i.e. an edge).
        """
        elements = list(self._elements.values())
        for elem in elements:
            elem.neighbors = []

        for i, e1 in enumerate(elements):
            for e2 in elements[i + 1 :]:
                if self._share_edge(e1.vertices, e2.vertices):
                    e1.neighbors.append(e2.id)
                    e2.neighbors.append(e1.id)

    @staticmethod
    def _share_edge(
        v1: np.ndarray,
        v2: np.ndarray,
        tol: float = 1e-10,
    ) -> bool:
        """Return ``True`` if *v1* and *v2* share >= 2 vertices."""
        shared = 0
        for p1 in v1:
            for p2 in v2:
                if np.linalg.norm(p1 - p2) < tol:
                    shared += 1
                    if shared >= 2:
                        return True
        return False

    # -- utilities ---------------------------------------------------

    def clone(self) -> MeshGraph:
        """Return a deep copy of the mesh (including retired elements)."""
        return copy.deepcopy(self)

    def element_sizes(self) -> dict[ElementID, float]:
        """Map every active element ID to its diameter."""
        return {
            eid: elem.size
            for eid, elem in self._elements.items()
        }

    def get_refinement_levels(self) -> dict[ElementID, int]:
        """Map every active element ID to its refinement level."""
        return {
            eid: elem.level
            for eid, elem in self._elements.items()
        }
