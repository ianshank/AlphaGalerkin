"""Tests for mesh graph."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.core.types import ElementID
from src.alphagalerkin.env.mesh_graph import Element, MeshGraph


class TestMeshGraphCreation:
    """Tests for mesh factory methods."""

    def test_create_uniform_quad(self) -> None:
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(3, 3),
        )
        assert mesh.num_elements == 9

    def test_create_uniform_tri(self) -> None:
        mesh = MeshGraph.create_uniform_tri(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        assert mesh.num_elements == 8  # 2*2*2

    def test_create_single_element(self) -> None:
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(1, 1),
        )
        assert mesh.num_elements == 1

    def test_non_unit_domain(self) -> None:
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 2.0), (-1.0, 1.0)),
            num_elements=(4, 4),
        )
        assert mesh.num_elements == 16


class TestMeshGraphRefinement:
    """Tests for h-refinement operations."""

    def test_h_refine_quad(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        eid = quad_mesh_2x2.element_ids[0]
        initial_count = quad_mesh_2x2.num_elements
        new_ids = quad_mesh_2x2.h_refine(eid)
        assert len(new_ids) == 4
        # -1 parent + 4 children = net +3
        assert (
            quad_mesh_2x2.num_elements == initial_count + 3
        )

    def test_h_refine_triangle(
        self, tri_mesh_small: MeshGraph,
    ) -> None:
        eid = tri_mesh_small.element_ids[0]
        initial_count = tri_mesh_small.num_elements
        new_ids = tri_mesh_small.h_refine(eid)
        assert len(new_ids) == 4
        assert (
            tri_mesh_small.num_elements == initial_count + 3
        )

    def test_children_have_incremented_level(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        eid = quad_mesh_2x2.element_ids[0]
        new_ids = quad_mesh_2x2.h_refine(eid)
        for cid in new_ids:
            child = quad_mesh_2x2.get_element(cid)
            assert child.level == 1

    def test_children_reference_parent(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        eid = quad_mesh_2x2.element_ids[0]
        new_ids = quad_mesh_2x2.h_refine(eid)
        for cid in new_ids:
            child = quad_mesh_2x2.get_element(cid)
            assert child.parent_id == eid


class TestMeshGraphProperties:
    """Tests for mesh properties and utilities."""

    def test_element_sizes_positive(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        sizes = quad_mesh_2x2.element_sizes()
        for eid, size in sizes.items():
            assert size > 0

    def test_clone_is_independent(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        cloned = quad_mesh_2x2.clone()
        assert (
            cloned.num_elements == quad_mesh_2x2.num_elements
        )
        # Modify clone, verify original unchanged
        eid = cloned.element_ids[0]
        cloned.h_refine(eid)
        assert (
            cloned.num_elements != quad_mesh_2x2.num_elements
        )

    def test_element_ids_sorted(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        ids = quad_mesh_2x2.element_ids
        assert ids == sorted(ids)

    def test_get_nonexistent_element_raises(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        with pytest.raises(KeyError):
            quad_mesh_2x2.get_element(
                ElementID("nonexistent")
            )

    def test_get_refinement_levels(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        levels = quad_mesh_2x2.get_refinement_levels()
        assert all(lvl == 0 for lvl in levels.values())

    def test_refinement_levels_after_refine(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        eid = quad_mesh_2x2.element_ids[0]
        new_ids = quad_mesh_2x2.h_refine(eid)
        levels = quad_mesh_2x2.get_refinement_levels()
        for nid in new_ids:
            assert levels[nid] == 1


class TestElement:
    """Tests for the Element dataclass."""

    def test_centroid_of_unit_quad(self) -> None:
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        elem = Element(
            id=ElementID("e0"), vertices=vertices,
        )
        centroid = elem.centroid
        np.testing.assert_allclose(centroid, [0.5, 0.5])

    def test_size_of_unit_quad(self) -> None:
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        elem = Element(
            id=ElementID("e0"), vertices=vertices,
        )
        # Diameter is the diagonal = sqrt(2)
        assert abs(elem.size - np.sqrt(2.0)) < 1e-10

    def test_size_of_right_triangle(self) -> None:
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        elem = Element(
            id=ElementID("e0"), vertices=vertices,
        )
        # Diameter is the hypotenuse = sqrt(2)
        assert abs(elem.size - np.sqrt(2.0)) < 1e-10


class TestMeshAdjacency:
    """Tests for adjacency detection."""

    def test_adjacent_elements_found(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        """In a 2x2 quad mesh, every element should have at least one neighbor."""
        for eid in quad_mesh_2x2.element_ids:
            elem = quad_mesh_2x2.get_element(eid)
            assert len(elem.neighbors) > 0

    def test_adjacency_is_symmetric(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        for eid in quad_mesh_2x2.element_ids:
            elem = quad_mesh_2x2.get_element(eid)
            for nid in elem.neighbors:
                nbr = quad_mesh_2x2.get_element(nid)
                assert eid in nbr.neighbors
