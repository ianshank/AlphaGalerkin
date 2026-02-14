"""Tests for h-coarsening (element merging).

Validates that :meth:`MeshGraph.h_coarsen` correctly restores a
parent element after h-refinement, and that
:class:`DiscretizationState` updates basis assignments accordingly.
"""

from __future__ import annotations

from src.alphagalerkin.core.types import ActionType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState


class TestHCoarsenAfterRefine:
    """Refine then coarsen should restore the parent element."""

    def test_h_coarsen_after_refine(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Refining and then coarsening restores original count."""
        original_count = quad_mesh_2x2.num_elements
        eid = quad_mesh_2x2.element_ids[0]

        # Refine
        child_ids = quad_mesh_2x2.h_refine(eid)
        assert quad_mesh_2x2.num_elements == original_count + 3

        # Coarsen using the first child
        parent_id = quad_mesh_2x2.h_coarsen(child_ids[0])
        assert parent_id is not None
        assert parent_id == eid
        assert quad_mesh_2x2.num_elements == original_count

    def test_h_coarsen_restores_parent_vertices(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Restored parent should have the same vertices."""
        eid = quad_mesh_2x2.element_ids[0]
        original_elem = quad_mesh_2x2.get_element(eid)
        original_vertices = original_elem.vertices.copy()

        child_ids = quad_mesh_2x2.h_refine(eid)
        parent_id = quad_mesh_2x2.h_coarsen(child_ids[0])

        assert parent_id is not None
        restored = quad_mesh_2x2.get_element(parent_id)
        assert restored.vertices.shape == original_vertices.shape
        # Vertices should match
        import numpy as np

        np.testing.assert_allclose(
            restored.vertices,
            original_vertices,
        )

    def test_h_coarsen_restores_parent_level(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Restored parent has original refinement level."""
        eid = quad_mesh_2x2.element_ids[0]
        child_ids = quad_mesh_2x2.h_refine(eid)
        parent_id = quad_mesh_2x2.h_coarsen(child_ids[0])

        assert parent_id is not None
        restored = quad_mesh_2x2.get_element(parent_id)
        assert restored.level == 0

    def test_h_coarsen_any_sibling_works(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Coarsening from any sibling should produce the same result."""
        eid = quad_mesh_2x2.element_ids[0]
        original_count = quad_mesh_2x2.num_elements

        child_ids = quad_mesh_2x2.h_refine(eid)
        # Use the last child instead of the first
        parent_id = quad_mesh_2x2.h_coarsen(child_ids[-1])
        assert parent_id is not None
        assert parent_id == eid
        assert quad_mesh_2x2.num_elements == original_count

    def test_h_coarsen_triangle_mesh(
        self,
        tri_mesh_small: MeshGraph,
    ) -> None:
        """Coarsening works on triangular meshes too."""
        original_count = tri_mesh_small.num_elements
        eid = tri_mesh_small.element_ids[0]

        child_ids = tri_mesh_small.h_refine(eid)
        assert tri_mesh_small.num_elements == original_count + 3

        parent_id = tri_mesh_small.h_coarsen(child_ids[0])
        assert parent_id is not None
        assert parent_id == eid
        assert tri_mesh_small.num_elements == original_count


class TestHCoarsenRootElement:
    """Coarsening a root element returns None."""

    def test_h_coarsen_root_element(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Root-level elements cannot be coarsened."""
        eid = quad_mesh_2x2.element_ids[0]
        result = quad_mesh_2x2.h_coarsen(eid)
        assert result is None

    def test_h_coarsen_root_preserves_mesh(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Attempting to coarsen a root element leaves mesh unchanged."""
        original_count = quad_mesh_2x2.num_elements
        eid = quad_mesh_2x2.element_ids[0]
        quad_mesh_2x2.h_coarsen(eid)
        assert quad_mesh_2x2.num_elements == original_count


class TestHCoarsenStateBasis:
    """Basis assignments update correctly after coarsening."""

    def test_h_coarsen_state_basis(
        self,
        initial_state: DiscretizationState,
    ) -> None:
        """After refine+coarsen, parent gets average basis order."""
        eid = initial_state.mesh.element_ids[0]
        # Refine
        refine_action = Action(eid, ActionType.H_REFINE, {})
        refined_state = initial_state.apply_action(refine_action)

        # Pick a child element
        child_eid = [
            e for e in refined_state.mesh.element_ids if e not in initial_state.mesh.element_ids
        ][0]

        # Coarsen
        coarsen_action = Action(
            child_eid,
            ActionType.H_COARSEN,
            {},
        )
        coarsened_state = refined_state.apply_action(
            coarsen_action,
        )

        # Parent should be restored with p=1 basis
        assert eid in coarsened_state.basis_assignments
        assert coarsened_state.basis_assignments[eid].polynomial_order == 1
        assert coarsened_state.validate()

    def test_h_coarsen_state_preserves_family(
        self,
        initial_state: DiscretizationState,
    ) -> None:
        """Restored parent inherits basis family from children."""
        eid = initial_state.mesh.element_ids[0]

        # Refine
        refine_action = Action(eid, ActionType.H_REFINE, {})
        refined_state = initial_state.apply_action(refine_action)

        # Change family on one child
        child_eids = [
            e for e in refined_state.mesh.element_ids if e not in initial_state.mesh.element_ids
        ]

        # Coarsen
        coarsen_action = Action(
            child_eids[0],
            ActionType.H_COARSEN,
            {},
        )
        coarsened_state = refined_state.apply_action(
            coarsen_action,
        )

        assert coarsened_state.basis_assignments[eid].basis_family == "lagrange"

    def test_h_coarsen_averages_mixed_orders(
        self,
        initial_state: DiscretizationState,
    ) -> None:
        """When siblings have different p, parent gets average."""
        eid = initial_state.mesh.element_ids[0]

        # Refine
        refine_action = Action(eid, ActionType.H_REFINE, {})
        refined_state = initial_state.apply_action(refine_action)

        # P-refine two of the children
        child_eids = [
            e for e in refined_state.mesh.element_ids if e not in initial_state.mesh.element_ids
        ]
        state2 = refined_state
        for child in child_eids[:2]:
            p_action = Action(child, ActionType.P_REFINE, {})
            state2 = state2.apply_action(p_action)

        # Now children have orders [2, 2, 1, 1] -> avg = 1.5 -> round = 2
        coarsen_action = Action(
            child_eids[0],
            ActionType.H_COARSEN,
            {},
        )
        coarsened_state = state2.apply_action(coarsen_action)

        assert eid in coarsened_state.basis_assignments
        assert coarsened_state.basis_assignments[eid].polynomial_order == 2

    def test_h_coarsen_noop_on_root_state(
        self,
        initial_state: DiscretizationState,
    ) -> None:
        """Coarsening a root element in state is effectively no-op."""
        eid = initial_state.mesh.element_ids[0]
        original_count = initial_state.mesh.num_elements

        coarsen_action = Action(
            eid,
            ActionType.H_COARSEN,
            {},
        )
        new_state = initial_state.apply_action(coarsen_action)

        assert new_state.mesh.num_elements == original_count
        assert new_state.validate()


class TestRetiredElementsPreserved:
    """_retired_elements should survive clone operations."""

    def test_retired_elements_preserved(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Cloning a mesh preserves _retired_elements."""
        eid = quad_mesh_2x2.element_ids[0]
        quad_mesh_2x2.h_refine(eid)

        # Parent should be in _retired_elements
        assert eid in quad_mesh_2x2._retired_elements

        # Clone and verify
        cloned = quad_mesh_2x2.clone()
        assert eid in cloned._retired_elements

        # Modify clone: should not affect original
        cloned._retired_elements.clear()
        assert eid in quad_mesh_2x2._retired_elements

    def test_retired_elements_independent_after_clone(
        self,
        quad_mesh_2x2: MeshGraph,
    ) -> None:
        """Retired elements in clone are independent objects."""
        eid = quad_mesh_2x2.element_ids[0]
        quad_mesh_2x2.h_refine(eid)

        cloned = quad_mesh_2x2.clone()

        # Coarsen on clone should not affect original
        child_id = cloned.element_ids[0]
        child = cloned.get_element(child_id)
        if child.parent_id == eid:
            cloned.h_coarsen(child_id)
            # Original should still have the retired element
            assert eid in quad_mesh_2x2._retired_elements

    def test_state_clone_preserves_retired(
        self,
        initial_state: DiscretizationState,
    ) -> None:
        """DiscretizationState.clone preserves retired elements."""
        eid = initial_state.mesh.element_ids[0]
        refine_action = Action(eid, ActionType.H_REFINE, {})
        refined = initial_state.apply_action(refine_action)

        # Refined state's mesh should have retired elements
        assert eid in refined.mesh._retired_elements

        # Clone and verify
        cloned = refined.clone()
        assert eid in cloned.mesh._retired_elements
