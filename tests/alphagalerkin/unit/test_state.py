"""Tests for discretization state."""
from __future__ import annotations

import torch

from src.alphagalerkin.core.types import ActionType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState


class TestDiscretizationState:
    """Core DiscretizationState behaviour."""

    def test_from_mesh_creates_valid_state(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        state = DiscretizationState.from_mesh(
            quad_mesh_2x2, initial_polynomial_order=1,
        )
        assert state.validate()
        assert state.dof_count > 0
        assert state.step == 0

    def test_h_refine_increases_elements(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.H_REFINE, {})
        new_state = initial_state.apply_action(action)
        assert (
            new_state.mesh.num_elements
            > initial_state.mesh.num_elements
        )

    def test_p_refine_increases_polynomial_order(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.P_REFINE, {})
        new_state = initial_state.apply_action(action)
        assert (
            new_state.basis_assignments[eid].polynomial_order
            == 2
        )

    def test_p_coarsen_decreases_polynomial_order(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        # First p-refine to order 2
        p_up = Action(eid, ActionType.P_REFINE, {})
        state2 = initial_state.apply_action(p_up)
        assert (
            state2.basis_assignments[eid].polynomial_order
            == 2
        )
        # Then p-coarsen back to 1
        p_down = Action(eid, ActionType.P_COARSEN, {})
        state3 = state2.apply_action(p_down)
        assert (
            state3.basis_assignments[eid].polynomial_order
            == 1
        )

    def test_p_coarsen_never_goes_below_one(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.P_COARSEN, {})
        new_state = initial_state.apply_action(action)
        assert (
            new_state.basis_assignments[eid].polynomial_order
            >= 1
        )

    def test_noop_preserves_state(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.NO_OP, {})
        new_state = initial_state.apply_action(action)
        assert (
            new_state.mesh.num_elements
            == initial_state.mesh.num_elements
        )
        assert new_state.step == initial_state.step + 1

    def test_apply_action_increments_step(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.NO_OP, {})
        s1 = initial_state.apply_action(action)
        s2 = s1.apply_action(action)
        assert s1.step == 1
        assert s2.step == 2

    def test_clone_produces_independent_copy(
        self, initial_state: DiscretizationState,
    ) -> None:
        cloned = initial_state.clone()
        assert cloned is not initial_state
        assert cloned.mesh is not initial_state.mesh
        assert cloned.step == initial_state.step

    def test_to_feature_tensor_shape(
        self, initial_state: DiscretizationState,
    ) -> None:
        features = initial_state.to_feature_tensor()
        assert (
            features.shape[0]
            == initial_state.mesh.num_elements
        )
        assert features.shape[1] == 8
        assert features.dtype == torch.float32

    def test_validate_catches_missing_basis(
        self, quad_mesh_2x2: MeshGraph,
    ) -> None:
        state = DiscretizationState(
            mesh=quad_mesh_2x2, basis_assignments={},
        )
        assert not state.validate()

    def test_swap_basis_changes_family(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.SWAP_BASIS,
            {"basis_family": "legendre"},
        )
        new_state = initial_state.apply_action(action)
        assert (
            new_state.basis_assignments[eid].basis_family
            == "legendre"
        )

    def test_h_refine_invalidates_solution(
        self, initial_state: DiscretizationState,
    ) -> None:
        """After a refinement, solution should be None."""
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.H_REFINE, {})
        new_state = initial_state.apply_action(action)
        assert new_state.solution is None

    def test_apply_action_does_not_mutate_original(
        self, initial_state: DiscretizationState,
    ) -> None:
        original_count = initial_state.mesh.num_elements
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.H_REFINE, {})
        _ = initial_state.apply_action(action)
        assert initial_state.mesh.num_elements == original_count
