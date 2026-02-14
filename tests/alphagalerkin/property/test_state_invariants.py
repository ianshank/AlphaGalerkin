"""Property-based tests for state invariants."""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.alphagalerkin.core.types import ActionType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState

_SUPPRESS = [HealthCheck.function_scoped_fixture]


class TestStateInvariants:
    """Hypothesis-driven invariant checks for state transitions."""

    def test_h_refine_increases_element_count(
        self, initial_state: DiscretizationState,
    ) -> None:
        eid = initial_state.mesh.element_ids[0]
        action = Action(eid, ActionType.H_REFINE, {})
        new_state = initial_state.apply_action(action)
        assert (
            new_state.mesh.num_elements
            > initial_state.mesh.num_elements
        )

    def test_clone_produces_independent_copy(
        self, initial_state: DiscretizationState,
    ) -> None:
        cloned = initial_state.clone()
        assert cloned is not initial_state
        assert cloned.mesh is not initial_state.mesh

    def test_state_valid_after_multiple_refinements(
        self, initial_state: DiscretizationState,
    ) -> None:
        state = initial_state
        for _ in range(3):
            eid = state.mesh.element_ids[0]
            action = Action(eid, ActionType.H_REFINE, {})
            state = state.apply_action(action)
        assert state.validate()
        assert state.dof_count > 0

    @given(
        n_steps=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=10,
        suppress_health_check=_SUPPRESS,
    )
    def test_step_counter_tracks_actions(
        self,
        initial_state: DiscretizationState,
        n_steps: int,
    ) -> None:
        """Step counter should match the number of applied actions."""
        state = initial_state
        for _ in range(n_steps):
            eid = state.mesh.element_ids[0]
            action = Action(eid, ActionType.NO_OP, {})
            state = state.apply_action(action)
        assert state.step == n_steps

    @given(
        n_refines=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=10,
        suppress_health_check=_SUPPRESS,
    )
    def test_dof_monotonically_increases_with_h_refine(
        self,
        initial_state: DiscretizationState,
        n_refines: int,
    ) -> None:
        """Each h-refinement should increase DOF count."""
        state = initial_state
        prev_dof = state.dof_count
        for _ in range(n_refines):
            eid = state.mesh.element_ids[0]
            action = Action(eid, ActionType.H_REFINE, {})
            state = state.apply_action(action)
            assert state.dof_count >= prev_dof
            prev_dof = state.dof_count

    @given(
        n_refines=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=10,
        suppress_health_check=_SUPPRESS,
    )
    def test_dof_increases_with_p_refine(
        self,
        initial_state: DiscretizationState,
        n_refines: int,
    ) -> None:
        """Each p-refinement should increase DOF count."""
        state = initial_state
        prev_dof = state.dof_count
        eid = state.mesh.element_ids[0]
        for _ in range(n_refines):
            action = Action(eid, ActionType.P_REFINE, {})
            state = state.apply_action(action)
            assert state.dof_count >= prev_dof
            prev_dof = state.dof_count

    def test_validate_after_mixed_refinements(
        self, initial_state: DiscretizationState,
    ) -> None:
        """State should remain valid after mixed h/p actions."""
        state = initial_state
        eid = state.mesh.element_ids[0]

        # h-refine
        state = state.apply_action(
            Action(eid, ActionType.H_REFINE, {}),
        )
        assert state.validate()

        # p-refine on a new element
        eid2 = state.mesh.element_ids[0]
        state = state.apply_action(
            Action(eid2, ActionType.P_REFINE, {}),
        )
        assert state.validate()

        # NO_OP
        state = state.apply_action(
            Action(eid2, ActionType.NO_OP, {}),
        )
        assert state.validate()

    @given(
        poly_order=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=10,
        suppress_health_check=_SUPPRESS,
    )
    def test_from_mesh_with_varying_poly_order(
        self, poly_order: int,
    ) -> None:
        """from_mesh with any valid poly order should produce a valid state."""
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(
            mesh, initial_polynomial_order=poly_order,
        )
        assert state.validate()
        assert state.dof_count > 0
