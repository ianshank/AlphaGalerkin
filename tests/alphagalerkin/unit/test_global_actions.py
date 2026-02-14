"""Tests for global (mesh-wide) discretization actions.

Validates ``REFINE_ALL_BOUNDARY``, ``COARSEN_ALL_INTERIOR``, and
``UNIFORM_P_REFINE`` action types, their validation logic, and
their presence in the action-masking output.
"""
from __future__ import annotations

import pytest

from src.alphagalerkin.core.config import EnvironmentConfig
from src.alphagalerkin.core.types import ActionType, ElementID
from src.alphagalerkin.env.actions import GLOBAL_ACTION_TYPES, Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker

# ---------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------

@pytest.fixture
def mesh_3x3() -> MeshGraph:
    """A 3x3 uniform quad mesh -- has clear interior / boundary."""
    return MeshGraph.create_uniform_quad(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        num_elements=(3, 3),
    )


@pytest.fixture
def state_3x3(mesh_3x3: MeshGraph) -> DiscretizationState:
    """Initial state on a 3x3 quad mesh, p=1."""
    return DiscretizationState.from_mesh(
        mesh=mesh_3x3,
        initial_polynomial_order=1,
    )


@pytest.fixture
def state_3x3_p2(mesh_3x3: MeshGraph) -> DiscretizationState:
    """State on a 3x3 quad mesh, p=2 (to test coarsening)."""
    return DiscretizationState.from_mesh(
        mesh=mesh_3x3,
        initial_polynomial_order=2,
    )


# ---------------------------------------------------------------
# REFINE_ALL_BOUNDARY
# ---------------------------------------------------------------

class TestRefineAllBoundary:
    """Boundary elements get h-refined by REFINE_ALL_BOUNDARY."""

    def test_refine_all_boundary(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """After REFINE_ALL_BOUNDARY, element count increases."""
        original_count = state_3x3.mesh.num_elements
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.REFINE_ALL_BOUNDARY,
            {},
        )
        new_state = state_3x3.apply_action(action)

        # In a 3x3 mesh, boundary elements (fewer than max
        # neighbors) should be refined, increasing element count.
        assert (
            new_state.mesh.num_elements > original_count
        )
        assert new_state.validate()

    def test_refine_all_boundary_preserves_interior(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Interior elements (max neighbors) stay at level 0.

        In a 3x3 quad mesh the single centre element has 4 neighbours
        (the maximum), so it should remain un-refined.
        """
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.REFINE_ALL_BOUNDARY,
            {},
        )
        new_state = state_3x3.apply_action(action)

        # Find any level-0 elements remaining -- the interior
        # element should still exist at level 0.
        level0 = [
            e
            for e in new_state.mesh.element_ids
            if new_state.mesh.get_element(e).level == 0
        ]
        assert len(level0) >= 1

    def test_refine_all_boundary_invalidates_solution(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Solution should be None after a topology change."""
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.REFINE_ALL_BOUNDARY,
            {},
        )
        new_state = state_3x3.apply_action(action)
        assert new_state.solution is None


# ---------------------------------------------------------------
# COARSEN_ALL_INTERIOR
# ---------------------------------------------------------------

class TestCoarsenAllInterior:
    """Interior elements get p-coarsened."""

    def test_coarsen_all_interior(
        self, state_3x3_p2: DiscretizationState,
    ) -> None:
        """Interior elements should have their p decremented."""
        eid = state_3x3_p2.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.COARSEN_ALL_INTERIOR,
            {},
        )
        new_state = state_3x3_p2.apply_action(action)

        # Find the interior element (max neighbor count)
        max_neighbors = max(
            len(new_state.mesh.get_element(e).neighbors)
            for e in new_state.mesh.element_ids
        )

        for e in new_state.mesh.element_ids:
            elem = new_state.mesh.get_element(e)
            basis = new_state.basis_assignments[e]
            if len(elem.neighbors) == max_neighbors:
                # Interior: should be coarsened from p=2 to p=1
                assert basis.polynomial_order == 1
            else:
                # Boundary: should remain at p=2
                assert basis.polynomial_order == 2

    def test_coarsen_all_interior_floor_at_one(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """P-coarsening should not go below p=1."""
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.COARSEN_ALL_INTERIOR,
            {},
        )
        new_state = state_3x3.apply_action(action)

        for e in new_state.mesh.element_ids:
            basis = new_state.basis_assignments[e]
            assert basis.polynomial_order >= 1

    def test_coarsen_all_interior_preserves_count(
        self, state_3x3_p2: DiscretizationState,
    ) -> None:
        """Element count should be unchanged (no topology change)."""
        original_count = state_3x3_p2.mesh.num_elements
        eid = state_3x3_p2.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.COARSEN_ALL_INTERIOR,
            {},
        )
        new_state = state_3x3_p2.apply_action(action)
        assert new_state.mesh.num_elements == original_count


# ---------------------------------------------------------------
# UNIFORM_P_REFINE
# ---------------------------------------------------------------

class TestUniformPRefine:
    """All elements increment polynomial order."""

    def test_uniform_p_refine(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Every element should go from p=1 to p=2."""
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.UNIFORM_P_REFINE,
            {},
        )
        new_state = state_3x3.apply_action(action)

        for e in new_state.mesh.element_ids:
            assert (
                new_state.basis_assignments[
                    e
                ].polynomial_order
                == 2
            )

    def test_uniform_p_refine_preserves_family(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Basis family should be preserved after uniform p-refine."""
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.UNIFORM_P_REFINE,
            {},
        )
        new_state = state_3x3.apply_action(action)

        for e in new_state.mesh.element_ids:
            assert (
                new_state.basis_assignments[e].basis_family
                == "lagrange"
            )

    def test_uniform_p_refine_preserves_count(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Element count unchanged (no topology change)."""
        original_count = state_3x3.mesh.num_elements
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.UNIFORM_P_REFINE,
            {},
        )
        new_state = state_3x3.apply_action(action)
        assert new_state.mesh.num_elements == original_count

    def test_uniform_p_refine_increments_step(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Step counter should increment."""
        eid = state_3x3.mesh.element_ids[0]
        action = Action(
            eid,
            ActionType.UNIFORM_P_REFINE,
            {},
        )
        new_state = state_3x3.apply_action(action)
        assert new_state.step == state_3x3.step + 1


# ---------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------

class TestGlobalActionValidate:
    """Global actions validate correctly."""

    def test_global_action_validate_on_nonempty_mesh(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Global actions are valid when mesh is non-empty."""
        eid = state_3x3.mesh.element_ids[0]
        for action_type in GLOBAL_ACTION_TYPES:
            action = Action(eid, action_type, {})
            assert action.validate(state_3x3)

    def test_global_action_validate_on_empty_mesh(
        self,
    ) -> None:
        """Global actions are invalid on an empty mesh."""
        empty_mesh = MeshGraph()
        state = DiscretizationState(
            mesh=empty_mesh, basis_assignments={},
        )
        for action_type in GLOBAL_ACTION_TYPES:
            action = Action(
                ElementID("e0"), action_type, {},
            )
            assert not action.validate(state)

    def test_global_actions_dont_require_element_in_mesh(
        self, state_3x3: DiscretizationState,
    ) -> None:
        """Global actions with a non-existent element_id are valid."""
        fake_eid = ElementID("nonexistent")
        for action_type in GLOBAL_ACTION_TYPES:
            action = Action(fake_eid, action_type, {})
            assert action.validate(state_3x3)


# ---------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------

class TestActionMaskingIncludesGlobals:
    """Global actions appear in valid_actions output."""

    def test_action_masking_includes_globals(
        self, initial_state: DiscretizationState,
    ) -> None:
        """ActionMasker should include all three global actions."""
        config = EnvironmentConfig()
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)

        action_types = {a.action_type for a in actions}
        assert (
            ActionType.REFINE_ALL_BOUNDARY in action_types
        )
        assert (
            ActionType.COARSEN_ALL_INTERIOR in action_types
        )
        assert ActionType.UNIFORM_P_REFINE in action_types

    def test_uniform_p_refine_blocked_at_budget(
        self, initial_state: DiscretizationState,
    ) -> None:
        """UNIFORM_P_REFINE should be blocked when at DOF budget."""
        config = EnvironmentConfig(
            max_dof=initial_state.dof_count,
        )
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)

        uniform_p = [
            a for a in actions
            if a.action_type == ActionType.UNIFORM_P_REFINE
        ]
        assert len(uniform_p) == 0

    def test_boundary_and_interior_always_available(
        self, initial_state: DiscretizationState,
    ) -> None:
        """Global actions available even at DOF budget."""
        config = EnvironmentConfig(
            max_dof=initial_state.dof_count,
        )
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)

        action_types = {a.action_type for a in actions}
        assert (
            ActionType.REFINE_ALL_BOUNDARY in action_types
        )
        assert (
            ActionType.COARSEN_ALL_INTERIOR in action_types
        )

    def test_global_actions_not_duplicated(
        self, initial_state: DiscretizationState,
    ) -> None:
        """Each global action type appears exactly once."""
        config = EnvironmentConfig()
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)

        for gtype in GLOBAL_ACTION_TYPES:
            count = sum(
                1 for a in actions
                if a.action_type == gtype
            )
            assert count == 1, (
                f"{gtype.value} appeared {count} times"
            )
