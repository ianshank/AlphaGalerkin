"""Tests for action definitions (src/alphagalerkin/env/actions.py)."""
from __future__ import annotations

import pytest

from src.alphagalerkin.core.types import (
    ActionType,
    ElementID,
)
from src.alphagalerkin.env.actions import (
    GLOBAL_ACTION_TYPES,
    Action,
)
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_simple_mesh() -> MeshGraph:
    """Create a 2x2 uniform quad mesh on [0,1]^2."""
    return MeshGraph.create_uniform_quad(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        num_elements=(2, 2),
    )


def _make_state(
    mesh: MeshGraph | None = None,
    polynomial_order: int = 1,
) -> DiscretizationState:
    """Build a DiscretizationState from a mesh."""
    if mesh is None:
        mesh = _make_simple_mesh()
    return DiscretizationState.from_mesh(
        mesh,
        initial_polynomial_order=polynomial_order,
    )


def _make_empty_state() -> DiscretizationState:
    """Build a state with an empty mesh."""
    mesh = MeshGraph()
    return DiscretizationState(
        mesh=mesh,
        basis_assignments={},
    )


# -------------------------------------------------------------------
# GLOBAL_ACTION_TYPES
# -------------------------------------------------------------------


class TestGlobalActionTypes:
    """Tests for the GLOBAL_ACTION_TYPES constant."""

    def test_is_frozenset(self) -> None:
        assert isinstance(GLOBAL_ACTION_TYPES, frozenset)

    def test_contains_refine_all_boundary(self) -> None:
        assert ActionType.REFINE_ALL_BOUNDARY in GLOBAL_ACTION_TYPES

    def test_contains_coarsen_all_interior(self) -> None:
        assert ActionType.COARSEN_ALL_INTERIOR in GLOBAL_ACTION_TYPES

    def test_contains_uniform_p_refine(self) -> None:
        assert ActionType.UNIFORM_P_REFINE in GLOBAL_ACTION_TYPES

    def test_does_not_contain_no_op(self) -> None:
        assert ActionType.NO_OP not in GLOBAL_ACTION_TYPES

    def test_does_not_contain_h_refine(self) -> None:
        assert ActionType.H_REFINE not in GLOBAL_ACTION_TYPES

    def test_exactly_three_members(self) -> None:
        assert len(GLOBAL_ACTION_TYPES) == 3


# -------------------------------------------------------------------
# Action immutability (frozen dataclass)
# -------------------------------------------------------------------


class TestActionImmutability:
    """Tests for the Action frozen dataclass."""

    def test_cannot_modify_action_type(self) -> None:
        action = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        with pytest.raises(AttributeError):
            action.action_type = ActionType.P_REFINE  # type: ignore[misc]

    def test_cannot_modify_element_id(self) -> None:
        action = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        with pytest.raises(AttributeError):
            action.element_id = ElementID("e1")  # type: ignore[misc]


# -------------------------------------------------------------------
# validate: NO_OP
# -------------------------------------------------------------------


class TestValidateNoOp:
    """NO_OP is always valid regardless of state."""

    def test_no_op_valid_with_elements(self) -> None:
        state = _make_state()
        action = Action(
            element_id=ElementID("doesnt_matter"),
            action_type=ActionType.NO_OP,
        )
        assert action.validate(state) is True

    def test_no_op_valid_with_empty_mesh(self) -> None:
        state = _make_empty_state()
        action = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.NO_OP,
        )
        assert action.validate(state) is True


# -------------------------------------------------------------------
# validate: global actions
# -------------------------------------------------------------------


class TestValidateGlobalActions:
    """Global actions need at least one element."""

    def test_global_action_valid_with_elements(self) -> None:
        state = _make_state()
        for action_type in GLOBAL_ACTION_TYPES:
            action = Action(
                element_id=ElementID("ignored"),
                action_type=action_type,
            )
            assert action.validate(state) is True, (
                f"{action_type} should be valid with non-empty mesh"
            )

    def test_global_action_invalid_with_empty_mesh(self) -> None:
        state = _make_empty_state()
        for action_type in GLOBAL_ACTION_TYPES:
            action = Action(
                element_id=ElementID("ignored"),
                action_type=action_type,
            )
            assert action.validate(state) is False, (
                f"{action_type} should be invalid with empty mesh"
            )


# -------------------------------------------------------------------
# validate: element_id not in mesh
# -------------------------------------------------------------------


class TestValidateElementNotInMesh:
    """Non-global actions targeting a missing element are invalid."""

    def test_h_refine_missing_element(self) -> None:
        state = _make_state()
        action = Action(
            element_id=ElementID("nonexistent"),
            action_type=ActionType.H_REFINE,
        )
        assert action.validate(state) is False

    def test_p_refine_missing_element(self) -> None:
        state = _make_state()
        action = Action(
            element_id=ElementID("nonexistent"),
            action_type=ActionType.P_REFINE,
        )
        assert action.validate(state) is False


# -------------------------------------------------------------------
# validate: H_COARSEN
# -------------------------------------------------------------------


class TestValidateHCoarsen:
    """H_COARSEN requires level > 0."""

    def test_h_coarsen_level_zero_invalid(self) -> None:
        """Root elements (level=0) cannot be coarsened."""
        state = _make_state()
        eid = state.mesh.element_ids[0]
        action = Action(
            element_id=eid,
            action_type=ActionType.H_COARSEN,
        )
        assert action.validate(state) is False

    def test_h_coarsen_level_gt_zero_valid(self) -> None:
        """Elements after refinement (level > 0) can be coarsened."""
        state = _make_state()
        eid = state.mesh.element_ids[0]
        # H-refine to create children at level 1
        new_state = state.apply_action(
            Action(element_id=eid, action_type=ActionType.H_REFINE)
        )
        # Find a child element (level > 0)
        child_eid = None
        for cid in new_state.mesh.element_ids:
            elem = new_state.mesh.get_element(cid)
            if elem.level > 0:
                child_eid = cid
                break
        assert child_eid is not None
        action = Action(
            element_id=child_eid,
            action_type=ActionType.H_COARSEN,
        )
        assert action.validate(new_state) is True


# -------------------------------------------------------------------
# validate: P_COARSEN
# -------------------------------------------------------------------


class TestValidatePCoarsen:
    """P_COARSEN requires basis with order > 1."""

    def test_p_coarsen_order_one_invalid(self) -> None:
        """Cannot coarsen polynomial order below 1."""
        state = _make_state(polynomial_order=1)
        eid = state.mesh.element_ids[0]
        action = Action(
            element_id=eid,
            action_type=ActionType.P_COARSEN,
        )
        assert action.validate(state) is False

    def test_p_coarsen_order_gt_one_valid(self) -> None:
        """Order > 1 allows coarsening."""
        state = _make_state(polynomial_order=3)
        eid = state.mesh.element_ids[0]
        action = Action(
            element_id=eid,
            action_type=ActionType.P_COARSEN,
        )
        assert action.validate(state) is True

    def test_p_coarsen_no_basis_invalid(self) -> None:
        """Element with no basis assignment cannot be p-coarsened."""
        state = _make_state()
        eid = state.mesh.element_ids[0]
        # Remove basis assignment for this element
        del state.basis_assignments[eid]
        action = Action(
            element_id=eid,
            action_type=ActionType.P_COARSEN,
        )
        assert action.validate(state) is False


# -------------------------------------------------------------------
# validate: H_REFINE
# -------------------------------------------------------------------


class TestValidateHRefine:
    """H_REFINE should return True when element exists."""

    def test_h_refine_valid(self) -> None:
        state = _make_state()
        eid = state.mesh.element_ids[0]
        action = Action(
            element_id=eid,
            action_type=ActionType.H_REFINE,
        )
        assert action.validate(state) is True


# -------------------------------------------------------------------
# __hash__ and __eq__
# -------------------------------------------------------------------


class TestHashAndEquality:
    """Tests for Action __hash__ and __eq__ methods."""

    def test_equal_actions(self) -> None:
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        a2 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        assert a1 == a2

    def test_different_action_type_not_equal(self) -> None:
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        a2 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.P_REFINE,
        )
        assert a1 != a2

    def test_different_element_id_not_equal(self) -> None:
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        a2 = Action(
            element_id=ElementID("e1"),
            action_type=ActionType.H_REFINE,
        )
        assert a1 != a2

    def test_hash_consistency(self) -> None:
        """Equal actions have the same hash."""
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        a2 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        assert hash(a1) == hash(a2)

    def test_different_actions_different_hash(self) -> None:
        """Different actions typically have different hashes."""
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        a2 = Action(
            element_id=ElementID("e1"),
            action_type=ActionType.P_REFINE,
        )
        # Not guaranteed by contract, but very likely for distinct inputs
        assert hash(a1) != hash(a2)

    def test_action_as_dict_key(self) -> None:
        a = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        d = {a: "value"}
        assert d[a] == "value"

    def test_action_in_set(self) -> None:
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        a2 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        s = {a1, a2}
        assert len(s) == 1

    def test_eq_with_non_action_returns_not_implemented(self) -> None:
        a = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.H_REFINE,
        )
        assert a.__eq__("not_an_action") is NotImplemented

    def test_params_ignored_in_equality(self) -> None:
        """Equality only checks element_id and action_type, not params."""
        a1 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.SWAP_BASIS,
            params={"basis_family": "lagrange"},
        )
        a2 = Action(
            element_id=ElementID("e0"),
            action_type=ActionType.SWAP_BASIS,
            params={"basis_family": "legendre"},
        )
        assert a1 == a2
