"""Tests for action masking."""
from __future__ import annotations

from src.alphagalerkin.core.config import EnvironmentConfig
from src.alphagalerkin.core.types import ActionType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker


class TestActionMasking:
    """Tests for the ActionMasker filtering logic."""

    def test_budget_exceeded_blocks_refinement(
        self, initial_state: DiscretizationState,
    ) -> None:
        """When DOF is at budget, refinement is blocked."""
        config = EnvironmentConfig(
            max_dof=initial_state.dof_count,
        )
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)
        refine_actions = [
            a for a in actions
            if a.action_type == ActionType.H_REFINE
        ]
        assert len(refine_actions) == 0

    def test_min_element_size_blocks_h_refine(
        self, initial_state: DiscretizationState,
    ) -> None:
        """When min_element_size is huge, h-refine blocked."""
        config = EnvironmentConfig(min_element_size=1e10)
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)
        h_refine = [
            a for a in actions
            if a.action_type == ActionType.H_REFINE
        ]
        assert len(h_refine) == 0

    def test_max_poly_order_blocks_p_refine(
        self, initial_state: DiscretizationState,
    ) -> None:
        """At max poly order, p-refine is blocked."""
        config = EnvironmentConfig(max_polynomial_order=1)
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)
        p_refine = [
            a for a in actions
            if a.action_type == ActionType.P_REFINE
        ]
        assert len(p_refine) == 0

    def test_coarsen_blocked_on_initial_mesh(
        self, initial_state: DiscretizationState,
    ) -> None:
        """Initial mesh elements have level=0, so h-coarsen is not available."""
        config = EnvironmentConfig()
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)
        coarsen = [
            a for a in actions
            if a.action_type == ActionType.H_COARSEN
        ]
        assert len(coarsen) == 0

    def test_noop_always_available(
        self, initial_state: DiscretizationState,
    ) -> None:
        """NO_OP should be available even at budget limit."""
        config = EnvironmentConfig(max_dof=10)
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)
        noop = [
            a for a in actions
            if a.action_type == ActionType.NO_OP
        ]
        assert len(noop) >= 1

    def test_p_coarsen_blocked_at_order_one(
        self, initial_state: DiscretizationState,
    ) -> None:
        """P-coarsen should not appear when poly order is 1."""
        config = EnvironmentConfig()
        masker = ActionMasker(config)
        actions = masker.valid_actions(initial_state)
        p_coarsen = [
            a for a in actions
            if a.action_type == ActionType.P_COARSEN
        ]
        assert len(p_coarsen) == 0


class TestPriorMasking:
    """Tests for prior masking / renormalization."""

    def test_mask_priors_removes_invalid(
        self, initial_state: DiscretizationState,
    ) -> None:
        config = EnvironmentConfig(max_dof=10)
        masker = ActionMasker(config)
        # Create a prior dict with an h-refine action
        eid = initial_state.mesh.element_ids[0]
        priors = {
            Action(eid, ActionType.H_REFINE, {}): 0.8,
            Action(eid, ActionType.NO_OP, {}): 0.2,
        }
        masked = masker.mask_priors(priors, initial_state)
        for action in masked:
            assert action.action_type != ActionType.H_REFINE

    def test_mask_priors_sums_to_one(
        self, initial_state: DiscretizationState,
    ) -> None:
        config = EnvironmentConfig()
        masker = ActionMasker(config)
        eid = initial_state.mesh.element_ids[0]
        priors = {
            Action(eid, ActionType.H_REFINE, {}): 0.5,
            Action(eid, ActionType.P_REFINE, {}): 0.3,
            Action(eid, ActionType.NO_OP, {}): 0.2,
        }
        masked = masker.mask_priors(priors, initial_state)
        if masked:
            total = sum(masked.values())
            assert abs(total - 1.0) < 1e-8
