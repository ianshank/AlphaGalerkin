"""Tests for MCTS node."""
from __future__ import annotations

import pytest

from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.node import MCTSNode


class TestMCTSNode:
    """Core MCTSNode behaviour."""

    def test_new_node_has_zero_visits(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.is_leaf

    def test_ucb_score_unexpanded_is_infinite(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        score = node.ucb_score(cpuct=1.4, parent_visits=10)
        assert score == float("inf")

    def test_ucb_score_increases_with_prior(
        self, initial_state: DiscretizationState,
    ) -> None:
        node_low = MCTSNode(
            state=initial_state, prior=0.1,
        )
        node_high = MCTSNode(
            state=initial_state, prior=0.9,
        )
        node_low.backup(0.5)
        node_high.backup(0.5)
        assert (
            node_high.ucb_score(1.4, 10)
            > node_low.ucb_score(1.4, 10)
        )

    def test_backup_increments_visit_count(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        node.backup(0.7)
        assert node.visit_count == 1
        node.backup(0.3)
        assert node.visit_count == 2

    def test_backup_accumulates_value(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        node.backup(0.6)
        node.backup(0.8)
        assert abs(node.mean_value - 0.7) < 1e-10

    def test_expand_creates_children(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        priors = {
            a: 1.0 / len(valid_actions)
            for a in valid_actions
        }
        node.expand(priors)
        assert len(node.children) == len(valid_actions)
        assert not node.is_leaf

    def test_expand_children_have_correct_priors(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        priors = {
            valid_actions[0]: 0.7,
            valid_actions[1]: 0.3,
        }
        node.expand(priors)
        child = node.children[valid_actions[0]]
        assert abs(child.prior - 0.7) < 1e-10

    def test_node_creates_child_state_on_expansion(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        priors = {valid_actions[0]: 1.0}
        node.expand(priors)
        child = node.children[valid_actions[0]]
        assert child.state is not node.state

    def test_select_best_child(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        priors = dict.fromkeys(valid_actions[:2], 0.5)
        node.expand(priors)
        node.backup(1.0)
        best = node.select_best_child(cpuct=1.4)
        assert best in node.children.values()

    def test_parent_reference_set_on_expand(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        priors = {valid_actions[0]: 1.0}
        node.expand(priors)
        child = node.children[valid_actions[0]]
        assert child.parent is node

    def test_action_from_parent_set_on_expand(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        action = valid_actions[0]
        priors = {action: 1.0}
        node.expand(priors)
        child = node.children[action]
        assert child.action_from_parent == action


class TestMCTSNodeEdgeCases:
    """Edge-case tests for MCTSNode."""

    def test_backup_negative_value(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        node.backup(-0.5)
        assert node.visit_count == 1
        assert abs(node.mean_value - (-0.5)) < 1e-10

    def test_expand_empty_priors_raises(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        with pytest.raises(ValueError, match="at least one"):
            node.expand({})

    def test_expand_twice_raises(
        self,
        initial_state: DiscretizationState,
        valid_actions: list[Action],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        priors = {valid_actions[0]: 1.0}
        node.expand(priors)
        with pytest.raises(RuntimeError, match="already expanded"):
            node.expand(priors)

    def test_select_best_child_no_children_raises(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=1.0)
        with pytest.raises(RuntimeError, match="No children"):
            node.select_best_child(cpuct=1.4)

    def test_mean_value_zero_when_unvisited(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        assert node.mean_value == 0.0

    def test_is_terminal_setter(
        self, initial_state: DiscretizationState,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        assert not node.is_terminal
        node.is_terminal = True
        assert node.is_terminal
