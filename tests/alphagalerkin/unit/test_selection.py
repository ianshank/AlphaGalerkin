"""Tests for MCTS selection strategies."""
from __future__ import annotations

import math

import pytest

from src.alphagalerkin.core.types import SelectionPolicy
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.node import MCTSNode
from src.alphagalerkin.mcts.selection import (
    SELECTION_STRATEGIES,
    get_selection_fn,
    puct_score,
    rave_score,
    ucb1_score,
)


def _make_node(
    prior: float = 0.5,
    visits: int = 0,
    total_value: float = 0.0,
) -> MCTSNode:
    """Helper to create a node with specific stats."""
    mesh = MeshGraph.create_uniform_quad(
        bounds=((0.0, 1.0), (0.0, 1.0)), num_elements=(2, 2),
    )
    state = DiscretizationState.from_mesh(mesh)
    node = MCTSNode(state=state, prior=prior)
    for _ in range(visits):
        node.backup(total_value / max(visits, 1))
    return node


class TestPUCTScore:
    """Tests for PUCT selection."""

    def test_unvisited_child_returns_inf(self) -> None:
        child = _make_node(prior=0.5, visits=0)
        assert puct_score(child, cpuct=2.5, parent_visits=10) == float("inf")

    def test_score_increases_with_prior(self) -> None:
        low = _make_node(prior=0.1, visits=5, total_value=2.5)
        high = _make_node(prior=0.9, visits=5, total_value=2.5)
        assert puct_score(high, 2.5, 100) > puct_score(low, 2.5, 100)

    def test_score_finite_after_visit(self) -> None:
        child = _make_node(prior=0.5, visits=3, total_value=1.5)
        score = puct_score(child, 2.5, 50)
        assert math.isfinite(score)

    def test_exploitation_component(self) -> None:
        """Higher value -> higher score."""
        low_v = _make_node(prior=0.5, visits=10, total_value=1.0)
        high_v = _make_node(prior=0.5, visits=10, total_value=9.0)
        assert puct_score(high_v, 0.0, 100) > puct_score(low_v, 0.0, 100)

    def test_zero_cpuct_pure_exploitation(self) -> None:
        """With cpuct=0, score equals mean value."""
        child = _make_node(prior=0.5, visits=10, total_value=8.0)
        score = puct_score(child, cpuct=0.0, parent_visits=10)
        assert score == pytest.approx(0.8, rel=1e-5)


class TestUCB1Score:
    """Tests for UCB1 selection."""

    def test_unvisited_returns_inf(self) -> None:
        child = _make_node(prior=0.5, visits=0)
        assert ucb1_score(child, cpuct=2.0, parent_visits=10) == float("inf")

    def test_score_finite_after_visit(self) -> None:
        child = _make_node(prior=0.5, visits=5, total_value=2.5)
        score = ucb1_score(child, 2.0, 50)
        assert math.isfinite(score)

    def test_exploration_decreases_with_visits(self) -> None:
        few = _make_node(prior=0.5, visits=2, total_value=1.0)
        many = _make_node(prior=0.5, visits=100, total_value=50.0)
        assert ucb1_score(few, 2.0, 200) > ucb1_score(many, 2.0, 200)

    def test_ucb1_ignores_prior(self) -> None:
        """UCB1 score does not depend on the prior probability."""
        child1 = _make_node(prior=0.1, visits=5, total_value=2.5)
        child2 = _make_node(prior=0.9, visits=5, total_value=2.5)
        score1 = ucb1_score(child1, cpuct=2.0, parent_visits=10)
        score2 = ucb1_score(child2, cpuct=2.0, parent_visits=10)
        assert score1 == pytest.approx(score2, rel=1e-5)

    def test_ucb1_formula(self) -> None:
        """Verify UCB1 formula: Q + c * sqrt(ln(N_parent) / N_child)."""
        child = _make_node(visits=4, total_value=2.0)
        cpuct = 1.5
        parent_visits = 20
        score = ucb1_score(child, cpuct, parent_visits)
        expected = 0.5 + 1.5 * math.sqrt(math.log(20) / 4)
        assert score == pytest.approx(expected, rel=1e-5)


class TestRAVEScore:
    """Tests for RAVE selection."""

    def test_rave_falls_back_to_puct(self) -> None:
        child = _make_node(prior=0.5, visits=5, total_value=2.5)
        assert rave_score(child, 2.5, 50) == puct_score(child, 2.5, 50)


class TestSelectionRegistry:
    """Tests for the selection strategy registry."""

    def test_all_policies_registered(self) -> None:
        for policy in SelectionPolicy:
            assert policy in SELECTION_STRATEGIES

    def test_get_selection_fn_returns_callable(self) -> None:
        fn = get_selection_fn(SelectionPolicy.PUCT)
        assert callable(fn)

    def test_get_selection_fn_puct(self) -> None:
        fn = get_selection_fn(SelectionPolicy.PUCT)
        assert fn is puct_score

    def test_get_selection_fn_ucb1(self) -> None:
        fn = get_selection_fn(SelectionPolicy.UCB1)
        assert fn is ucb1_score

    def test_get_selection_fn_rave(self) -> None:
        fn = get_selection_fn(SelectionPolicy.RAVE)
        assert fn is rave_score
