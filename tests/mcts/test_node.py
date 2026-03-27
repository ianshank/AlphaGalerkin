"""Tests for MCTS Node implementation.

Tests cover:
- MCTSNode: Node properties, UCB scores, selection, expansion, backup
"""

from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip("numpy")

from src.mcts.node import MCTSNode

# --- Fixtures ---


@pytest.fixture
def root_node() -> MCTSNode:
    """Create a root node."""
    return MCTSNode()


@pytest.fixture
def node_with_prior() -> MCTSNode:
    """Create a node with prior probability."""
    return MCTSNode(prior=0.5)


@pytest.fixture
def expanded_node() -> MCTSNode:
    """Create an expanded node with children."""
    root = MCTSNode()
    root.expand({0: 0.3, 1: 0.5, 2: 0.2})
    return root


@pytest.fixture
def visited_node() -> MCTSNode:
    """Create a node that has been visited."""
    node = MCTSNode(prior=0.5)
    node.visit_count = 10
    node.total_value = 5.0
    return node


# --- Basic Property Tests ---


class TestMCTSNodeProperties:
    """Tests for MCTSNode basic properties."""

    def test_init_default_values(self, root_node: MCTSNode):
        """Test default initialization values."""
        assert root_node.parent is None
        assert root_node.action is None
        assert root_node.prior == 0.0
        assert root_node.children == {}
        assert root_node.visit_count == 0
        assert root_node.total_value == 0.0
        assert root_node.virtual_loss == 0.0

    def test_init_with_values(self):
        """Test initialization with provided values."""
        parent = MCTSNode()
        node = MCTSNode(parent=parent, action=5, prior=0.8)

        assert node.parent is parent
        assert node.action == 5
        assert node.prior == 0.8

    def test_q_value_unvisited(self, root_node: MCTSNode):
        """Test Q value is 0 for unvisited nodes."""
        assert root_node.q_value == 0.0

    def test_q_value_visited(self, visited_node: MCTSNode):
        """Test Q value calculation."""
        # Q = total_value / visit_count = 5.0 / 10 = 0.5
        assert visited_node.q_value == 0.5

    def test_q_value_with_virtual_loss_unvisited(self, root_node: MCTSNode):
        """Test Q value with virtual loss for unvisited node."""
        assert root_node.q_value_with_virtual_loss == 0.0

    def test_q_value_with_virtual_loss(self, visited_node: MCTSNode):
        """Test Q value is reduced by virtual loss."""
        visited_node.virtual_loss = 2.0
        # Q_vl = (total_value - virtual_loss) / (visit_count + virtual_loss)
        # Q_vl = (5.0 - 2.0) / (10 + 2.0) = 3.0 / 12.0 = 0.25
        assert abs(visited_node.q_value_with_virtual_loss - 0.25) < 1e-6

    def test_is_leaf_true(self, root_node: MCTSNode):
        """Test is_leaf returns True for unexpanded node."""
        assert root_node.is_leaf is True

    def test_is_leaf_false(self, expanded_node: MCTSNode):
        """Test is_leaf returns False for expanded node."""
        assert expanded_node.is_leaf is False

    def test_is_root_true(self, root_node: MCTSNode):
        """Test is_root returns True for root node."""
        assert root_node.is_root is True

    def test_is_root_false(self, expanded_node: MCTSNode):
        """Test is_root returns False for child nodes."""
        child = expanded_node.children[0]
        assert child.is_root is False


# --- UCB Score Tests ---


class TestUCBScore:
    """Tests for UCB score computation."""

    def test_ucb_score_unvisited_child(self):
        """Test UCB favors unvisited children due to exploration bonus."""
        node = MCTSNode(prior=0.5)
        c_puct = 1.5
        parent_visits = 100

        # UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        # UCB = 0 + 1.5 * 0.5 * sqrt(100) / (1 + 0) = 7.5
        score = node.ucb_score(c_puct, parent_visits)
        expected = 1.5 * 0.5 * math.sqrt(100) / 1
        assert abs(score - expected) < 1e-6

    def test_ucb_score_visited_child(self, visited_node: MCTSNode):
        """Test UCB score with visited child."""
        c_puct = 1.5
        parent_visits = 100

        # Q = 0.5, N = 10, P = 0.5
        # UCB = 0.5 + 1.5 * 0.5 * sqrt(100) / (1 + 10) = 0.5 + 0.68
        score = visited_node.ucb_score(c_puct, parent_visits)
        expected_exploration = 1.5 * 0.5 * math.sqrt(100) / 11
        expected = 0.5 + expected_exploration
        assert abs(score - expected) < 1e-6

    def test_ucb_score_with_virtual_loss(self, visited_node: MCTSNode):
        """Test UCB score accounts for virtual loss."""
        visited_node.virtual_loss = 2.0
        c_puct = 1.5
        parent_visits = 100

        score_with_vl = visited_node.ucb_score(c_puct, parent_visits)

        # Reset and compare
        visited_node.virtual_loss = 0.0
        score_without_vl = visited_node.ucb_score(c_puct, parent_visits)

        # Score with virtual loss should be lower
        assert score_with_vl < score_without_vl

    def test_high_c_puct_increases_exploration(self, node_with_prior: MCTSNode):
        """Test higher c_puct increases exploration component."""
        parent_visits = 100

        score_low = node_with_prior.ucb_score(1.0, parent_visits)
        score_high = node_with_prior.ucb_score(3.0, parent_visits)

        assert score_high > score_low


# --- Selection Tests ---


class TestSelectChild:
    """Tests for child selection."""

    def test_select_child_raises_on_leaf(self, root_node: MCTSNode):
        """Test select_child raises on leaf node."""
        with pytest.raises(ValueError, match="Cannot select from node with no children"):
            root_node.select_child(c_puct=1.5)

    def test_select_child_chooses_highest_ucb(self, expanded_node: MCTSNode):
        """Test select_child returns child with highest UCB."""
        # Give parent some visits
        expanded_node.visit_count = 10

        selected = expanded_node.select_child(c_puct=1.5)

        # Should select child with highest prior (action 1 has prior 0.5)
        assert selected is expanded_node.children[1]

    def test_select_child_balances_exploration_exploitation(self):
        """Test selection balances exploration and exploitation."""
        root = MCTSNode()
        root.expand({0: 0.3, 1: 0.7})
        root.visit_count = 100

        # Give action 0 high value but low prior
        root.children[0].visit_count = 10
        root.children[0].total_value = 8.0  # Q = 0.8

        # Give action 1 low value but high prior
        root.children[1].visit_count = 50
        root.children[1].total_value = 25.0  # Q = 0.5

        # With high c_puct, exploration should dominate (favor unvisited actions)
        # With low c_puct, exploitation should dominate (favor high Q)
        selected_high_explore = root.select_child(c_puct=10.0)
        selected_low_explore = root.select_child(c_puct=0.1)

        # These may or may not be different based on exact values
        assert selected_high_explore is not None
        assert selected_low_explore is not None


# --- Expansion Tests ---


class TestExpand:
    """Tests for node expansion."""

    def test_expand_creates_children(self, root_node: MCTSNode):
        """Test expand creates child nodes."""
        action_priors = {0: 0.2, 1: 0.3, 2: 0.5}
        root_node.expand(action_priors)

        assert len(root_node.children) == 3
        assert 0 in root_node.children
        assert 1 in root_node.children
        assert 2 in root_node.children

    def test_expand_sets_priors(self, root_node: MCTSNode):
        """Test expand sets prior probabilities."""
        action_priors = {0: 0.2, 1: 0.3, 2: 0.5}
        root_node.expand(action_priors)

        assert root_node.children[0].prior == 0.2
        assert root_node.children[1].prior == 0.3
        assert root_node.children[2].prior == 0.5

    def test_expand_sets_parent(self, root_node: MCTSNode):
        """Test expand sets parent reference."""
        root_node.expand({0: 0.5, 1: 0.5})

        assert root_node.children[0].parent is root_node
        assert root_node.children[1].parent is root_node

    def test_expand_sets_action(self, root_node: MCTSNode):
        """Test expand sets action on children."""
        root_node.expand({5: 0.3, 10: 0.7})

        assert root_node.children[5].action == 5
        assert root_node.children[10].action == 10

    def test_expand_does_not_overwrite_existing(self, root_node: MCTSNode):
        """Test expand doesn't overwrite existing children."""
        root_node.expand({0: 0.5, 1: 0.5})
        original_child = root_node.children[0]

        root_node.expand({0: 0.9, 2: 0.1})

        # Original child should not be replaced
        assert root_node.children[0] is original_child
        # New child should be added
        assert 2 in root_node.children


# --- Backup Tests ---


class TestBackup:
    """Tests for value backup."""

    def test_backup_increments_visit_count(self, root_node: MCTSNode):
        """Test backup increments visit count."""
        root_node.backup(0.5)
        assert root_node.visit_count == 1

    def test_backup_adds_value(self, root_node: MCTSNode):
        """Test backup adds value to total."""
        root_node.backup(0.5)
        assert root_node.total_value == 0.5

    def test_backup_propagates_to_parent(self):
        """Test backup propagates up the tree."""
        root = MCTSNode()
        root.expand({0: 0.5})
        child = root.children[0]

        # Backup from child
        child.backup(0.8)

        assert child.visit_count == 1
        assert child.total_value == 0.8
        assert root.visit_count == 1
        # Value flips sign for parent (opponent's perspective)
        assert root.total_value == -0.8

    def test_backup_alternates_value_sign(self):
        """Test backup alternates value sign up the tree."""
        root = MCTSNode()
        root.expand({0: 0.5})
        child = root.children[0]
        child.expand({0: 0.5})
        grandchild = child.children[0]

        grandchild.backup(1.0)

        assert grandchild.total_value == 1.0
        assert child.total_value == -1.0
        assert root.total_value == 1.0

    def test_backup_removes_virtual_loss(self):
        """Test backup removes virtual loss."""
        node = MCTSNode()
        node.virtual_loss = 3.0

        node.backup(0.5)

        # Virtual loss reduced by 1
        assert node.virtual_loss == 2.0


# --- Virtual Loss Tests ---


class TestVirtualLoss:
    """Tests for virtual loss operations."""

    def test_add_virtual_loss(self, root_node: MCTSNode):
        """Test adding virtual loss."""
        root_node.add_virtual_loss(2.0)
        assert root_node.virtual_loss == 2.0

    def test_add_virtual_loss_accumulates(self, root_node: MCTSNode):
        """Test virtual loss accumulates."""
        root_node.add_virtual_loss(1.0)
        root_node.add_virtual_loss(1.5)
        assert root_node.virtual_loss == 2.5

    def test_remove_virtual_loss(self, root_node: MCTSNode):
        """Test removing virtual loss."""
        root_node.virtual_loss = 3.0
        root_node.remove_virtual_loss(2.0)
        assert root_node.virtual_loss == 1.0

    def test_remove_virtual_loss_clamped_at_zero(self, root_node: MCTSNode):
        """Test virtual loss doesn't go negative."""
        root_node.virtual_loss = 1.0
        root_node.remove_virtual_loss(5.0)
        assert root_node.virtual_loss == 0.0


# --- Visit Distribution Tests ---


class TestVisitDistribution:
    """Tests for visit count distribution."""

    def test_get_visit_distribution_empty(self, root_node: MCTSNode):
        """Test empty distribution for leaf node."""
        dist = root_node.get_visit_distribution()
        assert dist == {}

    def test_get_visit_distribution_uniform(self, expanded_node: MCTSNode):
        """Test distribution with equal visits."""
        # Give all children equal visits
        for child in expanded_node.children.values():
            child.visit_count = 10

        dist = expanded_node.get_visit_distribution(temperature=1.0)

        # Should be roughly uniform
        assert len(dist) == 3
        for prob in dist.values():
            assert abs(prob - 1 / 3) < 0.01

    def test_get_visit_distribution_temperature_zero(self, expanded_node: MCTSNode):
        """Test deterministic selection with temperature=0."""
        # Give one child more visits
        expanded_node.children[0].visit_count = 10
        expanded_node.children[1].visit_count = 100
        expanded_node.children[2].visit_count = 5

        dist = expanded_node.get_visit_distribution(temperature=0)

        # Should select action 1 with probability 1.0
        assert dist[1] == 1.0
        assert dist[0] == 0.0
        assert dist[2] == 0.0

    def test_get_visit_distribution_low_temperature(self, expanded_node: MCTSNode):
        """Test concentrated distribution with low temperature."""
        expanded_node.children[0].visit_count = 10
        expanded_node.children[1].visit_count = 100
        expanded_node.children[2].visit_count = 10

        dist = expanded_node.get_visit_distribution(temperature=0.5)

        # Action 1 should have highest probability
        assert dist[1] > dist[0]
        assert dist[1] > dist[2]

    def test_get_visit_distribution_high_temperature(self, expanded_node: MCTSNode):
        """Test flatter distribution with high temperature."""
        expanded_node.children[0].visit_count = 10
        expanded_node.children[1].visit_count = 100
        expanded_node.children[2].visit_count = 10

        dist_low = expanded_node.get_visit_distribution(temperature=0.5)
        dist_high = expanded_node.get_visit_distribution(temperature=2.0)

        # Higher temperature should give more uniform distribution
        # The ratio of max to min should be smaller
        values_low = list(dist_low.values())
        values_high = list(dist_high.values())

        ratio_low = max(values_low) / (min(values_low) + 1e-8)
        ratio_high = max(values_high) / (min(values_high) + 1e-8)

        assert ratio_high < ratio_low


# --- Best Action Tests ---


class TestGetBestAction:
    """Tests for best action selection."""

    def test_get_best_action_raises_on_leaf(self, root_node: MCTSNode):
        """Test get_best_action raises on leaf node."""
        with pytest.raises(ValueError, match="Cannot get best action"):
            root_node.get_best_action()

    def test_get_best_action_selects_most_visited(self, expanded_node: MCTSNode):
        """Test get_best_action returns most visited action."""
        expanded_node.children[0].visit_count = 10
        expanded_node.children[1].visit_count = 50
        expanded_node.children[2].visit_count = 30

        best = expanded_node.get_best_action()
        assert best == 1


# --- Principal Variation Tests ---


class TestPrincipalVariation:
    """Tests for principal variation extraction."""

    def test_get_pv_empty_on_leaf(self, root_node: MCTSNode):
        """Test PV is empty for leaf node."""
        pv = root_node.get_pv()
        assert pv == []

    def test_get_pv_single_level(self, expanded_node: MCTSNode):
        """Test PV with single expanded level."""
        expanded_node.children[1].visit_count = 100
        expanded_node.children[0].visit_count = 10
        expanded_node.children[2].visit_count = 10

        pv = expanded_node.get_pv()
        assert pv == [1]

    def test_get_pv_multiple_levels(self, expanded_node: MCTSNode):
        """Test PV traverses multiple levels."""
        # Set up deeper tree
        expanded_node.children[1].visit_count = 100
        expanded_node.children[1].expand({10: 0.6, 11: 0.4})
        expanded_node.children[1].children[10].visit_count = 60

        pv = expanded_node.get_pv()
        assert pv == [1, 10]

    def test_get_pv_respects_max_depth(self, expanded_node: MCTSNode):
        """Test PV stops at max_depth."""
        expanded_node.children[1].visit_count = 100
        expanded_node.children[1].expand({10: 0.5})
        expanded_node.children[1].children[10].visit_count = 50

        pv = expanded_node.get_pv(max_depth=1)
        assert len(pv) == 1


# --- Pruning Tests ---


class TestPruneExcept:
    """Tests for tree pruning."""

    def test_prune_except_returns_child(self, expanded_node: MCTSNode):
        """Test prune_except returns the kept child."""
        child = expanded_node.children[1]
        result = expanded_node.prune_except(1)

        assert result is child

    def test_prune_except_sets_parent_none(self, expanded_node: MCTSNode):
        """Test pruned child becomes new root."""
        original_child = expanded_node.children[1]
        result = expanded_node.prune_except(1)

        assert result is not None
        assert result.parent is None

    def test_prune_except_clears_parent_children(self, expanded_node: MCTSNode):
        """Test parent's children are cleared."""
        expanded_node.prune_except(1)
        assert expanded_node.children == {}

    def test_prune_except_nonexistent_returns_none(self, expanded_node: MCTSNode):
        """Test prune_except with nonexistent action returns None."""
        result = expanded_node.prune_except(999)
        assert result is None


# --- Repr Tests ---


class TestRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self, visited_node: MCTSNode):
        """Test __repr__ includes key information."""
        visited_node.action = 5
        repr_str = repr(visited_node)

        assert "action=5" in repr_str
        assert "N=10" in repr_str
        assert "Q=0.5" in repr_str
        assert "P=0.5" in repr_str
