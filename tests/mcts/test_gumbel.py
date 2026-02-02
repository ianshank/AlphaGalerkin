"""Tests for Gumbel MCTS implementation.

Tests cover:
- GumbelMCTSConfig: Configuration validation
- GumbelNode: Node properties and operations
- GumbelSearchResult: Result structure
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
pydantic = pytest.importorskip("pydantic")

from src.mcts.gumbel import (
    GumbelMCTSConfig,
    GumbelNode,
    GumbelSearchResult,
)

# --- GumbelMCTSConfig Tests ---


class TestGumbelMCTSConfig:
    """Tests for GumbelMCTSConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GumbelMCTSConfig()

        assert config.n_simulations == 800
        assert config.max_num_considered_actions == 16
        assert config.gumbel_scale == 1.0
        assert config.c_visit == 50.0
        assert config.c_scale == 1.0
        assert config.use_mixed_value is True
        assert config.discount == 1.0
        assert config.batch_size == 8

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = GumbelMCTSConfig(
            n_simulations=100,
            max_num_considered_actions=8,
            gumbel_scale=0.5,
            c_visit=25.0,
            batch_size=16,
        )

        assert config.n_simulations == 100
        assert config.max_num_considered_actions == 8
        assert config.gumbel_scale == 0.5
        assert config.c_visit == 25.0
        assert config.batch_size == 16

    def test_n_simulations_must_be_positive(self):
        """Test n_simulations must be >= 1."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(n_simulations=0)

    def test_max_num_considered_actions_must_be_positive(self):
        """Test max_num_considered_actions must be >= 1."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(max_num_considered_actions=0)

    def test_gumbel_scale_must_be_positive(self):
        """Test gumbel_scale must be > 0."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(gumbel_scale=0)

    def test_c_visit_must_be_positive(self):
        """Test c_visit must be > 0."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(c_visit=0)

    def test_c_scale_must_be_positive(self):
        """Test c_scale must be > 0."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(c_scale=-1)

    def test_discount_range(self):
        """Test discount must be in (0, 1]."""
        # Valid discount
        config = GumbelMCTSConfig(discount=0.99)
        assert config.discount == 0.99

        # Invalid: > 1
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(discount=1.5)

        # Invalid: <= 0
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(discount=0)

    def test_root_dirichlet_alpha_must_be_positive(self):
        """Test root_dirichlet_alpha must be > 0."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(root_dirichlet_alpha=0)

    def test_root_exploration_fraction_range(self):
        """Test root_exploration_fraction must be in [0, 1]."""
        # Valid
        config = GumbelMCTSConfig(root_exploration_fraction=0.5)
        assert config.root_exploration_fraction == 0.5

        # Valid boundaries
        GumbelMCTSConfig(root_exploration_fraction=0.0)
        GumbelMCTSConfig(root_exploration_fraction=1.0)

        # Invalid: > 1
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(root_exploration_fraction=1.5)

    def test_batch_size_must_be_positive(self):
        """Test batch_size must be >= 1."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(batch_size=0)

    def test_extra_fields_forbidden(self):
        """Test extra fields raise error."""
        with pytest.raises(pydantic.ValidationError):
            GumbelMCTSConfig(unknown_field=123)

    def test_assignment_validation(self):
        """Test validation on assignment."""
        config = GumbelMCTSConfig()

        # Valid assignment
        config.n_simulations = 500
        assert config.n_simulations == 500

        # Invalid assignment
        with pytest.raises(pydantic.ValidationError):
            config.n_simulations = -1


# --- GumbelNode Tests ---


class TestGumbelNode:
    """Tests for GumbelNode dataclass."""

    @pytest.fixture
    def default_node(self) -> GumbelNode:
        """Create a default node."""
        return GumbelNode()

    @pytest.fixture
    def visited_node(self) -> GumbelNode:
        """Create a visited node."""
        node = GumbelNode(prior=0.5, gumbel=1.5)
        node.visit_count = 10
        node.value_sum = 5.0
        return node

    def test_default_initialization(self, default_node: GumbelNode):
        """Test default node initialization."""
        assert default_node.state is None
        assert default_node.prior == 0.0
        assert default_node.gumbel == 0.0
        assert default_node.visit_count == 0
        assert default_node.value_sum == 0.0
        assert default_node.children == {}
        assert default_node._is_expanded is False
        assert default_node._terminal_value is None

    def test_custom_initialization(self):
        """Test node with custom values."""
        node = GumbelNode(prior=0.7, gumbel=2.5)

        assert node.prior == 0.7
        assert node.gumbel == 2.5

    def test_value_property_unvisited(self, default_node: GumbelNode):
        """Test value property for unvisited node."""
        assert default_node.value == 0.0

    def test_value_property_visited(self, visited_node: GumbelNode):
        """Test value property for visited node."""
        # value = value_sum / visit_count = 5.0 / 10 = 0.5
        assert visited_node.value == 0.5

    def test_is_expanded_property(self, default_node: GumbelNode):
        """Test is_expanded property."""
        assert default_node.is_expanded is False

        default_node._is_expanded = True
        assert default_node.is_expanded is True

    def test_is_terminal_property(self, default_node: GumbelNode):
        """Test is_terminal property."""
        assert default_node.is_terminal is False

        default_node._terminal_value = 1.0
        assert default_node.is_terminal is True

    def test_compute_completed_q_unvisited(self, default_node: GumbelNode):
        """Test completed Q for unvisited node."""
        q = default_node.compute_completed_q(c_visit=50.0, c_scale=1.0)
        assert q == 0.0

    def test_compute_completed_q_visited(self, visited_node: GumbelNode):
        """Test completed Q for visited node."""
        c_visit = 50.0
        c_scale = 1.0

        q = visited_node.compute_completed_q(c_visit, c_scale)

        # Expected: value + sigma * prior
        # sigma = c_scale * sqrt(c_visit) / (c_visit + visit_count)
        # sigma = 1.0 * sqrt(50) / (50 + 10) = 7.07 / 60 = 0.118
        # q = 0.5 + 0.118 * 0.5 = 0.559
        expected_sigma = c_scale * numpy.sqrt(c_visit) / (c_visit + 10)
        expected_q = 0.5 + expected_sigma * 0.5

        assert abs(q - expected_q) < 1e-4

    def test_compute_completed_q_c_visit_effect(self, visited_node: GumbelNode):
        """Test c_visit affects completed Q."""
        q_low = visited_node.compute_completed_q(c_visit=10.0, c_scale=1.0)
        q_high = visited_node.compute_completed_q(c_visit=100.0, c_scale=1.0)

        # Higher c_visit should increase exploration bonus
        # But the formula might cause different effects
        assert q_low != q_high

    def test_compute_completed_q_c_scale_effect(self, visited_node: GumbelNode):
        """Test c_scale affects completed Q."""
        q_low = visited_node.compute_completed_q(c_visit=50.0, c_scale=0.5)
        q_high = visited_node.compute_completed_q(c_visit=50.0, c_scale=2.0)

        # Higher c_scale should increase exploration bonus
        assert q_high > q_low

    def test_children_manipulation(self, default_node: GumbelNode):
        """Test adding children to node."""
        child1 = GumbelNode(prior=0.6)
        child2 = GumbelNode(prior=0.4)

        default_node.children[0] = child1
        default_node.children[1] = child2

        assert len(default_node.children) == 2
        assert default_node.children[0].prior == 0.6
        assert default_node.children[1].prior == 0.4


# --- GumbelSearchResult Tests ---


class TestGumbelSearchResult:
    """Tests for GumbelSearchResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> GumbelSearchResult:
        """Create a sample search result."""
        return GumbelSearchResult(
            action=5,
            policy=numpy.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
            value=0.6,
            root_value=0.5,
            visit_counts=numpy.array([10, 20, 30, 20, 10, 10]),
            q_values=numpy.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.4]),
            n_simulations=100,
        )

    def test_result_attributes(self, sample_result: GumbelSearchResult):
        """Test result stores all attributes."""
        assert sample_result.action == 5
        assert sample_result.value == 0.6
        assert sample_result.root_value == 0.5
        assert sample_result.n_simulations == 100

    def test_result_arrays(self, sample_result: GumbelSearchResult):
        """Test result stores numpy arrays correctly."""
        assert len(sample_result.policy) == 6
        assert len(sample_result.visit_counts) == 6
        assert len(sample_result.q_values) == 6

    def test_policy_sums_to_one(self, sample_result: GumbelSearchResult):
        """Test policy probabilities sum to approximately 1."""
        policy_sum = sample_result.policy.sum()
        assert abs(policy_sum - 1.0) < 0.01

    def test_selected_action_in_range(self, sample_result: GumbelSearchResult):
        """Test selected action is valid index."""
        assert 0 <= sample_result.action < len(sample_result.policy)


# --- Config Edge Cases ---


class TestGumbelMCTSConfigEdgeCases:
    """Edge case tests for configuration."""

    def test_minimum_valid_config(self):
        """Test minimum valid configuration."""
        config = GumbelMCTSConfig(
            n_simulations=1,
            max_num_considered_actions=1,
            batch_size=1,
        )
        assert config.n_simulations == 1

    def test_large_values_config(self):
        """Test configuration with large values."""
        config = GumbelMCTSConfig(
            n_simulations=100000,
            max_num_considered_actions=1000,
            c_visit=10000.0,
        )
        assert config.n_simulations == 100000

    def test_config_serialization(self):
        """Test configuration can be serialized and deserialized."""
        original = GumbelMCTSConfig(n_simulations=500, gumbel_scale=0.8)

        # Serialize to dict
        config_dict = original.model_dump()

        # Deserialize
        restored = GumbelMCTSConfig(**config_dict)

        assert restored.n_simulations == original.n_simulations
        assert restored.gumbel_scale == original.gumbel_scale


# --- GumbelNode Edge Cases ---


class TestGumbelNodeEdgeCases:
    """Edge case tests for GumbelNode."""

    def test_high_visit_count(self):
        """Test node with very high visit count."""
        node = GumbelNode(prior=0.5)
        node.visit_count = 1000000
        node.value_sum = 500000.0

        assert node.value == 0.5
        q = node.compute_completed_q(c_visit=50.0, c_scale=1.0)
        # Q should be close to value since sigma becomes small
        assert abs(q - 0.5) < 0.001

    def test_zero_prior(self):
        """Test node with zero prior."""
        node = GumbelNode(prior=0.0)
        node.visit_count = 10
        node.value_sum = 5.0

        # Should still compute without error
        q = node.compute_completed_q(c_visit=50.0, c_scale=1.0)
        # Q = value + 0 (since prior is 0)
        assert q == 0.5

    def test_negative_gumbel(self):
        """Test node with negative Gumbel noise."""
        node = GumbelNode(prior=0.5, gumbel=-2.5)
        assert node.gumbel == -2.5

    def test_negative_value_sum(self):
        """Test node with negative value sum."""
        node = GumbelNode(prior=0.5)
        node.visit_count = 10
        node.value_sum = -8.0

        assert node.value == -0.8
