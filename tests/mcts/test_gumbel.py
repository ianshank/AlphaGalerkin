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
    _gumbel_mixed_value,
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

    def test_compute_completed_q_unvisited_uses_mixed_value_estimate_when_given(
        self, default_node: GumbelNode
    ):
        """An unvisited node's completed Q is the supplied v_mix, not a flat 0.0.

        This is the exact knob ``GumbelMCTSConfig.use_mixed_value`` gates at
        its call sites in ``GumbelMCTS`` -- see ``TestGumbelMixedValue`` and
        ``test_gumbel_integration.py::TestSequentialHalvingConfigKnobs`` for
        proof that the config flag actually changes search behaviour.
        """
        q = default_node.compute_completed_q(c_visit=50.0, c_scale=1.0, mixed_value_estimate=0.42)
        assert q == 0.42

        # A visited node ignores mixed_value_estimate entirely -- it always
        # uses its own value + sigma * prior.
        visited = GumbelNode(prior=0.5)
        visited.visit_count = 10
        visited.value_sum = 5.0
        q_with = visited.compute_completed_q(c_visit=50.0, c_scale=1.0, mixed_value_estimate=99.0)
        q_without = visited.compute_completed_q(c_visit=50.0, c_scale=1.0)
        assert q_with == q_without

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


# --- _gumbel_mixed_value Tests ---
#
# ``_gumbel_mixed_value`` is the v_mix estimator that
# ``GumbelMCTSConfig.use_mixed_value`` gates. Before this fix nothing in
# ``src/mcts/gumbel.py`` computed it at all, so the flag was inert
# (docs/CODE_HYGIENE_AUDIT.md P2: "the *defining* feature of Gumbel
# AlphaZero -- setting it changes nothing").


class TestGumbelMixedValue:
    """Tests for the ``_gumbel_mixed_value`` value-mixing estimator."""

    def test_no_children_returns_raw_value(self):
        root = GumbelNode()
        assert _gumbel_mixed_value(root, raw_value=-2.0) == -2.0

    def test_no_visited_children_returns_raw_value(self):
        """With nothing visited yet there is nothing to mix in."""
        root = GumbelNode()
        root.children[0] = GumbelNode(prior=0.5)  # visit_count == 0
        root.children[1] = GumbelNode(prior=0.5)  # visit_count == 0
        assert _gumbel_mixed_value(root, raw_value=3.5) == 3.5

    def test_matches_hand_computed_formula(self):
        root = GumbelNode()
        a = GumbelNode(prior=0.3)
        a.visit_count = 2
        a.value_sum = 1.0  # value == 0.5
        b = GumbelNode(prior=0.7)
        b.visit_count = 1
        b.value_sum = -2.0  # value == -2.0
        root.children[0] = a
        root.children[1] = b

        raw_value = 1.0
        total_visits = 3  # 2 + 1
        weighted_q = (0.3 * 0.5 + 0.7 * -2.0) / (0.3 + 0.7)
        expected = (raw_value + total_visits * weighted_q) / (total_visits + 1)

        assert _gumbel_mixed_value(root, raw_value) == pytest.approx(expected)

    def test_unvisited_siblings_do_not_affect_the_mix(self):
        """Only visited children contribute to weighted_q's numerator/denominator."""
        root = GumbelNode()
        visited = GumbelNode(prior=0.4)
        visited.visit_count = 5
        visited.value_sum = 5.0  # value == 1.0
        unvisited = GumbelNode(prior=0.6)  # visit_count == 0, must be excluded
        root.children[0] = visited
        root.children[1] = unvisited

        raw_value = 0.0
        expected = (raw_value + 5 * 1.0) / (5 + 1)
        assert _gumbel_mixed_value(root, raw_value) == pytest.approx(expected)

    def test_zero_total_prior_among_visited_falls_back_to_raw_value(self):
        """Guards the division-by-~0 branch when every visited prior is ~0."""
        root = GumbelNode()
        visited = GumbelNode(prior=0.0)
        visited.visit_count = 1
        visited.value_sum = 99.0
        root.children[0] = visited

        assert _gumbel_mixed_value(root, raw_value=7.0) == 7.0

    def test_more_visits_pulls_the_mix_toward_weighted_q(self):
        """As a visited child's visit count grows, v_mix moves toward weighted_q.

        v_mix should move away from raw_value and toward the visit-weighted
        mean Q -- the correct limiting behaviour of an estimator that is a
        weighted average of the two, with raw_value acting as a
        single-pseudo-visit prior.
        """
        raw_value = 10.0

        def _mix(n_visits: int) -> float:
            root = GumbelNode()
            child = GumbelNode(prior=1.0)
            child.visit_count = n_visits
            child.value_sum = 0.0  # value == 0.0 regardless of visit count
            root.children[0] = child
            return _gumbel_mixed_value(root, raw_value)

        v_low = _mix(1)
        v_high = _mix(1000)
        assert abs(v_high - 0.0) < abs(v_low - 0.0)
        assert v_high == pytest.approx(0.0, abs=1e-2)


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


# --- Named-constant site binding (static guard) ---


class TestGumbelConstantSiteBinding:
    """Static guard that the two 1e-8 constants sit at the sites they name.

    ``GUMBEL_NORMALIZATION_EPSILON`` (inert division guard) and
    ``GUMBEL_LOG_PRIOR_FLOOR`` (an algorithmic knob that shifts action
    selection) share the same value today, so no purely numeric assertion can
    detect a swap. These tests parse ``src/mcts/gumbel.py`` and check which
    constant appears at which kind of expression, so swapping them -- or
    reintroducing a bare ``1e-8`` -- fails loudly.
    """

    NORMALIZATION_NAME = "GUMBEL_NORMALIZATION_EPSILON"
    LOG_PRIOR_NAME = "GUMBEL_LOG_PRIOR_FLOOR"
    EXPECTED_NORMALIZATION_SITES = 3
    EXPECTED_LOG_SITES = 3

    @staticmethod
    def _module_tree() -> ast.Module:
        import ast
        import inspect
        from pathlib import Path

        import src.mcts.gumbel as gumbel_module

        return ast.parse(Path(inspect.getfile(gumbel_module)).read_text(encoding="utf-8"))

    @classmethod
    def _sum_normalizer_names(cls) -> list[str]:
        """Names added to a ``<expr>.sum()`` call, i.e. denominators."""
        import ast

        names: list[str] = []
        for node in ast.walk(cls._module_tree()):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
                continue
            left = node.left
            is_sum_call = (
                isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute)
                and left.func.attr == "sum"
            )
            if is_sum_call and isinstance(node.right, ast.Name):
                names.append(node.right.id)
        return names

    @classmethod
    def _log_argument_floor_names(cls) -> list[str]:
        """Names added inside an ``np.log(... + <name>)`` argument."""
        import ast

        names: list[str] = []
        for node in ast.walk(cls._module_tree()):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "log"):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                if isinstance(arg.right, ast.Name):
                    names.append(arg.right.id)
        return names

    def test_every_sum_normalizer_uses_the_normalization_constant(self) -> None:
        """All `.sum() + X` denominators bind to GUMBEL_NORMALIZATION_EPSILON."""
        names = self._sum_normalizer_names()
        assert len(names) == self.EXPECTED_NORMALIZATION_SITES
        assert set(names) == {self.NORMALIZATION_NAME}

    def test_every_log_floor_uses_the_log_prior_constant(self) -> None:
        """All `np.log(p + X)` floors bind to GUMBEL_LOG_PRIOR_FLOOR."""
        names = self._log_argument_floor_names()
        assert len(names) == self.EXPECTED_LOG_SITES
        assert set(names) == {self.LOG_PRIOR_NAME}

    def test_no_bare_epsilon_literal_remains(self) -> None:
        """1e-8 appears only in the two constant definitions, never inline."""
        import ast

        tree = self._module_tree()
        definition_names = {self.NORMALIZATION_NAME, self.LOG_PRIOR_NAME}
        inline_literals = 0
        for top in tree.body:
            is_constant_def = (
                isinstance(top, ast.AnnAssign)
                and isinstance(top.target, ast.Name)
                and top.target.id in definition_names
            )
            if is_constant_def:
                continue
            for node in ast.walk(top):
                if isinstance(node, ast.Constant) and node.value == 1e-8:
                    inline_literals += 1
        assert inline_literals == 0

    def test_both_constants_are_module_level_and_equal_today(self) -> None:
        """Both exist as separate module attributes with the documented value."""
        from src.mcts.gumbel import GUMBEL_LOG_PRIOR_FLOOR, GUMBEL_NORMALIZATION_EPSILON

        assert GUMBEL_NORMALIZATION_EPSILON == 1e-8
        assert GUMBEL_LOG_PRIOR_FLOOR == 1e-8
