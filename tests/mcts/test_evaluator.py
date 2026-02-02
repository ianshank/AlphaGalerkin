"""Tests for MCTS evaluators.

Tests cover:
- RandomEvaluator: Baseline random evaluation
- EvaluationResult: Named tuple for results
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from src.mcts.evaluator import EvaluationResult, RandomEvaluator

# --- Fixtures ---


@pytest.fixture
def random_evaluator() -> RandomEvaluator:
    """Create a random evaluator for 82 actions (9x9 Go + pass)."""
    return RandomEvaluator(n_actions=82)


@pytest.fixture
def sample_state() -> numpy.ndarray:
    """Create a sample game state."""
    return numpy.random.randn(17, 9, 9).astype(numpy.float32)


# --- EvaluationResult Tests ---


class TestEvaluationResult:
    """Tests for EvaluationResult named tuple."""

    def test_creation(self):
        """Test creating evaluation result."""
        policy = numpy.ones(10, dtype=numpy.float32) / 10
        result = EvaluationResult(policy=policy, value=0.5)

        assert numpy.array_equal(result.policy, policy)
        assert result.value == 0.5

    def test_tuple_unpacking(self):
        """Test result can be unpacked as tuple."""
        policy = numpy.ones(10, dtype=numpy.float32) / 10
        result = EvaluationResult(policy=policy, value=-0.3)

        unpacked_policy, unpacked_value = result

        assert numpy.array_equal(unpacked_policy, policy)
        assert unpacked_value == -0.3

    def test_attribute_access(self):
        """Test attribute access works."""
        policy = numpy.zeros(5, dtype=numpy.float32)
        policy[2] = 1.0
        result = EvaluationResult(policy=policy, value=1.0)

        assert result.policy[2] == 1.0
        assert result.value == 1.0


# --- RandomEvaluator Tests ---


class TestRandomEvaluator:
    """Tests for RandomEvaluator."""

    def test_init(self, random_evaluator: RandomEvaluator):
        """Test initialization."""
        assert random_evaluator.n_actions == 82

    def test_evaluate_returns_result(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluate returns EvaluationResult."""
        legal_actions = [0, 1, 2, 10, 20]
        result = random_evaluator.evaluate(sample_state, legal_actions)

        assert isinstance(result, EvaluationResult)
        assert len(result.policy) == 82

    def test_evaluate_uniform_policy(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluate returns uniform policy over legal actions."""
        legal_actions = [0, 5, 10, 15]
        result = random_evaluator.evaluate(sample_state, legal_actions)

        # Legal actions should have uniform probability
        expected_prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            assert abs(result.policy[action] - expected_prob) < 1e-6

    def test_evaluate_zeros_illegal_actions(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test illegal actions have zero probability."""
        legal_actions = [0, 1, 2]
        result = random_evaluator.evaluate(sample_state, legal_actions)

        # Illegal actions should have zero probability
        for action in range(3, 82):
            assert result.policy[action] == 0.0

    def test_evaluate_value_is_zero(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluate returns zero value."""
        result = random_evaluator.evaluate(sample_state, [0, 1, 2])
        assert result.value == 0.0

    def test_evaluate_empty_legal_actions(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluate with empty legal actions."""
        result = random_evaluator.evaluate(sample_state, [])

        # All probabilities should be zero
        assert numpy.all(result.policy == 0.0)
        assert result.value == 0.0

    def test_evaluate_single_legal_action(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluate with single legal action."""
        result = random_evaluator.evaluate(sample_state, [42])

        assert result.policy[42] == 1.0
        # All other actions should be zero
        for action in range(82):
            if action != 42:
                assert result.policy[action] == 0.0

    def test_evaluate_all_legal_actions(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluate with all actions legal."""
        legal_actions = list(range(82))
        result = random_evaluator.evaluate(sample_state, legal_actions)

        expected_prob = 1.0 / 82
        for action in legal_actions:
            assert abs(result.policy[action] - expected_prob) < 1e-6

    def test_evaluate_batch_returns_list(
        self,
        random_evaluator: RandomEvaluator,
    ):
        """Test evaluate_batch returns list of results."""
        states = [numpy.random.randn(17, 9, 9).astype(numpy.float32) for _ in range(3)]
        legal_actions_batch = [[0, 1], [5, 10, 15], [0]]

        results = random_evaluator.evaluate_batch(states, legal_actions_batch)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, EvaluationResult)

    def test_evaluate_batch_empty(self, random_evaluator: RandomEvaluator):
        """Test evaluate_batch with empty input."""
        results = random_evaluator.evaluate_batch([], [])
        assert results == []

    def test_evaluate_batch_per_state_legal_actions(
        self,
        random_evaluator: RandomEvaluator,
    ):
        """Test evaluate_batch respects per-state legal actions."""
        states = [numpy.random.randn(17, 9, 9).astype(numpy.float32) for _ in range(2)]
        legal_actions_batch = [[0, 1, 2], [10, 20, 30, 40]]

        results = random_evaluator.evaluate_batch(states, legal_actions_batch)

        # First state: 3 legal actions
        assert abs(results[0].policy[0] - 1 / 3) < 1e-6
        assert results[0].policy[10] == 0.0

        # Second state: 4 legal actions
        assert abs(results[1].policy[10] - 1 / 4) < 1e-6
        assert results[1].policy[0] == 0.0

    def test_evaluate_ignores_state(self, random_evaluator: RandomEvaluator):
        """Test random evaluator ignores state content."""
        state1 = numpy.zeros((17, 9, 9), dtype=numpy.float32)
        state2 = numpy.ones((17, 9, 9), dtype=numpy.float32)
        legal_actions = [0, 1, 2]

        result1 = random_evaluator.evaluate(state1, legal_actions)
        result2 = random_evaluator.evaluate(state2, legal_actions)

        # Both should give same policy structure (uniform over legal)
        assert numpy.allclose(result1.policy, result2.policy)

    def test_policy_sums_to_one(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test policy probabilities sum to 1."""
        legal_actions = [0, 5, 10, 20, 50]
        result = random_evaluator.evaluate(sample_state, legal_actions)

        assert abs(result.policy.sum() - 1.0) < 1e-6

    def test_different_n_actions(self, sample_state: numpy.ndarray):
        """Test evaluator with different action counts."""
        evaluator_small = RandomEvaluator(n_actions=10)
        evaluator_large = RandomEvaluator(n_actions=100)

        result_small = evaluator_small.evaluate(sample_state, [0, 1, 2])
        result_large = evaluator_large.evaluate(sample_state, [0, 1, 2])

        assert len(result_small.policy) == 10
        assert len(result_large.policy) == 100


# --- Integration Tests ---


class TestEvaluatorIntegration:
    """Integration tests for evaluators."""

    def test_evaluator_protocol_compliance(self, random_evaluator: RandomEvaluator):
        """Test evaluator satisfies the Evaluator protocol."""
        # Should have evaluate method
        assert hasattr(random_evaluator, "evaluate")
        assert callable(random_evaluator.evaluate)

        # Should have evaluate_batch method
        assert hasattr(random_evaluator, "evaluate_batch")
        assert callable(random_evaluator.evaluate_batch)

    def test_can_use_for_mcts(
        self,
        random_evaluator: RandomEvaluator,
        sample_state: numpy.ndarray,
    ):
        """Test evaluator output can be used for MCTS."""
        legal_actions = [0, 1, 2, 3, 4]
        result = random_evaluator.evaluate(sample_state, legal_actions)

        # Create action priors from policy
        action_priors = {a: float(result.policy[a]) for a in legal_actions}

        # Should sum to approximately 1
        assert abs(sum(action_priors.values()) - 1.0) < 1e-6

        # Value should be in valid range
        assert -1.0 <= result.value <= 1.0
