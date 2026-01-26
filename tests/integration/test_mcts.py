"""Integration tests for MCTS with neural network."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config.schemas import OperatorConfig
from src.mcts.evaluator import FNetEvaluator, RandomEvaluator
from src.mcts.node import MCTSNode
from src.mcts.search import MCTS
from src.modeling.model import AlphaGalerkinModel
from src.tools.gtp import SimpleGoGame


class TestMCTSNode:
    """Tests for MCTS node."""

    def test_ucb_score_exploration(self) -> None:
        """Test that UCB score includes exploration bonus."""
        node = MCTSNode(prior=0.5)
        node.visit_count = 1
        node.total_value = 0.5

        # With more parent visits, exploration bonus increases
        score_low = node.ucb_score(c_puct=1.0, parent_visits=10)
        score_high = node.ucb_score(c_puct=1.0, parent_visits=100)

        assert score_high > score_low

    def test_backup_alternates_value(self) -> None:
        """Test that backup alternates value sign for two-player games."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0, prior=0.5)
        grandchild = MCTSNode(parent=child, action=1, prior=0.5)

        root.children[0] = child
        child.children[1] = grandchild

        # Backup value of 1.0 from grandchild
        grandchild.backup(1.0)

        # Grandchild: +1.0
        assert grandchild.total_value == 1.0

        # Child: -1.0 (opponent's perspective)
        assert child.total_value == -1.0

        # Root: +1.0 (back to original player)
        assert root.total_value == 1.0

    def test_visit_distribution_temperature(self) -> None:
        """Test visit distribution with temperature."""
        root = MCTSNode()
        root.children = {
            0: MCTSNode(parent=root, action=0, prior=0.5),
            1: MCTSNode(parent=root, action=1, prior=0.3),
            2: MCTSNode(parent=root, action=2, prior=0.2),
        }
        root.children[0].visit_count = 100
        root.children[1].visit_count = 50
        root.children[2].visit_count = 10

        # Temperature 0 should be deterministic
        dist_t0 = root.get_visit_distribution(temperature=0)
        assert dist_t0[0] == 1.0

        # Temperature 1 should be proportional to visits
        dist_t1 = root.get_visit_distribution(temperature=1)
        assert dist_t1[0] > dist_t1[1] > dist_t1[2]


class TestMCTSWithRandomEvaluator:
    """Tests for MCTS with random evaluator."""

    @pytest.fixture
    def evaluator(self) -> RandomEvaluator:
        """Create random evaluator."""
        return RandomEvaluator(n_actions=82)

    @pytest.fixture
    def mcts(self, evaluator: RandomEvaluator) -> MCTS:
        """Create MCTS with random evaluator."""
        return MCTS(
            evaluator=evaluator,
            n_simulations=50,
            c_puct=1.5,
        )

    @pytest.fixture
    def game(self) -> SimpleGoGame:
        """Create simple Go game."""
        return SimpleGoGame(board_size=9)

    def test_search_returns_distribution(
        self, mcts: MCTS, game: SimpleGoGame
    ) -> None:
        """Test that search returns valid distribution."""
        distribution = mcts.search(game)

        # Should have some actions
        assert len(distribution) > 0

        # Should sum to approximately 1
        total = sum(distribution.values())
        assert abs(total - 1.0) < 1e-5

    def test_get_action_returns_valid_move(
        self, mcts: MCTS, game: SimpleGoGame
    ) -> None:
        """Test that get_action returns valid action."""
        action = mcts.get_action(game, temperature=1.0)

        legal_actions = game.get_legal_actions()
        assert action in legal_actions

    def test_tree_reuse(self, mcts: MCTS, game: SimpleGoGame) -> None:
        """Test that tree is reused after advance."""
        # First search
        mcts.search(game)

        # Get best action and advance
        action = mcts.get_action(game, temperature=0)
        mcts.advance(action)

        # Root should now be the child node
        assert mcts._root is not None
        assert mcts._root.parent is None  # New root has no parent


class TestMCTSWithNeuralNetwork:
    """Tests for MCTS with AlphaGalerkin model."""

    @pytest.fixture
    def model(self) -> AlphaGalerkinModel:
        """Create model."""
        torch.manual_seed(42)
        config = OperatorConfig(
            d_model=64,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
            input_channels=17,
        )
        return AlphaGalerkinModel(config)

    @pytest.fixture
    def evaluator(self, model: AlphaGalerkinModel) -> FNetEvaluator:
        """Create FNet evaluator."""
        return FNetEvaluator(model, device="cpu", use_fast_path=True)

    @pytest.fixture
    def mcts(self, evaluator: FNetEvaluator) -> MCTS:
        """Create MCTS."""
        return MCTS(
            evaluator=evaluator,
            n_simulations=20,  # Reduced for testing speed
        )

    def test_neural_mcts_search(
        self, mcts: MCTS
    ) -> None:
        """Test MCTS search with neural network."""
        game = SimpleGoGame(board_size=9)

        distribution = mcts.search(game, add_noise=False)

        assert len(distribution) > 0
        assert all(p >= 0 for p in distribution.values())

    def test_neural_mcts_policy_guidance(
        self, mcts: MCTS
    ) -> None:
        """Test that neural network policy guides search."""
        game = SimpleGoGame(board_size=9)

        # Run search
        mcts.search(game, add_noise=False)

        # Check that root has children with priors
        root = mcts._root
        assert root is not None
        assert len(root.children) > 0

        # Children should have non-zero priors
        priors = [child.prior for child in root.children.values()]
        assert any(p > 0 for p in priors)
