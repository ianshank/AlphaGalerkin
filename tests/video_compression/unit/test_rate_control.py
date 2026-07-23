"""Unit tests for MCTS rate control module.

Tests the complete MCTS-based rate controller including:
- MCTSNode tree operations and UCB scoring
- MCTSRateController QP selection and tree search
- GOPPlanner bit allocation
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.video_compression.config import MCTSRateControlConfig, RateControlMode
from src.video_compression.mcts.networks import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.video_compression.mcts.rate_control import (
    GOPPlanner,
    MCTSNode,
    MCTSRateController,
    RateControlDecision,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def mcts_config() -> MCTSRateControlConfig:
    """Create test MCTS configuration with reasonable defaults."""
    return MCTSRateControlConfig(
        name="test_mcts",
        num_simulations=10,  # Reduced for test speed
        c_puct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        discount=0.99,
        gop_size=8,
        qp_min=0,
        qp_max=51,
        target_bitrate_kbps=2000.0,
        fps=30.0,
        rate_control_mode=RateControlMode.VBR,
    )


@pytest.fixture
def representation_net() -> RepresentationNetwork:
    """Create representation network for testing."""
    return RepresentationNetwork(latent_channels=192, state_dim=256, n_layers=2)


@pytest.fixture
def dynamics_net() -> DynamicsNetwork:
    """Create dynamics network for testing."""
    return DynamicsNetwork(state_dim=256, num_actions=52, n_layers=2)


@pytest.fixture
def prediction_net() -> PredictionNetwork:
    """Create prediction network for testing."""
    return PredictionNetwork(state_dim=256, num_actions=52, support_size=51, hidden_dim=256)


@pytest.fixture
def rate_controller(
    mcts_config: MCTSRateControlConfig,
    representation_net: RepresentationNetwork,
    dynamics_net: DynamicsNetwork,
    prediction_net: PredictionNetwork,
) -> MCTSRateController:
    """Create MCTS rate controller for testing."""
    return MCTSRateController(
        config=mcts_config,
        representation_net=representation_net,
        dynamics_net=dynamics_net,
        prediction_net=prediction_net,
        device="cpu",
    )


@pytest.fixture
def sample_latent() -> Tensor:
    """Create sample frame latent tensor."""
    return torch.randn(1, 192, 16, 16)


@pytest.fixture
def sample_state() -> Tensor:
    """Create sample hidden state tensor."""
    return torch.randn(1, 256)


# --------------------------------------------------------------------------
# MCTSNode Tests
# --------------------------------------------------------------------------


class TestMCTSNode:
    """Tests for MCTSNode tree node operations."""

    def test_node_initialization(self, sample_state: Tensor) -> None:
        """Test node initializes with correct default values."""
        node = MCTSNode(state=sample_state, prior=0.5)

        assert node.prior == 0.5
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.action is None
        assert node.parent is None
        assert len(node.children) == 0

    def test_node_value_zero_visits(self, sample_state: Tensor) -> None:
        """Test value is 0 when no visits."""
        node = MCTSNode(state=sample_state, prior=0.5)

        assert node.value == 0.0

    def test_node_value_with_visits(self, sample_state: Tensor) -> None:
        """Test value is correctly computed as mean."""
        node = MCTSNode(state=sample_state, prior=0.5)
        node.visit_count = 4
        node.value_sum = 10.0

        assert node.value == 2.5

    def test_ucb_score_unexplored_node(self, sample_state: Tensor) -> None:
        """Test UCB score is high for unexplored nodes (exploration bonus)."""
        parent = MCTSNode(state=sample_state, prior=1.0)
        parent.visit_count = 10

        child = MCTSNode(state=sample_state, prior=0.1, parent=parent)
        child.visit_count = 0

        score = child.ucb_score(c_puct=1.25)

        # Exploration term should dominate: c_puct * prior * sqrt(N_parent) / (1 + 0)
        expected_exploration = 1.25 * 0.1 * math.sqrt(10) / 1
        assert score == pytest.approx(expected_exploration, rel=0.01)

    def test_ucb_score_well_explored_node(self, sample_state: Tensor) -> None:
        """Test UCB score converges to value for well-explored nodes."""
        parent = MCTSNode(state=sample_state, prior=1.0)
        parent.visit_count = 100

        child = MCTSNode(state=sample_state, prior=0.1, parent=parent)
        child.visit_count = 50
        child.value_sum = 100.0  # Value = 2.0

        score = child.ucb_score(c_puct=1.25)

        # Value term should dominate
        # Exploration: 1.25 * 0.1 * sqrt(100) / 51 ≈ 0.024
        # Value: 2.0
        assert score > 1.9  # Should be close to 2.0
        assert score < 2.5

    def test_ucb_score_with_explicit_parent_visits(self, sample_state: Tensor) -> None:
        """Test UCB score with explicit parent visit count parameter."""
        node = MCTSNode(state=sample_state, prior=0.2)
        node.visit_count = 5
        node.value_sum = 5.0

        # Without parent, use explicit parent_visits
        score = node.ucb_score(c_puct=1.0, parent_visits=20)

        # Value = 1.0
        # Exploration = 1.0 * 0.2 * sqrt(20) / (1 + 5) ≈ 0.149
        expected = 1.0 + 0.2 * math.sqrt(20) / 6
        assert score == pytest.approx(expected, rel=0.01)

    def test_is_expanded_false(self, sample_state: Tensor) -> None:
        """Test is_expanded returns False for leaf nodes."""
        node = MCTSNode(state=sample_state, prior=0.5)

        assert not node.is_expanded()

    def test_is_expanded_true(self, sample_state: Tensor) -> None:
        """Test is_expanded returns True after adding children."""
        node = MCTSNode(state=sample_state, prior=0.5)
        node.children[0] = MCTSNode(state=sample_state, prior=0.1, action=0)

        assert node.is_expanded()


# --------------------------------------------------------------------------
# MCTSRateController Tests
# --------------------------------------------------------------------------


class TestMCTSRateController:
    """Tests for MCTSRateController QP selection."""

    def test_initialization(self, rate_controller: MCTSRateController) -> None:
        """Test rate controller initializes correctly."""
        assert rate_controller.frames_encoded == 0
        assert rate_controller.gop_position == 0
        assert rate_controller.device == "cpu"

    def test_compute_target_bits_cbr(
        self,
        representation_net: RepresentationNetwork,
        dynamics_net: DynamicsNetwork,
        prediction_net: PredictionNetwork,
    ) -> None:
        """Test target bits calculation for CBR mode."""
        config = MCTSRateControlConfig(
            name="cbr_test",
            rate_control_mode=RateControlMode.CBR,
            target_bitrate_kbps=2000.0,
            fps=30.0,
        )
        controller = MCTSRateController(
            config=config,
            representation_net=representation_net,
            dynamics_net=dynamics_net,
            prediction_net=prediction_net,
        )

        # 2000 kbps / 30 fps = ~66666 bits per frame
        expected = (2000.0 * 1000) / 30.0
        assert controller.target_bits_per_frame == pytest.approx(expected)

    def test_compute_target_bits_vbr(self, rate_controller: MCTSRateController) -> None:
        """Test target bits is infinite for VBR mode."""
        assert rate_controller.target_bits_per_frame == float("inf")

    def test_select_qp_returns_valid_range(
        self,
        rate_controller: MCTSRateController,
        sample_latent: Tensor,
    ) -> None:
        """Test that select_qp returns QP within valid range."""
        decision = rate_controller.select_qp(sample_latent, frame_type="P")

        assert isinstance(decision, RateControlDecision)
        assert rate_controller.config.qp_min <= decision.qp <= rate_controller.config.qp_max
        assert 0 <= decision.confidence <= 1.0
        assert decision.predicted_bits > 0
        assert decision.predicted_quality > 0

    def test_select_qp_updates_state(
        self,
        rate_controller: MCTSRateController,
        sample_latent: Tensor,
    ) -> None:
        """Test that select_qp updates internal state."""
        initial_frames = rate_controller.frames_encoded
        initial_gop_pos = rate_controller.gop_position

        rate_controller.select_qp(sample_latent, frame_type="I")

        assert rate_controller.frames_encoded == initial_frames + 1
        assert (
            rate_controller.gop_position == (initial_gop_pos + 1) % rate_controller.config.gop_size
        )

    def test_select_qp_gop_position_wraps(
        self,
        rate_controller: MCTSRateController,
        sample_latent: Tensor,
    ) -> None:
        """Test that GOP position wraps correctly."""
        gop_size = rate_controller.config.gop_size

        for i in range(gop_size):
            rate_controller.select_qp(sample_latent, frame_type="P")

        assert rate_controller.gop_position == 0
        assert rate_controller.frames_encoded == gop_size

    def test_estimate_bits_decreases_with_qp(
        self,
        rate_controller: MCTSRateController,
        sample_latent: Tensor,
    ) -> None:
        """Test that estimated bits decrease as QP increases."""
        bits_low_qp = rate_controller._estimate_bits(sample_latent, qp=10)
        bits_high_qp = rate_controller._estimate_bits(sample_latent, qp=40)

        assert bits_low_qp > bits_high_qp

    def test_estimate_quality_decreases_with_qp(self, rate_controller: MCTSRateController) -> None:
        """Test that estimated quality decreases as QP increases."""
        quality_low_qp = rate_controller._estimate_quality(qp=10)
        quality_high_qp = rate_controller._estimate_quality(qp=40)

        assert quality_low_qp > quality_high_qp
        assert quality_low_qp <= 50.0  # Max PSNR estimate
        assert quality_high_qp >= 20.0  # Min PSNR estimate

    def test_run_mcts_creates_tree(
        self,
        rate_controller: MCTSRateController,
        sample_state: Tensor,
    ) -> None:
        """Test that MCTS creates proper tree structure."""
        with torch.no_grad():
            root = rate_controller._run_mcts(sample_state, frame_type="P")

            assert isinstance(root, MCTSNode)
            assert root.is_expanded()
            assert root.visit_count >= 1

    def test_expand_node_creates_children(
        self,
        rate_controller: MCTSRateController,
        sample_state: Tensor,
        prediction_net: PredictionNetwork,
    ) -> None:
        """Test that _expand_node creates children for valid QP range."""
        with torch.no_grad():
            node = MCTSNode(state=sample_state, prior=1.0)
            prediction = prediction_net(sample_state)

            rate_controller._expand_node(node, prediction)

            qp_range = rate_controller.config.qp_max - rate_controller.config.qp_min + 1
            assert len(node.children) == qp_range
            assert all(
                rate_controller.config.qp_min <= action <= rate_controller.config.qp_max
                for action in node.children
            )

    def test_select_child_returns_best_ucb(
        self,
        rate_controller: MCTSRateController,
        sample_state: Tensor,
    ) -> None:
        """Test that _select_child selects child with highest UCB."""
        parent = MCTSNode(state=sample_state, prior=1.0)
        parent.visit_count = 10

        # Add children with known priors
        for action in range(3):
            child = MCTSNode(
                state=sample_state,
                prior=0.3 if action == 1 else 0.1,  # Child 1 has highest prior
                action=action,
                parent=parent,
            )
            parent.children[action] = child

        action, child = rate_controller._select_child(parent)

        # Child 1 should be selected (highest prior)
        assert action == 1
        assert child.prior == 0.3

    def test_add_exploration_noise_modifies_priors(
        self,
        rate_controller: MCTSRateController,
        sample_state: Tensor,
        prediction_net: PredictionNetwork,
    ) -> None:
        """Test that Dirichlet noise is added to root priors."""
        with torch.no_grad():
            root = MCTSNode(state=sample_state, prior=1.0)
            prediction = prediction_net(sample_state)
            rate_controller._expand_node(root, prediction)

            original_priors = {action: child.prior for action, child in root.children.items()}

            rate_controller._add_exploration_noise(root)

            # Priors should be modified
            modified = False
            for action, child in root.children.items():
                if child.prior != original_priors[action]:
                    modified = True
                    break

            assert modified

    def test_backpropagate_updates_path(
        self, rate_controller: MCTSRateController, sample_state: Tensor
    ) -> None:
        """Test that backpropagation updates all nodes in path."""
        # Create a simple path
        nodes = []
        for i in range(3):
            node = MCTSNode(state=sample_state, prior=0.5, action=i)
            if nodes:
                node.parent = nodes[-1]
            nodes.append(node)

        rate_controller._backpropagate(nodes, value=5.0)

        # All nodes should have updated counts and values
        for node in nodes:
            assert node.visit_count == 1
            assert node.value_sum > 0

    def test_select_action_greedy(
        self,
        representation_net: RepresentationNetwork,
        dynamics_net: DynamicsNetwork,
        prediction_net: PredictionNetwork,
        sample_state: Tensor,
    ) -> None:
        """Test greedy action selection with temperature=0."""
        config = MCTSRateControlConfig(
            name="greedy_test",
            temperature=0.0,
            num_simulations=5,
        )
        controller = MCTSRateController(
            config=config,
            representation_net=representation_net,
            dynamics_net=dynamics_net,
            prediction_net=prediction_net,
        )

        root = MCTSNode(state=sample_state, prior=1.0)
        root.visit_count = 10

        # Add children with different visit counts
        for action in range(3):
            child = MCTSNode(state=sample_state, prior=0.33, action=action, parent=root)
            child.visit_count = action + 1  # 1, 2, 3
            root.children[action] = child

        qp, confidence = controller._select_action(root)

        assert qp == 2  # Highest visit count


# --------------------------------------------------------------------------
# GOPPlanner Tests
# --------------------------------------------------------------------------


class TestGOPPlanner:
    """Tests for GOP-level bit allocation planning."""

    @pytest.fixture
    def gop_planner(
        self, mcts_config: MCTSRateControlConfig, rate_controller: MCTSRateController
    ) -> GOPPlanner:
        """Create GOP planner for testing."""
        return GOPPlanner(config=mcts_config, rate_controller=rate_controller)

    def test_plan_gop_returns_decisions(
        self, gop_planner: GOPPlanner, sample_latent: Tensor
    ) -> None:
        """Test that plan_gop returns decisions for all frames."""
        gop_size = 4
        frame_latents = [sample_latent.clone() for _ in range(gop_size)]

        decisions = gop_planner.plan_gop(frame_latents)

        assert len(decisions) == gop_size
        assert all(isinstance(d, RateControlDecision) for d in decisions)

    def test_get_frame_types_i_frame_first(self, gop_planner: GOPPlanner) -> None:
        """Test that first frame is always I-frame."""
        frame_types = gop_planner._get_frame_types(gop_size=4)

        assert frame_types[0] == "I"

    def test_get_frame_types_structure(self, gop_planner: GOPPlanner) -> None:
        """Test frame type pattern follows config."""
        gop_planner.config.use_b_frames = True
        gop_planner.config.b_frame_count = 2

        frame_types = gop_planner._get_frame_types(gop_size=8)

        # First frame is I
        assert frame_types[0] == "I"

        # Pattern should follow B-frame configuration
        assert all(ft in ("I", "P", "B") for ft in frame_types)

    def test_get_frame_types_no_b_frames(self, gop_planner: GOPPlanner) -> None:
        """Test frame types without B-frames."""
        gop_planner.config.use_b_frames = False

        frame_types = gop_planner._get_frame_types(gop_size=4)

        # Should only have I and P frames
        assert "B" not in frame_types
        assert frame_types[0] == "I"
        assert all(ft == "P" for ft in frame_types[1:])


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestMCTSRateControlIntegration:
    """Integration tests for MCTS rate control with codec."""

    def test_multiple_frames_encoding(
        self,
        rate_controller: MCTSRateController,
        sample_latent: Tensor,
    ) -> None:
        """Test rate controller handles multiple frames correctly."""
        decisions = []

        for i in range(10):
            frame_type = "I" if i == 0 else "P"
            decision = rate_controller.select_qp(sample_latent, frame_type)
            decisions.append(decision)

        assert len(decisions) == 10
        assert rate_controller.frames_encoded == 10

    def test_reproducibility_with_seed(
        self,
        mcts_config: MCTSRateControlConfig,
        sample_latent: Tensor,
    ) -> None:
        """Test that results are reproducible with same seed."""
        torch.manual_seed(42)
        repr_net1 = RepresentationNetwork(latent_channels=192, state_dim=256)
        dyn_net1 = DynamicsNetwork(state_dim=256, num_actions=52)
        pred_net1 = PredictionNetwork(state_dim=256, num_actions=52)

        controller1 = MCTSRateController(
            config=mcts_config,
            representation_net=repr_net1,
            dynamics_net=dyn_net1,
            prediction_net=pred_net1,
        )

        torch.manual_seed(42)
        repr_net2 = RepresentationNetwork(latent_channels=192, state_dim=256)
        dyn_net2 = DynamicsNetwork(state_dim=256, num_actions=52)
        pred_net2 = PredictionNetwork(state_dim=256, num_actions=52)

        controller2 = MCTSRateController(
            config=mcts_config,
            representation_net=repr_net2,
            dynamics_net=dyn_net2,
            prediction_net=pred_net2,
        )

        torch.manual_seed(123)
        decision1 = controller1.select_qp(sample_latent, "P")

        torch.manual_seed(123)
        decision2 = controller2.select_qp(sample_latent, "P")

        assert decision1.qp == decision2.qp
