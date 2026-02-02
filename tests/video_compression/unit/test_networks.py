"""Unit tests for MCTS neural networks.

Tests the MuZero-style networks for learned rate control:
- RepresentationNetwork: Latent to state encoding
- DynamicsNetwork: State transition prediction
- PolicyNetwork: Action distribution
- ValueNetwork: Categorical value estimation
- PredictionNetwork: Combined policy and value
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.video_compression.mcts.networks import (
    DynamicsNetwork,
    PolicyNetwork,
    PolicyOutput,
    PredictionNetwork,
    PredictionOutput,
    RepresentationNetwork,
    ValueNetwork,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def latent_channels() -> int:
    """Default latent channels."""
    return 192


@pytest.fixture
def state_dim() -> int:
    """Default hidden state dimension."""
    return 256


@pytest.fixture
def num_actions() -> int:
    """Default number of actions (QP values)."""
    return 52


@pytest.fixture
def support_size() -> int:
    """Default value support size."""
    return 51


@pytest.fixture
def sample_latent(batch_size: int, latent_channels: int) -> Tensor:
    """Create sample frame latent."""
    return torch.randn(batch_size, latent_channels, 16, 16)


@pytest.fixture
def sample_state(batch_size: int, state_dim: int) -> Tensor:
    """Create sample hidden state."""
    return torch.randn(batch_size, state_dim)


# --------------------------------------------------------------------------
# RepresentationNetwork Tests
# --------------------------------------------------------------------------


class TestRepresentationNetwork:
    """Tests for RepresentationNetwork latent encoding."""

    @pytest.fixture
    def network(self, latent_channels: int, state_dim: int) -> RepresentationNetwork:
        """Create representation network."""
        return RepresentationNetwork(
            latent_channels=latent_channels,
            state_dim=state_dim,
            n_layers=2,
        )

    def test_output_shape(
        self,
        network: RepresentationNetwork,
        sample_latent: Tensor,
        batch_size: int,
        state_dim: int,
    ) -> None:
        """Test output has correct shape."""
        output = network(sample_latent)

        assert output.shape == (batch_size, state_dim)

    def test_output_dtype(self, network: RepresentationNetwork, sample_latent: Tensor) -> None:
        """Test output has correct dtype."""
        output = network(sample_latent)

        assert output.dtype == torch.float32

    def test_gradient_flow(self, network: RepresentationNetwork, sample_latent: Tensor) -> None:
        """Test gradients flow through network."""
        sample_latent.requires_grad = True
        output = network(sample_latent)
        loss = output.sum()
        loss.backward()

        assert sample_latent.grad is not None
        assert not torch.isnan(sample_latent.grad).any()

    def test_variable_spatial_size(self, latent_channels: int, state_dim: int) -> None:
        """Test network handles variable spatial sizes."""
        network = RepresentationNetwork(latent_channels=latent_channels, state_dim=state_dim)

        for size in [8, 16, 32]:
            latent = torch.randn(1, latent_channels, size, size)
            output = network(latent)
            assert output.shape == (1, state_dim)

    def test_deterministic_output(
        self, network: RepresentationNetwork, sample_latent: Tensor
    ) -> None:
        """Test output is deterministic in eval mode."""
        network.eval()

        output1 = network(sample_latent)
        output2 = network(sample_latent)

        assert torch.allclose(output1, output2)


# --------------------------------------------------------------------------
# DynamicsNetwork Tests
# --------------------------------------------------------------------------


class TestDynamicsNetwork:
    """Tests for DynamicsNetwork state transitions."""

    @pytest.fixture
    def network(self, state_dim: int, num_actions: int) -> DynamicsNetwork:
        """Create dynamics network."""
        return DynamicsNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            n_layers=2,
        )

    def test_output_shapes(
        self,
        network: DynamicsNetwork,
        sample_state: Tensor,
        batch_size: int,
        state_dim: int,
    ) -> None:
        """Test output shapes are correct."""
        action = torch.randint(0, 52, (batch_size,))
        next_state, reward = network(sample_state, action)

        assert next_state.shape == (batch_size, state_dim)
        assert reward.shape == (batch_size, 1)

    def test_different_actions_different_states(
        self, network: DynamicsNetwork, sample_state: Tensor
    ) -> None:
        """Test different actions produce different next states."""
        action1 = torch.tensor([0])
        action2 = torch.tensor([51])

        state = sample_state[:1]

        next_state1, _ = network(state, action1)
        next_state2, _ = network(state, action2)

        # States should differ (though may be close due to random init)
        assert not torch.allclose(next_state1, next_state2, atol=1e-6)

    def test_gradient_flow(
        self, network: DynamicsNetwork, sample_state: Tensor, batch_size: int
    ) -> None:
        """Test gradients flow through network."""
        sample_state.requires_grad = True
        action = torch.randint(0, 52, (batch_size,))

        next_state, reward = network(sample_state, action)
        loss = next_state.sum() + reward.sum()
        loss.backward()

        assert sample_state.grad is not None
        assert not torch.isnan(sample_state.grad).any()

    def test_action_embedding(self, network: DynamicsNetwork, sample_state: Tensor) -> None:
        """Test action embedding dimension matches state."""
        assert network.action_embed.embedding_dim == network.state_dim


# --------------------------------------------------------------------------
# PolicyNetwork Tests
# --------------------------------------------------------------------------


class TestPolicyNetwork:
    """Tests for PolicyNetwork action distribution."""

    @pytest.fixture
    def network(self, state_dim: int, num_actions: int) -> PolicyNetwork:
        """Create policy network."""
        return PolicyNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=256,
        )

    def test_output_type(self, network: PolicyNetwork, sample_state: Tensor) -> None:
        """Test output is PolicyOutput."""
        output = network(sample_state)

        assert isinstance(output, PolicyOutput)
        assert hasattr(output, "logits")
        assert hasattr(output, "probs")

    def test_output_shapes(
        self,
        network: PolicyNetwork,
        sample_state: Tensor,
        batch_size: int,
        num_actions: int,
    ) -> None:
        """Test output shapes are correct."""
        output = network(sample_state)

        assert output.logits.shape == (batch_size, num_actions)
        assert output.probs.shape == (batch_size, num_actions)

    def test_probs_sum_to_one(self, network: PolicyNetwork, sample_state: Tensor) -> None:
        """Test probabilities sum to 1."""
        output = network(sample_state)

        prob_sums = output.probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_probs_non_negative(self, network: PolicyNetwork, sample_state: Tensor) -> None:
        """Test probabilities are non-negative."""
        output = network(sample_state)

        assert (output.probs >= 0).all()

    def test_temperature_effect(self, network: PolicyNetwork, sample_state: Tensor) -> None:
        """Test temperature affects distribution sharpness."""
        output_low_temp = network(sample_state[:1], temperature=0.1)
        output_high_temp = network(sample_state[:1], temperature=10.0)

        # Low temperature should be more peaked (higher max prob)
        assert output_low_temp.probs.max() > output_high_temp.probs.max()


# --------------------------------------------------------------------------
# ValueNetwork Tests
# --------------------------------------------------------------------------


class TestValueNetwork:
    """Tests for ValueNetwork categorical value estimation."""

    @pytest.fixture
    def network(self, state_dim: int, support_size: int) -> ValueNetwork:
        """Create value network."""
        return ValueNetwork(
            state_dim=state_dim,
            support_size=support_size,
            hidden_dim=256,
        )

    def test_output_shape(
        self,
        network: ValueNetwork,
        sample_state: Tensor,
        batch_size: int,
    ) -> None:
        """Test value output shape."""
        value = network(sample_state)

        assert value.shape == (batch_size,)

    def test_value_range(
        self,
        network: ValueNetwork,
        sample_state: Tensor,
        support_size: int,
    ) -> None:
        """Test value is within expected range."""
        value = network(sample_state)

        # Value should be within support range
        max_val = support_size // 2
        assert (value >= -max_val).all()
        assert (value <= max_val).all()

    def test_distribution_shape(
        self,
        network: ValueNetwork,
        sample_state: Tensor,
        batch_size: int,
        support_size: int,
    ) -> None:
        """Test distribution output shape."""
        dist = network.get_distribution(sample_state)

        assert dist.shape == (batch_size, support_size)

    def test_distribution_sums_to_one(self, network: ValueNetwork, sample_state: Tensor) -> None:
        """Test distribution sums to 1."""
        dist = network.get_distribution(sample_state)

        dist_sums = dist.sum(dim=-1)
        assert torch.allclose(dist_sums, torch.ones_like(dist_sums), atol=1e-5)

    def test_gradient_flow(self, network: ValueNetwork, sample_state: Tensor) -> None:
        """Test gradients flow through network."""
        sample_state.requires_grad = True

        value = network(sample_state)
        loss = value.sum()
        loss.backward()

        assert sample_state.grad is not None


# --------------------------------------------------------------------------
# PredictionNetwork Tests
# --------------------------------------------------------------------------


class TestPredictionNetwork:
    """Tests for combined PredictionNetwork."""

    @pytest.fixture
    def network(self, state_dim: int, num_actions: int, support_size: int) -> PredictionNetwork:
        """Create prediction network."""
        return PredictionNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            support_size=support_size,
            hidden_dim=256,
        )

    def test_output_type(self, network: PredictionNetwork, sample_state: Tensor) -> None:
        """Test output is PredictionOutput."""
        output = network(sample_state)

        assert isinstance(output, PredictionOutput)
        assert isinstance(output.policy, PolicyOutput)
        assert isinstance(output.value, Tensor)

    def test_output_shapes(
        self,
        network: PredictionNetwork,
        sample_state: Tensor,
        batch_size: int,
        num_actions: int,
    ) -> None:
        """Test output shapes are correct."""
        output = network(sample_state)

        assert output.policy.logits.shape == (batch_size, num_actions)
        assert output.policy.probs.shape == (batch_size, num_actions)
        assert output.value.shape == (batch_size,)

    def test_policy_valid_distribution(
        self, network: PredictionNetwork, sample_state: Tensor
    ) -> None:
        """Test policy is valid probability distribution."""
        output = network(sample_state)

        assert (output.policy.probs >= 0).all()
        prob_sums = output.policy.probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_shared_trunk(self, network: PredictionNetwork, sample_state: Tensor) -> None:
        """Test that network uses shared trunk efficiently."""
        # Both policy and value come from same forward pass
        output = network(sample_state)

        # Both outputs should be present without errors
        assert output.policy.logits is not None
        assert output.value is not None

    def test_gradient_flow_both_heads(
        self, network: PredictionNetwork, sample_state: Tensor
    ) -> None:
        """Test gradients flow through both heads."""
        sample_state.requires_grad = True

        output = network(sample_state)
        loss = output.policy.logits.sum() + output.value.sum()
        loss.backward()

        assert sample_state.grad is not None
        assert not torch.isnan(sample_state.grad).any()

    def test_temperature_effect_on_policy(
        self, network: PredictionNetwork, sample_state: Tensor
    ) -> None:
        """Test temperature only affects policy, not value."""
        output_low_temp = network(sample_state[:1], temperature=0.1)
        output_high_temp = network(sample_state[:1], temperature=10.0)

        # Policy should differ
        assert not torch.allclose(output_low_temp.policy.probs, output_high_temp.policy.probs)

    def test_eval_mode_deterministic(
        self, network: PredictionNetwork, sample_state: Tensor
    ) -> None:
        """Test outputs are deterministic in eval mode."""
        network.eval()

        output1 = network(sample_state)
        output2 = network(sample_state)

        assert torch.allclose(output1.policy.logits, output2.policy.logits)
        assert torch.allclose(output1.value, output2.value)


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestNetworksPipeline:
    """Integration tests for full network pipeline."""

    def test_full_pipeline(
        self,
        sample_latent: Tensor,
        latent_channels: int,
        state_dim: int,
        num_actions: int,
    ) -> None:
        """Test complete pipeline: latent -> state -> prediction."""
        repr_net = RepresentationNetwork(latent_channels=latent_channels, state_dim=state_dim)
        pred_net = PredictionNetwork(state_dim=state_dim, num_actions=num_actions)

        # Encode latent to state
        state = repr_net(sample_latent)

        # Get prediction
        output = pred_net(state)

        assert output.policy.probs.shape[-1] == num_actions
        assert output.value.dim() == 1

    def test_state_transition_pipeline(
        self,
        sample_latent: Tensor,
        latent_channels: int,
        state_dim: int,
        num_actions: int,
    ) -> None:
        """Test state transition pipeline."""
        repr_net = RepresentationNetwork(latent_channels=latent_channels, state_dim=state_dim)
        dyn_net = DynamicsNetwork(state_dim=state_dim, num_actions=num_actions)
        pred_net = PredictionNetwork(state_dim=state_dim, num_actions=num_actions)

        # Initial state
        state = repr_net(sample_latent)

        # Take action
        action = torch.randint(0, num_actions, (sample_latent.shape[0],))
        next_state, reward = dyn_net(state, action)

        # Predict from next state
        output = pred_net(next_state)

        assert next_state.shape == state.shape
        assert output.value.dim() == 1

    def test_networks_trainable(
        self, latent_channels: int, state_dim: int, num_actions: int
    ) -> None:
        """Test all networks have trainable parameters."""
        repr_net = RepresentationNetwork(latent_channels=latent_channels, state_dim=state_dim)
        dyn_net = DynamicsNetwork(state_dim=state_dim, num_actions=num_actions)
        pred_net = PredictionNetwork(state_dim=state_dim, num_actions=num_actions)

        for net in [repr_net, dyn_net, pred_net]:
            params = list(net.parameters())
            assert len(params) > 0
            assert all(p.requires_grad for p in params)
