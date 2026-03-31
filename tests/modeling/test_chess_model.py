"""Tests for chess-specific model architecture.

Validates the ActionPolicyHead and chess model configuration,
ensuring correct output shapes for chess's 4672-action space.
"""

from __future__ import annotations

import pytest
import torch

from config.schemas import OperatorConfig
from src.modeling.model import (
    ActionPolicyHead,
    AlphaGalerkinFast,
    AlphaGalerkinModel,
    PolicyHead,
)


class TestActionPolicyHead:
    """Tests for the ActionPolicyHead (dense action space)."""

    def test_output_shape(self) -> None:
        """Test ActionPolicyHead outputs correct shape."""
        head = ActionPolicyHead(d_model=64, action_space_size=4672)
        x = torch.randn(2, 64, 64)  # (batch, n_positions, d_model)
        out = head(x)
        assert out.shape == (2, 4672)

    def test_small_action_space(self) -> None:
        """Test with smaller action space."""
        head = ActionPolicyHead(d_model=32, action_space_size=100)
        x = torch.randn(1, 16, 32)
        out = head(x)
        assert out.shape == (1, 100)

    def test_custom_hidden_dim(self) -> None:
        """Test with custom hidden dimension."""
        head = ActionPolicyHead(d_model=64, action_space_size=4672, d_hidden=128)
        x = torch.randn(4, 64, 64)
        out = head(x)
        assert out.shape == (4, 4672)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the head."""
        head = ActionPolicyHead(d_model=32, action_space_size=100)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestPolicyHeadBackwardsCompat:
    """Ensure position-based PolicyHead still works for Go."""

    def test_go_policy_head_shape(self) -> None:
        """Test Go PolicyHead outputs n+1 logits."""
        head = PolicyHead(d_model=64)
        x = torch.randn(2, 81, 64)  # 9x9 board = 81 positions
        out = head(x)
        assert out.shape == (2, 82)  # 81 + 1 pass


class TestChessModelConfig:
    """Tests for chess OperatorConfig."""

    def test_chess_config_creation(self) -> None:
        """Test creating a chess operator config."""
        config = OperatorConfig(
            input_channels=119,
            action_space_size=4672,
            game_type="chess",
            d_model=64,
            d_key=16,
            d_value=16,
            d_ffn=128,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
        )
        assert config.input_channels == 119
        assert config.action_space_size == 4672
        assert config.game_type == "chess"

    def test_go_config_defaults(self) -> None:
        """Test that default config is still Go-compatible."""
        config = OperatorConfig()
        assert config.input_channels == 17
        assert config.action_space_size is None
        assert config.game_type == "go"


class TestChessModelForward:
    """Tests for full model forward pass with chess config."""

    @pytest.fixture
    def chess_config(self) -> OperatorConfig:
        """Create minimal chess config for testing."""
        return OperatorConfig(
            input_channels=119,
            action_space_size=4672,
            game_type="chess",
            d_model=64,
            d_key=16,
            d_value=16,
            d_ffn=128,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
            use_fnet_mixing=False,
        )

    def test_forward_output_shapes(self, chess_config: OperatorConfig) -> None:
        """Test model forward pass with chess input."""
        model = AlphaGalerkinModel(chess_config)
        model.eval()

        # Chess input: (batch, 119 channels, 8, 8)
        x = torch.randn(2, 119, 8, 8)
        with torch.no_grad():
            output = model(x)

        # Policy should be (batch, 4672)
        assert output.policy_logits.shape == (2, 4672)
        # Value should be (batch, 1)
        assert output.value.shape == (2, 1)
        # Value should be in [-1, 1]
        assert (output.value >= -1.0).all()
        assert (output.value <= 1.0).all()

    def test_forward_fast_output_shapes(self, chess_config: OperatorConfig) -> None:
        """Test fast forward pass (MCTS path)."""
        model = AlphaGalerkinModel(chess_config)
        model.eval()

        x = torch.randn(1, 119, 8, 8)
        with torch.no_grad():
            output = model.forward_fast(x)

        assert output.policy_logits.shape == (1, 4672)
        assert output.value.shape == (1, 1)

    def test_forward_with_lbb(self, chess_config: OperatorConfig) -> None:
        """Test forward pass with LBB constant computation."""
        model = AlphaGalerkinModel(chess_config)
        model.eval()

        x = torch.randn(1, 119, 8, 8)
        with torch.no_grad():
            output = model(x, return_lbb=True)

        assert output.policy_logits.shape == (1, 4672)
        assert output.value.shape == (1, 1)


class TestChessModelFast:
    """Tests for AlphaGalerkinFast with chess config."""

    def test_fast_model_chess(self) -> None:
        """Test fast model with chess action space."""
        config = OperatorConfig(
            input_channels=119,
            action_space_size=4672,
            game_type="chess",
            d_model=64,
            d_key=16,
            d_value=16,
            d_ffn=128,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
        )
        model = AlphaGalerkinFast(config, n_layers=2)
        model.eval()

        x = torch.randn(1, 119, 8, 8)
        with torch.no_grad():
            output = model(x)

        assert output.policy_logits.shape == (1, 4672)
        assert output.value.shape == (1, 1)


class TestGoModelBackwardsCompat:
    """Ensure Go model still works with default config."""

    def test_go_model_forward(self) -> None:
        """Test default Go model forward pass."""
        config = OperatorConfig(
            d_model=64,
            d_key=16,
            d_value=16,
            d_ffn=128,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
            use_fnet_mixing=False,
        )
        model = AlphaGalerkinModel(config)
        model.eval()

        # Go 9x9: (batch, 17 channels, 9, 9)
        x = torch.randn(1, 17, 9, 9)
        with torch.no_grad():
            output = model(x)

        # Go policy: 81 positions + 1 pass = 82
        assert output.policy_logits.shape == (1, 82)
        assert output.value.shape == (1, 1)
