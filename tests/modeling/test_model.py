"""Tests for AlphaGalerkin main model and component blocks."""

from __future__ import annotations

import pytest
import torch

from config.schemas import OperatorConfig
from src.modeling.model import (
    ActionPolicyHead,
    AlphaGalerkinFast,
    AlphaGalerkinModel,
    DenseHead,
    GalerkinBlock,
    ModelOutput,
    PolicyHead,
    ScaleNorm,
    SoftmaxBlock,
    ValueHead,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(**overrides: object) -> OperatorConfig:
    """Create a small OperatorConfig for fast testing.

    Defaults: d_model=16, n_heads=2, n_layers=1, input_channels=3.
    """
    defaults = dict(
        d_model=16,
        d_key=8,
        d_value=8,
        d_ffn=32,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=8,
        fourier_scale=1.0,
        use_fnet_mixing=True,
        fnet_dropout=0.0,
        lbb_beta_threshold=1e-6,
        norm_type="layernorm",
        input_channels=3,
        action_space_size=None,
    )
    defaults.update(overrides)
    return OperatorConfig(**defaults)  # type: ignore[arg-type]


BOARD_SIZE = 5
BATCH = 2
N_POSITIONS = BOARD_SIZE * BOARD_SIZE  # 25


# ---------------------------------------------------------------------------
# ScaleNorm
# ---------------------------------------------------------------------------


class TestScaleNorm:
    """Tests for ScaleNorm."""

    def test_forward_shape(self) -> None:
        """Test output shape matches input."""
        norm = ScaleNorm(d_model=16)
        x = torch.randn(BATCH, 8, 16)
        output = norm(x)
        assert output.shape == (BATCH, 8, 16)

    def test_normalizes_to_unit_scale(self) -> None:
        """Test that output has controlled norm."""
        norm = ScaleNorm(d_model=16)
        x = torch.randn(BATCH, 8, 16) * 100
        output = norm(x)
        norms = output.norm(dim=-1)
        assert norms.std() < norms.mean()

    def test_no_nan_with_zeros(self) -> None:
        """Test stability with zero input."""
        norm = ScaleNorm(d_model=16)
        x = torch.zeros(BATCH, 8, 16)
        output = norm(x)
        assert not torch.isnan(output).any()

    def test_learnable_scale(self) -> None:
        """Test that scale parameter is learnable."""
        norm = ScaleNorm(d_model=16)
        assert norm.scale.requires_grad


# ---------------------------------------------------------------------------
# GalerkinBlock
# ---------------------------------------------------------------------------


class TestGalerkinBlock:
    """Tests for GalerkinBlock."""

    def test_initialization(self) -> None:
        """Test block initialization."""
        block = GalerkinBlock(d_model=16, n_heads=2)
        assert block.attention is not None
        assert block.ffn is not None

    def test_forward_shape(self) -> None:
        """Test forward pass shape."""
        block = GalerkinBlock(d_model=16, n_heads=2)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = block(x)
        assert output.shape == (BATCH, N_POSITIONS, 16)

    def test_forward_with_lbb(self) -> None:
        """Test forward pass returning LBB constant."""
        block = GalerkinBlock(d_model=16, n_heads=2)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        result = block(x, return_lbb=True)
        assert isinstance(result, tuple)
        output, lbb = result
        assert output.shape == (BATCH, N_POSITIONS, 16)
        assert lbb.shape == (BATCH,)

    def test_different_norm_types(self) -> None:
        """Test with different normalization types."""
        for norm_type in ["layernorm", "scalenorm"]:
            block = GalerkinBlock(d_model=16, n_heads=2, norm_type=norm_type)
            x = torch.randn(BATCH, N_POSITIONS, 16)
            output = block(x)
            assert output.shape == (BATCH, N_POSITIONS, 16)

    def test_custom_ffn_dim(self) -> None:
        """Test with custom FFN dimension."""
        block = GalerkinBlock(d_model=16, n_heads=2, d_ffn=64)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = block(x)
        assert output.shape == (BATCH, N_POSITIONS, 16)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through block."""
        block = GalerkinBlock(d_model=16, n_heads=2)
        x = torch.randn(BATCH, N_POSITIONS, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SoftmaxBlock
# ---------------------------------------------------------------------------


class TestSoftmaxBlock:
    """Tests for SoftmaxBlock."""

    def test_initialization(self) -> None:
        """Test block initialization."""
        block = SoftmaxBlock(d_model=16, n_heads=2)
        assert block.attention is not None
        assert block.ffn is not None

    def test_forward_shape(self) -> None:
        """Test forward pass shape."""
        block = SoftmaxBlock(d_model=16, n_heads=2)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = block(x)
        assert output.shape == (BATCH, N_POSITIONS, 16)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through block."""
        block = SoftmaxBlock(d_model=16, n_heads=2)
        x = torch.randn(BATCH, N_POSITIONS, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# PolicyHead
# ---------------------------------------------------------------------------


class TestPolicyHead:
    """Tests for PolicyHead."""

    def test_forward_shape(self) -> None:
        """Test output shape includes pass move."""
        head = PolicyHead(d_model=16)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = head(x)
        assert output.shape == (BATCH, N_POSITIONS + 1)

    def test_forward_different_positions(self) -> None:
        """Test with different numbers of positions."""
        head = PolicyHead(d_model=16)
        for n_pos in [4, 9, 25]:
            x = torch.randn(BATCH, n_pos, 16)
            output = head(x)
            assert output.shape == (BATCH, n_pos + 1)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through policy head."""
        head = PolicyHead(d_model=16)
        x = torch.randn(BATCH, N_POSITIONS, 16, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_custom_hidden_dim(self) -> None:
        """Test with custom hidden dimension."""
        head = PolicyHead(d_model=16, d_hidden=32)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = head(x)
        assert output.shape == (BATCH, N_POSITIONS + 1)


# ---------------------------------------------------------------------------
# ActionPolicyHead
# ---------------------------------------------------------------------------


class TestActionPolicyHead:
    """Tests for ActionPolicyHead (dense action-space policy)."""

    def test_forward_shape(self) -> None:
        """Test output shape matches action space size."""
        head = ActionPolicyHead(d_model=16, action_space_size=100)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = head(x)
        assert output.shape == (BATCH, 100)

    def test_different_action_spaces(self) -> None:
        """Test with different action space sizes."""
        for action_size in [10, 100, 4672]:
            head = ActionPolicyHead(d_model=16, action_space_size=action_size)
            x = torch.randn(BATCH, N_POSITIONS, 16)
            output = head(x)
            assert output.shape == (BATCH, action_size)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through action policy head."""
        head = ActionPolicyHead(d_model=16, action_space_size=50)
        x = torch.randn(BATCH, N_POSITIONS, 16, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------


class TestValueHead:
    """Tests for ValueHead."""

    def test_forward_shape(self) -> None:
        """Test output shape is (batch, 1)."""
        head = ValueHead(d_model=16)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = head(x)
        assert output.shape == (BATCH, 1)

    def test_output_range(self) -> None:
        """Test output is in [-1, 1] due to tanh."""
        head = ValueHead(d_model=16)
        x = torch.randn(BATCH, N_POSITIONS, 16) * 10
        output = head(x)
        assert (output >= -1.0).all()
        assert (output <= 1.0).all()

    def test_gradient_flow(self) -> None:
        """Test gradient flows through value head."""
        head = ValueHead(d_model=16)
        x = torch.randn(BATCH, N_POSITIONS, 16, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# DenseHead
# ---------------------------------------------------------------------------


class TestDenseHead:
    """Tests for DenseHead."""

    def test_forward_shape(self) -> None:
        """Test output shape (batch, n, output_channels)."""
        head = DenseHead(d_model=16, output_channels=1)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = head(x)
        assert output.shape == (BATCH, N_POSITIONS, 1)

    def test_multi_channel_output(self) -> None:
        """Test with multiple output channels."""
        head = DenseHead(d_model=16, output_channels=3)
        x = torch.randn(BATCH, N_POSITIONS, 16)
        output = head(x)
        assert output.shape == (BATCH, N_POSITIONS, 3)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through dense head."""
        head = DenseHead(d_model=16)
        x = torch.randn(BATCH, N_POSITIONS, 16, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# AlphaGalerkinModel -- initialization
# ---------------------------------------------------------------------------


class TestAlphaGalerkinModelInit:
    """Tests for AlphaGalerkinModel initialization with various configs."""

    def test_default_small_config(self) -> None:
        """Test model initialization with default small config."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        assert model.config is config
        assert len(model.strategy_body) == 1
        assert len(model.tactical_head) == 1

    def test_without_fnet(self) -> None:
        """Test initialization without FNet mixing."""
        config = _small_config(use_fnet_mixing=False)
        model = AlphaGalerkinModel(config)
        assert model.fnet_layers is None

    def test_with_fnet(self) -> None:
        """Test initialization with FNet mixing."""
        config = _small_config(use_fnet_mixing=True, n_galerkin_layers=4)
        model = AlphaGalerkinModel(config)
        assert model.fnet_layers is not None
        assert len(model.fnet_layers) == 2  # n_galerkin_layers // 2

    def test_with_action_space_policy(self) -> None:
        """Test initialization with action-space policy head."""
        config = _small_config(action_space_size=100)
        model = AlphaGalerkinModel(config)
        assert isinstance(model.policy_head, ActionPolicyHead)

    def test_with_position_policy(self) -> None:
        """Test initialization with position-based policy head (Go)."""
        config = _small_config(action_space_size=None)
        model = AlphaGalerkinModel(config)
        assert isinstance(model.policy_head, PolicyHead)

    def test_different_norm_types(self) -> None:
        """Test initialization with different normalization types."""
        for norm in ["layernorm", "scalenorm"]:
            config = _small_config(norm_type=norm)
            model = AlphaGalerkinModel(config)
            assert model is not None

    def test_multiple_galerkin_layers(self) -> None:
        """Test initialization with multiple Galerkin layers."""
        config = _small_config(n_galerkin_layers=3, n_softmax_layers=2)
        model = AlphaGalerkinModel(config)
        assert len(model.strategy_body) == 3
        assert len(model.tactical_head) == 2


# ---------------------------------------------------------------------------
# AlphaGalerkinModel -- forward pass
# ---------------------------------------------------------------------------


class TestAlphaGalerkinModelForward:
    """Tests for AlphaGalerkinModel forward pass shapes and values."""

    def test_forward_output_type(self) -> None:
        """Test that forward returns ModelOutput."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert isinstance(output, ModelOutput)

    def test_forward_policy_shape(self) -> None:
        """Test policy logits shape (board positions + pass)."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert output.policy_logits.shape == (BATCH, N_POSITIONS + 1)

    def test_forward_value_shape(self) -> None:
        """Test value output shape."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert output.value.shape == (BATCH, 1)

    def test_forward_value_range(self) -> None:
        """Test value is in [-1, 1]."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert (output.value >= -1.0).all()
        assert (output.value <= 1.0).all()

    def test_forward_lbb_none_by_default(self) -> None:
        """Test LBB constant is None when not requested."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert output.lbb_constant is None

    def test_forward_with_lbb(self) -> None:
        """Test forward pass with LBB constant."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x, return_lbb=True)
        assert output.lbb_constant is not None
        assert output.lbb_constant.shape == (BATCH,)

    def test_no_nan_output(self) -> None:
        """Test no NaN values in model output."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert not torch.isnan(output.policy_logits).any()
        assert not torch.isnan(output.value).any()

    def test_action_space_policy_head(self) -> None:
        """Test model with action-space policy head (e.g., chess)."""
        config = _small_config(action_space_size=100)
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert output.policy_logits.shape == (BATCH, 100)


# ---------------------------------------------------------------------------
# AlphaGalerkinModel -- resolution independence
# ---------------------------------------------------------------------------


class TestAlphaGalerkinModelResolution:
    """Tests for resolution independence across different board sizes."""

    def test_handles_different_board_sizes(self) -> None:
        """Test model handles different board sizes."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        model.eval()

        for board_size in [3, 5, 7, 9]:
            x = torch.randn(BATCH, 3, board_size, board_size)
            output = model(x)
            n_pos = board_size * board_size
            assert output.policy_logits.shape == (BATCH, n_pos + 1)
            assert output.value.shape == (BATCH, 1)

    def test_fast_forward_different_sizes(self) -> None:
        """Test fast forward at different resolutions."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        model.eval()

        for board_size in [3, 5, 7]:
            x = torch.randn(1, 3, board_size, board_size)
            output = model.forward_fast(x)
            n_pos = board_size * board_size
            assert output.policy_logits.shape == (1, n_pos + 1)
            assert output.value.shape == (1, 1)

    def test_adapt_resolution(self) -> None:
        """Test resolution adaptation."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        model.adapt_resolution(source_size=5, target_size=9)
        assert model._training_resolution == 5

    def test_set_training_resolution(self) -> None:
        """Test setting training resolution."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        assert model.training_resolution is None
        model.set_training_resolution(5)
        assert model.training_resolution == 5

    def test_training_resolution_property(self) -> None:
        """Test training resolution property getter/setter."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        model.training_resolution = 9
        assert model.training_resolution == 9


# ---------------------------------------------------------------------------
# AlphaGalerkinModel -- gradient flow
# ---------------------------------------------------------------------------


class TestAlphaGalerkinModelGradients:
    """Tests for gradient flow through the full model."""

    def test_gradient_flows_through_model(self) -> None:
        """Test gradient flows through entire model."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
        output = model(x)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()
        assert x.grad is not None

    def test_all_parameters_receive_gradients(self) -> None:
        """Test all trainable parameters receive gradients."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_finite(self) -> None:
        """Test all gradients are finite (no NaN or Inf)."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_with_lbb(self) -> None:
        """Test gradient flow when computing LBB constant."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
        output = model(x, return_lbb=True)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# AlphaGalerkinModel -- forward_fast
# ---------------------------------------------------------------------------


class TestAlphaGalerkinModelForwardFast:
    """Tests for the fast forward pass (MCTS rollouts)."""

    def test_forward_fast_shape(self) -> None:
        """Test fast forward pass shapes."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model.forward_fast(x)
        assert isinstance(output, ModelOutput)
        assert output.policy_logits.shape == (BATCH, N_POSITIONS + 1)
        assert output.value.shape == (BATCH, 1)
        assert output.lbb_constant is None

    def test_forward_fast_without_fnet(self) -> None:
        """Test fast forward when FNet is disabled (falls back to Galerkin)."""
        config = _small_config(use_fnet_mixing=False)
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model.forward_fast(x)
        assert output.policy_logits.shape == (BATCH, N_POSITIONS + 1)
        assert output.value.shape == (BATCH, 1)

    def test_forward_fast_no_nan(self) -> None:
        """Test fast forward produces no NaN values."""
        config = _small_config()
        model = AlphaGalerkinModel(config)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model.forward_fast(x)
        assert not torch.isnan(output.policy_logits).any()
        assert not torch.isnan(output.value).any()


# ---------------------------------------------------------------------------
# AlphaGalerkinModel -- eval mode
# ---------------------------------------------------------------------------


class TestAlphaGalerkinModelEval:
    """Tests for eval-mode behavior."""

    def test_eval_mode_deterministic(self) -> None:
        """Test that eval mode produces deterministic outputs."""
        config = _small_config(fnet_dropout=0.0)
        model = AlphaGalerkinModel(config)
        model.eval()
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1.policy_logits, out2.policy_logits)
        assert torch.allclose(out1.value, out2.value)


# ---------------------------------------------------------------------------
# AlphaGalerkinFast
# ---------------------------------------------------------------------------


class TestAlphaGalerkinFast:
    """Tests for AlphaGalerkinFast (FNet-only model)."""

    def test_initialization(self) -> None:
        """Test fast model initialization."""
        config = _small_config()
        model = AlphaGalerkinFast(config, n_layers=2)
        assert model.fnet_stack is not None

    def test_forward_shape(self) -> None:
        """Test forward pass shapes."""
        config = _small_config()
        model = AlphaGalerkinFast(config, n_layers=2)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert isinstance(output, ModelOutput)
        assert output.policy_logits.shape == (BATCH, N_POSITIONS + 1)
        assert output.value.shape == (BATCH, 1)
        assert output.lbb_constant is None

    def test_resolution_independence(self) -> None:
        """Test fast model works at different resolutions."""
        config = _small_config()
        model = AlphaGalerkinFast(config, n_layers=2)
        model.eval()

        for board_size in [3, 5, 7]:
            x = torch.randn(1, 3, board_size, board_size)
            output = model(x)
            n_pos = board_size * board_size
            assert output.policy_logits.shape == (1, n_pos + 1)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through fast model."""
        config = _small_config()
        model = AlphaGalerkinFast(config, n_layers=2)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
        output = model(x)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()
        assert x.grad is not None

    def test_action_space_policy_head(self) -> None:
        """Test fast model with action-space policy head."""
        config = _small_config(action_space_size=50)
        model = AlphaGalerkinFast(config, n_layers=2)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert output.policy_logits.shape == (BATCH, 50)

    def test_no_nan_output(self) -> None:
        """Test no NaN values in output."""
        config = _small_config()
        model = AlphaGalerkinFast(config, n_layers=2)
        x = torch.randn(BATCH, 3, BOARD_SIZE, BOARD_SIZE)
        output = model(x)
        assert not torch.isnan(output.policy_logits).any()
        assert not torch.isnan(output.value).any()
