"""Tests for neural network module."""

from __future__ import annotations

import torch

from src.alphagalerkin.nn.backbone import (
    ElementBackbone,
    TransformerBlock,
)
from src.alphagalerkin.nn.encoder import MeshEncoder
from src.alphagalerkin.nn.feature_norm import RunningNorm
from src.alphagalerkin.nn.policy_head import PolicyHead
from src.alphagalerkin.nn.value_head import ValueHead


class TestMeshEncoder:
    """Tests for the MeshEncoder input projection."""

    def test_output_shape(self) -> None:
        encoder = MeshEncoder(
            input_features=8,
            hidden_dim=32,
        )
        x = torch.randn(2, 4, 8)
        out = encoder(x)
        assert out.shape == (2, 4, 32)

    def test_unbatched_input(self) -> None:
        encoder = MeshEncoder(
            input_features=8,
            hidden_dim=32,
        )
        x = torch.randn(4, 8)
        out = encoder(x)
        assert out.shape == (4, 32)


class TestTransformerBlock:
    """Tests for the self-attention transformer block."""

    def test_output_shape_batched(self) -> None:
        block = TransformerBlock(
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = block(x)
        assert out.shape == (2, 4, 32)

    def test_output_shape_unbatched(self) -> None:
        block = TransformerBlock(
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        x = torch.randn(4, 32)
        out = block(x)
        assert out.shape == (4, 32)

    def test_residual_connection(self) -> None:
        """Output should differ from input (non-trivial)."""
        block = TransformerBlock(
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        block.eval()
        x = torch.randn(1, 4, 32)
        out = block(x)
        assert not torch.allclose(out, x)


class TestElementBackbone:
    """Tests for the multi-layer transformer backbone."""

    def test_output_shape(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = backbone(x)
        assert out.shape == (2, 4, 32)

    def test_multiple_layers_applied(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=32,
            num_layers=3,
            num_heads=4,
            dropout=0.0,
        )
        assert len(backbone.layers) == 3


class TestPolicyHead:
    """Tests for the per-element policy head."""

    def test_output_shape(self) -> None:
        head = PolicyHead(
            hidden_dim=32,
            num_actions=7,
            hidden_dims=[16],
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = head(x)
        assert out.shape == (2, 4, 7)

    def test_output_is_log_prob(self) -> None:
        """Output should be log probabilities (all <= 0)."""
        head = PolicyHead(
            hidden_dim=32,
            num_actions=7,
            hidden_dims=[16],
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = head(x)
        assert (out <= 0.0 + 1e-6).all()

    def test_action_mask_zeros_out_invalid(self) -> None:
        head = PolicyHead(
            hidden_dim=32,
            num_actions=3,
            hidden_dims=[16],
            dropout=0.0,
        )
        x = torch.randn(1, 2, 32)
        mask = torch.tensor(
            [
                [[1, 0, 0], [0, 0, 1]],
            ],
            dtype=torch.float32,
        )
        out = head(x, action_mask=mask)
        # Masked positions should be -inf
        assert out[0, 0, 1] == float("-inf")
        assert out[0, 0, 2] == float("-inf")
        assert out[0, 1, 0] == float("-inf")
        assert out[0, 1, 1] == float("-inf")


class TestValueHead:
    """Tests for the global value head."""

    def test_output_shape_batched(self) -> None:
        head = ValueHead(
            hidden_dim=32,
            hidden_dims=[16],
            pooling="mean",
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = head(x)
        assert out.shape == (2, 1)

    def test_output_bounded(self) -> None:
        """Value head applies tanh -> output in [-1, 1]."""
        head = ValueHead(
            hidden_dim=32,
            hidden_dims=[16],
            pooling="mean",
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32) * 100
        out = head(x)
        assert (out >= -1.0 - 1e-6).all()
        assert (out <= 1.0 + 1e-6).all()

    def test_attention_pooling(self) -> None:
        head = ValueHead(
            hidden_dim=32,
            hidden_dims=[16],
            pooling="attention",
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = head(x)
        assert out.shape == (2, 1)

    def test_max_pooling(self) -> None:
        head = ValueHead(
            hidden_dim=32,
            hidden_dims=[16],
            pooling="max",
            dropout=0.0,
        )
        x = torch.randn(2, 4, 32)
        out = head(x)
        assert out.shape == (2, 1)

    def test_unbatched_input(self) -> None:
        head = ValueHead(
            hidden_dim=32,
            hidden_dims=[16],
            pooling="mean",
            dropout=0.0,
        )
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (1,)


class TestRunningNorm:
    """Tests for the online feature normalization."""

    def test_output_shape(self) -> None:
        norm = RunningNorm(num_features=8)
        x = torch.randn(2, 4, 8)
        out = norm(x)
        assert out.shape == (2, 4, 8)

    def test_training_updates_stats(self) -> None:
        norm = RunningNorm(num_features=8)
        norm.train()
        x = torch.ones(2, 4, 8) * 10.0
        _ = norm(x)
        assert norm.num_batches_tracked.item() == 1

    def test_eval_uses_stored_stats(self) -> None:
        norm = RunningNorm(num_features=8)
        norm.train()
        x = torch.ones(2, 4, 8) * 5.0
        _ = norm(x)
        norm.eval()
        out = norm(x)
        # Should normalize using running stats
        assert out.shape == x.shape
