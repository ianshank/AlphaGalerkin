"""Tests for backbone architecture modes."""
from __future__ import annotations

import pytest
import torch

from src.alphagalerkin.nn.backbone import (
    ElementBackbone,
    GalerkinBlock,
    TransformerBlock,
    _build_block,
)
from src.alphagalerkin.nn.fnet_block import FNetBlock


class TestBuildBlock:
    """Tests for the block factory."""

    def test_gat_returns_transformer(self) -> None:
        block = _build_block("gat", 64, 4, 0.1)
        assert isinstance(block, TransformerBlock)

    def test_gcn_returns_transformer(self) -> None:
        block = _build_block("gcn", 64, 4, 0.1)
        assert isinstance(block, TransformerBlock)

    def test_graphsage_returns_transformer(self) -> None:
        block = _build_block("graphsage", 64, 4, 0.1)
        assert isinstance(block, TransformerBlock)

    def test_galerkin_returns_galerkin_block(self) -> None:
        block = _build_block("galerkin", 64, 4, 0.1)
        assert isinstance(block, GalerkinBlock)

    def test_fnet_returns_fnet_block(self) -> None:
        block = _build_block("fnet", 64, 4, 0.1)
        assert isinstance(block, FNetBlock)

    def test_custom_returns_galerkin(self) -> None:
        block = _build_block("custom", 64, 4, 0.1)
        assert isinstance(block, GalerkinBlock)


class TestTransformerBlockForward:
    """Test TransformerBlock forward pass."""

    def test_output_shape(self) -> None:
        block = TransformerBlock(hidden_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_2d_input(self) -> None:
        block = TransformerBlock(hidden_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(10, 64)
        out = block(x)
        assert out.shape == (10, 64)


class TestGalerkinBlockForward:
    """Test GalerkinBlock forward pass."""

    def test_output_shape(self) -> None:
        block = GalerkinBlock(hidden_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_2d_input(self) -> None:
        block = GalerkinBlock(hidden_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(10, 64)
        out = block(x)
        assert out.shape == (10, 64)

    def test_gradients_flow(self) -> None:
        block = GalerkinBlock(hidden_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestElementBackbone:
    """Tests for ElementBackbone with different architectures."""

    def test_transformer_mode_forward(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=64, num_layers=2, num_heads=4,
            architecture="gat",
        )
        x = torch.randn(2, 10, 64)
        out = backbone(x)
        assert out.shape == (2, 10, 64)

    def test_galerkin_mode_forward(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=64, num_layers=2, num_heads=4,
            architecture="galerkin",
        )
        x = torch.randn(2, 10, 64)
        out = backbone(x)
        assert out.shape == (2, 10, 64)

    def test_fnet_mode_forward(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=64, num_layers=2, num_heads=4,
            architecture="fnet",
        )
        x = torch.randn(2, 10, 64)
        out = backbone(x)
        assert out.shape == (2, 10, 64)

    @pytest.mark.parametrize("arch", ["gat", "galerkin", "fnet"])
    def test_gradients_flow(self, arch: str) -> None:
        """Gradients propagate through backbone for all architectures."""
        backbone = ElementBackbone(
            hidden_dim=32, num_layers=2, num_heads=4,
            dropout=0.0, architecture=arch,
        )
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = backbone(x)
        out.sum().backward()
        assert x.grad is not None

    def test_architecture_stored(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=64, num_layers=2, architecture="galerkin",
        )
        assert backbone.architecture == "galerkin"

    def test_layer_count(self) -> None:
        backbone = ElementBackbone(
            hidden_dim=64, num_layers=3, architecture="fnet",
        )
        assert len(backbone.layers) == 3
