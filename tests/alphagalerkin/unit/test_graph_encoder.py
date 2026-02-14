"""Tests for the graph-based message-passing encoder."""
from __future__ import annotations

import torch

from src.alphagalerkin.core.config import GNNConfig, NetworkConfig
from src.alphagalerkin.core.types import GNNArchitecture
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.nn.graph_encoder import (
    GraphEncoder,
    MessagePassingLayer,
)
from src.alphagalerkin.nn.model import AlphaGalerkinNetwork

# -------------------------------------------------------------------
# MessagePassingLayer
# -------------------------------------------------------------------


class TestMessagePassingLayerShapes:
    """Verify output shapes of a single message-passing layer."""

    def test_batched_output_shape(self) -> None:
        """Output should preserve (batch, N, hidden_dim)."""
        hidden_dim = 32
        layer = MessagePassingLayer(hidden_dim=hidden_dim, dropout=0.0)
        batch, num_nodes = 2, 5
        x = torch.randn(batch, num_nodes, hidden_dim)
        adj = torch.eye(num_nodes).unsqueeze(0).expand(batch, -1, -1)

        out = layer(x, adj)

        assert out.shape == (batch, num_nodes, hidden_dim)

    def test_single_batch_output_shape(self) -> None:
        """Works with batch size 1."""
        hidden_dim = 16
        layer = MessagePassingLayer(hidden_dim=hidden_dim, dropout=0.0)
        x = torch.randn(1, 3, hidden_dim)
        adj = torch.ones(1, 3, 3)

        out = layer(x, adj)

        assert out.shape == (1, 3, hidden_dim)

    def test_output_dtype(self) -> None:
        hidden_dim = 16
        layer = MessagePassingLayer(hidden_dim=hidden_dim)
        x = torch.randn(1, 4, hidden_dim)
        adj = torch.zeros(1, 4, 4)

        out = layer(x, adj)

        assert out.dtype == torch.float32


# -------------------------------------------------------------------
# GraphEncoder
# -------------------------------------------------------------------


class TestGraphEncoderShapes:
    """Verify output shapes of the full GraphEncoder."""

    def test_batched_output_shape(self) -> None:
        """Batched input (B, N, F) -> (B, N, hidden_dim)."""
        encoder = GraphEncoder(
            input_features=8,
            hidden_dim=32,
            num_mp_layers=2,
            dropout=0.0,
        )
        batch, num_nodes = 3, 6
        x = torch.randn(batch, num_nodes, 8)
        adj = torch.eye(num_nodes).unsqueeze(0).expand(batch, -1, -1)

        out = encoder(x, adj)

        assert out.shape == (batch, num_nodes, 32)

    def test_unbatched_output_shape(self) -> None:
        """Unbatched input (N, F) -> (N, hidden_dim)."""
        encoder = GraphEncoder(
            input_features=8,
            hidden_dim=32,
            num_mp_layers=2,
            dropout=0.0,
        )
        num_nodes = 4
        x = torch.randn(num_nodes, 8)
        adj = torch.eye(num_nodes)

        out = encoder(x, adj)

        assert out.shape == (num_nodes, 32)

    def test_single_node(self) -> None:
        """A single-element mesh should still work."""
        encoder = GraphEncoder(
            input_features=8,
            hidden_dim=16,
            num_mp_layers=1,
            dropout=0.0,
        )
        x = torch.randn(1, 1, 8)
        adj = torch.zeros(1, 1, 1)

        out = encoder(x, adj)

        assert out.shape == (1, 1, 16)


class TestGraphEncoderAdjacencyEffect:
    """Adjacency structure must influence the encoder output."""

    def test_different_adjacency_gives_different_output(self) -> None:
        """Different adjacency with same input gives different output."""
        torch.manual_seed(123)
        encoder = GraphEncoder(
            input_features=8,
            hidden_dim=32,
            num_mp_layers=2,
            dropout=0.0,
        )
        encoder.eval()

        num_nodes = 4
        x = torch.randn(1, num_nodes, 8)

        # Fully disconnected graph (identity adjacency = self-loops only)
        adj_disconnected = torch.eye(num_nodes).unsqueeze(0)
        out_disconnected = encoder(x, adj_disconnected)

        # Fully connected graph
        adj_connected = torch.ones(1, num_nodes, num_nodes)
        out_connected = encoder(x, adj_connected)

        # They should differ since messages flow differently.
        assert not torch.allclose(
            out_disconnected, out_connected, atol=1e-6,
        ), "Outputs should differ with different adjacency structures"

    def test_zero_adjacency_no_messages(self) -> None:
        """With zero adjacency, only self-features contribute."""
        encoder = GraphEncoder(
            input_features=8,
            hidden_dim=16,
            num_mp_layers=1,
            dropout=0.0,
        )
        encoder.eval()

        x = torch.randn(1, 3, 8)
        adj_zero = torch.zeros(1, 3, 3)

        out = encoder(x, adj_zero)

        # Output should be valid (no NaN / Inf).
        assert torch.isfinite(out).all()


# -------------------------------------------------------------------
# Adjacency matrix from DiscretizationState
# -------------------------------------------------------------------


class TestAdjacencyMatrixFromState:
    """Test to_adjacency_matrix() on DiscretizationState."""

    def test_shape(self) -> None:
        """Adjacency matrix shape should be (N, N)."""
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(mesh)
        adj = state.to_adjacency_matrix()

        n = mesh.num_elements
        assert adj.shape == (n, n)

    def test_dtype_float32(self) -> None:
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(mesh)
        adj = state.to_adjacency_matrix()

        assert adj.dtype == torch.float32

    def test_symmetry(self) -> None:
        """Adjacency matrix must be symmetric."""
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(3, 3),
        )
        state = DiscretizationState.from_mesh(mesh)
        adj = state.to_adjacency_matrix()

        assert torch.equal(adj, adj.T)

    def test_binary_entries(self) -> None:
        """All entries should be 0.0 or 1.0."""
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(mesh)
        adj = state.to_adjacency_matrix()

        unique_vals = torch.unique(adj)
        for v in unique_vals:
            assert v.item() in (0.0, 1.0)

    def test_has_neighbors(self) -> None:
        """A 2x2 quad mesh should have edges (non-zero entries)."""
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(2, 2),
        )
        state = DiscretizationState.from_mesh(mesh)
        adj = state.to_adjacency_matrix()

        assert adj.sum().item() > 0.0

    def test_single_element_no_neighbors(self) -> None:
        """A single element mesh should have an all-zero adjacency."""
        mesh = MeshGraph.create_uniform_quad(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            num_elements=(1, 1),
        )
        state = DiscretizationState.from_mesh(mesh)
        adj = state.to_adjacency_matrix()

        assert adj.shape == (1, 1)
        assert adj.sum().item() == 0.0


# -------------------------------------------------------------------
# End-to-end: AlphaGalerkinNetwork with GraphEncoder
# -------------------------------------------------------------------


class TestModelWithGraphEncoder:
    """End-to-end forward pass through model with graph_mp architecture."""

    @staticmethod
    def _graph_mp_config() -> NetworkConfig:
        return NetworkConfig(
            input_features=8,
            gnn=GNNConfig(
                architecture=GNNArchitecture.GRAPH_MP,
                hidden_dim=32,
                num_layers=2,
                attention_heads=4,
                num_mp_layers=2,
            ),
            policy_head={"hidden_dims": [32, 16]},
            value_head={"hidden_dims": [32, 16]},
        )

    def test_graph_encoder_is_set(self) -> None:
        """GRAPH_MP mode should create a graph_encoder attribute."""
        config = self._graph_mp_config()
        model = AlphaGalerkinNetwork(config)

        assert model.graph_encoder is not None
        assert isinstance(model.graph_encoder, GraphEncoder)

    def test_non_graph_encoder_is_none(self) -> None:
        """Non-GRAPH_MP modes should have graph_encoder = None."""
        config = NetworkConfig(
            input_features=8,
            gnn=GNNConfig(
                architecture=GNNArchitecture.GAT,
                hidden_dim=32,
                num_layers=2,
                attention_heads=4,
            ),
            policy_head={"hidden_dims": [32, 16]},
            value_head={"hidden_dims": [32, 16]},
        )
        model = AlphaGalerkinNetwork(config)

        assert model.graph_encoder is None

    def test_forward_with_adjacency(self) -> None:
        """Forward pass with adjacency should produce valid output."""
        config = self._graph_mp_config()
        model = AlphaGalerkinNetwork(config)

        batch, num_elements = 2, 4
        features = torch.randn(batch, num_elements, 8)
        adj = torch.eye(num_elements).unsqueeze(0).expand(
            batch, -1, -1,
        )

        policy, value = model(features, adjacency=adj)

        assert policy.shape[0] == batch
        assert policy.shape[1] == num_elements
        assert value.shape == (batch, 1)

    def test_forward_without_adjacency_falls_back(self) -> None:
        """Without adjacency, GRAPH_MP model falls back to MLP encoder."""
        config = self._graph_mp_config()
        model = AlphaGalerkinNetwork(config)

        batch, num_elements = 2, 4
        features = torch.randn(batch, num_elements, 8)

        # No adjacency provided -> falls back to self.encoder (MLP).
        policy, value = model(features)

        assert policy.shape[0] == batch
        assert value.shape == (batch, 1)

    def test_backward_pass(self) -> None:
        """Gradients should flow through the graph encoder."""
        config = self._graph_mp_config()
        model = AlphaGalerkinNetwork(config)

        batch, num_elements = 2, 4
        features = torch.randn(batch, num_elements, 8)
        adj = torch.ones(batch, num_elements, num_elements)

        policy, value = model(features, adjacency=adj)
        loss = policy.sum() + value.sum()
        loss.backward()

        # Check that graph encoder parameters received gradients.
        for name, param in model.graph_encoder.named_parameters():
            assert param.grad is not None, (
                f"No gradient for graph_encoder.{name}"
            )
