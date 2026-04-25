"""Tests for continuous embeddings and Fourier features."""

import pytest
import torch

from src.modeling.embeddings import (
    ContinuousEmbedding,
    FourierFeatures,
    StoneEmbedding,
)


class TestFourierFeatures:
    """Tests for FourierFeatures."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        ff = FourierFeatures(n_features=16)
        assert ff.n_features == 16
        assert ff.include_coordinates is True

    def test_output_dim_with_coordinates(self) -> None:
        """Test output dimension includes raw coordinates."""
        ff = FourierFeatures(n_features=16, include_coordinates=True)
        # 2 * n_features (cos + sin) + 2 (raw coords)
        assert ff.output_dim == 2 * 16 + 2

    def test_output_dim_without_coordinates(self) -> None:
        """Test output dimension without raw coordinates."""
        ff = FourierFeatures(n_features=16, include_coordinates=False)
        assert ff.output_dim == 2 * 16

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        ff = FourierFeatures(n_features=16)
        coords = torch.randn(4, 8, 2)  # batch=4, n_points=8, 2D coords
        output = ff(coords)
        assert output.shape == (4, 8, ff.output_dim)

    def test_forward_single_sample(self) -> None:
        """Test forward with single sample."""
        ff = FourierFeatures(n_features=16)
        coords = torch.randn(1, 4, 2)
        output = ff(coords)
        assert output.shape == (1, 4, ff.output_dim)

    def test_includes_raw_coordinates(self) -> None:
        """Test that raw coordinates are concatenated when enabled."""
        ff = FourierFeatures(n_features=16, include_coordinates=True)
        coords = torch.randn(2, 8, 2)
        output = ff(coords)
        # First 2 dimensions should be the raw coordinates
        # (FourierFeatures concatenates [coords, fourier_features])
        assert torch.allclose(output[:, :, :2], coords)

    def test_different_scales(self) -> None:
        """Test with different frequency scales."""
        ff_small = FourierFeatures(n_features=16, scale=0.1)
        ff_large = FourierFeatures(n_features=16, scale=10.0)
        coords = torch.randn(2, 8, 2)
        out_small = ff_small(coords)
        out_large = ff_large(coords)
        # Different scales should produce different features
        assert not torch.allclose(out_small, out_large)

    def test_learnable_mode(self) -> None:
        """Test that learnable mode creates trainable parameters."""
        ff = FourierFeatures(n_features=16, learnable=True)
        # Should have learnable parameters in fourier_basis
        params = list(ff.parameters())
        assert len(params) > 0

    def test_gradient_flow(self) -> None:
        """Test gradient flows through Fourier features."""
        ff = FourierFeatures(n_features=16, learnable=True)
        coords = torch.randn(2, 8, 2, requires_grad=True)
        output = ff(coords)
        loss = output.sum()
        loss.backward()
        assert coords.grad is not None

    def test_no_nan_output(self) -> None:
        """Test no NaN values in output."""
        ff = FourierFeatures(n_features=16)
        coords = torch.randn(2, 8, 2)
        output = ff(coords)
        assert not torch.isnan(output).any()

    def test_different_n_features(self) -> None:
        """Test with various numbers of features."""
        for n_features in [4, 16, 64]:
            ff = FourierFeatures(n_features=n_features)
            coords = torch.randn(2, 8, 2)
            output = ff(coords)
            assert output.shape[2] == ff.output_dim


class TestContinuousEmbedding:
    """Tests for ContinuousEmbedding."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        assert emb.input_channels == 17
        assert emb.d_model == 16

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        x = torch.randn(4, 17, 3, 3)  # batch=4, channels=17, 3x3 board
        output = emb(x)
        assert output.shape == (4, 9, 16)  # 3*3=9 positions

    def test_forward_different_board_sizes(self) -> None:
        """Test resolution independence with different board sizes."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        for board_size in [3, 5, 9]:
            x = torch.randn(2, 17, board_size, board_size)
            output = emb(x)
            assert output.shape == (2, board_size * board_size, 16)

    def test_forward_with_explicit_coords(self) -> None:
        """Test forward pass with explicitly provided coordinates."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        x = torch.randn(2, 17, 3, 3)
        coords = torch.randn(2, 9, 2)  # Custom coordinates
        output = emb(x, coords=coords)
        assert output.shape == (2, 9, 16)

    def test_forward_without_coords(self) -> None:
        """Test forward pass creates coordinates automatically."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        x = torch.randn(2, 17, 5, 5)
        output = emb(x)  # No coords passed
        assert output.shape == (2, 25, 16)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through embedding."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        x = torch.randn(2, 17, 3, 3, requires_grad=True)
        output = emb(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in emb.parameters():
            assert param.grad is not None

    def test_no_nan_output(self) -> None:
        """Test no NaN in output."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        x = torch.randn(2, 17, 3, 3)
        output = emb(x)
        assert not torch.isnan(output).any()

    def test_learnable_positions_disabled(self) -> None:
        """Test with learnable positions disabled (default)."""
        emb = ContinuousEmbedding(
            input_channels=17,
            d_model=16,
            n_fourier_features=8,
            use_learnable_positions=False,
        )
        assert emb.use_learnable_positions is False
        x = torch.randn(2, 17, 3, 3)
        output = emb(x)
        assert output.shape == (2, 9, 16)

    def test_learnable_positions_enabled(self) -> None:
        """Test with learnable positions enabled."""
        emb = ContinuousEmbedding(
            input_channels=17,
            d_model=16,
            n_fourier_features=8,
            use_learnable_positions=True,
        )
        assert emb.use_learnable_positions is True
        x = torch.randn(2, 17, 3, 3)
        output = emb(x)
        assert output.shape == (2, 9, 16)

    def test_different_input_channels(self) -> None:
        """Test with various input channel counts."""
        for channels in [1, 4, 17]:
            emb = ContinuousEmbedding(
                input_channels=channels,
                d_model=16,
                n_fourier_features=8,
            )
            x = torch.randn(2, channels, 3, 3)
            output = emb(x)
            assert output.shape == (2, 9, 16)

    def test_non_square_board_requires_coords(self) -> None:
        """Test that non-square boards require explicit coordinates."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        x = torch.randn(2, 17, 3, 5)  # Non-square
        with pytest.raises(AssertionError):
            emb(x)  # No coords, non-square -> should fail


class TestStoneEmbedding:
    """Tests for StoneEmbedding."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        emb = StoneEmbedding(d_model=16)
        assert emb.d_model == 16

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        emb = StoneEmbedding(d_model=16, n_stone_types=5, n_special_features=8)
        assert emb.stone_embedding.num_embeddings == 5
        assert emb.feature_projection.in_features == 8

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        emb = StoneEmbedding(d_model=16, n_stone_types=3, n_special_features=14)
        # Stone types: 0=empty, 1=black, 2=white
        stone_types = torch.randint(0, 3, (4, 3, 3)).float()
        features = torch.randn(4, 14, 3, 3)
        output = emb(stone_types, features)
        assert output.shape == (4, 9, 16)  # 3*3=9 positions

    def test_forward_different_board_sizes(self) -> None:
        """Test with different board sizes."""
        emb = StoneEmbedding(d_model=16, n_stone_types=3, n_special_features=14)
        for board_size in [3, 5, 9]:
            stone_types = torch.randint(0, 3, (2, board_size, board_size)).float()
            features = torch.randn(2, 14, board_size, board_size)
            output = emb(stone_types, features)
            assert output.shape == (2, board_size * board_size, 16)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through stone embedding."""
        emb = StoneEmbedding(d_model=16, n_stone_types=3, n_special_features=14)
        stone_types = torch.randint(0, 3, (2, 3, 3)).float()
        features = torch.randn(2, 14, 3, 3, requires_grad=True)
        output = emb(stone_types, features)
        loss = output.sum()
        loss.backward()
        assert features.grad is not None

    def test_no_nan_output(self) -> None:
        """Test no NaN in output."""
        emb = StoneEmbedding(d_model=16)
        stone_types = torch.randint(0, 3, (2, 3, 3)).float()
        features = torch.randn(2, 14, 3, 3)
        output = emb(stone_types, features)
        assert not torch.isnan(output).any()

    def test_all_empty_board(self) -> None:
        """Test with all empty intersections."""
        emb = StoneEmbedding(d_model=16)
        stone_types = torch.zeros(2, 3, 3)  # All empty
        features = torch.randn(2, 14, 3, 3)
        output = emb(stone_types, features)
        assert output.shape == (2, 9, 16)
        assert not torch.isnan(output).any()


class TestEmbeddingIntegration:
    """Integration tests for embedding modules."""

    def test_fourier_features_in_pipeline(self) -> None:
        """Test Fourier features as part of a simple pipeline."""
        ff = FourierFeatures(n_features=16)
        linear = torch.nn.Linear(ff.output_dim, 8)

        coords = torch.randn(4, 8, 2)
        features = ff(coords)
        output = linear(features)
        assert output.shape == (4, 8, 8)

    def test_continuous_embedding_gradient_full_pipeline(self) -> None:
        """Test gradient flow through embedding into a downstream layer."""
        emb = ContinuousEmbedding(input_channels=17, d_model=16, n_fourier_features=8)
        linear = torch.nn.Linear(16, 1)

        x = torch.randn(2, 17, 3, 3, requires_grad=True)
        features = emb(x)
        output = linear(features)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
