"""Tests for FNet mixing blocks."""

import torch

from src.modeling.fnet import (
    FNetBlock,
    FNetMixing,
    FNetMixingLayer,
    FNetStack,
    GalerkinFNetHybrid,
)


class TestFNetMixing:
    """Tests for FNetMixing layer."""

    def test_initialization_2d(self) -> None:
        """Test initialization with 2D FFT mode."""
        mixing = FNetMixing(use_2d=True)
        assert mixing.use_2d is True

    def test_initialization_1d(self) -> None:
        """Test initialization with 1D FFT mode."""
        mixing = FNetMixing(use_2d=False)
        assert mixing.use_2d is False

    def test_forward_1d_shape(self) -> None:
        """Test 1D FFT mixing output shape."""
        mixing = FNetMixing(use_2d=False)
        x = torch.randn(4, 8, 16)
        output = mixing(x)
        assert output.shape == (4, 8, 16)

    def test_forward_2d_shape(self) -> None:
        """Test 2D FFT mixing output shape with board_size."""
        mixing = FNetMixing(use_2d=True)
        # seq_len must be board_size^2
        board_size = 3
        x = torch.randn(4, board_size * board_size, 16)
        output = mixing(x, board_size=board_size)
        assert output.shape == (4, board_size * board_size, 16)

    def test_forward_2d_different_board_sizes(self) -> None:
        """Test 2D mixing with various board sizes."""
        mixing = FNetMixing(use_2d=True)
        for board_size in [3, 5, 9]:
            seq_len = board_size * board_size
            x = torch.randn(2, seq_len, 16)
            output = mixing(x, board_size=board_size)
            assert output.shape == (2, seq_len, 16)

    def test_forward_2d_falls_back_to_1d_without_board_size(self) -> None:
        """Test that 2D mode falls back to 1D without board_size."""
        mixing = FNetMixing(use_2d=True)
        x = torch.randn(2, 8, 16)
        # No board_size -> should use 1D path
        output = mixing(x, board_size=None)
        assert output.shape == (2, 8, 16)

    def test_mixing_modifies_input(self) -> None:
        """Test that FFT mixing actually modifies the input."""
        mixing = FNetMixing(use_2d=False)
        x = torch.randn(2, 8, 16)
        output = mixing(x)
        # The real part of FFT is not identity, so output should differ
        assert not torch.allclose(output, x)

    def test_no_nan_output(self) -> None:
        """Test that mixing does not produce NaN values."""
        mixing = FNetMixing(use_2d=True)
        x = torch.randn(2, 9, 16)
        output = mixing(x, board_size=3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestFNetBlock:
    """Tests for FNetBlock."""

    def test_initialization(self) -> None:
        """Test block initialization."""
        block = FNetBlock(d_model=16)
        assert block.d_model == 16

    def test_initialization_custom_ffn(self) -> None:
        """Test initialization with custom FFN dimension."""
        block = FNetBlock(d_model=16, d_ffn=32)
        # Check FFN first linear has correct size
        assert block.ffn[0].in_features == 16
        assert block.ffn[0].out_features == 32

    def test_default_ffn_is_4x(self) -> None:
        """Test that default FFN dimension is 4x d_model."""
        block = FNetBlock(d_model=16)
        assert block.ffn[0].out_features == 64  # 4 * 16

    def test_forward_shape_with_board_size(self) -> None:
        """Test forward pass shape with board_size for 2D FFT."""
        block = FNetBlock(d_model=16, use_2d_fft=True)
        x = torch.randn(4, 9, 16)  # 3x3 board
        output = block(x, board_size=3)
        assert output.shape == (4, 9, 16)

    def test_forward_shape_without_board_size(self) -> None:
        """Test forward pass shape without board_size (1D path)."""
        block = FNetBlock(d_model=16, use_2d_fft=False)
        x = torch.randn(4, 8, 16)
        output = block(x)
        assert output.shape == (4, 8, 16)

    def test_residual_connection(self) -> None:
        """Test that residual connections are present."""
        block = FNetBlock(d_model=16, dropout=0.0)
        block.eval()
        x = torch.randn(2, 9, 16)
        output = block(x, board_size=3)
        # Output should not be zero (residual adds input back)
        assert output.norm() > 0

    def test_gradient_flow(self) -> None:
        """Test gradient flows through FNet block."""
        block = FNetBlock(d_model=16, use_2d_fft=False)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in block.parameters():
            assert param.grad is not None

    def test_no_nan_output(self) -> None:
        """Test no NaN in output."""
        block = FNetBlock(d_model=16)
        x = torch.randn(2, 9, 16)
        output = block(x, board_size=3)
        assert not torch.isnan(output).any()

    def test_different_dropout_rates(self) -> None:
        """Test initialization with various dropout rates."""
        for dropout in [0.0, 0.1, 0.5]:
            block = FNetBlock(d_model=16, dropout=dropout)
            x = torch.randn(2, 8, 16)
            output = block(x)
            assert output.shape == (2, 8, 16)


class TestFNetStack:
    """Tests for FNetStack."""

    def test_initialization(self) -> None:
        """Test stack initialization."""
        stack = FNetStack(d_model=16, n_layers=3)
        assert len(stack.layers) == 3

    def test_forward_shape(self) -> None:
        """Test forward pass shape."""
        stack = FNetStack(d_model=16, n_layers=3, use_2d_fft=True)
        x = torch.randn(4, 9, 16)
        output = stack(x, board_size=3)
        assert output.shape == (4, 9, 16)

    def test_forward_without_board_size(self) -> None:
        """Test forward pass without board_size."""
        stack = FNetStack(d_model=16, n_layers=2, use_2d_fft=False)
        x = torch.randn(2, 8, 16)
        output = stack(x)
        assert output.shape == (2, 8, 16)

    def test_single_layer(self) -> None:
        """Test stack with single layer."""
        stack = FNetStack(d_model=16, n_layers=1)
        x = torch.randn(2, 9, 16)
        output = stack(x, board_size=3)
        assert output.shape == (2, 9, 16)

    def test_gradient_flow_through_stack(self) -> None:
        """Test gradient flows through all layers."""
        stack = FNetStack(d_model=16, n_layers=3, use_2d_fft=False)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = stack(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        # All layers should receive gradients
        for layer in stack.layers:
            for param in layer.parameters():
                assert param.grad is not None

    def test_different_board_sizes(self) -> None:
        """Test resolution independence through the stack."""
        stack = FNetStack(d_model=16, n_layers=2, use_2d_fft=True)
        for board_size in [3, 5, 7]:
            seq_len = board_size * board_size
            x = torch.randn(2, seq_len, 16)
            output = stack(x, board_size=board_size)
            assert output.shape == (2, seq_len, 16)


class TestGalerkinFNetHybrid:
    """Tests for GalerkinFNetHybrid."""

    def test_initialization(self) -> None:
        """Test hybrid layer initialization."""
        hybrid = GalerkinFNetHybrid(d_model=16, n_heads=2)
        assert isinstance(hybrid.mix_ratio, torch.nn.Parameter)

    def test_forward_shape(self) -> None:
        """Test forward pass shape."""
        hybrid = GalerkinFNetHybrid(d_model=16, n_heads=2)
        x = torch.randn(2, 9, 16)
        output = hybrid(x, board_size=3)
        assert output.shape == (2, 9, 16)

    def test_forward_without_board_size(self) -> None:
        """Test forward pass without board_size (uses 1D FNet)."""
        hybrid = GalerkinFNetHybrid(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16)
        output = hybrid(x, board_size=None)
        assert output.shape == (2, 8, 16)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through hybrid layer."""
        hybrid = GalerkinFNetHybrid(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = hybrid(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert hybrid.mix_ratio.grad is not None

    def test_mix_ratio_affects_output(self) -> None:
        """Test that different mix ratios produce different outputs."""
        hybrid = GalerkinFNetHybrid(d_model=16, n_heads=2)
        hybrid.eval()
        x = torch.randn(2, 8, 16)

        with torch.no_grad():
            hybrid.mix_ratio.fill_(10.0)  # sigmoid(10) ~ 1 -> mostly FNet
            out_fnet = hybrid(x).clone()

            hybrid.mix_ratio.fill_(-10.0)  # sigmoid(-10) ~ 0 -> mostly Galerkin
            out_galerkin = hybrid(x).clone()

        assert not torch.allclose(out_fnet, out_galerkin)


class TestFNetMixingLayer:
    """Tests for FNetMixingLayer (backward-compatible alias)."""

    def test_initialization(self) -> None:
        """Test initialization."""
        layer = FNetMixingLayer(d_model=16)
        assert layer.d_model == 16

    def test_forward_shape(self) -> None:
        """Test forward pass shape."""
        layer = FNetMixingLayer(d_model=16)
        grid_size = 4
        x = torch.randn(2, grid_size * grid_size, 16)
        output = layer(x, grid_size=grid_size)
        assert output.shape == (2, grid_size * grid_size, 16)

    def test_includes_residual_and_norm(self) -> None:
        """Test that the layer includes residual connection and norm."""
        layer = FNetMixingLayer(d_model=16)
        layer.eval()
        x = torch.randn(2, 9, 16)
        output = layer(x, grid_size=3)
        # Due to residual + norm, output should differ from zero input
        assert output.norm() > 0

    def test_no_nan_output(self) -> None:
        """Test no NaN values in output."""
        layer = FNetMixingLayer(d_model=16)
        x = torch.randn(2, 16, 16)
        output = layer(x, grid_size=4)
        assert not torch.isnan(output).any()
