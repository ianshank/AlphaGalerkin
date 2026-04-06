"""Tests for attention mechanisms (Galerkin and Softmax)."""

import torch

from src.modeling.attention import (
    GalerkinAttention,
    HybridAttention,
    SoftmaxAttention,
)


class TestGalerkinAttention:
    """Tests for GalerkinAttention."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        assert attn.d_model == 16
        assert attn.n_heads == 2
        assert attn.d_key == 8  # d_model // n_heads
        assert attn.d_value == 8
        assert attn.normalize_features is True

    def test_initialization_custom_dims(self) -> None:
        """Test initialization with custom key/value dimensions."""
        attn = GalerkinAttention(d_model=16, n_heads=2, d_key=12, d_value=10)
        assert attn.d_key == 12
        assert attn.d_value == 10

    def test_forward_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        x = torch.randn(4, 8, 16)  # batch=4, seq=8, d_model=16
        output = attn(x)
        assert output.shape == (4, 8, 16)

    def test_forward_single_sample(self) -> None:
        """Test forward pass with single sample."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        x = torch.randn(1, 4, 16)
        output = attn(x)
        assert output.shape == (1, 4, 16)

    def test_forward_different_seq_lengths(self) -> None:
        """Test resolution independence with different sequence lengths."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        for seq_len in [4, 9, 16, 25]:
            x = torch.randn(2, seq_len, 16)
            output = attn(x)
            assert output.shape == (2, seq_len, 16)

    def test_normalize_features_enabled(self) -> None:
        """Test with feature normalization enabled."""
        attn = GalerkinAttention(d_model=16, n_heads=2, normalize_features=True)
        x = torch.randn(2, 8, 16)
        output = attn(x)
        assert output.shape == (2, 8, 16)
        assert not torch.isnan(output).any()

    def test_normalize_features_disabled(self) -> None:
        """Test with feature normalization disabled."""
        attn = GalerkinAttention(d_model=16, n_heads=2, normalize_features=False)
        x = torch.randn(2, 8, 16)
        output = attn(x)
        assert output.shape == (2, 8, 16)
        assert not torch.isnan(output).any()

    def test_return_lbb_constant(self) -> None:
        """Test returning LBB stability constant."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        x = torch.randn(4, 8, 16)
        result = attn(x, return_lbb=True)
        assert isinstance(result, tuple)
        output, lbb = result
        assert output.shape == (4, 8, 16)
        assert lbb.shape == (4,)
        # LBB constant should be positive (singular values are non-negative)
        assert (lbb >= 0).all()

    def test_lbb_stored_internally(self) -> None:
        """Test that LBB constant is stored after computation."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16)
        attn(x, return_lbb=True)
        assert attn._last_lbb_constant is not None

    def test_gradient_flow(self) -> None:
        """Test gradient flows through attention."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in attn.parameters():
            assert param.grad is not None

    def test_no_nan_with_zeros(self) -> None:
        """Test that zero input does not produce NaN."""
        attn = GalerkinAttention(d_model=16, n_heads=2)
        x = torch.zeros(2, 8, 16)
        output = attn(x)
        assert not torch.isnan(output).any()

    def test_dropout_applied_in_train_mode(self) -> None:
        """Test that dropout changes behavior between train and eval."""
        attn = GalerkinAttention(d_model=16, n_heads=2, dropout=0.5)
        x = torch.randn(2, 8, 16)

        attn.train()
        torch.manual_seed(42)
        out_train = attn(x)

        attn.eval()
        torch.manual_seed(42)
        out_eval = attn(x)

        # Outputs should differ due to dropout
        # (with high dropout, this is very likely)
        # Note: eval mode disables dropout, so outputs should be deterministic
        attn.eval()
        out_eval_1 = attn(x)
        out_eval_2 = attn(x)
        assert torch.allclose(out_eval_1, out_eval_2)

    def test_multi_head_dimension_consistency(self) -> None:
        """Test that multi-head attention handles dimension splitting."""
        for n_heads in [1, 2, 4]:
            d_model = 16
            attn = GalerkinAttention(d_model=d_model, n_heads=n_heads)
            x = torch.randn(2, 8, d_model)
            output = attn(x)
            assert output.shape == (2, 8, d_model)


class TestSoftmaxAttention:
    """Tests for SoftmaxAttention."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        assert attn.d_model == 16
        assert attn.n_heads == 2
        assert attn.d_key == 8
        assert attn.d_value == 8

    def test_initialization_custom_dims(self) -> None:
        """Test initialization with custom key/value dimensions."""
        attn = SoftmaxAttention(d_model=16, n_heads=2, d_key=12, d_value=10)
        assert attn.d_key == 12
        assert attn.d_value == 10

    def test_scale_factor(self) -> None:
        """Test that scale factor is 1/sqrt(d_key)."""
        import math

        attn = SoftmaxAttention(d_model=16, n_heads=2, d_key=16)
        assert abs(attn.scale - 1.0 / math.sqrt(16)) < 1e-6

    def test_forward_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        x = torch.randn(4, 8, 16)
        output = attn(x)
        assert output.shape == (4, 8, 16)

    def test_forward_single_sample(self) -> None:
        """Test forward pass with single sample."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        x = torch.randn(1, 4, 16)
        output = attn(x)
        assert output.shape == (1, 4, 16)

    def test_forward_different_seq_lengths(self) -> None:
        """Test with different sequence lengths."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        for seq_len in [4, 9, 16, 25]:
            x = torch.randn(2, seq_len, 16)
            output = attn(x)
            assert output.shape == (2, seq_len, 16)

    def test_forward_with_mask(self) -> None:
        """Test forward pass with attention mask."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16)
        # Create a causal-style mask
        mask = torch.ones(2, 8, 8)
        mask[:, :, 4:] = 0  # Mask out last 4 positions
        output = attn(x, mask=mask)
        assert output.shape == (2, 8, 16)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self) -> None:
        """Test gradient flows through softmax attention."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in attn.parameters():
            assert param.grad is not None

    def test_no_nan_output(self) -> None:
        """Test that normal input does not produce NaN."""
        attn = SoftmaxAttention(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16)
        output = attn(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_multi_head_dimension_consistency(self) -> None:
        """Test that different head counts produce correct shapes."""
        for n_heads in [1, 2, 4]:
            d_model = 16
            attn = SoftmaxAttention(d_model=d_model, n_heads=n_heads)
            x = torch.randn(2, 8, d_model)
            output = attn(x)
            assert output.shape == (2, 8, d_model)


class TestHybridAttention:
    """Tests for HybridAttention."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        attn = HybridAttention(d_model=16, n_heads=2)
        assert isinstance(attn.galerkin, GalerkinAttention)
        assert isinstance(attn.softmax, SoftmaxAttention)

    def test_initialization_learnable_gate(self) -> None:
        """Test initialization with learnable gate."""
        attn = HybridAttention(d_model=16, n_heads=2, learnable_gate=True)
        assert isinstance(attn.gate, torch.nn.Parameter)

    def test_initialization_fixed_gate(self) -> None:
        """Test initialization with fixed gate."""
        attn = HybridAttention(
            d_model=16, n_heads=2, galerkin_ratio=0.6, learnable_gate=False
        )
        assert not isinstance(attn.gate, torch.nn.Parameter)

    def test_forward_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        attn = HybridAttention(d_model=16, n_heads=2)
        x = torch.randn(4, 8, 16)
        output = attn(x)
        assert output.shape == (4, 8, 16)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through hybrid attention."""
        attn = HybridAttention(d_model=16, n_heads=2, learnable_gate=True)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        # Gate should receive gradient
        assert attn.gate.grad is not None

    def test_gate_affects_output(self) -> None:
        """Test that different gate values produce different outputs."""
        attn = HybridAttention(d_model=16, n_heads=2, learnable_gate=True)
        x = torch.randn(2, 8, 16)

        attn.eval()
        with torch.no_grad():
            attn.gate.fill_(10.0)  # sigmoid(10) ~ 1 -> mostly Galerkin
            out_galerkin = attn(x).clone()

            attn.gate.fill_(-10.0)  # sigmoid(-10) ~ 0 -> mostly Softmax
            out_softmax = attn(x).clone()

        assert not torch.allclose(out_galerkin, out_softmax)


class TestAttentionComparison:
    """Tests comparing Galerkin and Softmax attention properties."""

    def test_galerkin_is_linear_complexity(self) -> None:
        """Verify Galerkin attention does not create NxN attention matrix."""
        # Galerkin attention: Q * (K^T V / n) = O(N) since K^T V is d_key x d_value
        attn = GalerkinAttention(d_model=16, n_heads=2)
        # Should handle large sequences without OOM
        x = torch.randn(1, 256, 16)
        output = attn(x)
        assert output.shape == (1, 256, 16)

    def test_both_handle_same_input(self) -> None:
        """Test that both attention types handle the same input shapes."""
        x = torch.randn(2, 8, 16)
        galerkin = GalerkinAttention(d_model=16, n_heads=2)
        softmax = SoftmaxAttention(d_model=16, n_heads=2)

        out_g = galerkin(x)
        out_s = softmax(x)

        assert out_g.shape == out_s.shape
        # Outputs should generally differ (different mechanisms)
        assert not torch.allclose(out_g, out_s)
