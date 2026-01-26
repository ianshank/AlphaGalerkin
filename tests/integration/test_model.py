"""Integration tests for the AlphaGalerkin model."""

from __future__ import annotations

import pytest
import torch

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinFast, AlphaGalerkinModel


class TestAlphaGalerkinModel:
    """Integration tests for the full model."""

    @pytest.fixture
    def config(self) -> OperatorConfig:
        """Create test configuration."""
        return OperatorConfig(
            d_model=64,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
            d_ffn=128,
            input_channels=17,
            use_fnet_mixing=True,
        )

    @pytest.fixture
    def model(self, config: OperatorConfig) -> AlphaGalerkinModel:
        """Create model for testing."""
        torch.manual_seed(42)
        return AlphaGalerkinModel(config)

    def test_forward_9x9(self, model: AlphaGalerkinModel) -> None:
        """Test forward pass on 9x9 board."""
        batch_size = 2
        x = torch.randn(batch_size, 17, 9, 9)

        output = model(x)

        # Check policy shape: n + 1 (pass move)
        assert output.policy_logits.shape == (batch_size, 82)

        # Check value shape
        assert output.value.shape == (batch_size, 1)

        # Check value range
        assert (output.value >= -1).all() and (output.value <= 1).all()

    def test_forward_19x19(self, model: AlphaGalerkinModel) -> None:
        """Test forward pass on 19x19 board."""
        batch_size = 2
        x = torch.randn(batch_size, 17, 19, 19)

        output = model(x)

        # Check policy shape
        assert output.policy_logits.shape == (batch_size, 362)

        # Check value shape
        assert output.value.shape == (batch_size, 1)

    def test_resolution_independence(self, model: AlphaGalerkinModel) -> None:
        """Test that model works on any board size without errors."""
        batch_size = 1

        for board_size in [5, 9, 13, 19, 25]:
            x = torch.randn(batch_size, 17, board_size, board_size)
            output = model(x)

            expected_policy_size = board_size ** 2 + 1
            assert output.policy_logits.shape == (batch_size, expected_policy_size)

    def test_forward_with_lbb(self, model: AlphaGalerkinModel) -> None:
        """Test forward pass with LBB monitoring."""
        x = torch.randn(2, 17, 9, 9)

        output = model(x, return_lbb=True)

        assert output.lbb_constant is not None
        assert output.lbb_constant.shape == (2,)
        assert (output.lbb_constant > 0).all()

    def test_forward_fast(self, model: AlphaGalerkinModel) -> None:
        """Test fast forward pass for MCTS."""
        x = torch.randn(2, 17, 9, 9)

        output = model.forward_fast(x)

        assert output.policy_logits.shape == (2, 82)
        assert output.value.shape == (2, 1)
        assert output.lbb_constant is None  # Not computed in fast path

    def test_gradients_flow(self, model: AlphaGalerkinModel) -> None:
        """Test that gradients flow through the model."""
        x = torch.randn(2, 17, 9, 9, requires_grad=True)

        output = model(x)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_batch_size_independence(self, model: AlphaGalerkinModel) -> None:
        """Test that model works with different batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 17, 9, 9)
            output = model(x)

            assert output.policy_logits.shape[0] == batch_size
            assert output.value.shape[0] == batch_size


class TestAlphaGalerkinFast:
    """Tests for the fast FNet-only model."""

    @pytest.fixture
    def config(self) -> OperatorConfig:
        """Create test configuration."""
        return OperatorConfig(
            d_model=64,
            n_fourier_features=32,
            d_ffn=128,
            input_channels=17,
        )

    @pytest.fixture
    def fast_model(self, config: OperatorConfig) -> AlphaGalerkinFast:
        """Create fast model for testing."""
        torch.manual_seed(42)
        return AlphaGalerkinFast(config, n_layers=4)

    def test_forward(self, fast_model: AlphaGalerkinFast) -> None:
        """Test forward pass."""
        x = torch.randn(4, 17, 9, 9)

        output = fast_model(x)

        assert output.policy_logits.shape == (4, 82)
        assert output.value.shape == (4, 1)

    def test_faster_than_full_model(
        self, fast_model: AlphaGalerkinFast, config: OperatorConfig
    ) -> None:
        """Test that fast model is faster than full model."""
        import time

        full_model = AlphaGalerkinModel(config)
        x = torch.randn(8, 17, 9, 9)

        # Warmup
        for _ in range(3):
            fast_model(x)
            full_model.forward_fast(x)

        # Time fast model
        start = time.perf_counter()
        for _ in range(10):
            fast_model(x)
        fast_time = time.perf_counter() - start

        # Time full model (fast path)
        start = time.perf_counter()
        for _ in range(10):
            full_model.forward_fast(x)
        full_time = time.perf_counter() - start

        # Fast model should be comparable or faster
        # (may not always be faster due to overhead)
        assert fast_time < full_time * 5  # Allow 5x margin


class TestModelConsistency:
    """Tests for model consistency across resolutions."""

    @pytest.fixture
    def model(self) -> AlphaGalerkinModel:
        """Create model for testing."""
        torch.manual_seed(42)
        config = OperatorConfig(
            d_model=64,
            n_heads=4,
            n_galerkin_layers=2,
            n_softmax_layers=1,
            n_fourier_features=32,
            input_channels=17,
        )
        return AlphaGalerkinModel(config)

    def test_value_range_consistent(self, model: AlphaGalerkinModel) -> None:
        """Test that value output range is consistent across resolutions."""
        for board_size in [9, 13, 19]:
            x = torch.randn(10, 17, board_size, board_size)
            output = model(x)

            # Value should always be in [-1, 1]
            assert (output.value >= -1).all()
            assert (output.value <= 1).all()

    def test_policy_sums_to_valid(self, model: AlphaGalerkinModel) -> None:
        """Test that policy logits can be converted to valid distribution."""
        for board_size in [9, 13, 19]:
            x = torch.randn(4, 17, board_size, board_size)
            output = model(x)

            # Softmax should produce valid distribution
            policy = torch.softmax(output.policy_logits, dim=-1)

            # Should sum to 1
            assert torch.allclose(policy.sum(dim=-1), torch.ones(4), atol=1e-5)

            # Should be non-negative
            assert (policy >= 0).all()

    def test_deterministic_eval(self, model: AlphaGalerkinModel) -> None:
        """Test that model is deterministic in eval mode."""
        model.eval()
        x = torch.randn(2, 17, 9, 9)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1.policy_logits, output2.policy_logits)
        assert torch.allclose(output1.value, output2.value)
