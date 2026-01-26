"""Tests for gradient synchronization module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.distributed.config import DistributedConfig
from src.distributed.gradient_sync import (
    GradientAccumulator,
    GradientSynchronizer,
    SyncMetrics,
)


class SimpleModel(nn.Module):
    """Simple model for testing gradient synchronization."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def distributed_config() -> DistributedConfig:
    """Create a distributed config for testing."""
    return DistributedConfig(
        enabled=True,
        backend="gloo",
        world_size=2,
        gradient_accumulation_steps=1,
        gradient_compression=False,
    )


class TestSyncMetrics:
    """Tests for SyncMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = SyncMetrics()

        assert metrics.sync_time_ms == 0.0
        assert metrics.gradient_norm_before == 0.0
        assert metrics.gradient_norm_after == 0.0
        assert metrics.num_parameters == 0
        assert metrics.compression_ratio == 1.0

    def test_custom_values(self) -> None:
        """Test metrics with custom values."""
        metrics = SyncMetrics(
            sync_time_ms=10.5,
            gradient_norm_before=1.5,
            gradient_norm_after=1.2,
            num_parameters=1000,
            compression_ratio=2.0,
        )

        assert metrics.sync_time_ms == 10.5
        assert metrics.gradient_norm_before == 1.5
        assert metrics.gradient_norm_after == 1.2
        assert metrics.num_parameters == 1000
        assert metrics.compression_ratio == 2.0


class TestGradientSynchronizer:
    """Tests for GradientSynchronizer class."""

    def test_initialization(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test synchronizer initialization."""
        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, distributed_config)

            assert sync.model is simple_model
            assert sync.config is distributed_config
            assert sync.accumulation_counter == 0

    def test_synchronize_non_distributed_returns_empty_metrics(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test synchronize returns empty metrics when distributed is not initialized."""
        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, distributed_config)

            metrics = sync.synchronize()

            assert metrics.sync_time_ms == 0.0
            assert metrics.num_parameters == 0

    def test_synchronize_no_gradients(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test synchronize with no gradients attached."""
        with patch("torch.distributed.is_initialized", return_value=True):
            sync = GradientSynchronizer(simple_model, distributed_config)

            # No gradients attached
            metrics = sync.synchronize()

            assert metrics.num_parameters == 0

    def test_should_sync_accumulation_steps_one(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test should_sync with accumulation_steps=1."""
        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, distributed_config)

            # With accumulation_steps=1, should always sync
            assert sync.should_sync() is True

    def test_should_sync_with_accumulation(
        self,
        simple_model: SimpleModel,
    ) -> None:
        """Test should_sync with gradient accumulation."""
        config = DistributedConfig(
            enabled=True,
            backend="gloo",
            world_size=2,
            gradient_accumulation_steps=3,
        )

        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, config)

            # Initially counter is 0, (0 + 1) % 3 != 0
            assert sync.should_sync() is False

            sync.step()  # counter = 1
            assert sync.should_sync() is False

            sync.step()  # counter = 2
            assert sync.should_sync() is True  # (2 + 1) % 3 == 0

    def test_step_and_reset(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test step and reset counter methods."""
        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, distributed_config)

            assert sync.accumulation_counter == 0

            sync.step()
            assert sync.accumulation_counter == 1

            sync.step()
            assert sync.accumulation_counter == 2

            sync.reset()
            assert sync.accumulation_counter == 0

    def test_compute_grad_norm(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test gradient norm computation."""
        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, distributed_config)

            # Create known gradients
            grads = [torch.ones(10), torch.ones(20)]
            norm = sync._compute_grad_norm(grads)

            # Expected: sqrt(10 + 20) = sqrt(30)
            expected = (10.0 + 20.0) ** 0.5
            assert abs(norm - expected) < 0.01

    def test_all_reduce_scalar_non_distributed(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test scalar all-reduce without distributed."""
        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, distributed_config)

            # Should return input unchanged
            result = sync.all_reduce_scalar(5.0)
            assert result == 5.0

            result = sync.all_reduce_scalar(torch.tensor(3.0))
            assert result == 3.0


class TestGradientAccumulator:
    """Tests for GradientAccumulator helper class."""

    def test_scale_loss(self) -> None:
        """Test loss scaling for gradient accumulation."""
        accumulator = GradientAccumulator(accumulation_steps=4, scale_loss=True)

        scaled = accumulator.scale(4.0)
        assert scaled == 1.0  # 4.0 / 4

    def test_scale_loss_disabled(self) -> None:
        """Test loss scaling when disabled."""
        accumulator = GradientAccumulator(accumulation_steps=4, scale_loss=False)

        scaled = accumulator.scale(4.0)
        assert scaled == 4.0  # Unchanged

    def test_accumulate_and_average(self) -> None:
        """Test loss accumulation and averaging."""
        accumulator = GradientAccumulator(accumulation_steps=3)

        accumulator.accumulate(1.0)
        accumulator.accumulate(2.0)
        accumulator.accumulate(3.0)

        assert accumulator.accumulated_loss == 6.0
        assert accumulator.get_average_loss() == 2.0
        assert accumulator._step_count == 3

    def test_should_step(self) -> None:
        """Test optimizer step decision."""
        accumulator = GradientAccumulator(accumulation_steps=2)

        accumulator.accumulate(1.0)
        assert accumulator.should_step() is False

        accumulator.accumulate(1.0)
        assert accumulator.should_step() is True

    def test_reset(self) -> None:
        """Test accumulator reset."""
        accumulator = GradientAccumulator(accumulation_steps=2)

        accumulator.accumulate(1.0)
        accumulator.accumulate(1.0)

        accumulator.reset()

        assert accumulator.accumulated_loss == 0.0
        assert accumulator._step_count == 0


class TestGradientCompression:
    """Tests for gradient compression functionality."""

    def test_compress_decompress_roundtrip(
        self,
        simple_model: SimpleModel,
        distributed_config: DistributedConfig,
    ) -> None:
        """Test compression/decompression preserves top values."""
        config = DistributedConfig(
            enabled=True,
            backend="gloo",
            world_size=2,
            gradient_compression=True,
        )

        with patch("torch.distributed.is_initialized", return_value=False):
            sync = GradientSynchronizer(simple_model, config)

            # Create gradient tensor
            original = torch.randn(100)

            # Compress
            compressed, ratio = sync._compress_gradients(original)

            # Verify compression occurred
            assert compressed.numel() < original.numel()
            assert ratio > 1.0

            # Decompress
            decompressed = sync._decompress_gradients(compressed, original.numel())

            # Verify size matches
            assert decompressed.numel() == original.numel()

            # Top 10% values should be preserved
            _, top_indices = torch.topk(original.abs(), k=int(0.1 * original.numel()))
            for idx in top_indices:
                # Check that top values are approximately preserved
                assert decompressed[idx] != 0.0
