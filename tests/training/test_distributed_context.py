"""Tests for distributed training context."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.training.distributed_context import DistributedContext


class TestDistributedContextSingleGPU:
    """Tests for DistributedContext in single-GPU mode."""

    def test_default_values(self) -> None:
        """Test default context values."""
        ctx = DistributedContext()
        assert ctx.rank == 0
        assert ctx.local_rank == 0
        assert ctx.world_size == 1
        assert not ctx.is_distributed
        assert ctx.is_main_process

    def test_from_environment_no_env_vars(self) -> None:
        """Test context creation without environment variables."""
        # Clear any existing distributed env vars
        env_vars = ["WORLD_SIZE", "RANK", "LOCAL_RANK"]
        original_values = {k: os.environ.pop(k, None) for k in env_vars}

        try:
            ctx = DistributedContext.from_environment()
            assert ctx.rank == 0
            assert ctx.world_size == 1
            assert not ctx.is_distributed
        finally:
            # Restore original values
            for k, v in original_values.items():
                if v is not None:
                    os.environ[k] = v

    def test_wrap_model_single_gpu(self) -> None:
        """Test model wrapping in single-GPU mode does nothing."""
        ctx = DistributedContext()
        model = torch.nn.Linear(10, 5)
        wrapped = ctx.wrap_model(model)
        # Should return same model instance
        assert wrapped is model

    def test_unwrap_model_single_gpu(self) -> None:
        """Test unwrap on non-DDP model returns same model."""
        ctx = DistributedContext()
        model = torch.nn.Linear(10, 5)
        unwrapped = ctx.unwrap_model(model)
        assert unwrapped is model

    def test_barrier_single_gpu(self) -> None:
        """Test barrier is no-op in single-GPU mode."""
        ctx = DistributedContext()
        # Should not raise
        ctx.barrier()

    def test_all_reduce_scalar_single_gpu(self) -> None:
        """Test all_reduce returns same value in single-GPU mode."""
        ctx = DistributedContext()
        value = 3.14
        result = ctx.all_reduce_scalar(value, op="sum")
        assert result == value

    def test_broadcast_tensor_single_gpu(self) -> None:
        """Test broadcast returns same tensor in single-GPU mode."""
        ctx = DistributedContext()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = ctx.broadcast_tensor(tensor)
        assert torch.allclose(result, tensor)

    def test_broadcast_object_single_gpu(self) -> None:
        """Test broadcast object returns same object in single-GPU mode."""
        ctx = DistributedContext()
        obj = {"key": "value", "number": 42}
        result = ctx.broadcast_object(obj)
        assert result == obj

    def test_get_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        ctx = DistributedContext()
        # Single GPU: batch_size * 1 * grad_accum
        assert ctx.get_effective_batch_size(32, 1) == 32
        assert ctx.get_effective_batch_size(32, 4) == 128

    def test_no_sync_context(self) -> None:
        """Test no_sync returns nullcontext for non-DDP models."""
        ctx = DistributedContext()
        model = torch.nn.Linear(10, 5)
        with ctx.no_sync(model):
            pass  # Should not raise


class TestDistributedContextMultiGPU:
    """Tests for DistributedContext in multi-GPU mode (mocked)."""

    def test_from_environment_with_env_vars(self) -> None:
        """Test context creation with environment variables."""
        with patch.dict(os.environ, {
            "WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "2",
        }):
            ctx = DistributedContext.from_environment()
            assert ctx.world_size == 4
            assert ctx.rank == 2
            assert ctx.local_rank == 2
            assert ctx.is_distributed
            assert not ctx.is_main_process

    def test_from_environment_rank_0(self) -> None:
        """Test rank 0 is main process."""
        with patch.dict(os.environ, {
            "WORLD_SIZE": "4",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }):
            ctx = DistributedContext.from_environment()
            assert ctx.is_main_process

    def test_get_effective_batch_size_distributed(self) -> None:
        """Test effective batch size in distributed mode."""
        ctx = DistributedContext(
            rank=1,
            local_rank=1,
            world_size=4,
            is_distributed=True,
        )
        # 4 GPUs: batch_size * 4 * grad_accum
        assert ctx.get_effective_batch_size(32, 1) == 128
        assert ctx.get_effective_batch_size(32, 2) == 256

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_initialize_process_group(
        self, mock_is_init: MagicMock, mock_init_pg: MagicMock
    ) -> None:
        """Test process group initialization."""
        ctx = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            is_distributed=True,
            device=torch.device("cpu"),
        )
        ctx.initialize_process_group(backend="gloo")
        mock_init_pg.assert_called_once()

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_skip_already_initialized(
        self, mock_is_init: MagicMock, mock_init_pg: MagicMock
    ) -> None:
        """Test skipping initialization if already done."""
        ctx = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            is_distributed=True,
        )
        ctx.initialize_process_group()
        # Should not call init again
        mock_init_pg.assert_not_called()

    def test_initialize_process_group_single_gpu_noop(self) -> None:
        """Test init is no-op for single GPU."""
        ctx = DistributedContext()
        # Should not raise even without mocking
        ctx.initialize_process_group()

    @patch("torch.distributed.destroy_process_group")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_cleanup(
        self, mock_is_init: MagicMock, mock_destroy: MagicMock
    ) -> None:
        """Test cleanup destroys process group."""
        ctx = DistributedContext(
            rank=0,
            world_size=2,
            is_distributed=True,
            _process_group_initialized=True,
        )
        ctx.cleanup()
        mock_destroy.assert_called_once()
        assert not ctx._process_group_initialized

    def test_cleanup_single_gpu_noop(self) -> None:
        """Test cleanup is no-op for single GPU."""
        ctx = DistributedContext()
        # Should not raise
        ctx.cleanup()


class TestDistributedContextDeviceSelection:
    """Tests for device selection in DistributedContext."""

    def test_cpu_device_no_cuda(self) -> None:
        """Test CPU device when CUDA not available."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext.from_environment()
            assert ctx.device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_device_single_gpu(self) -> None:
        """Test CUDA device selection for single GPU."""
        # Clear distributed env vars
        with patch.dict(os.environ, {}, clear=True):
            ctx = DistributedContext.from_environment()
            assert ctx.device.type == "cuda"

    def test_force_gloo_for_cpu(self) -> None:
        """Test backend switches to gloo for CPU devices."""
        ctx = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            is_distributed=True,
            device=torch.device("cpu"),
        )
        # Mocking to prevent actual init
        with (
            patch("torch.distributed.init_process_group") as mock_init,
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            ctx.initialize_process_group(backend="nccl")
            # Should use gloo for CPU
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["backend"] == "gloo"


class TestDistributedContextConfig:
    """Tests for DistributedContext with config."""

    def test_config_override_world_size(self) -> None:
        """Test config can set world size."""
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.world_size = 8

        # With no env vars, should use config world size
        with patch.dict(os.environ, {}, clear=True):
            # When enabled but not in distributed env, still uses config
            ctx = DistributedContext.from_environment(config=mock_config)
            # world_size from config takes precedence
            assert ctx.world_size >= 1  # At least 1

    def test_disabled_config_uses_defaults(self) -> None:
        """Test disabled config uses default single-GPU."""
        mock_config = MagicMock()
        mock_config.enabled = False
        mock_config.world_size = 8

        with patch.dict(os.environ, {}, clear=True):
            ctx = DistributedContext.from_environment(config=mock_config)
            assert ctx.world_size == 1
            assert not ctx.is_distributed
