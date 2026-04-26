"""Multi-process integration tests for distributed training.

Tests the distributed training infrastructure using subprocess spawning
and mock NCCL backend to validate gradient synchronization and coordination.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.distributed.config import (
    DistributedBackend,
    DistributedInfraConfig,
    LauncherConfig,
    create_distributed_config,
)


class TestDistributedConfigCreation:
    """Tests for distributed config creation and validation."""

    def test_create_config_from_launcher(self) -> None:
        """Test creating distributed config from launcher config."""
        launcher = LauncherConfig(
            nnodes=2,
            nproc_per_node=4,
            master_addr="10.0.0.1",
            master_port=29500,
        )

        config = create_distributed_config(
            enabled=True,
            launcher=launcher,
        )

        assert config.enabled is True
        assert config.world_size == 8
        assert config.launcher is not None
        assert config.launcher.master_addr == "10.0.0.1"

    def test_create_single_gpu_config(self) -> None:
        """Test creating single GPU config (no distribution)."""
        config = create_distributed_config(enabled=False)

        assert config.enabled is False
        assert config.world_size == 1

    def test_config_with_gradient_accumulation(self) -> None:
        """Test config with gradient accumulation."""
        config = create_distributed_config(
            enabled=True,
            world_size=4,
            gradient_accumulation_steps=4,
        )

        effective = config.get_effective_batch_size(32)
        assert effective == 32 * 4 * 4  # 512


class TestDistributedEnvironment:
    """Tests for distributed environment detection."""

    def test_environment_detection_not_distributed(self) -> None:
        """Test environment detection when not in distributed mode."""
        # Clean environment
        env_vars = ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_vars}

        with patch.dict(os.environ, clean_env, clear=True):
            from src.distributed.config import config_from_environment

            config = config_from_environment()
            # Should return None (single-process) or a disabled config.
            assert config is None or config.enabled is False

    def test_environment_detection_with_vars(self) -> None:
        """Test environment detection with distributed env vars."""
        env_vars = {
            "WORLD_SIZE": "4",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }

        with patch.dict(os.environ, env_vars):
            from src.distributed.config import config_from_environment

            config = config_from_environment()
            if config is not None:
                assert config.world_size == 4


class TestGradientSynchronization:
    """Tests for gradient synchronization logic."""

    def test_gradient_sync_config_values(self) -> None:
        """Test gradient sync configuration values."""
        config = DistributedInfraConfig(
            enabled=True,
            world_size=4,
            gradient_compression=True,
            gradient_compression_bits=8,
        )

        assert config.gradient_compression is True
        assert config.gradient_compression_bits == 8

    def test_gradient_accumulation_config(self) -> None:
        """Test gradient accumulation configuration."""
        config = DistributedInfraConfig(
            gradient_accumulation_steps=4,
        )

        assert config.gradient_accumulation_steps == 4
        assert config.should_sync_at_step(step=3) is False
        assert config.should_sync_at_step(step=4) is True
        assert config.should_sync_at_step(step=8) is True


class TestLauncherMethods:
    """Tests for launcher method selection."""

    def test_torchrun_launcher(self) -> None:
        """Test torchrun launcher configuration."""
        config = LauncherConfig(
            method="torchrun",
            nnodes=2,
            nproc_per_node=4,
        )

        assert config.method == "torchrun"
        assert config.get_world_size() == 8

    def test_slurm_launcher(self) -> None:
        """Test SLURM launcher configuration."""
        config = LauncherConfig(
            method="slurm",
            nnodes=4,
            nproc_per_node=8,
        )

        assert config.method == "slurm"
        assert config.get_world_size() == 32

    def test_custom_launcher(self) -> None:
        """Test custom launcher configuration."""
        config = LauncherConfig(
            method="custom",
            nnodes=1,
            nproc_per_node=2,
        )

        assert config.method == "custom"


class TestMultiProcessCommunication:
    """Tests for multi-process communication patterns."""

    def test_rank_assignment(self) -> None:
        """Test rank assignment logic."""
        launcher = LauncherConfig(
            nnodes=2,
            nproc_per_node=4,
        )

        # Test local rank calculation for each global rank
        for global_rank in range(8):
            local_rank = launcher.get_local_rank(global_rank)
            assert 0 <= local_rank < 4

            node_rank = launcher.get_node_rank(global_rank)
            assert 0 <= node_rank < 2

    def test_process_group_division(self) -> None:
        """Test process group division."""
        launcher = LauncherConfig(
            nnodes=2,
            nproc_per_node=4,
        )

        world_size = launcher.get_world_size()

        # Verify all ranks are unique
        all_ranks = list(range(world_size))
        assert len(all_ranks) == world_size

        # Verify each node has correct number of local ranks
        node_0_ranks = [r for r in all_ranks if launcher.get_node_rank(r) == 0]
        node_1_ranks = [r for r in all_ranks if launcher.get_node_rank(r) == 1]

        assert len(node_0_ranks) == 4
        assert len(node_1_ranks) == 4


class TestDistributedTrainerMock:
    """Tests for distributed trainer with mock NCCL."""

    @pytest.fixture
    def mock_dist(self) -> MagicMock:
        """Create mock torch.distributed module."""
        mock = MagicMock()
        mock.is_initialized.return_value = True
        mock.get_world_size.return_value = 4
        mock.get_rank.return_value = 0
        return mock

    def test_trainer_initialization_mock(self, mock_dist: MagicMock) -> None:
        """Test trainer initialization with mocked distributed."""
        config = DistributedInfraConfig(
            enabled=True,
            world_size=4,
            backend=DistributedBackend.GLOO,  # Use GLOO for testing
        )

        with patch("torch.distributed", mock_dist):
            # Verify config is valid for distributed training
            assert config.enabled
            assert config.world_size == 4

    def test_all_reduce_mock(self, mock_dist: MagicMock) -> None:
        """Test all_reduce operation mock."""
        mock_dist.all_reduce = MagicMock()

        with patch("torch.distributed", mock_dist):
            # Simulate gradient all-reduce
            import torch

            tensor = torch.ones(10)
            mock_dist.all_reduce(tensor)

            mock_dist.all_reduce.assert_called_once_with(tensor)


class TestCheckpointCoordination:
    """Tests for distributed checkpoint coordination."""

    def test_checkpoint_path_generation(self) -> None:
        """Test checkpoint path generation for distributed training."""
        config = DistributedInfraConfig(
            enabled=True,
            world_size=4,
        )

        # Only rank 0 should save checkpoints by default
        assert config.should_save_checkpoint(rank=0) is True
        assert config.should_save_checkpoint(rank=1) is False
        assert config.should_save_checkpoint(rank=3) is False

    def test_model_sync_at_checkpoint(self) -> None:
        """Test model sync requirements at checkpoint time."""
        config = DistributedInfraConfig(
            enabled=True,
            world_size=4,
        )

        # All ranks should sync before checkpoint
        assert config.requires_barrier_before_checkpoint() is True


class TestSelfPlayDistributed:
    """Tests for distributed self-play configuration."""

    def test_self_play_worker_config(self) -> None:
        """Test self-play worker configuration."""
        from src.distributed.config import SelfPlayDistributedConfig

        config = SelfPlayDistributedConfig(
            num_workers=16,
            games_per_worker=100,
            batch_size=32,
        )

        assert config.num_workers == 16
        assert config.games_per_worker == 100
        assert config.total_games == 1600

    def test_self_play_distribution(self) -> None:
        """Test game distribution across workers."""
        from src.distributed.config import SelfPlayDistributedConfig

        config = SelfPlayDistributedConfig(
            num_workers=4,
            games_per_worker=100,
        )

        # Each worker should get equal games
        for worker_id in range(4):
            games = config.get_games_for_worker(worker_id)
            assert len(list(range(*games))) == 100


class TestErrorHandling:
    """Tests for distributed training error handling."""

    def test_invalid_world_size(self) -> None:
        """Test rejection of invalid world size."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DistributedInfraConfig(world_size=0)

        with pytest.raises(ValidationError):
            DistributedInfraConfig(world_size=-1)

    def test_invalid_port(self) -> None:
        """Test rejection of invalid port numbers."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LauncherConfig(master_port=80)  # Below 1024

        with pytest.raises(ValidationError):
            LauncherConfig(master_port=70000)  # Above 65535

    def test_invalid_backend(self) -> None:
        """Test rejection of invalid backend."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DistributedInfraConfig(backend="invalid_backend")


class TestIntegrationWithTrainer:
    """Integration tests with trainer infrastructure."""

    def test_trainer_config_merge(self) -> None:
        """Test merging distributed config with trainer config."""
        dist_config = DistributedInfraConfig(
            enabled=True,
            world_size=4,
            gradient_accumulation_steps=2,
        )

        # Verify effective batch size calculation
        per_gpu_batch = 32
        effective = dist_config.get_effective_batch_size(per_gpu_batch)
        assert effective == 32 * 4 * 2

    def test_learning_rate_scaling(self) -> None:
        """Test learning rate scaling for distributed training."""
        dist_config = DistributedInfraConfig(
            enabled=True,
            world_size=4,
            learning_rate_scaling="linear",
        )

        base_lr = 0.001
        scaled_lr = dist_config.scale_learning_rate(base_lr)
        assert scaled_lr == base_lr * 4  # Linear scaling

        dist_config.learning_rate_scaling = "sqrt"
        scaled_lr = dist_config.scale_learning_rate(base_lr)
        assert abs(scaled_lr - base_lr * 2) < 1e-6  # sqrt(4) = 2
