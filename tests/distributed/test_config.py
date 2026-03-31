"""Tests for distributed training configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.distributed.config import (
    DistributedBackend,
    DistributedInfraConfig,
    LauncherConfig,
    SelfPlayDistributedConfig,
    create_distributed_config,
    from_environment,
)


class TestDistributedInfraConfig:
    """Tests for DistributedInfraConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DistributedInfraConfig()

        assert config.enabled is False
        assert config.world_size == 1
        assert config.backend == DistributedBackend.NCCL
        assert config.gradient_accumulation_steps == 1
        assert config.sync_batch_norm is True

    def test_enabled_with_world_size(self) -> None:
        """Test that enabled=True requires world_size > 1."""
        config = DistributedInfraConfig(enabled=True, world_size=4)

        assert config.enabled is True
        assert config.world_size == 4

    def test_backend_validation(self) -> None:
        """Test backend enum validation."""
        config = DistributedInfraConfig(backend="gloo")
        assert config.backend == DistributedBackend.GLOO

        config = DistributedInfraConfig(backend=DistributedBackend.NCCL)
        assert config.backend == DistributedBackend.NCCL

    def test_invalid_world_size(self) -> None:
        """Test that world_size must be >= 1."""
        with pytest.raises(ValidationError):
            DistributedInfraConfig(world_size=0)

    def test_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        config = DistributedInfraConfig(
            world_size=4,
            gradient_accumulation_steps=2,
        )

        effective = config.get_effective_batch_size(per_gpu_batch_size=32)
        assert effective == 32 * 4 * 2  # 256

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            DistributedInfraConfig(unknown_field="value")


class TestLauncherConfig:
    """Tests for LauncherConfig."""

    def test_default_values(self) -> None:
        """Test default launcher configuration."""
        config = LauncherConfig()

        assert config.method == "torchrun"
        assert config.nnodes == 1
        assert config.nproc_per_node == 1
        assert config.master_addr == "localhost"
        assert config.master_port == 29500

    def test_world_size_calculation(self) -> None:
        """Test world size calculation."""
        config = LauncherConfig(nnodes=2, nproc_per_node=4)

        assert config.get_world_size() == 8

    def test_local_rank_calculation(self) -> None:
        """Test local rank calculation."""
        config = LauncherConfig(nproc_per_node=4)

        assert config.get_local_rank(global_rank=0) == 0
        assert config.get_local_rank(global_rank=1) == 1
        assert config.get_local_rank(global_rank=4) == 0
        assert config.get_local_rank(global_rank=5) == 1

    def test_port_validation(self) -> None:
        """Test port number validation."""
        with pytest.raises(ValidationError):
            LauncherConfig(master_port=1000)  # Below 1024

        with pytest.raises(ValidationError):
            LauncherConfig(master_port=70000)  # Above 65535


class TestSelfPlayDistributedConfig:
    """Tests for SelfPlayDistributedConfig."""

    def test_default_values(self) -> None:
        """Test default self-play configuration."""
        config = SelfPlayDistributedConfig()

        assert config.workers_per_node == 2
        assert config.games_per_worker == 50
        assert config.experience_sharing == "global"

    def test_experience_sharing_options(self) -> None:
        """Test experience sharing strategy options."""
        for strategy in ["local", "global", "hierarchical"]:
            config = SelfPlayDistributedConfig(experience_sharing=strategy)
            assert config.experience_sharing == strategy


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_distributed_config(self) -> None:
        """Test factory function for distributed config."""
        config = create_distributed_config(world_size=4, backend="gloo")

        assert config.enabled is True  # Auto-enabled for world_size > 1
        assert config.world_size == 4
        assert config.backend == DistributedBackend.GLOO

    def test_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable extraction returns config for distributed."""
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "8")

        config = from_environment()

        assert config is not None
        assert config.enabled is True
        assert config.world_size == 8

    def test_from_environment_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default values when env vars not set (non-distributed)."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        config = from_environment()

        # Single-process returns None
        assert config is None
