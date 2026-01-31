"""Tests for multi-node distributed training setup."""

from __future__ import annotations

import json
import os

import pytest

from src.vertex.multi_node import (
    DEFAULT_MASTER_PORT,
    NCCL_DEFAULTS,
    DistributedContext,
    VertexDistributedSetup,
    setup_distributed_training,
)

class TestDistributedContext:
    """Tests for DistributedContext."""

    def test_creation(self) -> None:
        """Test context creation."""
        ctx = DistributedContext(
            world_size=8,
            rank=3,
            local_rank=1,
            master_addr="10.0.0.1",
            master_port=29500,
            num_nodes=2,
            node_rank=1,
            gpus_per_node=4,
        )
        assert ctx.world_size == 8
        assert ctx.rank == 3
        assert ctx.local_rank == 1
        assert ctx.master_addr == "10.0.0.1"
        assert ctx.gpus_per_node == 4

    def test_is_main_process(self) -> None:
        """Test main process detection."""
        ctx_main = DistributedContext(
            world_size=4, rank=0, local_rank=0,
            master_addr="localhost", master_port=29500,
        )
        ctx_worker = DistributedContext(
            world_size=4, rank=2, local_rank=0,
            master_addr="localhost", master_port=29500,
        )
        assert ctx_main.is_main_process() is True
        assert ctx_worker.is_main_process() is False

    def test_is_local_main(self) -> None:
        """Test local main process detection."""
        ctx_local_main = DistributedContext(
            world_size=4, rank=2, local_rank=0,
            master_addr="localhost", master_port=29500,
        )
        ctx_local_worker = DistributedContext(
            world_size=4, rank=3, local_rank=1,
            master_addr="localhost", master_port=29500,
        )
        assert ctx_local_main.is_local_main() is True
        assert ctx_local_worker.is_local_main() is False

    def test_to_environment_vars(self) -> None:
        """Test environment variable conversion."""
        ctx = DistributedContext(
            world_size=4,
            rank=1,
            local_rank=1,
            master_addr="10.0.0.1",
            master_port=29501,
        )
        env = ctx.to_environment_vars()
        assert env["RANK"] == "1"
        assert env["WORLD_SIZE"] == "4"
        assert env["LOCAL_RANK"] == "1"
        assert env["MASTER_ADDR"] == "10.0.0.1"
        assert env["MASTER_PORT"] == "29501"

    def test_setup_environment(self) -> None:
        """Test environment setup."""
        ctx = DistributedContext(
            world_size=2,
            rank=0,
            local_rank=0,
            master_addr="10.0.0.1",
            master_port=29500,
        )

        # Clear any existing values
        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(key, None)

        ctx.setup_environment()

        assert os.environ["RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "2"
        assert os.environ["MASTER_ADDR"] == "10.0.0.1"

        # Cleanup
        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(key, None)


class TestVertexDistributedSetup:
    """Tests for VertexDistributedSetup."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self) -> None:
        """Clean up environment variables after each test."""
        yield
        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR", "MASTER_PORT", "CLUSTER_SPEC", "TF_CONFIG"]:
            os.environ.pop(key, None)
        for key in NCCL_DEFAULTS:
            os.environ.pop(key, None)

    def test_from_pytorch_env(self) -> None:
        """Test parsing PyTorch-style environment."""
        os.environ["RANK"] = "2"
        os.environ["WORLD_SIZE"] = "8"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "4"
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        os.environ["MASTER_PORT"] = "29501"

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.rank == 2
        assert ctx.world_size == 8
        assert ctx.local_rank == 0
        assert ctx.master_addr == "10.0.0.1"
        assert ctx.master_port == 29501
        assert ctx.gpus_per_node == 4
        assert ctx.num_nodes == 2

    def test_from_pytorch_env_minimal(self) -> None:
        """Test minimal PyTorch environment."""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.local_rank == 0
        assert ctx.master_addr == "localhost"

    def test_from_cluster_spec(self) -> None:
        """Test parsing CLUSTER_SPEC JSON."""
        cluster_spec = {
            "cluster": {
                "chief": ["10.0.0.1:29500"],
                "worker": ["10.0.0.2:29500", "10.0.0.3:29500"],
            },
            "task": {
                "type": "worker",
                "index": 1,
            },
        }
        os.environ["CLUSTER_SPEC"] = json.dumps(cluster_spec)

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.world_size == 3
        assert ctx.rank == 2  # 1 chief + index 1
        assert ctx.master_addr == "10.0.0.1"
        assert ctx.master_port == 29500

    def test_from_cluster_spec_chief(self) -> None:
        """Test parsing CLUSTER_SPEC for chief task."""
        cluster_spec = {
            "cluster": {
                "chief": ["10.0.0.1:29500"],
                "worker": ["10.0.0.2:29500"],
            },
            "task": {
                "type": "chief",
                "index": 0,
            },
        }
        os.environ["CLUSTER_SPEC"] = json.dumps(cluster_spec)

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.rank == 0  # Chief is rank 0
        assert ctx.world_size == 2

    def test_from_cluster_spec_workers_only(self) -> None:
        """Test CLUSTER_SPEC with only workers (no chief)."""
        cluster_spec = {
            "cluster": {
                "worker": ["10.0.0.1:29500", "10.0.0.2:29500"],
            },
            "task": {
                "type": "worker",
                "index": 0,
            },
        }
        os.environ["CLUSTER_SPEC"] = json.dumps(cluster_spec)

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.world_size == 2
        assert ctx.rank == 0
        assert ctx.master_addr == "10.0.0.1"

    def test_from_cluster_spec_invalid_json(self) -> None:
        """Test handling invalid CLUSTER_SPEC JSON."""
        os.environ["CLUSTER_SPEC"] = "not valid json"

        ctx = VertexDistributedSetup.setup_from_environment()

        # Should fall back to defaults
        assert ctx.world_size == 1
        assert ctx.rank == 0

    def test_from_tf_config(self) -> None:
        """Test parsing TF_CONFIG JSON."""
        tf_config = {
            "cluster": {
                "chief": ["10.0.0.1:29500"],
                "worker": ["10.0.0.2:29500", "10.0.0.3:29500"],
            },
            "task": {
                "type": "worker",
                "index": 0,
            },
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.world_size == 3
        assert ctx.rank == 1  # First worker after chief
        assert ctx.master_addr == "10.0.0.1"

    def test_from_tf_config_chief(self) -> None:
        """Test TF_CONFIG for chief task."""
        tf_config = {
            "cluster": {
                "chief": ["10.0.0.1:29500"],
                "worker": ["10.0.0.2:29500"],
            },
            "task": {
                "type": "chief",
                "index": 0,
            },
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)

        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.rank == 0
        assert ctx.world_size == 2

    def test_default_single_node(self) -> None:
        """Test default to single-node when no config found."""
        # No environment variables set
        ctx = VertexDistributedSetup.setup_from_environment()

        assert ctx.world_size == 1
        assert ctx.rank == 0
        assert ctx.local_rank == 0
        assert ctx.master_addr == "localhost"
        assert ctx.master_port == DEFAULT_MASTER_PORT

    def test_get_master_addr(self) -> None:
        """Test getting master address."""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "192.168.1.100"

        addr = VertexDistributedSetup.get_master_addr()

        assert addr == "192.168.1.100"

    def test_configure_nccl_for_vertex(self) -> None:
        """Test NCCL configuration."""
        # Clear NCCL variables
        for key in NCCL_DEFAULTS:
            os.environ.pop(key, None)

        VertexDistributedSetup.configure_nccl_for_vertex()

        for key, expected_value in NCCL_DEFAULTS.items():
            assert os.environ.get(key) == expected_value

    def test_configure_nccl_respects_existing(self) -> None:
        """Test NCCL config doesn't override existing values."""
        os.environ["NCCL_DEBUG"] = "INFO"

        VertexDistributedSetup.configure_nccl_for_vertex()

        # Should keep existing value
        assert os.environ["NCCL_DEBUG"] == "INFO"

    def test_get_local_ip(self) -> None:
        """Test getting local IP."""
        ip = VertexDistributedSetup.get_local_ip()
        # Should be a valid IP format
        parts = ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert part.isdigit()
            assert 0 <= int(part) <= 255

    def test_find_free_port(self) -> None:
        """Test finding free port."""
        port = VertexDistributedSetup.find_free_port(start_port=40000)
        assert port >= 40000
        assert port < 40100


class TestSetupDistributedTraining:
    """Tests for setup_distributed_training convenience function."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self) -> None:
        """Clean up environment variables after each test."""
        yield
        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(key, None)
        for key in NCCL_DEFAULTS:
            os.environ.pop(key, None)

    def test_full_setup(self) -> None:
        """Test complete distributed training setup."""
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        ctx = setup_distributed_training()

        # Verify context
        assert ctx.rank == 1
        assert ctx.world_size == 4

        # Verify NCCL configured
        assert "NCCL_IB_DISABLE" in os.environ

        # Verify environment set
        assert os.environ["RANK"] == "1"
