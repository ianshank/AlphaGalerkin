"""Multi-node distributed training setup for Vertex AI.

This module provides utilities for configuring PyTorch distributed
training from Vertex AI environment variables, enabling seamless
multi-node training.

Vertex AI sets various environment variables to configure distributed
training, including:
- CLUSTER_SPEC: JSON with worker addresses
- RANK, WORLD_SIZE, LOCAL_RANK: PyTorch distributed config
- Various TF_CONFIG variables (compatible with TensorFlow)

Example:
    from src.vertex.multi_node import VertexDistributedSetup

    # At training script startup
    VertexDistributedSetup.configure_nccl_for_vertex()
    ctx = VertexDistributedSetup.setup_from_environment()

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ctx.master_addr}:{ctx.master_port}",
        world_size=ctx.world_size,
        rank=ctx.rank,
    )
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default port for distributed training
DEFAULT_MASTER_PORT = 29500

# NCCL environment variable defaults for GCP
NCCL_DEFAULTS = {
    "NCCL_IB_DISABLE": "1",  # Disable InfiniBand (not available on GCP)
    "NCCL_SOCKET_IFNAME": "eth0",  # Use eth0 for communication
    "NCCL_DEBUG": "WARN",  # Reduce log verbosity
    "NCCL_P2P_DISABLE": "0",  # Enable P2P (NVLink if available)
    "NCCL_SHM_DISABLE": "0",  # Enable shared memory
}


@dataclass
class DistributedContext:
    """Distributed training context from Vertex AI.

    This dataclass holds all the information needed to configure
    PyTorch distributed training.

    Attributes:
        world_size: Total number of processes across all nodes.
        rank: Global rank of this process (0 to world_size-1).
        local_rank: Rank within this node (0 to gpus_per_node-1).
        master_addr: IP address of the master node.
        master_port: Port for master node communication.
        num_nodes: Number of nodes in the cluster.
        node_rank: Rank of this node (0 to num_nodes-1).
        gpus_per_node: Number of GPUs on each node.
    """

    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: int
    num_nodes: int = 1
    node_rank: int = 0
    gpus_per_node: int = 1

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def is_local_main(self) -> bool:
        """Check if this is the local main process (local_rank 0)."""
        return self.local_rank == 0

    def to_environment_vars(self) -> dict[str, str]:
        """Convert to environment variables for subprocess."""
        return {
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }

    def setup_environment(self) -> None:
        """Set environment variables for distributed training."""
        for key, value in self.to_environment_vars().items():
            os.environ[key] = value
        logger.info(
            "distributed_environment_configured",
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            master_addr=self.master_addr,
        )


class VertexDistributedSetup:
    """Configure distributed training on Vertex AI.

    This class provides static methods for setting up PyTorch
    distributed training from Vertex AI environment variables.

    Vertex AI supports multiple ways to configure distributed training:
    1. PyTorch-style env vars (RANK, WORLD_SIZE, etc.)
    2. CLUSTER_SPEC JSON (compatible with TensorFlow)
    3. Cloud ML Engine style (TF_CONFIG)

    This class handles all three methods automatically.
    """

    @staticmethod
    def setup_from_environment() -> DistributedContext:
        """Configure DDP from Vertex AI environment variables.

        Detects the environment configuration method and extracts
        all necessary distributed training parameters.

        Returns:
            DistributedContext with all configuration values.
        """
        # Check for PyTorch-style env vars first (most common)
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return VertexDistributedSetup._from_pytorch_env()

        # Check for CLUSTER_SPEC (Vertex AI custom jobs)
        if "CLUSTER_SPEC" in os.environ:
            return VertexDistributedSetup._from_cluster_spec()

        # Check for TF_CONFIG (Cloud ML Engine style)
        if "TF_CONFIG" in os.environ:
            return VertexDistributedSetup._from_tf_config()

        # Default to single-node single-GPU
        logger.warning(
            "no_distributed_config_found",
            hint="Running in single-node mode",
        )
        return DistributedContext(
            world_size=1,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=DEFAULT_MASTER_PORT,
            num_nodes=1,
            node_rank=0,
            gpus_per_node=1,
        )

    @staticmethod
    def _from_pytorch_env() -> DistributedContext:
        """Parse PyTorch-style environment variables."""
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT))

        # Calculate derived values
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if gpus_per_node == 0:
            gpus_per_node = 1
        num_nodes = world_size // gpus_per_node if gpus_per_node > 0 else 1
        node_rank = rank // gpus_per_node if gpus_per_node > 0 else 0

        logger.info(
            "distributed_config_from_pytorch_env",
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            master_addr=master_addr,
        )

        return DistributedContext(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
            num_nodes=num_nodes,
            node_rank=node_rank,
            gpus_per_node=gpus_per_node,
        )

    @staticmethod
    def _from_cluster_spec() -> DistributedContext:
        """Parse CLUSTER_SPEC JSON (Vertex AI format).

        CLUSTER_SPEC format:
        {
            "cluster": {
                "worker": ["host1:port1", "host2:port2"],
                "chief": ["host0:port0"]  # Optional
            },
            "task": {
                "type": "worker",
                "index": 0
            }
        }
        """
        cluster_spec_str = os.environ.get("CLUSTER_SPEC", "{}")
        try:
            spec = json.loads(cluster_spec_str)
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_cluster_spec", spec=cluster_spec_str)
            return DistributedContext(
                world_size=1,
                rank=0,
                local_rank=0,
                master_addr="localhost",
                master_port=DEFAULT_MASTER_PORT,
            )

        cluster = spec.get("cluster", {})
        task = spec.get("task", {})

        # Get all workers
        workers = cluster.get("worker", [])
        chief = cluster.get("chief", [])

        # Chief is typically the master
        all_hosts = chief + workers
        if not all_hosts:
            return DistributedContext(
                world_size=1,
                rank=0,
                local_rank=0,
                master_addr="localhost",
                master_port=DEFAULT_MASTER_PORT,
            )

        # Parse master address
        master_host = all_hosts[0]
        if ":" in master_host:
            master_addr, port_str = master_host.rsplit(":", 1)
            master_port = int(port_str)
        else:
            master_addr = master_host
            master_port = DEFAULT_MASTER_PORT

        # Determine rank
        task_type = task.get("type", "worker")
        task_index = task.get("index", 0)

        if task_type == "chief":
            rank = 0
        else:
            # Workers start after chief
            rank = len(chief) + task_index

        world_size = len(all_hosts)

        logger.info(
            "distributed_config_from_cluster_spec",
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            task_type=task_type,
        )

        return DistributedContext(
            world_size=world_size,
            rank=rank,
            local_rank=0,  # Assume single GPU per task for CLUSTER_SPEC
            master_addr=master_addr,
            master_port=master_port,
            num_nodes=world_size,
            node_rank=rank,
            gpus_per_node=1,
        )

    @staticmethod
    def _from_tf_config() -> DistributedContext:
        """Parse TF_CONFIG JSON (Cloud ML Engine style).

        TF_CONFIG format:
        {
            "cluster": {
                "chief": ["host:port"],
                "worker": ["host1:port1", "host2:port2"]
            },
            "task": {
                "type": "worker",
                "index": 1
            }
        }
        """
        tf_config_str = os.environ.get("TF_CONFIG", "{}")
        try:
            config = json.loads(tf_config_str)
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_tf_config", config=tf_config_str)
            return DistributedContext(
                world_size=1,
                rank=0,
                local_rank=0,
                master_addr="localhost",
                master_port=DEFAULT_MASTER_PORT,
            )

        cluster = config.get("cluster", {})
        task = config.get("task", {})

        chiefs = cluster.get("chief", [])
        workers = cluster.get("worker", [])
        all_hosts = chiefs + workers

        if not all_hosts:
            return DistributedContext(
                world_size=1,
                rank=0,
                local_rank=0,
                master_addr="localhost",
                master_port=DEFAULT_MASTER_PORT,
            )

        # Master is first chief or first worker
        master_host = all_hosts[0]
        if ":" in master_host:
            master_addr, port_str = master_host.rsplit(":", 1)
            master_port = int(port_str)
        else:
            master_addr = master_host
            master_port = DEFAULT_MASTER_PORT

        task_type = task.get("type", "worker")
        task_index = task.get("index", 0)

        if task_type == "chief":
            rank = task_index
        else:
            rank = len(chiefs) + task_index

        world_size = len(all_hosts)

        logger.info(
            "distributed_config_from_tf_config",
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
        )

        return DistributedContext(
            world_size=world_size,
            rank=rank,
            local_rank=0,
            master_addr=master_addr,
            master_port=master_port,
            num_nodes=world_size,
            node_rank=rank,
            gpus_per_node=1,
        )

    @staticmethod
    def get_master_addr() -> str:
        """Extract master address from environment.

        Returns:
            Master node IP address.
        """
        ctx = VertexDistributedSetup.setup_from_environment()
        return ctx.master_addr

    @staticmethod
    def configure_nccl_for_vertex() -> None:
        """Set NCCL environment variables for Vertex AI networking.

        This should be called early in the training script, before
        initializing the distributed process group.
        """
        for key, value in NCCL_DEFAULTS.items():
            if key not in os.environ:
                os.environ[key] = value

        logger.debug(
            "nccl_configured_for_vertex",
            settings={k: os.environ.get(k) for k in NCCL_DEFAULTS},
        )

    @staticmethod
    def get_local_ip() -> str:
        """Get the local IP address of this node.

        Returns:
            Local IP address string.
        """
        try:
            # Create a socket to determine the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                # Doesn't need to be reachable
                s.connect(("10.254.254.254", 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    @staticmethod
    def find_free_port(start_port: int = 29500, max_attempts: int = 100) -> int:
        """Find a free port for distributed communication.

        Args:
            start_port: Port to start searching from.
            max_attempts: Maximum number of ports to try.

        Returns:
            Available port number.

        Raises:
            RuntimeError: If no free port found.
        """
        for offset in range(max_attempts):
            port = start_port + offset
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", port))
                s.close()
                return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


def setup_distributed_training() -> DistributedContext:
    """Convenience function to set up distributed training.

    This function:
    1. Configures NCCL for Vertex AI networking
    2. Extracts distributed configuration from environment
    3. Sets up environment variables

    Returns:
        DistributedContext ready for torch.distributed.init_process_group

    Example:
        from src.vertex.multi_node import setup_distributed_training
        import torch.distributed as dist

        ctx = setup_distributed_training()
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ctx.master_addr}:{ctx.master_port}",
            world_size=ctx.world_size,
            rank=ctx.rank,
        )
    """
    VertexDistributedSetup.configure_nccl_for_vertex()
    ctx = VertexDistributedSetup.setup_from_environment()
    ctx.setup_environment()
    return ctx
