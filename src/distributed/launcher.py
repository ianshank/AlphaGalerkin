"""Distributed training launcher utilities.

This module provides utilities for launching distributed training jobs
across multiple nodes using torchrun or custom launch methods.

Features:
    - torchrun integration
    - SLURM cluster support
    - Automatic environment configuration
    - Elastic training support
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.distributed.config import LauncherConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


@dataclass
class LaunchResult:
    """Result of a launch operation."""

    success: bool
    return_code: int
    processes: list[subprocess.Popen]
    error_message: str | None = None


class DistributedLauncher:
    """Launcher for distributed training jobs.

    Handles the complexity of setting up distributed training environments
    and launching processes across nodes.

    Attributes:
        config: Launcher configuration.
        script_path: Path to training script.
        script_args: Arguments to pass to training script.

    """

    def __init__(
        self,
        config: LauncherConfig,
        script_path: str | Path,
        script_args: Sequence[str] | None = None,
    ) -> None:
        """Initialize launcher.

        Args:
            config: Launcher configuration.
            script_path: Path to the training script.
            script_args: Arguments to pass to the script.

        """
        self.config = config
        self.script_path = Path(script_path)
        self.script_args = list(script_args) if script_args else []

        self._logger = structlog.get_logger(__name__).bind(
            method=config.method,
            nnodes=config.nnodes,
            nproc_per_node=config.nproc_per_node,
        )

    def launch(self) -> LaunchResult:
        """Launch distributed training based on configured method.

        Returns:
            LaunchResult with status and process information.

        """
        self._logger.info("launching_distributed_training")

        if self.config.method == "torchrun":
            return self._launch_torchrun()
        elif self.config.method == "slurm":
            return self._launch_slurm()
        else:
            return self._launch_custom()

    def _launch_torchrun(self) -> LaunchResult:
        """Launch using torchrun.

        Returns:
            LaunchResult with torchrun process.

        """
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={self.config.nnodes}",
            f"--nproc_per_node={self.config.nproc_per_node}",
            f"--node_rank={self.config.node_rank}",
            f"--master_addr={self.config.master_addr}",
            f"--master_port={self.config.master_port}",
            f"--rdzv_backend={self.config.rdzv_backend}",
        ]

        if self.config.rdzv_endpoint:
            cmd.append(f"--rdzv_endpoint={self.config.rdzv_endpoint}")

        if self.config.max_restarts > 0:
            cmd.append(f"--max_restarts={self.config.max_restarts}")

        cmd.append(str(self.script_path))
        cmd.extend(self.script_args)

        self._logger.debug("torchrun_command", cmd=" ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for completion
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                self._logger.error(
                    "torchrun_failed",
                    return_code=process.returncode,
                    stderr=stderr,
                )
                return LaunchResult(
                    success=False,
                    return_code=process.returncode,
                    processes=[process],
                    error_message=stderr,
                )

            return LaunchResult(
                success=True,
                return_code=0,
                processes=[process],
            )

        except Exception as e:
            self._logger.error("torchrun_exception", error=str(e))
            return LaunchResult(
                success=False,
                return_code=-1,
                processes=[],
                error_message=str(e),
            )

    def _launch_slurm(self) -> LaunchResult:
        """Launch using SLURM.

        Returns:
            LaunchResult with SLURM job information.

        """
        # Generate SLURM script
        slurm_script = self._generate_slurm_script()
        script_path = Path("/tmp/alphagalerkin_slurm.sh")
        script_path.write_text(slurm_script)
        script_path.chmod(0o755)

        cmd = ["sbatch", str(script_path)]

        self._logger.debug("slurm_command", cmd=" ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                return LaunchResult(
                    success=False,
                    return_code=process.returncode,
                    processes=[process],
                    error_message=stderr,
                )

            self._logger.info("slurm_job_submitted", output=stdout.strip())

            return LaunchResult(
                success=True,
                return_code=0,
                processes=[process],
            )

        except Exception as e:
            self._logger.error("slurm_exception", error=str(e))
            return LaunchResult(
                success=False,
                return_code=-1,
                processes=[],
                error_message=str(e),
            )

    def _generate_slurm_script(self) -> str:
        """Generate SLURM batch script.

        Returns:
            SLURM script content.

        """
        world_size = self.config.get_world_size()
        script_args_str = " ".join(self.script_args)

        return f"""#!/bin/bash
#SBATCH --job-name=alphagalerkin
#SBATCH --nodes={self.config.nnodes}
#SBATCH --ntasks-per-node={self.config.nproc_per_node}
#SBATCH --gpus-per-node={self.config.nproc_per_node}
#SBATCH --time=24:00:00

# Set up environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT={self.config.master_port}
export WORLD_SIZE={world_size}

# Launch with srun
srun --ntasks-per-node={self.config.nproc_per_node} \\
     python -m torch.distributed.run \\
     --nnodes={self.config.nnodes} \\
     --nproc_per_node={self.config.nproc_per_node} \\
     --rdzv_backend=c10d \\
     --rdzv_endpoint=$MASTER_ADDR:{self.config.master_port} \\
     {self.script_path} {script_args_str}
"""

    def _launch_custom(self) -> LaunchResult:
        """Launch using custom method.

        Returns:
            LaunchResult with custom process.

        """
        # Set environment variables
        env = os.environ.copy()
        env.update(
            {
                "MASTER_ADDR": self.config.master_addr,
                "MASTER_PORT": str(self.config.master_port),
                "WORLD_SIZE": str(self.config.get_world_size()),
                "RANK": str(self.config.node_rank * self.config.nproc_per_node),
                "LOCAL_RANK": "0",
            }
        )

        cmd = [sys.executable, str(self.script_path)] + self.script_args

        self._logger.debug("custom_command", cmd=" ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                return LaunchResult(
                    success=False,
                    return_code=process.returncode,
                    processes=[process],
                    error_message=stderr,
                )

            return LaunchResult(
                success=True,
                return_code=0,
                processes=[process],
            )

        except Exception as e:
            self._logger.error("custom_launch_exception", error=str(e))
            return LaunchResult(
                success=False,
                return_code=-1,
                processes=[],
                error_message=str(e),
            )

    @staticmethod
    def get_environment_info() -> dict[str, Any]:
        """Get distributed environment information.

        Returns:
            Dictionary with environment details.

        """
        return {
            "MASTER_ADDR": os.environ.get("MASTER_ADDR", "not set"),
            "MASTER_PORT": os.environ.get("MASTER_PORT", "not set"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE", "not set"),
            "RANK": os.environ.get("RANK", "not set"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK", "not set"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            # SLURM-specific
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "not set"),
            "SLURM_NODEID": os.environ.get("SLURM_NODEID", "not set"),
            "SLURM_PROCID": os.environ.get("SLURM_PROCID", "not set"),
        }


def create_launcher(
    script_path: str | Path,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    method: str = "torchrun",
    script_args: Sequence[str] | None = None,
    **kwargs: Any,
) -> DistributedLauncher:
    """Factory function to create a distributed launcher.

    Args:
        script_path: Path to training script.
        nnodes: Number of nodes.
        nproc_per_node: Processes per node.
        method: Launch method.
        script_args: Arguments for the script.
        **kwargs: Additional launcher config options.

    Returns:
        Configured DistributedLauncher instance.

    """
    config = LauncherConfig(
        method=method,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        **kwargs,
    )

    return DistributedLauncher(
        config=config,
        script_path=script_path,
        script_args=script_args,
    )
