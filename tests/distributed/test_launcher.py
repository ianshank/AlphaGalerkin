"""Tests for distributed training launcher.

Tests cover:
- DistributedLauncher: Initialization, command construction, launch methods
- LaunchResult: Result dataclass
- create_launcher: Factory function
- Environment info retrieval
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.distributed.config import LauncherConfig
from src.distributed.launcher import (
    DistributedLauncher,
    LaunchResult,
    create_launcher,
)

# --- LaunchResult Tests ---


class TestLaunchResult:
    """Tests for LaunchResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful launch result."""
        result = LaunchResult(success=True, return_code=0, processes=[])

        assert result.success is True
        assert result.return_code == 0
        assert result.error_message is None

    def test_failed_result(self) -> None:
        """Test creating a failed launch result."""
        result = LaunchResult(
            success=False,
            return_code=1,
            processes=[],
            error_message="Process failed",
        )

        assert result.success is False
        assert result.return_code == 1
        assert result.error_message == "Process failed"


# --- DistributedLauncher Tests ---


class TestDistributedLauncher:
    """Tests for DistributedLauncher."""

    @pytest.fixture
    def default_config(self) -> LauncherConfig:
        """Create default launcher config."""
        return LauncherConfig()

    @pytest.fixture
    def multi_node_config(self) -> LauncherConfig:
        """Create multi-node launcher config."""
        return LauncherConfig(
            nnodes=2,
            nproc_per_node=4,
            master_addr="10.0.0.1",
            master_port=29500,
        )

    @pytest.fixture
    def script_path(self, tmp_path: Path) -> Path:
        """Create a dummy training script."""
        script = tmp_path / "train.py"
        script.write_text("print('training')")
        return script

    def test_initialization(self, default_config: LauncherConfig, script_path: Path) -> None:
        """Test launcher initialization."""
        launcher = DistributedLauncher(
            config=default_config,
            script_path=script_path,
            script_args=["--lr", "0.001"],
        )

        assert launcher.config == default_config
        assert launcher.script_path == script_path
        assert launcher.script_args == ["--lr", "0.001"]

    def test_initialization_no_args(
        self, default_config: LauncherConfig, script_path: Path
    ) -> None:
        """Test launcher initialization without script args."""
        launcher = DistributedLauncher(
            config=default_config,
            script_path=script_path,
        )

        assert launcher.script_args == []

    @patch("subprocess.Popen")
    def test_launch_torchrun(
        self, mock_popen: MagicMock, default_config: LauncherConfig, script_path: Path
    ) -> None:
        """Test torchrun launch constructs correct command."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("output", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        launcher = DistributedLauncher(config=default_config, script_path=script_path)
        result = launcher.launch()

        assert result.success is True
        assert result.return_code == 0

        # Verify command includes torchrun arguments
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert sys.executable in cmd[0]
        assert "-m" in cmd
        assert "torch.distributed.run" in cmd
        assert f"--nnodes={default_config.nnodes}" in cmd
        assert f"--nproc_per_node={default_config.nproc_per_node}" in cmd
        assert str(script_path) in cmd

    @patch("subprocess.Popen")
    def test_launch_torchrun_with_args(
        self, mock_popen: MagicMock, default_config: LauncherConfig, script_path: Path
    ) -> None:
        """Test torchrun launch passes script args."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        launcher = DistributedLauncher(
            config=default_config,
            script_path=script_path,
            script_args=["--batch-size", "32"],
        )
        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert "--batch-size" in cmd
        assert "32" in cmd

    @patch("subprocess.Popen")
    def test_launch_torchrun_with_rdzv_endpoint(
        self, mock_popen: MagicMock, script_path: Path
    ) -> None:
        """Test torchrun launch includes rdzv_endpoint when set."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = LauncherConfig(rdzv_endpoint="10.0.0.1:29500")
        launcher = DistributedLauncher(config=config, script_path=script_path)
        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert "--rdzv_endpoint=10.0.0.1:29500" in cmd

    @patch("subprocess.Popen")
    def test_launch_torchrun_with_max_restarts(
        self, mock_popen: MagicMock, script_path: Path
    ) -> None:
        """Test torchrun launch includes max_restarts when > 0."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = LauncherConfig(max_restarts=3)
        launcher = DistributedLauncher(config=config, script_path=script_path)
        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert "--max_restarts=3" in cmd

    @patch("subprocess.Popen")
    def test_launch_torchrun_failure(
        self, mock_popen: MagicMock, default_config: LauncherConfig, script_path: Path
    ) -> None:
        """Test handling of torchrun process failure."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "CUDA OOM error")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        launcher = DistributedLauncher(config=default_config, script_path=script_path)
        result = launcher.launch()

        assert result.success is False
        assert result.return_code == 1
        assert "CUDA OOM error" in result.error_message

    @patch("subprocess.Popen")
    def test_launch_torchrun_exception(
        self, mock_popen: MagicMock, default_config: LauncherConfig, script_path: Path
    ) -> None:
        """Test handling of subprocess exception."""
        mock_popen.side_effect = FileNotFoundError("torchrun not found")

        launcher = DistributedLauncher(config=default_config, script_path=script_path)
        result = launcher.launch()

        assert result.success is False
        assert result.return_code == -1
        assert "torchrun not found" in result.error_message
        assert result.processes == []

    @patch("subprocess.Popen")
    def test_launch_slurm(self, mock_popen: MagicMock, script_path: Path) -> None:
        """Test SLURM launch method."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Submitted batch job 12345", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = LauncherConfig(method="slurm", nnodes=2, nproc_per_node=4)
        launcher = DistributedLauncher(config=config, script_path=script_path)
        result = launcher.launch()

        assert result.success is True
        call_args = mock_popen.call_args[0][0]
        assert "sbatch" in call_args

    @patch("subprocess.Popen")
    def test_launch_slurm_failure(self, mock_popen: MagicMock, script_path: Path) -> None:
        """Test SLURM launch failure."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "sbatch: error: invalid partition")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        config = LauncherConfig(method="slurm")
        launcher = DistributedLauncher(config=config, script_path=script_path)
        result = launcher.launch()

        assert result.success is False

    @patch("subprocess.Popen")
    def test_launch_slurm_exception(self, mock_popen: MagicMock, script_path: Path) -> None:
        """Test SLURM launch with sbatch not found."""
        mock_popen.side_effect = FileNotFoundError("sbatch not found")

        config = LauncherConfig(method="slurm")
        launcher = DistributedLauncher(config=config, script_path=script_path)
        result = launcher.launch()

        assert result.success is False
        assert result.return_code == -1

    @patch("subprocess.Popen")
    def test_launch_custom(self, mock_popen: MagicMock, script_path: Path) -> None:
        """Test custom launch method sets environment variables."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = LauncherConfig(
            method="custom",
            nnodes=1,
            nproc_per_node=2,
            master_addr="localhost",
            master_port=29500,
        )
        launcher = DistributedLauncher(config=config, script_path=script_path)
        result = launcher.launch()

        assert result.success is True

        # Verify environment variables were set
        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs["env"]
        assert env["MASTER_ADDR"] == "localhost"
        assert env["MASTER_PORT"] == "29500"
        assert env["WORLD_SIZE"] == "2"
        assert env["RANK"] == "0"
        assert env["LOCAL_RANK"] == "0"

    @patch("subprocess.Popen")
    def test_launch_custom_failure(self, mock_popen: MagicMock, script_path: Path) -> None:
        """Test custom launch failure."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "error occurred")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        config = LauncherConfig(method="custom")
        launcher = DistributedLauncher(config=config, script_path=script_path)
        result = launcher.launch()

        assert result.success is False

    @patch("subprocess.Popen")
    def test_launch_custom_exception(self, mock_popen: MagicMock, script_path: Path) -> None:
        """Test custom launch with exception."""
        mock_popen.side_effect = OSError("Permission denied")

        config = LauncherConfig(method="custom")
        launcher = DistributedLauncher(config=config, script_path=script_path)
        result = launcher.launch()

        assert result.success is False
        assert result.return_code == -1
        assert "Permission denied" in result.error_message

    def test_generate_slurm_script(self, script_path: Path) -> None:
        """Test SLURM script generation content."""
        config = LauncherConfig(
            method="slurm",
            nnodes=2,
            nproc_per_node=4,
            master_port=29500,
        )
        launcher = DistributedLauncher(config=config, script_path=script_path)

        script = launcher._generate_slurm_script()

        assert "#!/bin/bash" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --ntasks-per-node=4" in script
        assert "#SBATCH --gpus-per-node=4" in script
        assert "WORLD_SIZE=8" in script
        assert str(script_path) in script

    def test_get_environment_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment info retrieval."""
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)

        info = DistributedLauncher.get_environment_info()

        assert info["MASTER_ADDR"] == "10.0.0.1"
        assert info["RANK"] == "0"
        assert info["WORLD_SIZE"] == "4"
        assert info["SLURM_JOB_ID"] == "not set"

    def test_get_environment_info_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment info when vars not set."""
        for var in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]:
            monkeypatch.delenv(var, raising=False)

        info = DistributedLauncher.get_environment_info()

        assert info["MASTER_ADDR"] == "not set"
        assert info["WORLD_SIZE"] == "not set"

    @patch("subprocess.Popen")
    def test_multi_node_torchrun_command(
        self, mock_popen: MagicMock, multi_node_config: LauncherConfig, script_path: Path
    ) -> None:
        """Test torchrun command for multi-node setup."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        launcher = DistributedLauncher(config=multi_node_config, script_path=script_path)
        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert "--nnodes=2" in cmd
        assert "--nproc_per_node=4" in cmd
        assert "--master_addr=10.0.0.1" in cmd
        assert "--master_port=29500" in cmd


# --- Factory Tests ---


class TestCreateLauncher:
    """Tests for create_launcher factory function."""

    def test_factory_default(self, tmp_path: Path) -> None:
        """Test factory with default settings."""
        script = tmp_path / "train.py"
        script.write_text("pass")

        launcher = create_launcher(script_path=script)

        assert isinstance(launcher, DistributedLauncher)
        assert launcher.config.method == "torchrun"
        assert launcher.config.nnodes == 1
        assert launcher.config.nproc_per_node == 1

    def test_factory_custom(self, tmp_path: Path) -> None:
        """Test factory with custom settings."""
        script = tmp_path / "train.py"
        script.write_text("pass")

        launcher = create_launcher(
            script_path=script,
            nnodes=2,
            nproc_per_node=4,
            method="slurm",
            script_args=["--epochs", "10"],
        )

        assert launcher.config.method == "slurm"
        assert launcher.config.nnodes == 2
        assert launcher.config.nproc_per_node == 4
        assert launcher.script_args == ["--epochs", "10"]
