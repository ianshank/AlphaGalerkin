"""Tests for distributed training launcher utilities."""

from __future__ import annotations

import subprocess
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_launcher_config() -> LauncherConfig:
    """Create a default launcher config for torchrun."""
    return LauncherConfig(
        method="torchrun",
        nnodes=1,
        nproc_per_node=2,
        master_addr="localhost",
        master_port=29500,
    )


@pytest.fixture
def slurm_launcher_config() -> LauncherConfig:
    """Create a SLURM launcher config."""
    return LauncherConfig(
        method="slurm",
        nnodes=2,
        nproc_per_node=4,
        master_addr="10.0.0.1",
        master_port=29500,
    )


@pytest.fixture
def custom_launcher_config() -> LauncherConfig:
    """Create a custom launcher config."""
    return LauncherConfig(
        method="custom",
        nnodes=1,
        nproc_per_node=1,
        master_addr="localhost",
        master_port=30000,
    )


@pytest.fixture
def script_path(tmp_path: Path) -> Path:
    """Create a dummy training script."""
    script = tmp_path / "train.py"
    script.write_text("print('training')")
    return script


@pytest.fixture
def script_args() -> list[str]:
    """Provide sample script arguments."""
    return ["--batch-size", "32", "--lr", "0.001"]


def _make_mock_process(
    returncode: int = 0,
    stdout: str = "ok",
    stderr: str = "",
) -> MagicMock:
    """Build a mock subprocess.Popen that behaves like a completed process."""
    proc = MagicMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.wait.return_value = returncode
    return proc


# ---------------------------------------------------------------------------
# TestDistributedLauncher
# ---------------------------------------------------------------------------


class TestDistributedLauncherInit:
    """Tests for DistributedLauncher construction."""

    def test_init_stores_config(
        self,
        default_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """Launcher stores config, script_path, and script_args."""
        launcher = DistributedLauncher(
            config=default_launcher_config,
            script_path=script_path,
            script_args=["--epochs", "10"],
        )

        assert launcher.config is default_launcher_config
        assert launcher.script_path == script_path
        assert launcher.script_args == ["--epochs", "10"]

    def test_init_defaults_empty_args(
        self,
        default_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """When script_args is None, it defaults to an empty list."""
        launcher = DistributedLauncher(
            config=default_launcher_config,
            script_path=script_path,
        )
        assert launcher.script_args == []


# ---------------------------------------------------------------------------
# Torchrun
# ---------------------------------------------------------------------------


class TestTorchrunLaunch:
    """Tests for torchrun command generation and execution."""

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_torchrun_command_components(
        self,
        mock_popen: MagicMock,
        default_launcher_config: LauncherConfig,
        script_path: Path,
        script_args: list[str],
    ) -> None:
        """Verify the torchrun command contains all expected flags."""
        mock_popen.return_value = _make_mock_process()
        launcher = DistributedLauncher(
            config=default_launcher_config,
            script_path=script_path,
            script_args=script_args,
        )

        result = launcher.launch()

        assert result.success is True
        cmd = mock_popen.call_args[0][0]
        assert sys.executable in cmd
        assert "-m" in cmd
        assert "torch.distributed.run" in cmd
        assert f"--nnodes={default_launcher_config.nnodes}" in cmd
        assert f"--nproc_per_node={default_launcher_config.nproc_per_node}" in cmd
        assert f"--master_addr={default_launcher_config.master_addr}" in cmd
        assert f"--master_port={default_launcher_config.master_port}" in cmd
        assert str(script_path) in cmd
        for arg in script_args:
            assert arg in cmd

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_torchrun_includes_rdzv_endpoint(
        self,
        mock_popen: MagicMock,
        script_path: Path,
    ) -> None:
        """When rdzv_endpoint is set, it appears in the command."""
        config = LauncherConfig(
            method="torchrun",
            rdzv_endpoint="10.0.0.1:29500",
        )
        mock_popen.return_value = _make_mock_process()
        launcher = DistributedLauncher(config=config, script_path=script_path)

        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert f"--rdzv_endpoint={config.rdzv_endpoint}" in cmd

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_torchrun_includes_max_restarts(
        self,
        mock_popen: MagicMock,
        script_path: Path,
    ) -> None:
        """When max_restarts > 0, the flag is included."""
        config = LauncherConfig(method="torchrun", max_restarts=3)
        mock_popen.return_value = _make_mock_process()
        launcher = DistributedLauncher(config=config, script_path=script_path)

        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert "--max_restarts=3" in cmd

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_torchrun_omits_max_restarts_when_zero(
        self,
        mock_popen: MagicMock,
        script_path: Path,
    ) -> None:
        """When max_restarts == 0, the flag is omitted."""
        config = LauncherConfig(method="torchrun", max_restarts=0)
        mock_popen.return_value = _make_mock_process()
        launcher = DistributedLauncher(config=config, script_path=script_path)

        launcher.launch()

        cmd = mock_popen.call_args[0][0]
        assert all("--max_restarts" not in c for c in cmd)

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_torchrun_failure_returns_error(
        self,
        mock_popen: MagicMock,
        default_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """A non-zero return code is reported in LaunchResult."""
        mock_popen.return_value = _make_mock_process(
            returncode=1,
            stderr="CUDA OOM",
        )
        launcher = DistributedLauncher(
            config=default_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is False
        assert result.return_code == 1
        assert result.error_message == "CUDA OOM"

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_torchrun_exception_returns_error(
        self,
        mock_popen: MagicMock,
        default_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """An exception during Popen is caught and returned."""
        mock_popen.side_effect = FileNotFoundError("torchrun not found")
        launcher = DistributedLauncher(
            config=default_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is False
        assert result.return_code == -1
        assert "torchrun not found" in (result.error_message or "")


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------


class TestSlurmLaunch:
    """Tests for SLURM script generation and submission."""

    def test_slurm_script_content(
        self,
        slurm_launcher_config: LauncherConfig,
        script_path: Path,
        script_args: list[str],
    ) -> None:
        """Verify the generated SLURM script contains expected directives."""
        launcher = DistributedLauncher(
            config=slurm_launcher_config,
            script_path=script_path,
            script_args=script_args,
        )

        script = launcher._generate_slurm_script()

        assert "#!/bin/bash" in script
        assert f"#SBATCH --nodes={slurm_launcher_config.nnodes}" in script
        assert f"#SBATCH --ntasks-per-node={slurm_launcher_config.nproc_per_node}" in script
        assert f"#SBATCH --gpus-per-node={slurm_launcher_config.nproc_per_node}" in script
        assert f"MASTER_PORT={slurm_launcher_config.master_port}" in script
        world_size = slurm_launcher_config.get_world_size()
        assert f"WORLD_SIZE={world_size}" in script
        assert str(script_path) in script
        for arg in script_args:
            assert arg in script

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_slurm_launch_calls_sbatch(
        self,
        mock_popen: MagicMock,
        slurm_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """Launch with method=slurm calls sbatch."""
        mock_popen.return_value = _make_mock_process(
            stdout="Submitted batch job 12345",
        )
        launcher = DistributedLauncher(
            config=slurm_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is True
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "sbatch"

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_slurm_failure(
        self,
        mock_popen: MagicMock,
        slurm_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """Sbatch failure is propagated."""
        mock_popen.return_value = _make_mock_process(
            returncode=1,
            stderr="sbatch: error: Batch job submission failed",
        )
        launcher = DistributedLauncher(
            config=slurm_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is False
        assert "submission failed" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Custom launcher
# ---------------------------------------------------------------------------


class TestCustomLaunch:
    """Tests for custom launcher method."""

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_custom_sets_environment(
        self,
        mock_popen: MagicMock,
        custom_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """Custom launcher sets MASTER_ADDR, WORLD_SIZE, etc. in env."""
        mock_popen.return_value = _make_mock_process()
        launcher = DistributedLauncher(
            config=custom_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is True
        env = mock_popen.call_args[1]["env"]
        assert env["MASTER_ADDR"] == custom_launcher_config.master_addr
        assert env["MASTER_PORT"] == str(custom_launcher_config.master_port)
        world_size = custom_launcher_config.get_world_size()
        assert env["WORLD_SIZE"] == str(world_size)

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_custom_failure(
        self,
        mock_popen: MagicMock,
        custom_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """Non-zero return code from custom launcher."""
        mock_popen.return_value = _make_mock_process(returncode=2, stderr="crash")
        launcher = DistributedLauncher(
            config=custom_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is False
        assert result.return_code == 2

    @patch("src.distributed.launcher.subprocess.Popen")
    def test_custom_exception(
        self,
        mock_popen: MagicMock,
        custom_launcher_config: LauncherConfig,
        script_path: Path,
    ) -> None:
        """Exception in custom launcher is handled gracefully."""
        mock_popen.side_effect = OSError("permission denied")
        launcher = DistributedLauncher(
            config=custom_launcher_config,
            script_path=script_path,
        )

        result = launcher.launch()

        assert result.success is False
        assert result.return_code == -1
        assert "permission denied" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestLauncherConfigValidation:
    """Tests that config constraints propagate correctly to launcher."""

    @pytest.mark.parametrize(
        "method",
        ["torchrun", "slurm", "custom"],
    )
    def test_method_dispatch(
        self,
        method: str,
        script_path: Path,
    ) -> None:
        """Each valid method is accepted and stored."""
        config = LauncherConfig(method=method)
        launcher = DistributedLauncher(config=config, script_path=script_path)
        assert launcher.config.method == method

    @pytest.mark.parametrize(
        ("nnodes", "nproc", "expected_world"),
        [
            (1, 1, 1),
            (1, 4, 4),
            (2, 4, 8),
            (4, 8, 32),
        ],
    )
    def test_world_size_propagation(
        self,
        nnodes: int,
        nproc: int,
        expected_world: int,
        script_path: Path,
    ) -> None:
        """World size is computed from nnodes * nproc_per_node."""
        config = LauncherConfig(nnodes=nnodes, nproc_per_node=nproc)
        launcher = DistributedLauncher(config=config, script_path=script_path)
        assert launcher.config.get_world_size() == expected_world


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateLauncher:
    """Tests for the create_launcher factory."""

    def test_creates_launcher_with_defaults(self, script_path: Path) -> None:
        """Factory returns a correctly configured launcher."""
        launcher = create_launcher(script_path=script_path)
        assert isinstance(launcher, DistributedLauncher)
        assert launcher.config.method == "torchrun"
        assert launcher.config.nnodes == 1

    def test_creates_launcher_with_overrides(self, script_path: Path) -> None:
        """Factory propagates keyword arguments to config."""
        launcher = create_launcher(
            script_path=script_path,
            nnodes=2,
            nproc_per_node=4,
            method="slurm",
            script_args=["--fast"],
        )
        assert launcher.config.method == "slurm"
        assert launcher.config.nnodes == 2
        assert launcher.config.nproc_per_node == 4
        assert launcher.script_args == ["--fast"]


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


class TestGetEnvironmentInfo:
    """Tests for environment info utility."""

    def test_returns_expected_keys(self) -> None:
        """All expected environment keys are present."""
        info = DistributedLauncher.get_environment_info()
        expected_keys = {
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_NODEID",
            "SLURM_PROCID",
        }
        assert expected_keys == set(info.keys())

    def test_reads_set_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When environment variables are set, they are reported."""
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.5")
        monkeypatch.setenv("RANK", "3")

        info = DistributedLauncher.get_environment_info()
        assert info["MASTER_ADDR"] == "10.0.0.5"
        assert info["RANK"] == "3"


# ---------------------------------------------------------------------------
# LaunchResult dataclass
# ---------------------------------------------------------------------------


class TestLaunchResult:
    """Tests for the LaunchResult dataclass."""

    def test_default_error_message_is_none(self) -> None:
        """error_message defaults to None for success."""
        result = LaunchResult(success=True, return_code=0, processes=[])
        assert result.error_message is None

    def test_stores_processes(self) -> None:
        """Processes list is stored."""
        proc = MagicMock(spec=subprocess.Popen)
        result = LaunchResult(success=True, return_code=0, processes=[proc])
        assert len(result.processes) == 1
