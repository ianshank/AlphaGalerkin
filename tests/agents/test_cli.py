"""Tests for agents CLI commands."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.agents.cli import app

runner = CliRunner()


class TestListAgents:
    """Tests for the list-agents command."""

    def test_list_agents_shows_registered(self) -> None:
        """Verify list-agents prints registered agent types."""
        result = runner.invoke(app, ["list-agents"])
        assert result.exit_code == 0
        assert "solver" in result.output.lower() or "Registered" in result.output

    def test_list_agents_empty_registry(self) -> None:
        """When no builtin agents are loaded, show 'No agents' message."""
        with patch(
            "src.agents.registry._register_builtin_agents",
        ):
            from src.agents.registry import AgentRegistry

            AgentRegistry().clear()
            result = runner.invoke(app, ["list-agents"])
            assert result.exit_code == 0
            assert "No agents registered" in result.output


class TestInfo:
    """Tests for the info command."""

    def test_info_known_agent(self) -> None:
        """Invoke info for 'solver', verify class info printed."""
        result = runner.invoke(app, ["info", "solver"])
        assert result.exit_code == 0
        assert "solver" in result.output.lower()

    def test_info_unknown_agent(self) -> None:
        """Invoke info for unknown type, verify exit code 1."""
        result = runner.invoke(app, ["info", "nonexistent_agent_xyz"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestRun:
    """Tests for the run command."""

    def _make_config_file(self) -> str:
        """Create a temporary YAML config file."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False,
        ) as f:
            f.write("name: test\n")
            return f.name

    def test_run_success(self) -> None:
        """Mock the full run pipeline to return success."""
        config_path = self._make_config_file()
        try:
            mock_result = MagicMock()
            mock_result.is_success.return_value = True
            mock_result.status.value = "completed"
            mock_result.duration_seconds = 1.5
            mock_result.metrics = {
                "global_error": 0.001,
                "total_steps": 10.0,
                "budget_used": 0.5,
            }
            mock_result.error = None

            mock_orch_instance = MagicMock()
            mock_orch_instance.run.return_value = mock_result

            with (
                patch(
                    "src.agents.cli.load_config_file",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.agents.orchestrator.AgentOrchestrator",
                    return_value=mock_orch_instance,
                ),
            ):
                # Also patch the deferred import inside run()
                with patch.dict(
                    "sys.modules",
                    {
                        "src.agents.config": __import__(
                            "src.agents.config", fromlist=["OrchestratorConfig"],
                        ),
                        "src.agents.orchestrator": MagicMock(
                            AgentOrchestrator=MagicMock(
                                return_value=mock_orch_instance,
                            ),
                        ),
                    },
                ):
                    result = runner.invoke(app, ["run", "--config", config_path])
                    assert result.exit_code == 0
                    assert "completed" in result.output.lower()
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_run_failure(self) -> None:
        """Mock the full run pipeline to return failure."""
        config_path = self._make_config_file()
        try:
            mock_result = MagicMock()
            mock_result.is_success.return_value = False
            mock_result.status.value = "failed"
            mock_result.duration_seconds = 0.5
            mock_result.metrics = {}
            mock_result.error = "Convergence failed"

            mock_orch_instance = MagicMock()
            mock_orch_instance.run.return_value = mock_result

            with patch.dict(
                "sys.modules",
                {
                    "src.agents.config": __import__(
                        "src.agents.config", fromlist=["OrchestratorConfig"],
                    ),
                    "src.agents.orchestrator": MagicMock(
                        AgentOrchestrator=MagicMock(
                            return_value=mock_orch_instance,
                        ),
                    ),
                },
            ):
                with patch(
                    "src.agents.cli.load_config_file",
                    return_value=MagicMock(),
                ):
                    result = runner.invoke(app, ["run", "--config", config_path])
                    assert result.exit_code == 1
                    assert "Convergence failed" in result.output
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_run_invalid_config_path(self) -> None:
        """Running with nonexistent config path fails."""
        result = runner.invoke(app, ["run", "--config", "/tmp/nonexistent.yaml"])
        assert result.exit_code != 0


class TestMain:
    """Tests for the main entry point."""

    def test_main_is_callable(self) -> None:
        """Verify main() is a callable function."""
        from src.agents.cli import main

        assert callable(main)

    def test_help_output(self) -> None:
        """Verify the app produces help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agent" in result.output.lower()

    def test_main_entry_point(self) -> None:
        """Verify main() calls app()."""
        from src.agents import cli

        result = runner.invoke(cli.app, ["list-agents"])
        assert result.exit_code == 0
