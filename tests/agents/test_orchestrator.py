"""Tests for AgentOrchestrator BaseExecutable integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.config import MultiPhysicsConfig, OrchestratorConfig
from src.agents.orchestrator import AgentOrchestrator
from src.pde.config import PDEConfig, PDEType
from src.templates.base import ExecutionStatus


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""

    @pytest.fixture
    def simple_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            name="test_orch",
            multi_physics=MultiPhysicsConfig(
                name="single",
                physics=[
                    PDEConfig(name="poisson", pde_type=PDEType.POISSON),
                ],
            ),
        )

    def test_creation(self, simple_config: OrchestratorConfig) -> None:
        orch = AgentOrchestrator(simple_config)
        assert orch._executable_name == "agent_orchestrator"

    def test_execute_returns_result(self, simple_config: OrchestratorConfig) -> None:
        orch = AgentOrchestrator(simple_config)
        result = orch.execute()
        assert result.status == ExecutionStatus.COMPLETED
        assert "total_steps" in result.metrics
        assert "error_history" in result.artifacts

    def test_execute_with_run_wrapper(self, simple_config: OrchestratorConfig) -> None:
        orch = AgentOrchestrator(simple_config)
        result = orch.run()
        assert result.status == ExecutionStatus.COMPLETED
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    def test_execute_failure_handled(self) -> None:
        """Bad config should result in FAILED status, not exception."""
        # Create config that will fail during execution
        config = OrchestratorConfig(
            name="fail",
            multi_physics=MultiPhysicsConfig(
                name="single",
                physics=[
                    PDEConfig(name="poisson", pde_type=PDEType.POISSON),
                ],
            ),
        )
        orch = AgentOrchestrator(config)

        with patch("src.agents.orchestrator.MetaAgent") as mock_meta_cls:
            mock_meta = MagicMock()
            mock_meta.setup.side_effect = RuntimeError("Test error")
            mock_meta_cls.return_value = mock_meta

            result = orch.execute()
            assert result.status == ExecutionStatus.FAILED
            assert "Test error" in result.error

    def test_artifacts_populated(self, simple_config: OrchestratorConfig) -> None:
        orch = AgentOrchestrator(simple_config)
        result = orch.execute()
        assert "error_history" in result.artifacts
        assert "n_subproblems" in result.artifacts
        assert "subproblem_names" in result.artifacts

    def test_metrics_populated(self, simple_config: OrchestratorConfig) -> None:
        orch = AgentOrchestrator(simple_config)
        result = orch.execute()
        assert "total_steps" in result.metrics
        assert "budget_used" in result.metrics

    def test_full_orchestrator_config(
        self,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        orch = AgentOrchestrator(sample_orchestrator_config)
        result = orch.execute()
        assert result.status == ExecutionStatus.COMPLETED

    def test_with_game_factory(self, simple_config: OrchestratorConfig) -> None:
        mock_game = MagicMock()
        mock_evaluator = MagicMock()

        orch = AgentOrchestrator(
            simple_config,
            game_factory=lambda spec: mock_game,
            evaluator_factory=lambda game: mock_evaluator,
        )

        with (
            patch("src.pde.mcts_adapter.PDEGameAdapter") as mock_adapter_cls,
            patch("src.mcts.search.MCTS") as mock_mcts_cls,
        ):
            mock_adapter = MagicMock()
            mock_adapter.current_error = 0.001
            mock_adapter.is_terminal.return_value = True
            mock_adapter.error_reduction = 0.999
            mock_adapter.state.dof = 10
            mock_adapter_cls.return_value = mock_adapter
            mock_mcts_cls.return_value = MagicMock()

            result = orch.execute()
            assert result.status == ExecutionStatus.COMPLETED
