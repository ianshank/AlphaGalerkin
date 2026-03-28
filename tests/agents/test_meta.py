"""Tests for MetaAgent orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.config import (
    DecompositionConfig,
    DecompositionStrategy,
    MultiPhysicsConfig,
    OrchestratorConfig,
    SolverAgentConfig,
)
from src.agents.decomposition import SubproblemSpec
from src.agents.message import MessageBus
from src.agents.meta import MetaAgent
from src.pde.config import PDEConfig, PDEType
from src.templates.base import ExecutionStatus


class TestMetaAgent:
    """Tests for MetaAgent multi-physics orchestration."""

    @pytest.fixture
    def simple_orch_config(
        self,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> OrchestratorConfig:
        return sample_orchestrator_config

    @pytest.fixture
    def meta_no_factories(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> MetaAgent:
        """MetaAgent without game/evaluator factories (no real solvers)."""
        return MetaAgent(
            config=simple_orch_config,
            message_bus=message_bus,
            agent_id="test_meta",
        )

    def test_initial_state(self, meta_no_factories: MetaAgent) -> None:
        assert meta_no_factories.state.step == 0
        assert meta_no_factories.solver_agents == {}

    def test_setup_without_factories(self, meta_no_factories: MetaAgent) -> None:
        meta_no_factories.setup()
        # Without factories, no solver agents are created
        assert len(meta_no_factories.solver_agents) == 0
        assert len(meta_no_factories.subproblems) > 0
        assert meta_no_factories.coupling_agent is not None

    def test_setup_with_factories(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        mock_game = MagicMock()
        mock_evaluator = MagicMock()

        meta = MetaAgent(
            config=simple_orch_config,
            message_bus=message_bus,
            game_factory=lambda spec: mock_game,
            evaluator_factory=lambda game: mock_evaluator,
        )

        with patch("src.pde.mcts_adapter.PDEGameAdapter") as mock_adapter_cls, \
             patch("src.mcts.search.MCTS") as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 1.0
            mock_adapter_cls.return_value = mock_adapter
            mock_mcts_cls.return_value = MagicMock()

            meta.setup()
            assert len(meta.solver_agents) > 0

    def test_step_without_solvers(self, meta_no_factories: MetaAgent) -> None:
        meta_no_factories.setup()
        state = meta_no_factories.step()
        assert state.step == 1

    def test_step_with_mock_solvers(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(
            config=simple_orch_config,
            message_bus=message_bus,
        )
        meta.setup()

        # Manually add mock solvers
        mock_solver = MagicMock()
        mock_solver.is_terminal = False
        mock_solver.current_error = 0.5
        mock_solver.state.error_history = [1.0, 0.5]
        mock_solver.state.budget_used = 0.1
        mock_solver.get_metrics.return_value = {"error": 0.5}
        mock_solver.step.return_value = MagicMock()

        meta._solver_agents = {"sub_0": mock_solver}
        meta._stall_counters = {"sub_0": 0}

        state = meta.step()
        mock_solver.step.assert_called_once()
        assert state.step == 1

    def test_global_convergence_all_terminal(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(config=simple_orch_config, message_bus=message_bus)
        meta.setup()

        # No solvers → trivially converged
        assert meta._check_global_convergence()

    def test_global_convergence_solver_not_done(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(config=simple_orch_config, message_bus=message_bus)
        meta.setup()

        mock_solver = MagicMock()
        mock_solver.is_terminal = False
        meta._solver_agents = {"sub_0": mock_solver}

        assert not meta._check_global_convergence()

    def test_stall_detection(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(config=simple_orch_config, message_bus=message_bus)
        meta.setup()

        mock_solver = MagicMock()
        mock_solver.is_terminal = False
        mock_solver.current_error = 0.5  # Never changes
        mock_solver.state.error_history = [0.5]
        mock_solver.state.budget_used = 0.1
        mock_solver.get_metrics.return_value = {"error": 0.5}

        meta._solver_agents = {"stalled": mock_solver}
        meta._stall_counters = {"stalled": 0}

        for _ in range(6):
            meta.step()

        # Should have triggered stall handling (counter >= 5)
        assert meta._stall_counters.get("stalled", 0) < meta._stall_threshold

    def test_update_global_state(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(config=simple_orch_config, message_bus=message_bus)
        meta.setup()

        mock_solver_a = MagicMock()
        mock_solver_a.state.error_history = [0.5]
        mock_solver_a.state.budget_used = 0.2

        mock_solver_b = MagicMock()
        mock_solver_b.state.error_history = [0.3]
        mock_solver_b.state.budget_used = 0.1

        meta._solver_agents = {"a": mock_solver_a, "b": mock_solver_b}
        meta._update_global_state()

        assert len(meta.state.error_history) == 1
        assert meta.state.error_history[-1] == 0.5  # max of 0.5, 0.3
        assert meta.state.budget_used == pytest.approx(0.3)

    def test_get_metrics(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(config=simple_orch_config, message_bus=message_bus)
        meta.setup()

        mock_solver = MagicMock()
        mock_solver.is_terminal = True
        mock_solver.get_metrics.return_value = {"error": 0.01, "step": 10.0}
        meta._solver_agents = {"sub": mock_solver}
        meta._state.error_history = [0.01]

        metrics = meta.get_metrics()
        assert "global_step" in metrics
        assert "n_active_solvers" in metrics
        assert "n_total_solvers" in metrics
        assert "global_error" in metrics
        assert "solver_sub_error" in metrics

    def test_reset(
        self,
        simple_orch_config: OrchestratorConfig,
        message_bus: MessageBus,
    ) -> None:
        meta = MetaAgent(config=simple_orch_config, message_bus=message_bus)
        meta.setup()

        mock_solver = MagicMock()
        meta._solver_agents = {"sub": mock_solver}
        meta._stall_counters = {"sub": 3}
        meta._state.step = 10

        meta.reset()
        mock_solver.reset.assert_called_once()
        assert meta.state.step == 0
        assert meta._stall_counters["sub"] == 0

    def test_single_physics_no_coupling(self) -> None:
        """Single physics should work without coupling agent."""
        single_config = MultiPhysicsConfig(
            name="single",
            physics=[
                PDEConfig(
                    name="poisson",
                    pde_type=PDEType.POISSON,
                ),
            ],
        )
        orch = OrchestratorConfig(
            name="single_orch",
            multi_physics=single_config,
        )
        meta = MetaAgent(config=orch)
        meta.setup()
        # Single physics → no coupling agent needed
        # decomposition produces 1 subproblem
        assert len(meta.subproblems) >= 1
