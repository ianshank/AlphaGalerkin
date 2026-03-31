"""Tests for SolverAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.config import MessageType, SolverAgentConfig
from src.agents.message import MessageBus
from src.agents.solver import SolverAgent
from src.templates.base import ExecutionStatus

# Patch targets: the modules where setup()/step() import from
_ADAPTER_PATH = "src.pde.mcts_adapter.PDEGameAdapter"
_MCTS_PATH = "src.mcts.search.MCTS"


class TestSolverAgent:
    """Tests for SolverAgent wrapping PDEGame + MCTS."""

    @pytest.fixture
    def solver(
        self,
        sample_solver_config: SolverAgentConfig,
        mock_pde_game: MagicMock,
        mock_evaluator: MagicMock,
        message_bus: MessageBus,
    ) -> SolverAgent:
        """Create a SolverAgent with mocked dependencies."""
        message_bus.subscribe("broadcast_target")
        return SolverAgent(
            config=sample_solver_config,
            pde_game=mock_pde_game,
            evaluator=mock_evaluator,
            message_bus=message_bus,
            agent_id="test_solver",
        )

    def test_initial_state(self, solver: SolverAgent) -> None:
        assert solver.state.status == ExecutionStatus.PENDING
        assert solver.state.step == 0

    def test_setup_creates_adapter_and_mcts(self, solver: SolverAgent) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter_inst = MagicMock()
            mock_adapter_inst.current_error = 1.0
            mock_adapter_cls.return_value = mock_adapter_inst
            mock_mcts_cls.return_value = MagicMock()

            solver.setup()

            mock_adapter_cls.assert_called_once()
            mock_mcts_cls.assert_called_once()
            assert solver.adapter is not None
            assert len(solver.state.error_history) > 0

    def test_solver_config_property(self, solver: SolverAgent) -> None:
        assert solver.solver_config.game_mode == "basis_selection"
        assert solver.solver_config.n_simulations == 10

    def test_current_error_without_adapter(self, solver: SolverAgent) -> None:
        assert solver.current_error == float("inf")

    def test_error_reduction_without_adapter(self, solver: SolverAgent) -> None:
        assert solver.error_reduction == 0.0

    def test_temperature_schedule(self, solver: SolverAgent) -> None:
        # Step 0: should be at temperature_start
        assert solver._compute_temperature() == 1.0

        # After some steps
        solver._state.step = 25
        temp = solver._compute_temperature()
        assert 0.1 < temp < 1.0

        # At decay_steps: should be at temperature_end
        solver._state.step = 50
        temp = solver._compute_temperature()
        assert temp == pytest.approx(0.1, abs=0.01)

        # Beyond decay_steps: clamp to temperature_end
        solver._state.step = 100
        temp = solver._compute_temperature()
        assert temp == pytest.approx(0.1, abs=0.01)

    def test_step_with_mocked_mcts(self, solver: SolverAgent) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 0.8
            mock_adapter.is_terminal.return_value = False
            mock_adapter.state.dof = 30
            mock_adapter.error_reduction = 0.2
            mock_adapter_cls.return_value = mock_adapter

            mock_mcts = MagicMock()
            mock_mcts.search.return_value = {0: 0.5, 1: 0.3, 2: 0.2}
            mock_mcts.get_action.return_value = 0
            mock_mcts_cls.return_value = mock_mcts

            solver.setup()
            state = solver.step()

            mock_mcts.search.assert_called_once()
            mock_mcts.get_action.assert_called_once()
            mock_adapter.apply_action.assert_called_once_with(0)
            mock_mcts.advance.assert_called_once_with(0)
            assert state.step == 1

    def test_step_publishes_message(
        self,
        solver: SolverAgent,
        message_bus: MessageBus,
    ) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 0.5
            mock_adapter.is_terminal.return_value = False
            mock_adapter.state.dof = 25
            mock_adapter.error_reduction = 0.5
            mock_adapter_cls.return_value = mock_adapter

            mock_mcts = MagicMock()
            mock_mcts.search.return_value = {0: 1.0}
            mock_mcts.get_action.return_value = 0
            mock_mcts_cls.return_value = mock_mcts

            solver.setup()
            solver.step()

            # Broadcast message should reach subscribed agents
            msgs = message_bus.receive("broadcast_target")
            assert len(msgs) >= 1
            assert msgs[0].message_type == MessageType.STATE_UPDATE

    def test_metrics(self, solver: SolverAgent) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 0.5
            mock_adapter.is_terminal.return_value = False
            mock_adapter.state.dof = 30
            mock_adapter.error_reduction = 0.5
            mock_adapter_cls.return_value = mock_adapter
            mock_mcts_cls.return_value = MagicMock(
                search=MagicMock(return_value={0: 1.0}),
                get_action=MagicMock(return_value=0),
            )

            solver.setup()
            solver.step()
            metrics = solver.get_metrics()

            assert "error" in metrics
            assert "error_reduction" in metrics
            assert "dof" in metrics
            assert "step" in metrics
            assert "budget_used" in metrics

    def test_reset(self, solver: SolverAgent) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 1.0
            mock_adapter_cls.return_value = mock_adapter
            mock_mcts_cls.return_value = MagicMock()

            solver.setup()
            solver._state.step = 5
            solver.reset()

            mock_adapter.reset.assert_called_once()
            assert solver.state.step == 0
            assert solver.state.budget_used == 0.0

    def test_terminal_on_adapter_terminal(self, solver: SolverAgent) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 0.001
            mock_adapter.is_terminal.return_value = True
            mock_adapter_cls.return_value = mock_adapter
            mock_mcts_cls.return_value = MagicMock()

            solver.setup()
            state = solver.step()
            assert state.status == ExecutionStatus.COMPLETED

    def test_reset_calls_mcts_reset(self, solver: SolverAgent) -> None:
        """Verify reset() calls MCTS.reset() when available."""
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 1.0
            mock_adapter_cls.return_value = mock_adapter

            mock_mcts = MagicMock()
            mock_mcts.reset = MagicMock()  # Ensure reset exists
            mock_mcts_cls.return_value = mock_mcts

            solver.setup()
            solver.reset()

            mock_mcts.reset.assert_called_once()

    def test_budget_tracking(self, solver: SolverAgent) -> None:
        with patch(_ADAPTER_PATH) as mock_adapter_cls, patch(_MCTS_PATH) as mock_mcts_cls:
            mock_adapter = MagicMock()
            mock_adapter.current_error = 0.9
            mock_adapter.is_terminal.return_value = False
            mock_adapter.state.dof = 25
            mock_adapter.error_reduction = 0.1
            mock_adapter_cls.return_value = mock_adapter
            mock_mcts_cls.return_value = MagicMock(
                search=MagicMock(return_value={0: 1.0}),
                get_action=MagicMock(return_value=0),
            )

            solver.setup()
            solver.step()
            assert solver.state.budget_used == pytest.approx(0.01)
            assert solver.state.budget_remaining == pytest.approx(0.99)
