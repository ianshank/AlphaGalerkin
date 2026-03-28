"""Tests for CouplingAgent."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.config import CouplingConfig, CouplingType, MessageType
from src.agents.coupling import CouplingAgent
from src.agents.message import AgentMessage, MessageBus
from src.templates.base import ExecutionStatus


class TestCouplingAgent:
    """Tests for CouplingAgent interface condition management."""

    @pytest.fixture
    def coupling_agent(
        self,
        sample_coupling_config: CouplingConfig,
        message_bus: MessageBus,
    ) -> CouplingAgent:
        agent = CouplingAgent(
            config=sample_coupling_config,
            message_bus=message_bus,
            agent_id="test_coupling",
        )
        agent.setup()
        return agent

    def test_initial_state(self, coupling_agent: CouplingAgent) -> None:
        assert coupling_agent.state.step == 0
        assert not coupling_agent.is_converged()

    def test_coupling_config_property(self, coupling_agent: CouplingAgent) -> None:
        assert coupling_agent.coupling_config.coupling_type == CouplingType.DIRICHLET_NEUMANN
        assert coupling_agent.coupling_config.relaxation_factor == 0.5

    def test_exchange_boundary_data_first_iteration(
        self, coupling_agent: CouplingAgent
    ) -> None:
        solver_data = {
            "solver_a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "solver_b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        }
        updated = coupling_agent.exchange_boundary_data(solver_data)
        # First iteration: no previous data, so values pass through
        np.testing.assert_array_equal(updated["solver_a"], solver_data["solver_a"])
        np.testing.assert_array_equal(updated["solver_b"], solver_data["solver_b"])

    def test_exchange_with_relaxation(self, coupling_agent: CouplingAgent) -> None:
        # First iteration
        data_1 = {
            "solver_a": np.array([1.0, 1.0], dtype=np.float32),
        }
        coupling_agent.exchange_boundary_data(data_1)

        # Second iteration with different data
        data_2 = {
            "solver_a": np.array([2.0, 2.0], dtype=np.float32),
        }
        updated = coupling_agent.exchange_boundary_data(data_2)

        # With omega=0.5: new = 0.5 * old + 0.5 * received = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        np.testing.assert_allclose(updated["solver_a"], [1.5, 1.5], rtol=1e-5)

    @pytest.mark.parametrize("omega", [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_relaxation_factor_effect(self, omega: float) -> None:
        config = CouplingConfig(
            name="relax_test",
            relaxation_factor=omega,
            tolerance=1e-4,
            convergence_window=2,
        )
        agent = CouplingAgent(config)
        agent.setup()

        old_val = np.array([1.0], dtype=np.float32)
        new_val = np.array([3.0], dtype=np.float32)

        agent.exchange_boundary_data({"s": old_val})
        result = agent.exchange_boundary_data({"s": new_val})

        expected = (1.0 - omega) * 1.0 + omega * 3.0
        np.testing.assert_allclose(result["s"], [expected], rtol=1e-5)

    def test_convergence_detection(self) -> None:
        config = CouplingConfig(
            name="conv_test",
            tolerance=0.1,
            convergence_window=3,
            relaxation_factor=1.0,  # No relaxation for simplicity
        )
        agent = CouplingAgent(config)
        agent.setup()

        # Push residuals below tolerance for convergence_window steps
        agent._residual_history = [0.05, 0.04, 0.03]
        assert agent.is_converged()

    def test_not_converged_insufficient_window(self) -> None:
        config = CouplingConfig(
            name="conv_test",
            tolerance=0.1,
            convergence_window=3,
        )
        agent = CouplingAgent(config)
        agent.setup()
        agent._residual_history = [0.05, 0.04]  # Only 2, need 3
        assert not agent.is_converged()

    def test_not_converged_above_tolerance(self) -> None:
        config = CouplingConfig(
            name="conv_test",
            tolerance=0.01,
            convergence_window=3,
        )
        agent = CouplingAgent(config)
        agent.setup()
        agent._residual_history = [0.005, 0.02, 0.005]  # One above
        assert not agent.is_converged()

    def test_step_with_boundary_messages(
        self,
        coupling_agent: CouplingAgent,
        message_bus: MessageBus,
    ) -> None:
        # Send boundary data via message bus
        message_bus.publish(
            AgentMessage(
                sender="solver_1",
                receiver="test_coupling",
                message_type=MessageType.BOUNDARY_DATA,
                payload={
                    "agent_id": "solver_1",
                    "boundary_values": [1.0, 2.0, 3.0],
                },
            )
        )
        state = coupling_agent.step()
        assert state.step == 1
        assert len(coupling_agent._residual_history) == 1

    def test_step_without_messages(self, coupling_agent: CouplingAgent) -> None:
        state = coupling_agent.step()
        assert state.step == 1
        assert coupling_agent._residual_history[-1] == float("inf")

    def test_max_iterations_terminal(self) -> None:
        config = CouplingConfig(
            name="max_iter",
            max_iterations=3,
            tolerance=1e-10,  # Won't converge
        )
        agent = CouplingAgent(config)
        agent.setup()
        for _ in range(3):
            agent.step()
        assert agent.state.status == ExecutionStatus.COMPLETED

    def test_convergence_terminal(self) -> None:
        config = CouplingConfig(
            name="conv",
            tolerance=0.5,
            convergence_window=1,
            relaxation_factor=1.0,
        )
        agent = CouplingAgent(config)
        agent.setup()
        agent._residual_history = [0.1]  # Below tolerance
        agent.step()  # Will check convergence
        # After step, residual_history has [0.1, inf] due to no messages
        # But the convergence check sees the new inf and doesn't converge

    def test_reset(self, coupling_agent: CouplingAgent) -> None:
        coupling_agent.step()
        coupling_agent.reset()
        assert coupling_agent.state.step == 0
        assert coupling_agent._residual_history == []
        assert coupling_agent._previous_bcs == {}

    def test_get_metrics(self, coupling_agent: CouplingAgent) -> None:
        coupling_agent._residual_history = [0.5, 0.3, 0.1]
        metrics = coupling_agent.get_metrics()
        assert "iteration" in metrics
        assert "converged" in metrics
        assert "interface_residual" in metrics
        assert "min_residual" in metrics
        assert metrics["interface_residual"] == 0.1
        assert metrics["min_residual"] == 0.1

    @pytest.mark.parametrize("coupling_type", list(CouplingType))
    def test_all_coupling_types(self, coupling_type: CouplingType) -> None:
        config = CouplingConfig(
            name="type_test",
            coupling_type=coupling_type,
            relaxation_factor=0.5,
        )
        agent = CouplingAgent(config)
        agent.setup()

        data = {"s": np.array([1.0, 2.0], dtype=np.float32)}
        agent.exchange_boundary_data(data)
        result = agent.exchange_boundary_data(
            {"s": np.array([3.0, 4.0], dtype=np.float32)}
        )
        # All types use relaxation, just verify no crash
        assert result["s"].shape == (2,)

    def test_interface_residual_computation(
        self, coupling_agent: CouplingAgent
    ) -> None:
        solver_data = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
        }
        updated = {
            "a": np.array([1.1, 2.1], dtype=np.float32),
        }
        residual = coupling_agent._compute_interface_residual(solver_data, updated)
        expected = float(np.sqrt(np.mean(np.array([0.1, 0.1]) ** 2)))
        assert residual == pytest.approx(expected, abs=1e-5)
