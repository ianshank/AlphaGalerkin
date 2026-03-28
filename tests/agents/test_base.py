"""Tests for BaseAgent and AgentState."""

from __future__ import annotations

import pytest

from src.agents.base import AgentState, BaseAgent
from src.agents.config import AgentConfig, AgentType, MessageType
from src.agents.message import MessageBus
from src.templates.base import ExecutionStatus
from tests.agents.conftest import ConcreteTestAgent


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_default_values(self) -> None:
        state = AgentState(agent_id="test")
        assert state.agent_id == "test"
        assert state.status == ExecutionStatus.PENDING
        assert state.step == 0
        assert state.metrics == {}
        assert state.error_history == []
        assert state.budget_used == 0.0
        assert state.budget_remaining == 1.0

    def test_custom_values(self) -> None:
        state = AgentState(
            agent_id="custom",
            status=ExecutionStatus.RUNNING,
            step=5,
            metrics={"loss": 0.5},
            error_history=[1.0, 0.5, 0.25],
            budget_used=0.3,
            budget_remaining=0.7,
        )
        assert state.step == 5
        assert state.metrics["loss"] == 0.5
        assert len(state.error_history) == 3


class TestBaseAgent:
    """Tests for BaseAgent ABC via ConcreteTestAgent."""

    def test_auto_generated_id(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(config=sample_agent_config)
        assert len(agent.agent_id) > 0

    def test_explicit_id(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config, agent_id="explicit_id"
        )
        assert agent.agent_id == "explicit_id"

    def test_unique_ids(self, sample_agent_config: AgentConfig) -> None:
        a1 = ConcreteTestAgent(config=sample_agent_config)
        a2 = ConcreteTestAgent(config=sample_agent_config)
        assert a1.agent_id != a2.agent_id

    def test_initial_state(self, concrete_agent: ConcreteTestAgent) -> None:
        assert concrete_agent.state.status == ExecutionStatus.PENDING
        assert concrete_agent.state.step == 0
        assert concrete_agent.state.budget_remaining == 1.0

    def test_is_active_before_run(self, concrete_agent: ConcreteTestAgent) -> None:
        assert not concrete_agent.is_active

    def test_is_terminal_initially_false(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert not concrete_agent.is_terminal

    def test_step_increments(self, concrete_agent: ConcreteTestAgent) -> None:
        concrete_agent.setup()
        state = concrete_agent.step()
        assert state.step == 1
        assert len(state.error_history) == 2  # initial + one step

    def test_budget_tracking(self, concrete_agent: ConcreteTestAgent) -> None:
        concrete_agent.setup()
        concrete_agent.step()
        assert concrete_agent.state.budget_used > 0
        assert concrete_agent.state.budget_remaining < 1.0

    def test_budget_exhaustion(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config.with_overrides(computational_budget=0.05),
            steps_to_terminal=1,
        )
        agent.setup()
        agent.step()
        assert agent.state.budget_remaining == 0.0
        assert agent.is_terminal

    def test_error_convergence(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config.with_overrides(error_tolerance=0.5),
            error_per_step=0.6,
        )
        agent.setup()
        agent.step()
        # After one step: 1.0 * 0.4 = 0.4 < 0.5
        assert agent.state.error_history[-1] < 0.5
        assert agent.is_terminal

    def test_max_steps_terminal(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config.with_overrides(max_steps=2),
            error_per_step=0.01,  # Won't converge fast
        )
        agent.setup()
        agent.step()
        agent.step()
        assert agent.state.step >= 2
        assert agent.is_terminal

    def test_run_lifecycle(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config.with_overrides(
                max_steps=10, error_tolerance=0.001
            ),
            steps_to_terminal=10,
            error_per_step=0.5,
        )
        final_state = agent.run(max_steps=10)
        assert final_state.status == ExecutionStatus.COMPLETED
        assert final_state.step > 0

    def test_run_respects_max_steps(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config.with_overrides(max_steps=1000),
            error_per_step=0.01,
        )
        final_state = agent.run(max_steps=3)
        assert final_state.step <= 3

    def test_get_metrics(self, concrete_agent: ConcreteTestAgent) -> None:
        concrete_agent.setup()
        concrete_agent.step()
        metrics = concrete_agent.get_metrics()
        assert "step" in metrics
        assert "error" in metrics

    def test_reset(self, concrete_agent: ConcreteTestAgent) -> None:
        concrete_agent.setup()
        concrete_agent.step()
        concrete_agent.step()
        assert concrete_agent.state.step == 2
        concrete_agent.reset()
        assert concrete_agent.state.step == 0
        assert concrete_agent.state.budget_used == 0.0

    def test_message_send(
        self,
        concrete_agent: ConcreteTestAgent,
        message_bus: MessageBus,
    ) -> None:
        message_bus.subscribe("receiver")
        concrete_agent.send_message(
            receiver="receiver",
            msg_type=MessageType.STATE_UPDATE,
            payload={"test": True},
        )
        received = message_bus.receive("receiver")
        assert len(received) == 1
        assert received[0].payload["test"] is True

    def test_message_receive(
        self,
        concrete_agent: ConcreteTestAgent,
        message_bus: MessageBus,
    ) -> None:
        from src.agents.message import AgentMessage

        message_bus.publish(
            AgentMessage(
                sender="external",
                receiver=concrete_agent.agent_id,
                message_type=MessageType.BOUNDARY_DATA,
                payload={"values": [1.0, 2.0]},
            )
        )
        received = concrete_agent.receive_messages(MessageType.BOUNDARY_DATA)
        assert len(received) == 1
        assert received[0].payload["values"] == [1.0, 2.0]

    def test_message_without_bus(self, sample_agent_config: AgentConfig) -> None:
        agent = ConcreteTestAgent(config=sample_agent_config, message_bus=None)
        agent.send_message("receiver", MessageType.STATE_UPDATE)
        received = agent.receive_messages()
        assert received == []

    def test_update_budget(self, concrete_agent: ConcreteTestAgent) -> None:
        concrete_agent.update_budget(0.3)
        assert concrete_agent.state.budget_used == pytest.approx(0.3)
        assert concrete_agent.state.budget_remaining == pytest.approx(0.7)

    def test_update_budget_clamps_to_zero(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        concrete_agent.update_budget(2.0)
        assert concrete_agent.state.budget_remaining == 0.0

    def test_status_completed_on_successful_run(
        self, sample_agent_config: AgentConfig
    ) -> None:
        agent = ConcreteTestAgent(
            config=sample_agent_config.with_overrides(max_steps=3),
        )
        result = agent.run()
        assert result.status == ExecutionStatus.COMPLETED
