"""Base agent abstraction for physics problem solving.

Provides the ``BaseAgent`` ABC and ``AgentState`` dataclass that all
concrete agent types inherit from. Agents follow a setup → step → terminal
lifecycle with optional message bus integration.

Example:
    from src.agents.base import BaseAgent, AgentState
    from src.agents.config import AgentConfig, AgentType

    class MyAgent(BaseAgent):
        def setup(self) -> None:
            pass

        def step(self) -> AgentState:
            self._state.step += 1
            return self._state

        def reset(self) -> None:
            self._state = self._create_initial_state()

        def get_metrics(self) -> dict[str, float]:
            return {"step": float(self._state.step)}

"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agents.config import MessageType
from src.templates.base import ExecutionStatus
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import AgentConfig
    from src.agents.message import AgentMessage, MessageBus

AgentLogger = create_logger_class("Agent")


@dataclass
class AgentState:
    """Snapshot of an agent's current state.

    Attributes:
        agent_id: Unique identifier for this agent.
        status: Current execution status.
        step: Number of steps executed so far.
        metrics: Named metric values.
        error_history: Error estimates over time.
        budget_used: Total budget consumed.
        budget_remaining: Budget left before exhaustion.

    """

    agent_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    error_history: list[float] = field(default_factory=list)
    budget_used: float = 0.0
    budget_remaining: float = 1.0


class BaseAgent(ABC):
    """Abstract base class for all agents in the orchestration framework.

    Concrete subclasses must implement ``setup``, ``step``, ``reset``,
    and ``get_metrics``. The ``run`` method provides a main loop that
    repeatedly calls ``step`` until a terminal condition is met.

    Args:
        config: Agent configuration.
        message_bus: Optional message bus for inter-agent communication.
        agent_id: Optional explicit ID (auto-generated if omitted).

    """

    def __init__(
        self,
        config: AgentConfig,
        message_bus: MessageBus | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.config = config
        self._message_bus = message_bus
        self._agent_id = agent_id or str(uuid.uuid4())
        self._state = self._create_initial_state()
        self._logger = AgentLogger(
            component=type(self).__name__,
            run_id=self._agent_id,
        )

        if self._message_bus is not None:
            self._message_bus.subscribe(self._agent_id)

    def _create_initial_state(self) -> AgentState:
        """Create a fresh initial state from config."""
        return AgentState(
            agent_id=self._agent_id,
            status=ExecutionStatus.PENDING,
            budget_remaining=self.config.computational_budget,
        )

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        return self._agent_id

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the agent is currently running."""
        return self._state.status == ExecutionStatus.RUNNING

    @property
    def is_terminal(self) -> bool:
        """Whether the agent has reached a terminal state.

        Terminal conditions:
        - Status is COMPLETED or FAILED
        - Budget exhausted
        - Max steps reached
        - Error below tolerance (with at least one error recorded)

        """
        return (
            self._state.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
            or self._state.budget_remaining <= 0
            or self._state.step >= self.config.max_steps
            or (
                bool(self._state.error_history)
                and self._state.error_history[-1] < self.config.error_tolerance
            )
        )

    @abstractmethod
    def setup(self) -> None:
        """Initialize agent resources. Called before the first step."""

    @abstractmethod
    def step(self) -> AgentState:
        """Execute one agent step and return updated state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent to initial state."""

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Return current metric values."""

    # ------------------------------------------------------------------
    # Lifecycle hooks
    #
    # Optional extension points around ``setup`` and each ``step`` in the
    # ``run`` loop. All four default to no-ops, so overriding none of them
    # preserves the historical behaviour exactly — existing agents and tests
    # are unaffected. Subclasses may override any subset to attach telemetry,
    # adaptive strategy switching, or resource management without touching the
    # core loop.
    # ------------------------------------------------------------------

    def pre_setup(self) -> None:
        """Hook called immediately before :meth:`setup`. No-op by default."""

    def post_setup(self) -> None:
        """Hook called immediately after :meth:`setup`. No-op by default."""

    def pre_step(self) -> None:
        """Hook called before each :meth:`step`. No-op by default."""

    def post_step(self) -> None:
        """Hook called after each :meth:`step` (and metric refresh). No-op by default."""

    def run(self, max_steps: int | None = None) -> AgentState:
        """Main loop: call step() until terminal or max_steps reached.

        Args:
            max_steps: Override for config.max_steps.

        Returns:
            Final agent state.

        """
        step_limit = max_steps if max_steps is not None else self.config.max_steps
        # Opt-in wall-clock deadline. Enforcement is gated on
        # ``config.enforce_timeout`` (default False) so the default path never
        # reads the clock and behaviour is unchanged for existing agents.
        deadline = (
            time.monotonic() + self.config.timeout_seconds
            if getattr(self.config, "enforce_timeout", False)
            else None
        )
        self._state.status = ExecutionStatus.RUNNING
        self._logger.info(
            "agent_run_started",
            max_steps=step_limit,
            budget=self._state.budget_remaining,
            enforce_timeout=deadline is not None,
        )

        try:
            self.pre_setup()
            self.setup()
            self.post_setup()
            while not self.is_terminal and self._state.step < step_limit:
                if deadline is not None and time.monotonic() >= deadline:
                    self._state.status = ExecutionStatus.TIMEOUT
                    self._logger.warning(
                        "agent_run_timeout",
                        timeout_seconds=self.config.timeout_seconds,
                        steps=self._state.step,
                    )
                    break
                self.pre_step()
                self._state = self.step()
                self._state.metrics = self.get_metrics()
                self.post_step()

            if self._state.status not in (
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.TIMEOUT,
            ):
                self._state.status = ExecutionStatus.COMPLETED

        except Exception as e:
            self._state.status = ExecutionStatus.FAILED
            self._state.metrics["error_message"] = 0.0  # flag only
            self._logger.exception("agent_run_failed", error=str(e))
            raise

        self._logger.info(
            "agent_run_finished",
            status=self._state.status.value,
            steps=self._state.step,
            budget_used=self._state.budget_used,
        )
        return self._state

    def update_budget(self, used: float) -> None:
        """Deduct from computational budget.

        Args:
            used: Amount of budget consumed.

        """
        self._state.budget_used += used
        self._state.budget_remaining = max(
            0.0, self.config.computational_budget - self._state.budget_used
        )

    def send_message(
        self,
        receiver: str,
        msg_type: MessageType,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Send a message via the message bus.

        Args:
            receiver: Target agent ID.
            msg_type: Message category.
            payload: Data to include in the message.

        """
        if self._message_bus is None:
            return

        from src.agents.message import AgentMessage

        message = AgentMessage(
            sender=self._agent_id,
            receiver=receiver,
            message_type=msg_type,
            payload=payload or {},
        )
        if receiver == "*":
            self._message_bus.broadcast(message)
        else:
            self._message_bus.publish(message)

    def receive_messages(
        self,
        msg_type: MessageType | None = None,
    ) -> list[AgentMessage]:
        """Receive messages from the message bus.

        Args:
            msg_type: Optional filter for message type.

        Returns:
            List of messages (empty if no bus or no messages).

        """
        if self._message_bus is None:
            return []
        return self._message_bus.receive(self._agent_id, msg_type)
