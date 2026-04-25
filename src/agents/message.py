"""Inter-agent communication via message bus.

Provides a thread-safe pub/sub message passing system for agents
to exchange state updates, boundary data, and convergence signals.

Example:
    from src.agents.message import MessageBus, AgentMessage
    from src.agents.config import MessageBusConfig, MessageType

    bus = MessageBus(MessageBusConfig(name="bus"))
    bus.subscribe("solver_1")
    bus.subscribe("solver_2")

    msg = AgentMessage(
        sender="solver_1",
        receiver="solver_2",
        message_type=MessageType.BOUNDARY_DATA,
        payload={"values": [1.0, 2.0]},
    )
    bus.publish(msg)
    received = bus.receive("solver_2")

"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agents.config import MessageType
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import MessageBusConfig

MessageLogger = create_logger_class("MessageBus")


@dataclass
class AgentMessage:
    r"""A message exchanged between agents.

    Attributes:
        sender: ID of the sending agent.
        receiver: ID of the receiving agent (or \"*\" for broadcast).
        message_type: Category of the message.
        payload: Arbitrary data carried by the message.
        timestamp: Auto-populated creation time.
        message_id: Auto-populated unique identifier.

    """

    sender: str
    receiver: str
    message_type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class MessageBus:
    """Thread-safe pub/sub message bus for inter-agent communication.

    Each subscribed agent has a bounded queue. When a queue exceeds
    ``buffer_size``, the oldest messages are dropped.

    Attributes:
        config: Bus configuration controlling buffer size and logging.

    """

    def __init__(self, config: MessageBusConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._queues: dict[str, deque[AgentMessage]] = {}
        self._logger = MessageLogger("bus")

    def subscribe(self, agent_id: str) -> None:
        """Register an agent to receive messages.

        Args:
            agent_id: Unique identifier for the agent.

        """
        with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = deque(maxlen=self.config.buffer_size)
                if self.config.enable_logging:
                    self._logger.info("agent_subscribed", agent_id=agent_id)

    def unsubscribe(self, agent_id: str) -> None:
        """Remove an agent's subscription and clear its queue.

        Args:
            agent_id: Unique identifier for the agent.

        """
        with self._lock:
            self._queues.pop(agent_id, None)
            if self.config.enable_logging:
                self._logger.info("agent_unsubscribed", agent_id=agent_id)

    def publish(self, message: AgentMessage) -> None:
        """Send a message to a specific agent.

        If the receiver is ``"*"``, the message is broadcast to all
        subscribed agents except the sender.  Otherwise, if the receiver
        is not subscribed, the message is silently dropped.

        Args:
            message: The message to deliver.

        """
        if message.receiver == "*":
            self.broadcast(message)
            return

        with self._lock:
            if message.receiver in self._queues:
                queue = self._queues[message.receiver]
                if len(queue) == queue.maxlen:
                    if self.config.enable_logging:
                        self._logger.debug(
                            "buffer_overflow",
                            agent_id=message.receiver,
                            dropped=1,
                        )
                queue.append(message)
                if self.config.enable_logging:
                    self._logger.debug(
                        "message_published",
                        sender=message.sender,
                        receiver=message.receiver,
                        message_type=message.message_type.value,
                    )

    def broadcast(self, message: AgentMessage) -> None:
        """Send a message to all subscribed agents except the sender.

        Args:
            message: The message to broadcast.

        """
        with self._lock:
            for agent_id, queue in self._queues.items():
                if agent_id != message.sender:
                    # deepcopy so subscribers cannot mutate each other's payload
                    queue.append(copy.deepcopy(message))
            if self.config.enable_logging:
                self._logger.debug(
                    "message_broadcast",
                    sender=message.sender,
                    message_type=message.message_type.value,
                    recipients=len(self._queues) - (1 if message.sender in self._queues else 0),
                )

    def receive(
        self,
        agent_id: str,
        message_type: MessageType | None = None,
    ) -> list[AgentMessage]:
        """Drain messages for an agent, optionally filtered by type.

        Matching messages are removed from the queue. Non-matching
        messages remain for future retrieval.

        Args:
            agent_id: Agent whose messages to retrieve.
            message_type: Optional filter for message type.

        Returns:
            List of messages (possibly empty).

        """
        with self._lock:
            queue = self._queues.get(agent_id)
            if queue is None:
                return []

            if message_type is None:
                messages = list(queue)
                queue.clear()
                return messages

            matched: list[AgentMessage] = []
            remaining: list[AgentMessage] = []
            for msg in queue:
                if msg.message_type == message_type:
                    matched.append(msg)
                else:
                    remaining.append(msg)
            queue.clear()
            queue.extend(remaining)
            return matched

    def peek(self, agent_id: str) -> int:
        """Count pending messages for an agent.

        Args:
            agent_id: Agent whose queue to check.

        Returns:
            Number of pending messages (0 if not subscribed).

        """
        with self._lock:
            queue = self._queues.get(agent_id)
            return len(queue) if queue is not None else 0

    def clear(self, agent_id: str | None = None) -> None:
        """Clear message queues.

        Args:
            agent_id: If provided, clear only that agent's queue.
                If None, clear all queues.

        """
        with self._lock:
            if agent_id is not None:
                queue = self._queues.get(agent_id)
                if queue is not None:
                    queue.clear()
            else:
                for queue in self._queues.values():
                    queue.clear()

    @property
    def subscribers(self) -> list[str]:
        """List of subscribed agent IDs."""
        with self._lock:
            return list(self._queues.keys())
