"""Tests for inter-agent message bus."""

from __future__ import annotations

import threading

import pytest

from src.agents.config import MessageBusConfig, MessageType
from src.agents.message import AgentMessage, MessageBus


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_creation(self) -> None:
        msg = AgentMessage(
            sender="a",
            receiver="b",
            message_type=MessageType.STATE_UPDATE,
            payload={"key": "value"},
        )
        assert msg.sender == "a"
        assert msg.receiver == "b"
        assert msg.message_type == MessageType.STATE_UPDATE
        assert msg.payload == {"key": "value"}
        assert msg.timestamp > 0
        assert len(msg.message_id) > 0

    def test_unique_ids(self) -> None:
        m1 = AgentMessage(sender="a", receiver="b", message_type=MessageType.STATE_UPDATE)
        m2 = AgentMessage(sender="a", receiver="b", message_type=MessageType.STATE_UPDATE)
        assert m1.message_id != m2.message_id

    def test_default_payload(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", message_type=MessageType.STATE_UPDATE)
        assert msg.payload == {}


class TestMessageBus:
    """Tests for MessageBus."""

    @pytest.fixture
    def bus(self) -> MessageBus:
        config = MessageBusConfig(name="test", buffer_size=100)
        return MessageBus(config)

    def test_subscribe_and_receive(self, bus: MessageBus) -> None:
        bus.subscribe("agent_1")
        msg = AgentMessage(
            sender="sender", receiver="agent_1", message_type=MessageType.STATE_UPDATE
        )
        bus.publish(msg)
        received = bus.receive("agent_1")
        assert len(received) == 1
        assert received[0].sender == "sender"

    def test_receive_drains_queue(self, bus: MessageBus) -> None:
        bus.subscribe("agent_1")
        bus.publish(
            AgentMessage(sender="s", receiver="agent_1", message_type=MessageType.STATE_UPDATE)
        )
        assert len(bus.receive("agent_1")) == 1
        assert len(bus.receive("agent_1")) == 0

    def test_unsubscribed_receive_empty(self, bus: MessageBus) -> None:
        assert bus.receive("nonexistent") == []

    def test_unsubscribed_publish_dropped(self, bus: MessageBus) -> None:
        bus.publish(
            AgentMessage(sender="s", receiver="nonexistent", message_type=MessageType.STATE_UPDATE)
        )
        # No error, message silently dropped

    def test_multiple_subscribers(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.subscribe("b")
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        bus.publish(AgentMessage(sender="s", receiver="b", message_type=MessageType.BOUNDARY_DATA))
        assert len(bus.receive("a")) == 1
        assert len(bus.receive("b")) == 1

    def test_broadcast(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.subscribe("b")
        bus.subscribe("sender")
        msg = AgentMessage(sender="sender", receiver="*", message_type=MessageType.STATE_UPDATE)
        bus.broadcast(msg)
        assert len(bus.receive("a")) == 1
        assert len(bus.receive("b")) == 1
        assert len(bus.receive("sender")) == 0  # sender excluded

    def test_message_type_filtering(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.BOUNDARY_DATA))
        state_msgs = bus.receive("a", MessageType.STATE_UPDATE)
        assert len(state_msgs) == 1
        assert state_msgs[0].message_type == MessageType.STATE_UPDATE

        # Remaining message still in queue
        remaining = bus.receive("a")
        assert len(remaining) == 1
        assert remaining[0].message_type == MessageType.BOUNDARY_DATA

    def test_buffer_overflow(self) -> None:
        config = MessageBusConfig(name="small", buffer_size=3)
        bus = MessageBus(config)
        bus.subscribe("a")
        for i in range(5):
            bus.publish(
                AgentMessage(
                    sender="s",
                    receiver="a",
                    message_type=MessageType.STATE_UPDATE,
                    payload={"i": i},
                )
            )
        received = bus.receive("a")
        assert len(received) == 3
        # Oldest messages dropped
        assert received[0].payload["i"] == 2

    def test_peek(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        assert bus.peek("a") == 0
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        assert bus.peek("a") == 1
        assert bus.peek("nonexistent") == 0

    def test_clear_specific(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.subscribe("b")
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        bus.publish(AgentMessage(sender="s", receiver="b", message_type=MessageType.STATE_UPDATE))
        bus.clear("a")
        assert bus.peek("a") == 0
        assert bus.peek("b") == 1

    def test_clear_all(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.subscribe("b")
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        bus.publish(AgentMessage(sender="s", receiver="b", message_type=MessageType.STATE_UPDATE))
        bus.clear()
        assert bus.peek("a") == 0
        assert bus.peek("b") == 0

    def test_unsubscribe(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.unsubscribe("a")
        assert "a" not in bus.subscribers
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        assert bus.receive("a") == []

    def test_subscribers_list(self, bus: MessageBus) -> None:
        bus.subscribe("a")
        bus.subscribe("b")
        assert sorted(bus.subscribers) == ["a", "b"]

    def test_thread_safety(self, bus: MessageBus) -> None:
        bus.subscribe("target")
        errors: list[Exception] = []
        n_messages = 50

        def publish_messages(sender_id: str) -> None:
            try:
                for i in range(n_messages):
                    bus.publish(
                        AgentMessage(
                            sender=sender_id,
                            receiver="target",
                            message_type=MessageType.STATE_UPDATE,
                            payload={"i": i},
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publish_messages, args=(f"sender_{j}",)) for j in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        received = bus.receive("target")
        assert len(received) == min(100, 5 * n_messages)

    def test_logging_enabled(self) -> None:
        config = MessageBusConfig(name="logged", buffer_size=10, enable_logging=True)
        bus = MessageBus(config)
        bus.subscribe("a")
        bus.publish(AgentMessage(sender="s", receiver="a", message_type=MessageType.STATE_UPDATE))
        # Should not raise
        assert bus.peek("a") == 1

    def test_unsubscribe_with_logging(self) -> None:
        """Unsubscribe with enable_logging=True triggers log."""
        config = MessageBusConfig(name="logged", buffer_size=10, enable_logging=True)
        bus = MessageBus(config)
        bus.subscribe("a")
        bus.unsubscribe("a")
        # No error, agent removed
        assert "a" not in bus.subscribers

    def test_unsubscribe_unknown_agent(self) -> None:
        """Unsubscribing an agent that was never subscribed does not raise."""
        config = MessageBusConfig(name="safe", buffer_size=10, enable_logging=True)
        bus = MessageBus(config)
        bus.unsubscribe("never_existed")  # Should not raise

    def test_broadcast_with_logging(self) -> None:
        """Broadcast with enable_logging=True triggers log."""
        config = MessageBusConfig(name="logged", buffer_size=10, enable_logging=True)
        bus = MessageBus(config)
        bus.subscribe("a")
        bus.subscribe("b")
        msg = AgentMessage(
            sender="sender",
            receiver="*",
            message_type=MessageType.STATE_UPDATE,
        )
        bus.broadcast(msg)
        assert bus.peek("a") == 1
        assert bus.peek("b") == 1

    def test_broadcast_no_subscribers(self) -> None:
        """Broadcast with no subscribers does not raise."""
        config = MessageBusConfig(name="empty", buffer_size=10, enable_logging=True)
        bus = MessageBus(config)
        msg = AgentMessage(
            sender="sender",
            receiver="*",
            message_type=MessageType.STATE_UPDATE,
        )
        bus.broadcast(msg)  # No error
