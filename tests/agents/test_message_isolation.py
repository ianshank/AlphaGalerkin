"""Regression tests for :class:`MessageBus.broadcast` payload isolation.

The bus must deep-copy each broadcast envelope so that a subscriber that
mutates its received ``payload`` does not corrupt the copies handed to
other subscribers (gemini-code-assist review on PR #53,
``src/agents/message.py:152``).
"""

from __future__ import annotations

import pytest

from src.agents.config import MessageBusConfig, MessageType
from src.agents.message import AgentMessage, MessageBus


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus(MessageBusConfig(name="iso-bus", buffer_size=8, enable_logging=False))


def test_broadcast_isolates_payload_dict(bus: MessageBus) -> None:
    """Mutating one subscriber's payload must not affect any other subscriber."""
    bus.subscribe("a")
    bus.subscribe("b")
    bus.subscribe("c")

    payload = {"values": [1.0, 2.0], "meta": {"step": 0}}
    msg = AgentMessage(
        sender="a",
        receiver="*",
        message_type=MessageType.BOUNDARY_DATA,
        payload=payload,
    )
    bus.broadcast(msg)

    received_b = bus.receive("b")
    received_c = bus.receive("c")
    assert len(received_b) == 1
    assert len(received_c) == 1

    # Mutate b's payload (top-level + nested)
    received_b[0].payload["values"].append(99.0)
    received_b[0].payload["meta"]["step"] = 999

    # c's payload must remain untouched
    assert received_c[0].payload["values"] == [1.0, 2.0]
    assert received_c[0].payload["meta"] == {"step": 0}

    # And the original sender-side dict must remain untouched too
    assert payload["values"] == [1.0, 2.0]
    assert payload["meta"] == {"step": 0}


def test_broadcast_skips_sender(bus: MessageBus) -> None:
    """Broadcast should not deliver the message back to the sender."""
    bus.subscribe("a")
    bus.subscribe("b")

    msg = AgentMessage(
        sender="a",
        receiver="*",
        message_type=MessageType.STATE_UPDATE,
        payload={"x": 1},
    )
    bus.broadcast(msg)

    assert bus.receive("a") == []
    assert len(bus.receive("b")) == 1


def test_broadcast_message_ids_are_preserved(bus: MessageBus) -> None:
    """Deepcopy must preserve message_id so causality tracking still works."""
    bus.subscribe("a")
    bus.subscribe("b")
    bus.subscribe("c")

    msg = AgentMessage(
        sender="a",
        receiver="*",
        message_type=MessageType.CONVERGENCE_CHECK,
        payload={"residual": 1e-6},
    )
    bus.broadcast(msg)

    rb = bus.receive("b")[0]
    rc = bus.receive("c")[0]
    assert rb.message_id == msg.message_id == rc.message_id
