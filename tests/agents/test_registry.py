"""Tests for agent registration and discovery."""

from __future__ import annotations

import threading

import pytest

from src.agents.base import AgentState, BaseAgent
from src.agents.config import AgentConfig, AgentType
from src.agents.registry import AgentRegistry, get_agent, list_agents, register_agent


class _DummyAgent(BaseAgent):
    """Minimal agent for registry testing."""

    def setup(self) -> None:
        pass

    def step(self) -> AgentState:
        self._state.step += 1
        return self._state

    def reset(self) -> None:
        self._state = self._create_initial_state()

    def get_metrics(self) -> dict[str, float]:
        return {}


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_via_decorator(self) -> None:
        @register_agent("test_dummy")
        class DecoratedAgent(_DummyAgent):
            pass

        assert AgentRegistry().is_registered("test_dummy")
        assert AgentRegistry().get("test_dummy") is DecoratedAgent

    def test_register_direct(self) -> None:
        class DirectAgent(_DummyAgent):
            pass

        AgentRegistry().register("direct_agent", DirectAgent)
        assert AgentRegistry().is_registered("direct_agent")

    def test_duplicate_raises(self) -> None:
        register_agent("dup_agent")(_DummyAgent)
        with pytest.raises(ValueError):
            register_agent("dup_agent")(_DummyAgent)

    def test_get_unregistered_returns_none(self) -> None:
        assert AgentRegistry().get("nonexistent") is None

    def test_get_or_raise_unknown(self) -> None:
        with pytest.raises(KeyError):
            AgentRegistry().get_or_raise("nonexistent")

    def test_get_agent_convenience(self) -> None:
        register_agent("conv_agent")(_DummyAgent)
        cls = get_agent("conv_agent")
        assert cls is _DummyAgent

    def test_get_agent_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_agent("totally_unknown")

    def test_list_agents(self) -> None:
        register_agent("list_a")(_DummyAgent)

        class AnotherAgent(_DummyAgent):
            pass

        register_agent("list_b")(AnotherAgent)
        agents = list_agents()
        assert "list_a" in agents
        assert "list_b" in agents

    def test_clear_for_isolation(self) -> None:
        register_agent("cleared")(_DummyAgent)
        assert AgentRegistry().is_registered("cleared")
        AgentRegistry().clear()
        assert not AgentRegistry().is_registered("cleared")

    def test_thread_safety(self) -> None:
        errors: list[Exception] = []

        def register_in_thread(name: str) -> None:
            try:

                class ThreadAgent(_DummyAgent):
                    pass

                AgentRegistry().register(name, ThreadAgent)
            except ValueError:
                pass  # Expected for duplicates
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_in_thread, args=(f"thread_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All unique names should be registered
        for i in range(10):
            assert AgentRegistry().is_registered(f"thread_{i}")

    def test_singleton_behavior(self) -> None:
        r1 = AgentRegistry()
        r2 = AgentRegistry()
        assert r1 is r2

    def test_len(self) -> None:
        register_agent("len_a")(_DummyAgent)

        class B(_DummyAgent):
            pass

        register_agent("len_b")(B)
        assert len(AgentRegistry()) >= 2

    def test_non_base_agent_rejected(self) -> None:
        class NotAnAgent:
            pass

        with pytest.raises(TypeError):
            register_agent("bad")(NotAnAgent)  # type: ignore[arg-type]
