"""Tests for BaseAgent lifecycle hooks and opt-in timeout enforcement.

These guard the WS1B agentic-hardening additions:

- ``pre_setup`` / ``post_setup`` / ``pre_step`` / ``post_step`` are no-op
  extension points by default (backwards compatible) and fire in a defined
  order in ``run()``.
- ``AgentConfig.enforce_timeout`` gates wall-clock enforcement; when False
  (the default) the run loop never reads the clock, preserving historical
  behaviour; when True a run that exceeds ``timeout_seconds`` stops with
  ``ExecutionStatus.TIMEOUT``.
"""

from __future__ import annotations

import pytest

from src.agents import base as agent_base
from src.agents.base import AgentState, BaseAgent
from src.agents.config import AgentConfig, AgentType
from src.templates.base import ExecutionStatus


class _PlainAgent(BaseAgent):
    """Minimal agent that overrides no hooks (backwards-compat probe)."""

    def setup(self) -> None:
        self._did_setup = True

    def step(self) -> AgentState:
        self._state.step += 1
        return self._state

    def reset(self) -> None:
        self._state = self._create_initial_state()

    def get_metrics(self) -> dict[str, float]:
        return {"step": float(self._state.step)}


class _ProbeAgent(_PlainAgent):
    """Agent that records every hook + step call to assert ordering."""

    def __init__(self, config: AgentConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)  # type: ignore[arg-type]
        self.events: list[str] = []

    def setup(self) -> None:
        self.events.append("setup")

    def step(self) -> AgentState:
        self.events.append("step")
        self._state.step += 1
        return self._state

    def pre_setup(self) -> None:
        self.events.append("pre_setup")

    def post_setup(self) -> None:
        self.events.append("post_setup")

    def pre_step(self) -> None:
        self.events.append("pre_step")

    def post_step(self) -> None:
        self.events.append("post_step")


def _config(**overrides: object) -> AgentConfig:
    params: dict[str, object] = {
        "name": "probe",
        "agent_type": AgentType.META,
        "max_steps": 2,
    }
    params.update(overrides)
    return AgentConfig(**params)  # type: ignore[arg-type]


class TestLifecycleHooks:
    def test_default_hooks_are_noops(self) -> None:
        """A plain agent overriding no hooks runs to completion unchanged."""
        agent = _PlainAgent(_config(max_steps=3))
        state = agent.run()
        assert state.status == ExecutionStatus.COMPLETED
        assert state.step == 3

    def test_base_hook_methods_return_none(self) -> None:
        """The base hook implementations exist and are pure no-ops."""
        agent = _PlainAgent(_config())
        assert agent.pre_setup() is None
        assert agent.post_setup() is None
        assert agent.pre_step() is None
        assert agent.post_step() is None

    def test_hooks_fire_in_order(self) -> None:
        """Hooks wrap setup and every step in the documented order."""
        agent = _ProbeAgent(_config(max_steps=2))
        agent.run()
        assert agent.events == [
            "pre_setup",
            "setup",
            "post_setup",
            "pre_step",
            "step",
            "post_step",
            "pre_step",
            "step",
            "post_step",
        ]


class TestTimeoutEnforcement:
    def test_timeout_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With enforce_timeout=False the clock is never read; all steps run."""
        calls = {"n": 0}

        def _boom() -> float:  # pragma: no cover - must never be called
            calls["n"] += 1
            return 0.0

        monkeypatch.setattr(agent_base.time, "monotonic", _boom)
        agent = _PlainAgent(_config(max_steps=3, enforce_timeout=False))
        state = agent.run()
        assert state.status == ExecutionStatus.COMPLETED
        assert state.step == 3
        assert calls["n"] == 0  # deadline branch skipped entirely

    def test_timeout_enforced_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With enforce_timeout=True a run past the deadline stops with TIMEOUT."""
        # monotonic() sequence: deadline calc (0.0), iter-1 check (0.0 < 1 ok),
        # iter-2 check (5.0 >= 1 -> timeout) before a 2nd step runs.
        seq = iter([0.0, 0.0, 5.0, 5.0, 5.0])

        monkeypatch.setattr(agent_base.time, "monotonic", lambda: next(seq))
        agent = _ProbeAgent(_config(max_steps=10, enforce_timeout=True, timeout_seconds=1))
        state = agent.run()

        assert state.status == ExecutionStatus.TIMEOUT
        assert state.step == 1  # exactly one step executed before the deadline
        assert agent.events.count("step") == 1
        # is_terminal must recognise TIMEOUT as terminal (status consistency).
        assert agent.is_terminal
