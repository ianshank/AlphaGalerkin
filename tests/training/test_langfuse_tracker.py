"""Tests for the Langfuse experiment tracker (mocks the langfuse SDK; no torch).

Covers the no-op degradation paths (disabled / no credentials), the mocked
live path (events + scores + artifact + summary + flush), and the critical
shutdown-safety contract: the ``atexit`` flush must never log.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.training.langfuse_tracker import (
    ENV_PUBLIC_KEY,
    ENV_SECRET_KEY,
    LangfuseTracker,
    create_tracker,
)


class _FakeTrace:
    def __init__(self) -> None:
        self.id = "trace-123"
        self.events: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._trace = _FakeTrace()
        self.flush_count = 0

    def trace(self, **kwargs: Any) -> _FakeTrace:
        self._trace.trace_kwargs = kwargs  # type: ignore[attr-defined]
        return self._trace

    def flush(self) -> None:
        self.flush_count += 1


@pytest.fixture
def fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Install a fake ``langfuse`` module and valid credentials."""
    module = SimpleNamespace(Langfuse=_FakeClient)
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv(ENV_PUBLIC_KEY, "pk-test")
    monkeypatch.setenv(ENV_SECRET_KEY, "sk-test")
    return _FakeClient


def _training_metrics(step: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        step=step,
        total_loss=1.0,
        policy_loss=0.5,
        value_loss=0.3,
        lbb_loss=0.2,
        lbb_constant=0.9,
        gradient_norm=1.5,
        learning_rate=1e-3,
        buffer_size=100,
        games_generated=4,
        step_time_ms=12.0,
    )


def test_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_PUBLIC_KEY, raising=False)
    monkeypatch.delenv(ENV_SECRET_KEY, raising=False)
    tracker = LangfuseTracker(config={"enabled": False})
    assert tracker.is_enabled is False
    tracker.log_metrics({"loss": 1.0}, step=1)  # must not raise
    tracker.finish()


def test_no_credentials_degrades_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_PUBLIC_KEY, raising=False)
    monkeypatch.delenv(ENV_SECRET_KEY, raising=False)
    tracker = LangfuseTracker(config={"enabled": True})
    assert tracker.is_enabled is False
    assert tracker.run_id is None
    tracker.log_metrics({"loss": 1.0})  # no-op, must not raise


def test_initializes_with_credentials(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True, "project": "p", "run_name": "r"})
    assert tracker.is_enabled is True
    assert tracker.run_id == "trace-123"
    assert tracker.run_name == "r"


def test_log_metrics_emits_event_and_scores(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True})
    tracker.log_metrics({"loss": 0.5, "name": "x", "flag": True}, step=3)
    trace = tracker.run
    assert len(trace.events) == 1
    assert trace.events[0]["metadata"]["step"] == 3
    # Numeric -> score; bool and str -> not scored.
    scored = {s["name"] for s in trace.scores}
    assert scored == {"loss"}


def test_log_training_step_respects_interval(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True, "log_interval": 10})
    tracker.log_training_step(_training_metrics(step=7))  # 7 % 10 != 0 -> skipped
    assert tracker.run.events == []
    tracker.log_training_step(_training_metrics(step=10))  # logged
    assert len(tracker.run.events) == 1


def test_log_model_artifact_records_path_no_upload(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True})
    tracker.log_model_artifact("/tmp/ckpt.pt", name="best", aliases=["best"])
    events = [e for e in tracker.run.events if e["name"] == "model_artifact"]
    assert len(events) == 1
    assert events[0]["metadata"]["path"] == "/tmp/ckpt.pt"
    assert events[0]["metadata"]["aliases"] == ["best"]


def test_log_summary_sets_trace_output(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True})
    tracker.log_summary({"final_loss": 0.1})
    assert tracker.run.updates[-1]["output"] == {"final_loss": 0.1}


def test_noop_methods_do_not_raise(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True})
    tracker.watch_model(object())
    tracker.log_histogram("h", [1, 2, 3])
    tracker.define_metric("m")
    tracker.set_step_offset(5)
    # set_step_offset shifts subsequent steps.
    tracker.log_metrics({"v": 1.0}, step=0)
    assert tracker.run.events[-1]["metadata"]["step"] == 5


def test_finish_flushes_and_is_idempotent(fake_langfuse: type[_FakeClient]) -> None:
    tracker = LangfuseTracker(config={"enabled": True})
    client = tracker._client
    tracker.finish()
    tracker.finish()  # idempotent
    assert client.flush_count == 1
    assert tracker.is_enabled is False


def test_atexit_flush_never_logs_and_swallows_errors(
    fake_langfuse: type[_FakeClient],
) -> None:
    tracker = LangfuseTracker(config={"enabled": True})

    # A flush that raises must not propagate out of the atexit handler.
    def _boom() -> None:
        raise RuntimeError("stream closed")

    tracker._client.flush = _boom  # type: ignore[method-assign]
    tracker._atexit_flush()  # must not raise


def test_create_tracker_factory(fake_langfuse: type[_FakeClient]) -> None:
    tracker = create_tracker({"enabled": True}, {"lr": 1e-3})
    assert isinstance(tracker, LangfuseTracker)
    assert tracker.is_enabled is True
