"""Lifecycle hooks and event dispatchers for agent execution loops.

Provides extensible, thread-safe hook interfaces to intercept and customize
agent lifecycle events (initialization, pre-step, post-step, convergence,
checkpointing, error handling) without modifying agent core logic.

Example:
    from src.agents.lifecycle_hooks import (
        HookManager,
        LoggingHook,
        MetricsCollectorHook,
        EarlyStoppingHook,
    )

    hook_manager = HookManager()
    hook_manager.register(LoggingHook())
    hook_manager.register(MetricsCollectorHook())
    hook_manager.register(EarlyStoppingHook(patience=5, min_delta=1e-4))

    # In agent loop:
    hook_manager.trigger_pre_step(step=0, context={"state": current_state})

"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from src.templates.logging import create_logger_class

HookLogger = create_logger_class("LifecycleHooks")


class LifecycleHook(ABC):
    """Abstract base class for all agent execution lifecycle hooks."""

    @abstractmethod
    def on_init(self, context: dict[str, Any]) -> None:
        """Called when the agent/loop initializes."""

    @abstractmethod
    def on_pre_step(self, step: int, context: dict[str, Any]) -> None:
        """Called immediately before executing a step."""

    @abstractmethod
    def on_post_step(self, step: int, result: Any, context: dict[str, Any]) -> None:
        """Called immediately after executing a step."""

    @abstractmethod
    def on_error(self, step: int, error: Exception, context: dict[str, Any]) -> None:
        """Called when an exception is raised during step execution."""

    @abstractmethod
    def on_complete(self, total_steps: int, context: dict[str, Any]) -> None:
        """Called upon successful completion of the loop."""


class BaseLifecycleHook(LifecycleHook):
    """Default no-op implementation of LifecycleHook for selective overriding."""

    def on_init(self, context: dict[str, Any]) -> None:
        """Default no-op init."""

    def on_pre_step(self, step: int, context: dict[str, Any]) -> None:
        """Default no-op pre-step."""

    def on_post_step(self, step: int, result: Any, context: dict[str, Any]) -> None:
        """Default no-op post-step."""

    def on_error(self, step: int, error: Exception, context: dict[str, Any]) -> None:
        """Default no-op error handler."""

    def on_complete(self, total_steps: int, context: dict[str, Any]) -> None:
        """Default no-op completion."""


class LoggingHook(BaseLifecycleHook):
    """Lifecycle hook that logs structured events at each lifecycle stage."""

    def __init__(self, name: str = "AgentLoop") -> None:
        self.name = name
        self.logger = HookLogger(component=name)

    def on_init(self, context: dict[str, Any]) -> None:
        """Log loop initialization."""
        self.logger.info("loop_initialized", name=self.name, **context)

    def on_pre_step(self, step: int, context: dict[str, Any]) -> None:
        """Log pre-step."""
        self.logger.debug("pre_step", step=step, **context)

    def on_post_step(self, step: int, result: Any, context: dict[str, Any]) -> None:
        """Log post-step."""
        self.logger.debug("post_step", step=step, result=str(result), **context)

    def on_error(self, step: int, error: Exception, context: dict[str, Any]) -> None:
        """Log error."""
        self.logger.error("step_error", step=step, error=str(error), exc_info=True, **context)

    def on_complete(self, total_steps: int, context: dict[str, Any]) -> None:
        """Log loop completion."""
        self.logger.info("loop_completed", total_steps=total_steps, **context)


class MetricsCollectorHook(BaseLifecycleHook):
    """Lifecycle hook that records step durations and numeric metrics."""

    def __init__(self) -> None:
        self.step_times: list[float] = []
        self.metrics_history: list[dict[str, Any]] = []
        self._start_time: float = 0.0
        self._step_start: float = 0.0

    def on_init(self, context: dict[str, Any]) -> None:
        """Initialize metric buffers."""
        self.step_times.clear()
        self.metrics_history.clear()
        self._start_time = time.perf_counter()

    def on_pre_step(self, step: int, context: dict[str, Any]) -> None:
        """Record step start timestamp."""
        self._step_start = time.perf_counter()

    def on_post_step(self, step: int, result: Any, context: dict[str, Any]) -> None:
        """Record step duration and metrics."""
        duration = time.perf_counter() - self._step_start
        self.step_times.append(duration)
        step_metrics = {k: v for k, v in context.items() if isinstance(v, (int, float, bool))}
        step_metrics["step"] = step
        step_metrics["duration_s"] = duration
        self.metrics_history.append(step_metrics)

    @property
    def total_duration_s(self) -> float:
        """Total execution time from initialization."""
        if not self._start_time:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def mean_step_duration_s(self) -> float:
        """Average duration per step."""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)


class EarlyStoppingHook(BaseLifecycleHook):
    """Lifecycle hook for early stopping based on metric convergence."""

    def __init__(
        self,
        metric_key: str = "residual",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.metric_key = metric_key
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.wait_count: int = 0
        self.should_stop: bool = False

    def on_init(self, context: dict[str, Any]) -> None:
        """Reset early stopping counters."""
        self.best_score = None
        self.wait_count = 0
        self.should_stop = False

    def on_post_step(self, step: int, result: Any, context: dict[str, Any]) -> None:
        """Evaluate metric against convergence criteria."""
        if self.metric_key not in context:
            return

        current = float(context[self.metric_key])
        if self.best_score is None:
            self.best_score = current
            return

        if self.mode == "min":
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.should_stop = True


class HookManager:
    """Thread-safe manager for dispatching lifecycle events to registered hooks."""

    def __init__(self) -> None:
        self._hooks: list[LifecycleHook] = []
        self._lock = threading.Lock()

    def register(self, hook: LifecycleHook) -> None:
        """Register a new lifecycle hook."""
        with self._lock:
            if hook not in self._hooks:
                self._hooks.append(hook)

    def unregister(self, hook: LifecycleHook) -> None:
        """Unregister an existing lifecycle hook."""
        with self._lock:
            if hook in self._hooks:
                self._hooks.remove(hook)

    def trigger_init(self, context: dict[str, Any]) -> None:
        """Dispatch init event."""
        with self._lock:
            hooks = list(self._hooks)
        for hook in hooks:
            hook.on_init(context)

    def trigger_pre_step(self, step: int, context: dict[str, Any]) -> None:
        """Dispatch pre-step event."""
        with self._lock:
            hooks = list(self._hooks)
        for hook in hooks:
            hook.on_pre_step(step, context)

    def trigger_post_step(self, step: int, result: Any, context: dict[str, Any]) -> None:
        """Dispatch post-step event."""
        with self._lock:
            hooks = list(self._hooks)
        for hook in hooks:
            hook.on_post_step(step, result, context)

    def trigger_error(self, step: int, error: Exception, context: dict[str, Any]) -> None:
        """Dispatch error event."""
        with self._lock:
            hooks = list(self._hooks)
        for hook in hooks:
            hook.on_error(step, error, context)

    def trigger_complete(self, total_steps: int, context: dict[str, Any]) -> None:
        """Dispatch complete event."""
        with self._lock:
            hooks = list(self._hooks)
        for hook in hooks:
            hook.on_complete(total_steps, context)
