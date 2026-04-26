"""Tests for the trainer callback base abstractions and registry.

Covers:

* :class:`Callback` default no-op semantics.
* :class:`CallbackContext` immutability / defaults.
* :class:`CallbackRegistry` registration, retrieval, duplicate guard.
* :class:`CallbackSpec` Pydantic validation.
* :func:`build_callbacks_from_specs` instantiation, error paths.
* Dispatch via :class:`BaseTrainer` helpers — including exception
  isolation (one failing callback must not abort the others).

The tests deliberately avoid spinning up any real model / data: a
trivial ``DummyTrainer`` subclass exercises the dispatch surface.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from src.training.base_trainer import BaseTrainer, BaseTrainerConfig
from src.training.callbacks import (
    Callback,
    CallbackContext,
    CallbackRegistry,
    CallbackSpec,
    build_callbacks_from_specs,
    register_callback,
)
from src.training.callbacks.base import _ensure_builtin_callbacks_imported

# ---------------------------------------------------------------------------
# Fixtures: registry isolation per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry() -> Any:
    """Restore the registry after each test so registrations don't leak."""
    registry = CallbackRegistry()
    snapshot = registry.get_all()
    yield
    registry.clear()
    for name, cls in snapshot.items():
        registry.register(name, cls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCallback(Callback):
    """Callback that records every event it sees."""

    def __init__(self, tag: str = "rec") -> None:
        self.tag = tag
        self.events: list[tuple[str, CallbackContext]] = []

    def on_train_start(self, ctx: CallbackContext) -> None:
        self.events.append(("on_train_start", ctx))

    def on_step_end(self, ctx: CallbackContext) -> None:
        self.events.append(("on_step_end", ctx))

    def on_evaluation(self, ctx: CallbackContext) -> None:
        self.events.append(("on_evaluation", ctx))

    def on_checkpoint(self, ctx: CallbackContext) -> None:
        self.events.append(("on_checkpoint", ctx))

    def on_train_end(self, ctx: CallbackContext) -> None:
        self.events.append(("on_train_end", ctx))


class _RaisingCallback(Callback):
    """Callback whose hooks always raise."""

    def on_step_end(self, ctx: CallbackContext) -> None:
        raise RuntimeError("intentional")


class _DummyTrainer(BaseTrainer[BaseTrainerConfig]):
    """Minimal concrete trainer for testing dispatch.

    Implements the abstract hooks with trivial bodies; we only exercise
    the callback machinery, not real training.
    """

    def compute_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        loss = self.model(torch.zeros(1)).sum()
        return loss, {}

    def generate_data(self) -> Any:
        return None

    def evaluate(self) -> dict[str, float]:
        return {}


def _make_trainer(callbacks: list[Callback] | None = None) -> _DummyTrainer:
    """Construct a `_DummyTrainer` with a tiny linear model on CPU."""
    model = nn.Linear(1, 1, bias=False)
    config = BaseTrainerConfig(
        name="dummy",
        learning_rate=1e-3,
        warmup_steps=0,
        total_steps=10,
        use_amp=False,
    )
    return _DummyTrainer(
        model=model,
        config=config,
        device="cpu",
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# CallbackContext
# ---------------------------------------------------------------------------


class TestCallbackContext:
    """:class:`CallbackContext` is an immutable carrier."""

    def test_defaults(self) -> None:
        ctx = CallbackContext(step=0)
        assert ctx.step == 0
        assert ctx.metrics == {}
        assert ctx.model is None
        assert ctx.trainer is None
        assert ctx.extras == {}

    def test_is_frozen(self) -> None:
        ctx = CallbackContext(step=1, metrics={"loss": 0.5})
        with pytest.raises((AttributeError, Exception)):
            ctx.step = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Callback default behaviour
# ---------------------------------------------------------------------------


class TestCallbackDefaults:
    """The base :class:`Callback` provides no-op defaults."""

    def test_all_hooks_are_noops(self) -> None:
        cb = Callback()
        ctx = CallbackContext(step=0)
        # Should not raise
        cb.on_train_start(ctx)
        cb.on_step_end(ctx)
        cb.on_evaluation(ctx)
        cb.on_checkpoint(ctx)
        cb.on_train_end(ctx)


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


class TestCallbackRegistry:
    """Decorator-based registration against the singleton registry."""

    def test_register_and_lookup(self) -> None:
        @register_callback("rec_test")
        class _MyCallback(_RecordingCallback):
            pass

        assert "rec_test" in CallbackRegistry()
        cls = CallbackRegistry().get_or_raise("rec_test")
        assert issubclass(cls, _RecordingCallback)

    def test_duplicate_registration_raises(self) -> None:
        register_callback("dup")(_RecordingCallback)
        with pytest.raises(ValueError, match="already registered"):
            register_callback("dup")(_RecordingCallback)

    def test_register_rejects_non_callback(self) -> None:
        class NotACallback:
            pass

        with pytest.raises(TypeError, match="must inherit from"):
            register_callback("bad")(NotACallback)

    def test_get_or_raise_lists_available(self) -> None:
        register_callback("a")(_RecordingCallback)
        with pytest.raises(KeyError, match="Available"):
            CallbackRegistry().get_or_raise("missing")


# ---------------------------------------------------------------------------
# CallbackSpec
# ---------------------------------------------------------------------------


class TestCallbackSpec:
    """Pydantic validation of :class:`CallbackSpec`."""

    def test_valid_spec(self) -> None:
        spec = CallbackSpec(name="x", params={"k": 1})
        assert spec.name == "x"
        assert spec.params == {"k": 1}

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CallbackSpec(name="", params={})

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CallbackSpec(name="x", params={}, extra="bad")  # type: ignore[call-arg]

    def test_default_params_empty(self) -> None:
        spec = CallbackSpec(name="x")
        assert spec.params == {}


# ---------------------------------------------------------------------------
# build_callbacks_from_specs
# ---------------------------------------------------------------------------


class TestBuildCallbacks:
    """Resolution of specs -> instances."""

    def test_empty_returns_empty(self) -> None:
        assert build_callbacks_from_specs([]) == []

    def test_resolves_registered_spec(self) -> None:
        register_callback("rec_build")(_RecordingCallback)
        cbs = build_callbacks_from_specs(
            [CallbackSpec(name="rec_build", params={"tag": "from_spec"})]
        )
        assert len(cbs) == 1
        assert isinstance(cbs[0], _RecordingCallback)
        assert cbs[0].tag == "from_spec"

    def test_unknown_callback_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            build_callbacks_from_specs([CallbackSpec(name="nonexistent")])

    def test_invalid_params_raises_typeerror(self) -> None:
        register_callback("rec_bad_params")(_RecordingCallback)
        with pytest.raises(TypeError, match="Failed to instantiate"):
            build_callbacks_from_specs(
                [CallbackSpec(name="rec_bad_params", params={"unknown_kwarg": 1})]
            )

    def test_non_callback_subclass_in_registry_rejected(self) -> None:
        """Defensively rejects an entry whose class produces a non-Callback."""
        # ``register_callback`` already enforces inheritance; we simulate a
        # bypass by using the lower-level decorator directly.
        from src.training.callbacks.base import _register_callback_decorator

        class _Fake:
            def __init__(self) -> None:
                pass

        _register_callback_decorator("fake_non_callback")(_Fake)
        with pytest.raises(TypeError, match="non-Callback instance"):
            build_callbacks_from_specs([CallbackSpec(name="fake_non_callback")])

    def test_builtin_modules_are_imported(self) -> None:
        # _ensure_builtin_callbacks_imported is idempotent and must succeed
        # even when the registry already has entries.
        _ensure_builtin_callbacks_imported()
        _ensure_builtin_callbacks_imported()

    def test_builtin_module_import_failure_is_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing built-in module import must not abort registry use."""
        import importlib

        from src.training.callbacks import base as base_module

        monkeypatch.setattr(
            base_module,
            "BUILTIN_CALLBACK_MODULES",
            ("nonexistent_module_xyz",),
        )

        original = importlib.import_module

        def _blow_up(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "nonexistent_module_xyz":
                raise ImportError("forced missing module")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", _blow_up)
        # Should not raise even though the module is missing
        base_module._ensure_builtin_callbacks_imported()
        # Still resolves valid callbacks afterwards
        register_callback("ok_after_failure")(_RecordingCallback)
        assert "ok_after_failure" in CallbackRegistry()


# ---------------------------------------------------------------------------
# Trainer dispatch integration
# ---------------------------------------------------------------------------


class TestTrainerDispatch:
    """Verify :class:`BaseTrainer` correctly dispatches to callbacks."""

    def test_default_callbacks_empty(self) -> None:
        trainer = _make_trainer()
        assert trainer.callbacks == []

    def test_callbacks_constructor_arg_stored(self) -> None:
        cb = _RecordingCallback()
        trainer = _make_trainer(callbacks=[cb])
        assert trainer.callbacks == [cb]

    def test_add_remove_callback_roundtrip(self) -> None:
        trainer = _make_trainer()
        cb = _RecordingCallback()
        trainer.add_callback(cb)
        assert cb in trainer.callbacks
        assert trainer.remove_callback(cb) is True
        assert cb not in trainer.callbacks
        assert trainer.remove_callback(cb) is False  # already removed

    def test_add_callback_rejects_non_callback(self) -> None:
        trainer = _make_trainer()
        with pytest.raises(TypeError, match="Expected Callback"):
            trainer.add_callback("not a callback")  # type: ignore[arg-type]

    def test_dispatch_invokes_each_callback(self) -> None:
        cb1, cb2 = _RecordingCallback("a"), _RecordingCallback("b")
        trainer = _make_trainer(callbacks=[cb1, cb2])
        ctx = trainer._build_callback_context(step=5, metrics={"loss": 0.1})
        trainer._dispatch_callback("on_step_end", ctx)
        assert cb1.events[-1][0] == "on_step_end"
        assert cb2.events[-1][0] == "on_step_end"
        assert cb1.events[-1][1].step == 5

    def test_dispatch_isolates_callback_exceptions(self) -> None:
        good, bad, also_good = (
            _RecordingCallback("good"),
            _RaisingCallback(),
            _RecordingCallback("also"),
        )
        trainer = _make_trainer(callbacks=[good, bad, also_good])
        ctx = trainer._build_callback_context(step=1)
        # Should not raise — exception isolated and logged
        trainer._dispatch_callback("on_step_end", ctx)
        # Healthy callbacks still observed the event
        assert good.events[-1][0] == "on_step_end"
        assert also_good.events[-1][0] == "on_step_end"

    def test_dispatch_unknown_event_is_silent(self) -> None:
        cb = _RecordingCallback()
        trainer = _make_trainer(callbacks=[cb])
        ctx = trainer._build_callback_context(step=0)
        # Hooks the callback hasn't implemented should not raise
        trainer._dispatch_callback("on_made_up_event", ctx)
        assert cb.events == []  # nothing recorded

    def test_build_context_uses_global_step_default(self) -> None:
        trainer = _make_trainer()
        trainer.global_step = 42
        ctx = trainer._build_callback_context()
        assert ctx.step == 42
        assert ctx.model is trainer.model
        assert ctx.trainer is trainer
