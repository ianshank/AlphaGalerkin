"""Base callback abstraction and registry for AlphaGalerkin trainers.

This module defines:

* :class:`Callback` — abstract base with no-op default implementations
  for every lifecycle hook.  Subclasses override only the events they
  care about.
* :class:`CallbackContext` — frozen dataclass passed to every hook,
  bundling ``step``, ``metrics``, ``model``, ``trainer`` and an
  ``extras`` dict.
* :class:`CallbackSpec` — Pydantic config model describing a callback
  to instantiate by name.  Used in ``TrainingConfig.callbacks``.
* :class:`CallbackRegistry` and :func:`register_callback` — thread-safe
  singleton registry built on top of the existing
  :func:`src.templates.registry.create_registry` template.
* :func:`build_callbacks_from_specs` — factory that resolves a list of
  :class:`CallbackSpec` to instantiated :class:`Callback` objects,
  importing built-in callback modules first so their registrations
  exist.

The lifecycle hooks deliberately avoid coupling to any concrete
trainer.  ``CallbackContext.trainer`` is typed as :class:`object` to
avoid a circular import; callbacks that need typed access can do
``cast(BaseTrainer[Any], ctx.trainer)``.

All hyperparameters are surfaced via Pydantic, so no hardcoded values
appear in user-facing YAML configurations.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import Module

from src.templates.registry import create_typed_registry

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Callback context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallbackContext:
    """Immutable snapshot passed to every callback hook.

    Attributes:
        step: Current training step (0-indexed).
        metrics: Mapping of scalar metric name to value for the most
            recent step or evaluation.  Always non-None — empty dict if
            no metrics are available.
        model: Model under training, or ``None`` for events that do
            not have a model (rare).
        trainer: Reference to the trainer instance dispatching the
            event.  Typed as :class:`object` to avoid circular imports;
            cast at the call site if you need typed access.
        extras: Free-form per-event payload (e.g. checkpoint paths,
            evaluation game results).  Always non-None.

    """

    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    model: Module | None = None
    trainer: object | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Callback ABC
# ---------------------------------------------------------------------------


class Callback:
    """Base class for AlphaGalerkin trainer callbacks.

    Subclasses override the lifecycle hooks they care about.  Every
    method has a no-op default so subclasses only need to implement
    the events they handle.

    Lifecycle order during a single ``Trainer.train()`` invocation::

        on_train_start(ctx)            # once
        for step in steps:
            ...                        # training step
            on_step_end(ctx)           # once per step
            on_evaluation(ctx)         # zero or more (opt-in)
            on_checkpoint(ctx)         # zero or more (opt-in)
        on_train_end(ctx)              # once

    Callbacks must be safe to invoke even if the trainer is in an
    unusual state (e.g. mid-epoch interruption).  Hooks should never
    raise — exceptions are caught by the trainer's dispatcher and
    logged but do not abort training.
    """

    def on_train_start(self, ctx: CallbackContext) -> None:  # noqa: B027
        """Called once before the first training step.

        Args:
            ctx: Callback context.

        """

    def on_step_end(self, ctx: CallbackContext) -> None:  # noqa: B027
        """Called after each training step.

        Args:
            ctx: Callback context.  ``ctx.metrics`` contains the loss
                and any per-component metrics for this step.

        """

    def on_evaluation(self, ctx: CallbackContext) -> None:  # noqa: B027
        """Called after an evaluation pass.

        Args:
            ctx: Callback context.  ``ctx.metrics`` contains evaluation
                metrics.  ``ctx.extras`` may contain per-game results.

        """

    def on_checkpoint(self, ctx: CallbackContext) -> None:  # noqa: B027
        """Called after a checkpoint is saved.

        Args:
            ctx: Callback context.  ``ctx.extras["path"]`` is the
                checkpoint path on disk.

        """

    def on_train_end(self, ctx: CallbackContext) -> None:  # noqa: B027
        """Called once after the final training step.

        Args:
            ctx: Callback context with the most recent metrics.

        """


# ---------------------------------------------------------------------------
# Callback registry (typed, since Callback is the base)
# ---------------------------------------------------------------------------


# We use ``create_typed_registry`` rather than ``create_registry`` because
# ``isinstance(callback_cls, Callback)`` is not a static check and we want
# the same registry shape used by other AlphaGalerkin modules.
CallbackRegistry, _register_callback_decorator = create_typed_registry("Callback")


def register_callback(name: str) -> Any:
    """Register a callback class under ``name`` in the global registry.

    This is a thin wrapper around the registry decorator that adds an
    inheritance check.  Callback classes must inherit from
    :class:`Callback`.

    Args:
        name: Name to register under.  Must be unique within the
            registry.

    Returns:
        Decorator that registers the class.

    Raises:
        TypeError: If the decorated class is not a :class:`Callback`
            subclass.

    """

    def decorator(cls: type) -> type:
        if not issubclass(cls, Callback):
            raise TypeError(
                f"Callback {cls.__name__} must inherit from " f"src.training.callbacks.Callback"
            )
        return _register_callback_decorator(name)(cls)

    return decorator


# ---------------------------------------------------------------------------
# Pydantic spec
# ---------------------------------------------------------------------------


class CallbackSpec(BaseModel):
    """Configuration for a single callback to instantiate.

    Attributes:
        name: Registered callback name (looked up in
            :class:`CallbackRegistry`).
        params: Keyword arguments forwarded to the callback's
            constructor.  Defaults to empty dict.

    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Name of a callback registered via @register_callback.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the callback constructor.",
    )


# ---------------------------------------------------------------------------
# Built-in callback modules
# ---------------------------------------------------------------------------


# When users name a built-in callback in their config, the registry
# may be empty if the module hasn't been imported yet.  This list is
# imported lazily by ``build_callbacks_from_specs`` to ensure side-
# effect registration happens before lookup.  Adding a new built-in
# callback module is O(1): just append to this list.
BUILTIN_CALLBACK_MODULES: tuple[str, ...] = ("src.training.callbacks.lbb_monitor",)


def _ensure_builtin_callbacks_imported() -> None:
    """Import built-in callback modules to populate the registry.

    Importing a module that fails (e.g. optional dependency missing)
    is logged but does not raise — only callbacks the user actually
    requests will be looked up, and a missing one yields a clear
    KeyError from the registry.
    """
    for mod in BUILTIN_CALLBACK_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as exc:  # pragma: no cover - defensive
            logger.debug(
                "callback_module_import_failed",
                module=mod,
                error=str(exc),
            )


def build_callbacks_from_specs(specs: list[CallbackSpec]) -> list[Callback]:
    """Resolve a list of :class:`CallbackSpec` to instantiated callbacks.

    Args:
        specs: List of callback specifications.  May be empty.

    Returns:
        List of instantiated :class:`Callback` objects in the same
        order as ``specs``.  Empty list if ``specs`` is empty.

    Raises:
        KeyError: If any spec names a callback not in
            :class:`CallbackRegistry`.
        TypeError: If a spec's ``params`` are incompatible with the
            callback's constructor.

    """
    if not specs:
        return []

    _ensure_builtin_callbacks_imported()

    registry = CallbackRegistry()
    callbacks: list[Callback] = []
    for spec in specs:
        cls = registry.get_or_raise(spec.name)
        try:
            instance = cls(**spec.params)
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate callback '{spec.name}' with " f"params {spec.params}: {exc}"
            ) from exc
        if not isinstance(instance, Callback):
            raise TypeError(
                f"Registered callback '{spec.name}' produced a "
                f"non-Callback instance ({type(instance).__name__})."
            )
        callbacks.append(instance)
        logger.debug(
            "callback_instantiated",
            name=spec.name,
            cls=f"{cls.__module__}.{cls.__name__}",
        )
    return callbacks
