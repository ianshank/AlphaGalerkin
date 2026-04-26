"""Trainer callback subsystem for AlphaGalerkin.

Callbacks let downstream code react to lifecycle events of any
:class:`~src.training.base_trainer.BaseTrainer` subclass without having
to subclass the trainer.  They are the canonical extension point for
monitoring (LBB stability, custom logging), structured reporting, and
side-channel data collection during training.

Typical usage::

    from src.training.callbacks import (
        Callback,
        CallbackContext,
        CallbackRegistry,
        CallbackSpec,
        build_callbacks_from_specs,
        register_callback,
    )

    @register_callback("my_logger")
    class MyLogger(Callback):
        def on_step_end(self, ctx: CallbackContext) -> None:
            print(ctx.step, ctx.metrics["loss"])

    specs = [CallbackSpec(name="my_logger", params={})]
    callbacks = build_callbacks_from_specs(specs)

The registry is populated by importing ``src.training.callbacks`` (this
module re-exports concrete callbacks via lazy imports inside
``build_callbacks_from_specs`` so the registry is always live before
specs are resolved).

This package is the *only* sanctioned place to extend the trainer with
new lifecycle hooks. All values are configuration-driven (Pydantic
``CallbackSpec``) — no hardcoded callback class names appear in user
configs.
"""

from __future__ import annotations

from src.training.callbacks.base import (
    BUILTIN_CALLBACK_MODULES,
    Callback,
    CallbackContext,
    CallbackRegistry,
    CallbackSpec,
    build_callbacks_from_specs,
    register_callback,
)

__all__ = [
    "BUILTIN_CALLBACK_MODULES",
    "Callback",
    "CallbackContext",
    "CallbackRegistry",
    "CallbackSpec",
    "build_callbacks_from_specs",
    "register_callback",
]
