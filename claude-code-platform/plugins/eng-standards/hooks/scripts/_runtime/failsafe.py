"""Fail-safe execution wrapper for hook scripts (stdlib-only).

Hooks run inside every consumer session; an uncaught crash must never
block work — unless the hook is *gating*, in which case an unexpected
failure fails CLOSED with exit 2: a gate that crashes must not silently
allow what it was asked to block.

Because gating is usually a runtime tunable (env/config), ``gating`` may
be a callable resolved *at crash time*. The resolver is itself guarded:
if it raises, gating falls back to ``False`` and the resolution failure
is logged, so a broken config cannot turn a warn-only hook into a
session blocker.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from . import constants
from .jsonlog import configure_logging

#: A hook body: receives a configured logger, returns its exit code.
HookMain = Callable[[logging.Logger], int]

#: Static flag or runtime resolver for the gating contract.
GatingSpec = bool | Callable[[], bool]


def _resolve_gating(gating: GatingSpec, logger: logging.Logger) -> bool:
    if not callable(gating):
        return gating
    try:
        return bool(gating())
    except Exception:  # fail-safe boundary: a broken resolver must not gate
        logger.exception("hook_gating_resolution_failed")
        return False


def run_failsafe(main: HookMain, *, component: str, gating: GatingSpec = False) -> int:
    """Run ``main`` with crash containment; returns the process exit code."""
    logger = configure_logging(component)
    try:
        return main(logger)
    except Exception:
        effective_gating = _resolve_gating(gating, logger)
        logger.exception(
            "hook_failsafe_triggered",
            extra={"gating": effective_gating, "component_name": component},
        )
        return constants.EXIT_BLOCK if effective_gating else constants.EXIT_OK


def main_entry(main: HookMain, *, component: str, gating: GatingSpec = False) -> None:
    """Process entrypoint: run fail-safe and exit with the resulting code."""
    sys.exit(run_failsafe(main, component=component, gating=gating))
