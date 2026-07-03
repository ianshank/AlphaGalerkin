"""Fail-safe execution wrapper for hook scripts (stdlib-only).

Hooks run inside every consumer session; an uncaught crash must never
block work. :func:`run_failsafe` logs any unexpected exception as a
structured event and exits 0 (warn-only) unless the hook is explicitly
gating, in which case an unexpected failure fails CLOSED with exit 2 —
a gate that crashes must not silently allow what it was asked to block.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from . import constants
from .jsonlog import configure_logging

#: A hook body: receives a configured logger, returns its exit code.
HookMain = Callable[[logging.Logger], int]


def run_failsafe(main: HookMain, *, component: str, gating: bool = False) -> int:
    """Run ``main`` with crash containment; returns the process exit code."""
    logger = configure_logging(component)
    try:
        return main(logger)
    except Exception:
        logger.exception(
            "hook_failsafe_triggered",
            extra={"gating": gating, "component_name": component},
        )
        return constants.EXIT_BLOCK if gating else constants.EXIT_OK


def main_entry(main: HookMain, *, component: str, gating: bool = False) -> None:
    """Process entrypoint: run fail-safe and exit with the resulting code."""
    sys.exit(run_failsafe(main, component=component, gating=gating))
