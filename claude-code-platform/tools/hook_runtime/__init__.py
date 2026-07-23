"""CCP hook runtime — stdlib-only shared library for plugin hook scripts.

This package is the canonical source of truth; ``python -m
tools.sync_runtime`` vendors it verbatim into each plugin as
``hooks/scripts/_runtime/`` (ADR-0002), because installed plugins are
cached per-directory and cannot import anything outside their own root.
CI parity-gates the vendored copies against this package.

Contract (enforced by the validate stdlib-import gate):
- imports from the Python standard library only;
- intra-package imports are relative, so the package works under both
  its canonical name (``tools.hook_runtime``) and its vendored name
  (``_runtime``).
"""

from .constants import (
    EXIT_BLOCK,
    EXIT_OK,
    RUNTIME_VERSION,
)
from .failsafe import HookMain, main_entry, run_failsafe
from .jsonlog import JsonLineFormatter, configure_logging, resolve_level
from .models import HookInput, HookInputError, parse_hook_input
from .tunables import TunablesError, apply_env_overrides, load_tunables, plugin_root

__all__ = [
    "EXIT_BLOCK",
    "EXIT_OK",
    "RUNTIME_VERSION",
    "HookInput",
    "HookInputError",
    "HookMain",
    "JsonLineFormatter",
    "TunablesError",
    "apply_env_overrides",
    "configure_logging",
    "load_tunables",
    "main_entry",
    "parse_hook_input",
    "plugin_root",
    "resolve_level",
    "run_failsafe",
]
