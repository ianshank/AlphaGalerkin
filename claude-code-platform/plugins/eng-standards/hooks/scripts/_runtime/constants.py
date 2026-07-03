"""Shared constants for the CCP hook runtime.

Every environment-variable name, default, and exit code used by hook
scripts lives here so no script carries inline literals. This package is
stdlib-only by contract (ADR-0002): it is vendored into each plugin and
executes on consumer machines where no third-party dependency resolution
exists.
"""

from __future__ import annotations

from typing import Final

#: Prefix for all tunable-override environment variables.
ENV_PREFIX: Final[str] = "CCP_"

#: Log level override, e.g. ``CCP_LOG_LEVEL=DEBUG``.
ENV_LOG_LEVEL: Final[str] = "CCP_LOG_LEVEL"

#: Debug switch; any truthy value forces DEBUG level regardless of
#: :data:`ENV_LOG_LEVEL`.
ENV_DEBUG: Final[str] = "CCP_DEBUG"

#: Set by Claude Code to the plugin's installation directory.
ENV_PLUGIN_ROOT: Final[str] = "CLAUDE_PLUGIN_ROOT"

DEFAULT_LOG_LEVEL: Final[str] = "INFO"

#: Values (lower-cased) treated as boolean true in env overrides.
TRUTHY_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

#: Successful / non-blocking hook exit.
EXIT_OK: Final[int] = 0

#: Blocking hook exit (Claude Code treats exit 2 as a gate).
EXIT_BLOCK: Final[int] = 2

#: Plugin-relative path of the tunables defaults file.
DEFAULT_TUNABLES_RELPATH: Final[str] = "config/defaults.json"

#: Highest tunables-file schema version this runtime understands.
TUNABLES_SCHEMA_VERSION: Final[int] = 1

#: Version of the hook runtime itself (bump on any behavioural change;
#: re-vendor with ``python -m tools.sync_runtime`` afterwards).
RUNTIME_VERSION: Final[str] = "0.1.0"
