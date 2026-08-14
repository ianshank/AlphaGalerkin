"""Shared fixtures for ``tests/poc/``.

``structlog``'s global configuration is a process-wide singleton that
``test_logging.py`` mutates without restoring: it calls ``configure_logging()``
repeatedly (ending at ``level="ERROR", json_format=True``) with no teardown.
That swaps the logger factory to ``structlog.stdlib.LoggerFactory``, routing
every subsequent ``logger.warning(...)`` in the process into stdlib logging,
where pytest's capture handler swallows it. The observable effect is that
warnings such as ``vram_probe_failed`` / ``interpolator_build_failed`` render
fine when their test file runs alone and vanish in a full run — so a ``caplog``
assertion on them would pass or fail purely by collection order.

**Deliberately NOT here: a ``ScenarioRegistry`` snapshot/restore fixture.**
The registry has the same shape of leak (several modules here clear it via local
``autouse`` fixtures and never restore), and ``test_charter_alignment.py``
documents the hazard and works around it by reading
``ScenarioRegistry().list_scenarios()`` in a subprocess. A package-level
snapshot/restore fixture was tried and **reverted — it made the end state
strictly worse**, measurably:

    # after tests/poc/test_cli_commands.py, probing the live registry
    no fixture (baseline) -> 10 real scenarios still registered
    clear + restore       -> []            (catastrophic: see below)
    restore-if-missing    -> ['alpha','beta']  (real scenarios lost)

The cause is that these modules also purge ``sys.modules['src.poc.scenarios*']``
so their ``@scenario`` decorators re-fire on re-import. A per-test snapshot taken
*before* that re-import is empty or partial, so restoring it removes
registrations the test legitimately created — and with ``sys.modules`` now
repopulated the decorators cannot fire again. Any per-test restore fights the
local fixtures rather than complementing them.

Closing this properly means reworking those local fixtures (they are not
homogeneous — two purge ``sys.modules``, two re-register for identity checks),
which is tracked as **B16** in ``docs/CODE_HYGIENE_AUDIT.md``. Until then the
subprocess workaround in the charter guard remains the correct mitigation.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def _snapshot_structlog_config() -> Iterator[None]:
    """Restore structlog's global configuration after each test.

    ``structlog.get_config()`` returns the live config dict; re-applying it
    through ``structlog.configure(**config)`` restores the exact processor
    chain, wrapper class, logger factory and cache setting. When a test never
    touches the configuration this is a no-op round-trip.
    """
    saved = structlog.get_config().copy()
    was_configured = structlog.is_configured()
    yield
    if was_configured:
        structlog.configure(**saved)
    else:
        structlog.reset_defaults()
