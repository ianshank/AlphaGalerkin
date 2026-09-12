"""Shared fixtures for the security suite.

Every test here runs from a fresh temporary working directory. The suite's
subject is untrusted input (checkpoint payloads, YAML, GTP commands), and a
test that resolves a relative path against whatever ``os.getcwd()`` happens
to be is an accident-of-environment test: ``../../etc/passwd`` resolves to a
missing file from the repository root and to the real ``/etc/passwd`` from two
levels below ``/``. Pinning the CWD makes the outcome the same everywhere
(CLAUDE.md Next Steps, ``test_path_traversal_in_config``; plan R-02 / 6.6).

The tests that need the repository itself already anchor on
``Path(__file__)``, so they are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each security test from its own empty temporary directory."""
    monkeypatch.chdir(tmp_path)
