"""Tests for intercept game registration."""

from __future__ import annotations


class TestRegisterGames:
    def test_import_succeeds(self) -> None:
        """Importing register_games should not raise."""
        import src.intercept.register_games  # noqa: F401

    def test_no_crash_on_reimport(self) -> None:
        """Multiple imports should be safe (idempotent)."""
        import src.intercept.register_games  # noqa: F401
