"""Tests for src/engines/protocol.py.

Covers the exception hierarchy, EngineProtocol enum, EngineInfo TypedDict,
and the concrete default methods on BaseEngine (new_game, __enter__, __exit__).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.engines.protocol import (
    BaseEngine,
    EngineCrashError,
    EngineError,
    EngineInfo,
    EngineProtocol,
    EngineStartupError,
    EngineTimeoutError,
)

# ---------------------------------------------------------------------------
# EngineProtocol enum
# ---------------------------------------------------------------------------


class TestEngineProtocol:
    """Tests for the EngineProtocol string enum."""

    def test_uci_value(self) -> None:
        assert EngineProtocol.UCI == "uci"
        assert EngineProtocol.UCI.value == "uci"

    def test_cecp_value(self) -> None:
        assert EngineProtocol.CECP == "cecp"
        assert EngineProtocol.CECP.value == "cecp"

    def test_from_string(self) -> None:
        assert EngineProtocol("uci") is EngineProtocol.UCI
        assert EngineProtocol("cecp") is EngineProtocol.CECP

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            EngineProtocol("invalid")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the exception class hierarchy and raisability."""

    def test_engine_error_is_exception(self) -> None:
        err = EngineError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    def test_startup_error_is_engine_error(self) -> None:
        err = EngineStartupError("failed to start")
        assert isinstance(err, EngineError)
        assert isinstance(err, Exception)

    def test_timeout_error_is_engine_error(self) -> None:
        err = EngineTimeoutError("timed out")
        assert isinstance(err, EngineError)
        assert isinstance(err, Exception)

    def test_crash_error_is_engine_error(self) -> None:
        err = EngineCrashError("process died")
        assert isinstance(err, EngineError)
        assert isinstance(err, Exception)

    def test_timeout_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(EngineTimeoutError, match="search timeout"):
            raise EngineTimeoutError("search timeout")

    def test_timeout_caught_as_engine_error(self) -> None:
        """EngineTimeoutError is catchable as the base EngineError."""
        with pytest.raises(EngineError):
            raise EngineTimeoutError("caught as base")

    def test_startup_caught_as_engine_error(self) -> None:
        with pytest.raises(EngineError):
            raise EngineStartupError("caught as base")

    def test_crash_caught_as_engine_error(self) -> None:
        with pytest.raises(EngineError):
            raise EngineCrashError("caught as base")


# ---------------------------------------------------------------------------
# EngineInfo TypedDict
# ---------------------------------------------------------------------------


class TestEngineInfo:
    """Verify EngineInfo TypedDict structure."""

    def test_empty_dict_is_valid(self) -> None:
        info: EngineInfo = {}
        assert isinstance(info, dict)

    def test_all_fields_accepted(self) -> None:
        info: EngineInfo = {
            "depth": 20,
            "seldepth": 25,
            "score_cp": 35,
            "score_mate": 0,
            "pv": ["e2e4", "e7e5"],
            "nodes": 100_000,
            "nps": 500_000,
            "time_ms": 200,
            "multipv": 1,
            "hashfull": 512,
        }
        assert info["depth"] == 20
        assert info["pv"] == ["e2e4", "e7e5"]

    def test_partial_fields_accepted(self) -> None:
        info: EngineInfo = {"depth": 10, "score_cp": -50}
        assert info.get("depth") == 10
        assert info.get("nodes") is None


# ---------------------------------------------------------------------------
# BaseEngine concrete methods (new_game, __enter__, __exit__)
# ---------------------------------------------------------------------------


class _ConcreteEngine(BaseEngine):
    """Minimal concrete subclass to test BaseEngine default methods."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def is_ready(self) -> bool:
        return self.started

    def set_position(self, fen: str, moves: list[str] | None = None) -> None:
        pass

    def go(self, **kwargs: Any) -> tuple[str, EngineInfo]:
        return ("e2e4", {})

    def quit(self) -> None:
        self.quit_called = True


class TestBaseEngineDefaults:
    """Tests for the default concrete methods on BaseEngine."""

    def test_new_game_is_noop(self) -> None:
        """new_game() has a default no-op implementation; it should not raise."""
        engine = _ConcreteEngine()
        engine.start()
        engine.new_game()  # must not raise

    def test_context_manager_calls_start(self) -> None:
        engine = _ConcreteEngine()
        assert not engine.started
        with engine:
            assert engine.started

    def test_context_manager_calls_quit_on_exit(self) -> None:
        engine = _ConcreteEngine()
        with engine:
            pass
        assert engine.quit_called

    def test_context_manager_calls_quit_on_exception(self) -> None:
        engine = _ConcreteEngine()
        with pytest.raises(ValueError):
            with engine:
                raise ValueError("test error")
        assert engine.quit_called

    def test_enter_returns_engine_instance(self) -> None:
        engine = _ConcreteEngine()
        result = engine.__enter__()
        engine.__exit__(None, None, None)
        assert result is engine
