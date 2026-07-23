"""Unit tests for the fail-safe wrapper (tools.hook_runtime.failsafe)."""

from __future__ import annotations

import json
import logging

import pytest

from tools.hook_runtime import constants
from tools.hook_runtime.failsafe import main_entry, run_failsafe


def test_returns_main_exit_code() -> None:
    assert run_failsafe(lambda _log: 7, component="t.ok") == 7


def test_exception_warn_only_exits_ok(capsys: pytest.CaptureFixture[str]) -> None:
    def boom(_log: logging.Logger) -> int:
        raise RuntimeError("unexpected")

    assert run_failsafe(boom, component="t.warn") == constants.EXIT_OK
    stderr = capsys.readouterr().err
    document = json.loads(stderr.splitlines()[-1])
    assert document["event"] == "hook_failsafe_triggered"
    assert document["gating"] is False
    assert "RuntimeError" in document["exception"]


def test_exception_gating_fails_closed(capsys: pytest.CaptureFixture[str]) -> None:
    def boom(_log: logging.Logger) -> int:
        raise RuntimeError("unexpected")

    assert run_failsafe(boom, component="t.gate", gating=True) == constants.EXIT_BLOCK
    document = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert document["gating"] is True


def test_main_receives_configured_logger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def body(log: logging.Logger) -> int:
        log.info("hello", extra={"k": 1})
        return constants.EXIT_OK

    run_failsafe(body, component="t.logger")
    document = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert document == {**document, "event": "hello", "k": 1, "component": "t.logger"}


def test_main_entry_exits_with_code() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main_entry(lambda _log: 3, component="t.exit")
    assert excinfo.value.code == 3
