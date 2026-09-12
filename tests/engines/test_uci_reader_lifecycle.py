"""Regression guards for the ``UCIEngine`` stdout reader's memory and lifetime.

Root cause of the leak that killed the CI runner (CLAUDE.md Next Steps, "The
E2E tier cannot run as one process"): ``UCIEngine.start()`` spawned a daemon
thread running ``for line in process.stdout: queue.put(line)`` into an
**unbounded** ``queue.Queue`` with no stop signal, and the thread's frame held
``self``. Against a real engine the pipe blocks, so the producer is throttled
by the engine's own output rate. Against the E2E mock -- whose ``stdout`` is
an infinite generator that never blocks -- the thread spun at CPU speed
forever, every ``str`` it produced was retained by the queue, and the engine
object (hence the queue) could never be collected because the running frame
referenced it. Each test that built an engine added another spinning thread,
which is why RSS climbed *during tests that allocate nothing* and accelerated
through the file. Retaining chain::

    threading._active -> Thread -> frame(_read_stdout) -> self (UCIEngine)
        -> self._stdout_queue -> deque -> str x millions

Three properties now hold and are each pinned here against a mock with the
E2E generator's shape (``RunawayProcess``):

1. the queue is bounded, so a runaway producer applies backpressure instead
   of growing the process (``test_reader_applies_backpressure_...``);
2. ``quit()`` stops the reader thread (``test_quit_stops_the_reader_thread``);
3. an engine that is simply dropped -- the E2E tests never call ``quit()`` --
   is garbage-collectable and its reader exits on its own
   (``test_dropped_engine_is_collectable_and_its_reader_exits``).

Mutation evidence: reverting ``src/engines/uci.py`` alone (keeping the config
fields) turns 1, 2 and 3 red; they were verified red before the fix landed.
"""

from __future__ import annotations

import gc
import io
import threading
import time
import weakref
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import src.engines.uci as uci_module
from src.constants import (
    DEFAULT_UCI_READER_JOIN_TIMEOUT_SECONDS,
    DEFAULT_UCI_READER_POLL_SECONDS,
    DEFAULT_UCI_STDOUT_QUEUE_MAXSIZE,
)
from src.engines.config import UCIConfig
from src.engines.uci import UCIEngine

#: Small bound so the backpressure path is reached in milliseconds.
SMALL_QUEUE_MAXSIZE: int = 64

#: Poll interval for the reader in these tests; short so stop is prompt.
FAST_POLL_SECONDS: float = 0.01

#: Wall-clock ceiling on every wait loop below (generous; the fixed code
#: satisfies each condition in well under a second).
WAIT_DEADLINE_SECONDS: float = 5.0

#: Sleep between polls of a wait loop.
WAIT_STEP_SECONDS: float = 0.005

#: Lines the UCI handshake consumes before ``start()`` returns
#: (``id name``, ``id author``, ``uciok``, ``readyok``).
HANDSHAKE_LINES: int = 4

#: A producer may hold one line it has read but not yet enqueued.
IN_FLIGHT_LINES: int = 1


class RunawayProcess:
    """Mock ``Popen`` whose stdout never blocks and never ends.

    This is the shape of ``tests/e2e/test_chess_engine_e2e.py::MultiMoveEngine``:
    a handshake followed by an infinite ``info`` / ``bestmove`` / ``readyok``
    cycle. ``lines_yielded`` counts how far the reader has pulled the
    generator, which is the memory the old unbounded queue would have kept.
    """

    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stderr = io.StringIO()
        self.returncode: int | None = None
        self.lines_yielded = 0
        self.stdout: Iterator[str] = self._lines()

    def _lines(self) -> Iterator[str]:
        for line in ("id name Runaway\n", "id author Test\n", "uciok\n", "readyok\n"):
            self.lines_yielded += 1
            yield line
        while True:
            for line in ("info depth 1 score cp 0 nodes 1\n", "bestmove e2e4\n", "readyok\n"):
                self.lines_yielded += 1
                yield line

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9

    def terminate(self) -> None:
        self.returncode = -15


def _config(**overrides: object) -> UCIConfig:
    base: dict[str, object] = {
        "name": "runaway",
        "engine_path": Path("/fake/engine"),
        "depth_limit": 1,
        "reader_poll_seconds": FAST_POLL_SECONDS,
    }
    base.update(overrides)
    return UCIConfig(**base)  # type: ignore[arg-type]


def _start(config: UCIConfig) -> tuple[UCIEngine, RunawayProcess]:
    proc = RunawayProcess()
    with patch("subprocess.Popen", return_value=proc):
        engine = UCIEngine(config)
        engine.start()
    return engine, proc


def _wait_until(predicate: object, *, deadline: float = WAIT_DEADLINE_SECONDS) -> bool:
    """Poll *predicate* (a zero-arg callable) until true or *deadline* elapses."""
    assert callable(predicate)
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(WAIT_STEP_SECONDS)
    return bool(predicate())


class TestReaderIsBounded:
    """Property 1: a runaway producer is throttled, not buffered without bound."""

    def test_reader_applies_backpressure_instead_of_buffering_without_bound(self) -> None:
        engine, proc = _start(_config(stdout_queue_maxsize=SMALL_QUEUE_MAXSIZE))
        try:
            queue = engine._stdout_queue
            # Either the bound engages (fixed) or the producer blows past it
            # (pre-fix: the queue was unbounded and grew at CPU speed).
            _wait_until(
                lambda: (
                    queue.qsize() >= SMALL_QUEUE_MAXSIZE
                    or proc.lines_yielded > HANDSHAKE_LINES + 4 * SMALL_QUEUE_MAXSIZE
                )
            )
            # Let the producer keep running for a few poll intervals: if the
            # bound is real, nothing below moves.
            time.sleep(FAST_POLL_SECONDS * 10)

            assert queue.qsize() <= SMALL_QUEUE_MAXSIZE, (
                f"queue holds {queue.qsize()} lines, bound is {SMALL_QUEUE_MAXSIZE}"
            )
            assert proc.lines_yielded <= HANDSHAKE_LINES + SMALL_QUEUE_MAXSIZE + IN_FLIGHT_LINES, (
                f"producer pulled {proc.lines_yielded} lines; the reader is not applying "
                "backpressure"
            )
        finally:
            engine.quit()

    def test_bounded_queue_preserves_command_response_ordering(self) -> None:
        """Backpressure must not change the protocol: three searches, three answers."""
        # A bound smaller than one search response (3 lines) forces the reader
        # to block *inside* a response, which is the hardest case.
        engine, _ = _start(_config(stdout_queue_maxsize=2))
        try:
            for _ in range(3):
                engine.set_position("startpos")
                move, info = engine.go(depth=1)
                assert move == "e2e4"
                assert info["score_cp"] == 0
        finally:
            engine.quit()

    def test_backpressure_is_logged_once_per_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = MagicMock()
        monkeypatch.setattr(uci_module, "logger", spy)
        engine, _ = _start(_config(stdout_queue_maxsize=SMALL_QUEUE_MAXSIZE))
        try:

            def backpressure_calls() -> list[object]:
                return [
                    call
                    for call in spy.warning.call_args_list
                    if call.args[0] == "engine_stdout_backpressure"
                ]

            assert _wait_until(lambda: len(backpressure_calls()) >= 1)
            # Keep the producer blocked through many more poll timeouts.
            time.sleep(FAST_POLL_SECONDS * 20)
            calls = backpressure_calls()
            assert len(calls) == 1, f"expected one backpressure warning, got {len(calls)}"
            assert calls[0].kwargs == {"engine": "runaway", "maxsize": SMALL_QUEUE_MAXSIZE}
        finally:
            engine.quit()


class TestReaderLifetime:
    """Properties 2 and 3: the reader stops, and the engine is collectable."""

    def test_quit_stops_the_reader_thread(self) -> None:
        engine, _ = _start(_config())
        thread = engine._reader_thread
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()

        engine.quit()

        thread.join(timeout=WAIT_DEADLINE_SECONDS)
        assert not thread.is_alive(), "reader thread survived quit()"
        assert engine._reader_thread is None

    def test_dropped_engine_is_collectable_and_its_reader_exits(self) -> None:
        """The E2E pattern: build an engine, never call quit(), let it go out of scope."""
        engine, proc = _start(_config())
        thread = engine._reader_thread
        assert isinstance(thread, threading.Thread)
        ref = weakref.ref(engine)

        del engine
        gc.collect()

        assert ref() is None, "reader thread is keeping the engine (and its queue) alive"
        thread.join(timeout=WAIT_DEADLINE_SECONDS)
        assert not thread.is_alive(), "reader thread outlived the engine it served"
        # __del__ also released the (mock) process.
        assert proc.returncode is not None

    def test_kill_process_stops_the_reader(self) -> None:
        engine, _ = _start(_config())
        thread = engine._reader_thread
        assert isinstance(thread, threading.Thread)

        engine._kill_process()

        thread.join(timeout=WAIT_DEADLINE_SECONDS)
        assert not thread.is_alive()

    def test_restart_after_quit_gets_a_fresh_queue_and_reader(self) -> None:
        engine, _ = _start(_config())
        first_queue = engine._stdout_queue
        first_thread = engine._reader_thread
        engine.quit()

        with patch("subprocess.Popen", return_value=RunawayProcess()):
            engine.start()
        try:
            assert engine._stdout_queue is not first_queue
            assert engine._reader_thread is not first_thread
            move, _ = engine.go(depth=1)
            assert move == "e2e4"
        finally:
            engine.quit()


class TestConfigSurface:
    """Every knob is a typed, validated field with the constant as its default."""

    def test_defaults_come_from_constants(self) -> None:
        config = UCIConfig(name="c", engine_path=Path("/fake"), depth_limit=1)
        assert config.stdout_queue_maxsize == DEFAULT_UCI_STDOUT_QUEUE_MAXSIZE
        assert config.reader_poll_seconds == DEFAULT_UCI_READER_POLL_SECONDS
        assert config.reader_join_timeout_seconds == DEFAULT_UCI_READER_JOIN_TIMEOUT_SECONDS

    def test_default_bound_is_finite(self) -> None:
        assert DEFAULT_UCI_STDOUT_QUEUE_MAXSIZE > 0

    def test_zero_is_the_documented_unbounded_opt_out(self) -> None:
        config = _config(stdout_queue_maxsize=0)
        engine = UCIEngine(config)
        assert engine._stdout_queue.maxsize == 0

    @pytest.mark.parametrize(
        "field, value",
        [
            ("stdout_queue_maxsize", -1),
            ("reader_poll_seconds", 0.0),
            ("reader_join_timeout_seconds", 0.0),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            _config(**{field: value})

    def test_old_config_dicts_still_parse(self) -> None:
        """Additive fields: a pre-existing config without them is unchanged."""
        config = UCIConfig.model_validate(
            {"name": "legacy", "engine_path": "/fake", "depth_limit": 3}
        )
        assert config.stdout_queue_maxsize == DEFAULT_UCI_STDOUT_QUEUE_MAXSIZE
