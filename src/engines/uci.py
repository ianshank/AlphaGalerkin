"""UCI (Universal Chess Interface) protocol implementation.

Communicates with UCI-compatible chess engines (e.g., Stockfish, Leela)
via subprocess stdin/stdout. Handles engine lifecycle, command formatting,
response parsing, and error recovery.

Protocol reference: https://www.wbec-ridderkerk.nl/html/UCIProtocol.html
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from queue import Empty, Full, Queue
from typing import IO, Any

import structlog

from src.engines.config import UCIConfig
from src.engines.protocol import (
    BaseEngine,
    EngineCrashError,
    EngineInfo,
    EngineStartupError,
    EngineTimeoutError,
)

logger = structlog.get_logger(__name__)


def _pump_stdout(
    stream: IO[str],
    sink: Queue[str],
    stop_event: threading.Event,
    poll_seconds: float,
    engine_label: str,
) -> None:
    """Copy *stream* lines into *sink* until EOF or *stop_event* is set.

    Runs on the daemon reader thread. Deliberately a module-level function
    that receives only what it needs: it must **not** hold a reference to the
    :class:`UCIEngine`, or the engine can never be garbage-collected while
    the thread lives -- and the thread lives as long as the stream does.
    That reference cycle (thread frame -> engine -> queue -> lines) is what
    kept every mock engine in the E2E chess tier alive and allocating.

    *sink* is bounded (see ``UCIConfig.stdout_queue_maxsize``); when full the
    put blocks for *poll_seconds* and re-checks *stop_event*, so a producer
    that outruns the consumer applies backpressure instead of growing the
    process, and ``quit()`` can end the thread even while it is blocked.

    Args:
        stream: The engine's stdout (text mode).
        sink: Queue drained by ``UCIEngine._read_until``.
        stop_event: Set by ``quit()`` / ``_kill_process()`` / ``__del__``.
        poll_seconds: Timeout per blocked put before re-checking the flag.
        engine_label: Config name, for log context only.

    """
    backpressure_logged = False
    try:
        for line in stream:
            while not stop_event.is_set():
                try:
                    sink.put(line, timeout=poll_seconds)
                    break
                except Full:
                    if not backpressure_logged:
                        backpressure_logged = True
                        logger.warning(
                            "engine_stdout_backpressure",
                            engine=engine_label,
                            maxsize=sink.maxsize,
                        )
            if stop_event.is_set():
                break
    except ValueError:
        # Stream closed underneath us.
        pass
    logger.debug("engine_reader_stopped", engine=engine_label, stopped=stop_event.is_set())


class UCIEngine(BaseEngine):
    """UCI protocol engine communicating via subprocess.

    Manages the engine process lifecycle, sends UCI commands,
    and parses responses. Thread-safe stdout reading via
    a daemon reader thread.

    Args:
        config: UCI engine configuration.

    """

    def __init__(self, config: UCIConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._stop_event = threading.Event()
        self._stdout_queue: Queue[str] = Queue(maxsize=config.stdout_queue_maxsize)
        self._reader_thread: threading.Thread | None = None
        self._started = False
        self._engine_name: str = "unknown"
        self._engine_author: str = "unknown"

    @property
    def config(self) -> UCIConfig:
        """Get engine configuration."""
        return self._config

    @property
    def engine_name(self) -> str:
        """Get the engine's reported name."""
        return self._engine_name

    def start(self) -> None:
        """Start the engine process and complete UCI handshake.

        Spawns the engine binary, sends 'uci', waits for 'uciok',
        configures options, and confirms readiness with 'isready'/'readyok'.

        Raises:
            EngineStartupError: If the engine fails to start or handshake.

        """
        if self._started:
            return

        engine_path = str(self._config.engine_path)
        timeout = self._config.startup_timeout_seconds

        try:
            self._process = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise EngineStartupError(f"Engine binary not found: {engine_path}") from e
        except PermissionError as e:
            raise EngineStartupError(f"Permission denied executing: {engine_path}") from e
        except OSError as e:
            raise EngineStartupError(f"Failed to start engine: {e}") from e

        stdout = self._process.stdout
        if stdout is None:  # pragma: no cover - Popen(stdout=PIPE) always provides one
            self._kill_process()
            raise EngineStartupError("Engine process has no stdout pipe")

        # Start stdout reader thread. Fresh queue + stop event per start so a
        # reader left over from a previous session (quit() then start()) can
        # neither feed stale lines into this one nor be stopped by mistake.
        self._stop_event = threading.Event()
        self._stdout_queue = Queue(maxsize=self._config.stdout_queue_maxsize)
        self._reader_thread = threading.Thread(
            target=_pump_stdout,
            args=(
                stdout,
                self._stdout_queue,
                self._stop_event,
                self._config.reader_poll_seconds,
                self._config.name,
            ),
            daemon=True,
            name=f"uci-reader-{self._config.name}",
        )
        self._reader_thread.start()

        # UCI handshake
        self._send("uci")

        try:
            lines = self._read_until(
                lambda line: line.strip() == "uciok",
                timeout=timeout,
            )
        except EngineTimeoutError as e:
            self._kill_process()
            raise EngineStartupError(
                f"Engine did not respond with 'uciok' within {timeout}s"
            ) from e

        # Parse engine identity
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("id name "):
                self._engine_name = stripped[len("id name ") :]
            elif stripped.startswith("id author "):
                self._engine_author = stripped[len("id author ") :]

        # Configure UCI options
        if self._config.hash_mb != 64:
            self._send(f"setoption name Hash value {self._config.hash_mb}")
        if self._config.threads != 1:
            self._send(f"setoption name Threads value {self._config.threads}")

        for opt_name, opt_value in self._config.options.items():
            if isinstance(opt_value, bool):
                val_str = "true" if opt_value else "false"
            else:
                val_str = str(opt_value)
            self._send(f"setoption name {opt_name} value {val_str}")

        # Confirm readiness
        self._send("isready")
        try:
            self._read_until(
                lambda line: line.strip() == "readyok",
                timeout=timeout,
            )
        except EngineTimeoutError as e:
            self._kill_process()
            raise EngineStartupError(
                f"Engine did not respond with 'readyok' within {timeout}s"
            ) from e

        self._started = True
        logger.info(
            "engine_started",
            engine=self._engine_name,
            author=self._engine_author,
            path=engine_path,
        )

    def stop(self) -> None:
        """Send 'stop' command to halt the current search."""
        if self._process and self._started:
            self._send("stop")

    def is_ready(self) -> bool:
        """Check engine responsiveness via isready/readyok.

        Returns:
            True if engine responds within timeout.

        """
        if not self._started or not self._process:
            return False

        if self._process.poll() is not None:
            return False

        self._send("isready")
        try:
            self._read_until(
                lambda line: line.strip() == "readyok",
                timeout=self._config.startup_timeout_seconds,
            )
            return True
        except (EngineTimeoutError, EngineCrashError):
            return False

    def set_position(
        self,
        fen: str,
        moves: list[str] | None = None,
    ) -> None:
        """Set the board position for the next search.

        Args:
            fen: FEN string describing the position.
            moves: Optional list of UCI moves from the position.

        """
        self._ensure_started()

        cmd = f"position fen {fen}"
        if moves:
            moves_str = " ".join(moves)
            cmd += f" moves {moves_str}"

        self._send(cmd)

    def go(self, **kwargs: Any) -> tuple[str, EngineInfo]:
        """Start searching and return the best move.

        Uses search limits from config unless overridden via kwargs.
        Supported kwargs: depth, nodes, movetime, wtime, btime, winc, binc.

        Args:
            **kwargs: Search parameter overrides.

        Returns:
            Tuple of (best_move_uci, engine_info).

        Raises:
            EngineTimeoutError: If search exceeds move timeout.
            EngineCrashError: If engine process dies during search.

        """
        self._ensure_started()

        # Build go command from config defaults + overrides
        parts = ["go"]

        depth = kwargs.get("depth", self._config.depth_limit)
        nodes = kwargs.get("nodes", self._config.nodes_limit)
        movetime = kwargs.get("movetime", self._config.movetime_ms)

        if depth is not None:
            parts.append(f"depth {depth}")
        if nodes is not None:
            parts.append(f"nodes {nodes}")
        if movetime is not None:
            parts.append(f"movetime {movetime}")

        # Time control parameters (passthrough)
        for param in ("wtime", "btime", "winc", "binc", "movestogo"):
            if param in kwargs:
                parts.append(f"{param} {kwargs[param]}")

        self._send(" ".join(parts))

        # Parse info lines until bestmove
        info: EngineInfo = {}
        timeout = self._config.move_timeout_seconds

        try:
            lines = self._read_until(
                lambda line: line.strip().startswith("bestmove"),
                timeout=timeout,
            )
        except EngineTimeoutError:
            self.stop()
            raise

        # Parse accumulated info from search
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("info "):
                info = self._parse_info_line(stripped, info)

        # Extract bestmove from last line
        bestmove_line = lines[-1].strip() if lines else ""
        best_move = self._parse_bestmove(bestmove_line)

        logger.debug(
            "engine_move",
            move=best_move,
            depth=info.get("depth"),
            score_cp=info.get("score_cp"),
            nodes=info.get("nodes"),
        )

        return best_move, info

    def new_game(self) -> None:
        """Signal the start of a new game."""
        if self._started:
            self._send("ucinewgame")
            # Wait for engine to be ready after clearing state
            self._send("isready")
            try:
                self._read_until(
                    lambda line: line.strip() == "readyok",
                    timeout=self._config.startup_timeout_seconds,
                )
            except EngineTimeoutError:
                logger.warning("engine_not_ready_after_newgame")

    def quit(self) -> None:
        """Terminate the engine process gracefully."""
        if self._process is None:
            return

        try:
            self._send("quit")
            # Wait for process to exit
            self._process.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError):
            self._kill_process()
        finally:
            self._started = False
            self._process = None
            reader_stopped = self._stop_reader()

        logger.debug("engine_quit", engine=self._engine_name, reader_stopped=reader_stopped)

    def _send(self, command: str) -> None:
        """Send a command to the engine via stdin.

        Args:
            command: UCI command string.

        Raises:
            EngineCrashError: If the engine process is not running.

        """
        if self._process is None or self._process.stdin is None:
            raise EngineCrashError("Engine process is not running")

        if self._process.poll() is not None:
            raise EngineCrashError(f"Engine process exited with code {self._process.returncode}")

        try:
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
            logger.debug("engine_send", command=command)
        except BrokenPipeError as e:
            raise EngineCrashError("Engine process pipe is broken") from e

    def _stop_reader(self) -> bool:
        """Signal the stdout reader to exit and wait briefly for it.

        Returns:
            True if no reader is running afterwards; False if it is still
            alive after ``reader_join_timeout_seconds`` (a reader blocked on
            a live pipe cannot be interrupted from Python and exits on its
            own once the process closes stdout).

        """
        self._stop_event.set()
        thread = self._reader_thread
        if thread is None or thread is threading.current_thread():
            return True
        if thread.is_alive():
            thread.join(timeout=self._config.reader_join_timeout_seconds)
        if thread.is_alive():
            logger.warning(
                "engine_reader_still_alive",
                engine=self._engine_name,
                join_timeout_seconds=self._config.reader_join_timeout_seconds,
            )
            return False
        self._reader_thread = None
        return True

    def _read_until(
        self,
        predicate: Callable[[str], bool],
        timeout: float,
    ) -> list[str]:
        """Read lines from stdout until predicate matches or timeout.

        Args:
            predicate: Function that returns True when done reading.
            timeout: Maximum seconds to wait.

        Returns:
            List of lines read (including the matching line).

        Raises:
            EngineTimeoutError: If timeout reached without predicate match.
            EngineCrashError: If process exits during read.

        """
        lines: list[str] = []

        while True:
            # Check if process is still alive
            if self._process and self._process.poll() is not None:
                raise EngineCrashError(
                    f"Engine process exited with code {self._process.returncode}"
                )

            try:
                line = self._stdout_queue.get(timeout=timeout)
            except Empty as e:
                raise EngineTimeoutError(f"Engine did not respond within {timeout}s") from e

            lines.append(line)
            logger.debug("engine_recv", line=line.strip())

            if predicate(line):
                return lines

    def _parse_info_line(self, line: str, info: EngineInfo) -> EngineInfo:
        """Parse a UCI 'info' line into EngineInfo dict.

        Args:
            line: Full info line (e.g., "info depth 20 score cp 35 nodes 1234").
            info: Existing info dict to update.

        Returns:
            Updated EngineInfo.

        """
        tokens = line.split()
        i = 1  # Skip "info"

        while i < len(tokens):
            token = tokens[i]

            if token == "depth" and i + 1 < len(tokens):
                info["depth"] = int(tokens[i + 1])
                i += 2
            elif token == "seldepth" and i + 1 < len(tokens):
                info["seldepth"] = int(tokens[i + 1])
                i += 2
            elif token == "score" and i + 2 < len(tokens):
                score_type = tokens[i + 1]
                if score_type == "cp":
                    info["score_cp"] = int(tokens[i + 2])
                    i += 3
                elif score_type == "mate":
                    info["score_mate"] = int(tokens[i + 2])
                    i += 3
                else:
                    i += 1
            elif token == "nodes" and i + 1 < len(tokens):
                info["nodes"] = int(tokens[i + 1])
                i += 2
            elif token == "nps" and i + 1 < len(tokens):
                info["nps"] = int(tokens[i + 1])
                i += 2
            elif token == "time" and i + 1 < len(tokens):
                info["time_ms"] = int(tokens[i + 1])
                i += 2
            elif token == "multipv" and i + 1 < len(tokens):
                info["multipv"] = int(tokens[i + 1])
                i += 2
            elif token == "hashfull" and i + 1 < len(tokens):
                info["hashfull"] = int(tokens[i + 1])
                i += 2
            elif token == "pv":
                # PV extends to end of line
                info["pv"] = tokens[i + 1 :]
                break
            else:
                i += 1

        return info

    def _parse_bestmove(self, line: str) -> str:
        """Parse the bestmove line.

        Args:
            line: Line starting with "bestmove" (e.g., "bestmove e2e4 ponder d7d5").

        Returns:
            Best move in UCI notation.

        Raises:
            EngineCrashError: If the bestmove line is malformed.

        """
        tokens = line.split()
        if len(tokens) < 2 or tokens[0] != "bestmove":
            raise EngineCrashError(f"Malformed bestmove line: {line!r}")

        return tokens[1]

    def _ensure_started(self) -> None:
        """Verify the engine is started and running.

        Raises:
            EngineStartupError: If the engine has not been started.
            EngineCrashError: If the engine process has died.

        """
        if not self._started:
            raise EngineStartupError("Engine has not been started. Call start() first.")

        if self._process and self._process.poll() is not None:
            self._started = False
            raise EngineCrashError(f"Engine process exited with code {self._process.returncode}")

    def _kill_process(self) -> None:
        """Force-kill the engine process and release its stdout reader."""
        self._stop_event.set()
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def __del__(self) -> None:
        """Ensure the engine process and its stdout reader are released.

        Only signals the reader (no join): finalizers run on whatever thread
        triggered collection and must not block. The reader exits within one
        ``reader_poll_seconds`` if blocked on a full queue, or on the next
        line / EOF otherwise.
        """
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            self._kill_process()
