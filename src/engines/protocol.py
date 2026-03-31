"""Abstract engine protocol and shared types.

Defines the interface for communicating with external chess engines,
independent of the specific protocol (UCI, CECP, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypedDict


class EngineProtocol(str, Enum):
    """Supported engine communication protocols."""

    UCI = "uci"
    CECP = "cecp"


class EngineInfo(TypedDict, total=False):
    """Information returned by an engine during/after search.

    All fields are optional since engines may not report all info.
    """

    depth: int
    seldepth: int
    score_cp: int
    score_mate: int
    pv: list[str]
    nodes: int
    nps: int
    time_ms: int
    multipv: int
    hashfull: int


class EngineError(Exception):
    """Base exception for engine-related errors."""


class EngineStartupError(EngineError):
    """Engine failed to start or complete handshake."""


class EngineTimeoutError(EngineError):
    """Engine did not respond within the configured timeout."""


class EngineCrashError(EngineError):
    """Engine process terminated unexpectedly."""


class BaseEngine(ABC):
    """Abstract interface for external chess engines.

    Implementations must handle subprocess lifecycle management
    and protocol-specific communication. Use as a context manager
    to ensure proper resource cleanup.

    Example:
        with UCIEngine(config) as engine:
            engine.set_position(fen)
            best_move, info = engine.go(depth=20)

    """

    @abstractmethod
    def start(self) -> None:
        """Start the engine process and complete protocol handshake.

        Raises:
            EngineStartupError: If the engine fails to start.

        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the current search (if running).

        This does not terminate the engine process.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the engine is ready to accept commands.

        Returns:
            True if the engine is responsive.

        """

    @abstractmethod
    def set_position(
        self,
        fen: str,
        moves: list[str] | None = None,
    ) -> None:
        """Set the board position for the next search.

        Args:
            fen: FEN string describing the position.
            moves: Optional list of moves from the FEN position
                in UCI notation (e.g., ["e2e4", "e7e5"]).

        """

    @abstractmethod
    def go(self, **kwargs: Any) -> tuple[str, EngineInfo]:
        """Start searching and return the best move.

        Keyword args are protocol-specific (e.g., depth, nodes, movetime).

        Args:
            **kwargs: Search parameters (depth, nodes, movetime, etc.).

        Returns:
            Tuple of (best_move_uci, search_info).

        Raises:
            EngineTimeoutError: If search exceeds timeout.
            EngineCrashError: If engine process dies during search.

        """

    @abstractmethod
    def quit(self) -> None:
        """Terminate the engine process gracefully.

        Sends quit command and waits for process to exit.
        Forces termination if the process does not exit cleanly.
        """

    def new_game(self) -> None:
        """Signal the start of a new game.

        Optional: engines may use this to clear internal state.
        Default implementation does nothing.
        """

    def __enter__(self) -> BaseEngine:
        """Start the engine on context entry."""
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Quit the engine on context exit."""
        self.quit()
