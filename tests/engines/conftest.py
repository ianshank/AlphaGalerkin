"""Shared fixtures for engine tests.

Provides mock UCI engine subprocess, mock engine instances,
and chess game fixtures. No real engine binary is required
for unit tests.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from src.engines.config import EloConfig, MatchConfig, UCIConfig
from src.engines.uci import UCIEngine
from src.games.chess import ChessGame


@pytest.fixture
def chess_game() -> ChessGame:
    """Create a ChessGame instance."""
    return ChessGame()


@pytest.fixture
def uci_config() -> UCIConfig:
    """Create a UCI config with depth limit for testing."""
    return UCIConfig(
        name="test_engine",
        engine_path=Path("/fake/stockfish"),
        depth_limit=10,
        hash_mb=16,
        threads=1,
    )


@pytest.fixture
def elo_config() -> EloConfig:
    """Create an Elo config for testing."""
    return EloConfig(name="test_elo")


@pytest.fixture
def match_config() -> MatchConfig:
    """Create a match config for testing."""
    return MatchConfig(
        name="test_match",
        n_games=4,
        max_moves=50,
    )


class MockUCIProcess:
    """Simulates a UCI engine subprocess.

    Responds to UCI commands with configurable responses.
    """

    def __init__(
        self,
        bestmove: str = "e2e4",
        score_cp: int = 35,
        engine_name: str = "MockFish",
    ) -> None:
        self.bestmove = bestmove
        self.score_cp = score_cp
        self.engine_name = engine_name
        self.stdin = io.StringIO()
        self.stderr = io.StringIO()
        self.returncode: int | None = None
        self._killed = False

        # Pre-build response lines
        self._responses = self._build_responses()
        self.stdout = iter(self._responses)

    def _build_responses(self) -> list[str]:
        """Build the full sequence of UCI responses."""
        return [
            f"id name {self.engine_name}\n",
            "id author Test\n",
            "uciok\n",
            "readyok\n",
            # After ucinewgame + isready
            "readyok\n",
            # Search output
            f"info depth 10 score cp {self.score_cp} nodes 12345 nps 1000000\n",
            f"bestmove {self.bestmove}\n",
            # Extra readyok for subsequent isready calls
            "readyok\n",
            "readyok\n",
            "readyok\n",
        ]

    def poll(self) -> int | None:
        """Check if process is running."""
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process to exit."""
        self.returncode = 0
        return 0

    def kill(self) -> None:
        """Kill the process."""
        self._killed = True
        self.returncode = -9

    def terminate(self) -> None:
        """Terminate the process."""
        self.returncode = -15


@pytest.fixture
def mock_uci_process() -> MockUCIProcess:
    """Create a mock UCI engine process."""
    return MockUCIProcess()


@pytest.fixture
def mock_uci_engine(
    uci_config: UCIConfig,
    mock_uci_process: MockUCIProcess,
) -> UCIEngine:
    """Create a UCIEngine with mocked subprocess.

    The engine is started and ready for use.
    """
    with patch("subprocess.Popen", return_value=mock_uci_process):
        engine = UCIEngine(uci_config)
        engine.start()
        return engine
