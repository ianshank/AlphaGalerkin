"""Tests for engine evaluator adapter.

Tests the bridge between UCI engines and the MCTS Evaluator protocol.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.engines.adapter import MATE_VALUE, EngineEvaluator
from src.engines.config import UCIConfig
from src.engines.protocol import EngineInfo
from src.games.chess import ChessGame


@pytest.fixture
def chess_game() -> ChessGame:
    return ChessGame()


@pytest.fixture
def uci_config() -> UCIConfig:
    return UCIConfig(
        name="test",
        engine_path=Path("/fake/engine"),
        depth_limit=10,
    )


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock BaseEngine."""
    engine = MagicMock()
    engine.go.return_value = ("e2e4", {"depth": 10, "score_cp": 50})
    return engine


@pytest.fixture
def adapter(
    mock_engine: MagicMock,
    chess_game: ChessGame,
    uci_config: UCIConfig,
) -> EngineEvaluator:
    return EngineEvaluator(
        engine=mock_engine,
        game=chess_game,
        config=uci_config,
    )


class TestEngineEvaluator:
    """Tests for EngineEvaluator."""

    def test_evaluate_with_state(self, adapter: EngineEvaluator, chess_game: ChessGame) -> None:
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)

        result = adapter.evaluate(tensor, legal)

        assert result.policy.shape[0] == chess_game.action_space_size
        assert sum(result.policy) == pytest.approx(1.0, abs=0.01)
        assert -1.0 <= result.value <= 1.0

    def test_evaluate_without_state_uniform(
        self, adapter: EngineEvaluator, chess_game: ChessGame
    ) -> None:
        state = chess_game.initial_state()
        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)

        # Don't call set_state — should fallback to uniform
        result = adapter.evaluate(tensor, legal)

        # Uniform policy over legal moves
        for action in legal:
            assert result.policy[action] == pytest.approx(1.0 / len(legal), abs=0.001)
        assert result.value == 0.0

    def test_evaluate_batch(self, adapter: EngineEvaluator, chess_game: ChessGame) -> None:
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)

        results = adapter.evaluate_batch([tensor, tensor], [legal, legal])
        assert len(results) == 2


class TestScoreToValue:
    """Tests for centipawn-to-value conversion."""

    @pytest.fixture
    def adapter(
        self,
        chess_game: ChessGame,
        uci_config: UCIConfig,
    ) -> EngineEvaluator:
        engine = MagicMock()
        return EngineEvaluator(engine, chess_game, uci_config)

    def test_zero_cp_is_neutral(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_cp": 0}
        assert adapter._score_to_value(info) == pytest.approx(0.0)

    def test_positive_cp_positive_value(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_cp": 100}
        value = adapter._score_to_value(info)
        assert value > 0
        assert value < 1.0

    def test_negative_cp_negative_value(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_cp": -100}
        value = adapter._score_to_value(info)
        assert value < 0
        assert value > -1.0

    def test_large_cp_near_one(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_cp": 1000}
        value = adapter._score_to_value(info)
        assert value > 0.95

    def test_mate_winning(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_mate": 3}
        assert adapter._score_to_value(info) == pytest.approx(MATE_VALUE)

    def test_mate_losing(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_mate": -2}
        assert adapter._score_to_value(info) == pytest.approx(-MATE_VALUE)

    def test_no_score_neutral(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"depth": 10}
        assert adapter._score_to_value(info) == 0.0

    def test_mate_takes_priority_over_cp(self, adapter: EngineEvaluator) -> None:
        info: EngineInfo = {"score_mate": 1, "score_cp": -500}
        assert adapter._score_to_value(info) == pytest.approx(MATE_VALUE)
