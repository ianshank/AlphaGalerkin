"""Tests for engine evaluator adapter.

Tests the bridge between UCI engines and the MCTS Evaluator protocol,
including score conversion, illegal move fallback, and batch evaluation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
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

    def test_evaluate_illegal_engine_move_fallback(
        self, chess_game: ChessGame, uci_config: UCIConfig
    ) -> None:
        """When engine returns an illegal move, adapter should fallback to uniform."""
        engine = MagicMock()
        # Return a move that won't be in the legal actions we'll pass
        engine.go.return_value = ("a1a8", {"depth": 10, "score_cp": 100})

        adapter = EngineEvaluator(engine, chess_game, uci_config)
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()
        # Use a restricted set of legal actions that doesn't include the engine's move
        legal = [0, 1, 2]  # Arbitrary small set

        result = adapter.evaluate(tensor, legal)

        # Should be uniform over legal actions
        for action in legal:
            assert result.policy[action] == pytest.approx(1.0 / len(legal), abs=0.001)

    def test_evaluate_empty_legal_actions(
        self, adapter: EngineEvaluator, chess_game: ChessGame
    ) -> None:
        """Evaluate with empty legal actions should not crash."""
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()

        result = adapter.evaluate(tensor, [])
        # All zeros policy, value from engine score
        assert np.sum(result.policy) == pytest.approx(0.0)

    def test_set_state_updates_state(self, adapter: EngineEvaluator, chess_game: ChessGame) -> None:
        """Calling set_state multiple times should update the state."""
        state1 = chess_game.initial_state()
        adapter.set_state(state1)
        assert adapter._current_state is state1

        # Apply a move to get a different state
        legal = chess_game.get_legal_actions(state1)
        state2 = chess_game.apply_action(state1, legal[0])
        adapter.set_state(state2)
        assert adapter._current_state is state2

    def test_go_kwargs_from_config_depth(self, chess_game: ChessGame) -> None:
        """Config depth_limit should be passed as depth kwarg to engine.go()."""
        engine = MagicMock()
        engine.go.return_value = ("e2e4", {"depth": 15, "score_cp": 0})

        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/engine"),
            depth_limit=15,
        )
        adapter = EngineEvaluator(engine, chess_game, config)
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)
        adapter.evaluate(tensor, legal)

        engine.go.assert_called_once_with(depth=15)

    def test_go_kwargs_from_config_nodes(self, chess_game: ChessGame) -> None:
        """Config nodes_limit should be passed as nodes kwarg."""
        engine = MagicMock()
        engine.go.return_value = ("e2e4", {"depth": 10, "score_cp": 0})

        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/engine"),
            nodes_limit=100000,
        )
        adapter = EngineEvaluator(engine, chess_game, config)
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)
        adapter.evaluate(tensor, legal)

        engine.go.assert_called_once_with(nodes=100000)

    def test_go_kwargs_from_config_movetime(self, chess_game: ChessGame) -> None:
        """Config movetime_ms should be passed as movetime kwarg."""
        engine = MagicMock()
        engine.go.return_value = ("e2e4", {"depth": 10, "score_cp": 0})

        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/engine"),
            movetime_ms=1000,
        )
        adapter = EngineEvaluator(engine, chess_game, config)
        state = chess_game.initial_state()
        adapter.set_state(state)

        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)
        adapter.evaluate(tensor, legal)

        engine.go.assert_called_once_with(movetime=1000)

    def test_custom_cp_scale(self, chess_game: ChessGame) -> None:
        """Custom cp_scale should affect value conversion."""
        engine = MagicMock()
        engine.go.return_value = ("e2e4", {"score_cp": 300})

        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/engine"),
            depth_limit=10,
        )

        # Default scale (300) → tanh(300/300) = tanh(1) ≈ 0.76
        adapter_default = EngineEvaluator(engine, chess_game, config)
        state = chess_game.initial_state()
        adapter_default.set_state(state)
        tensor = chess_game.to_tensor(state).numpy()
        legal = chess_game.get_legal_actions(state)
        result_default = adapter_default.evaluate(tensor, legal)

        # Custom scale (600) → tanh(300/600) = tanh(0.5) ≈ 0.46
        adapter_custom = EngineEvaluator(engine, chess_game, config, cp_scale=600.0)
        adapter_custom.set_state(state)
        result_custom = adapter_custom.evaluate(tensor, legal)

        assert result_default.value > result_custom.value


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

    def test_symmetry(self, adapter: EngineEvaluator) -> None:
        """Score conversion should be antisymmetric."""
        info_pos: EngineInfo = {"score_cp": 200}
        info_neg: EngineInfo = {"score_cp": -200}
        val_pos = adapter._score_to_value(info_pos)
        val_neg = adapter._score_to_value(info_neg)
        assert val_pos == pytest.approx(-val_neg)
