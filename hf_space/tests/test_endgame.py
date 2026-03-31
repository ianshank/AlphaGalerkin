"""Tests for endgame detection module.

Verifies that EndgameDetector correctly identifies when the AI should pass
to properly end the game when human passes in endgame situations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.board import EndgameConfig
from src.endgame import EndgameAnalysis, EndgameDetector

if TYPE_CHECKING:
    from src.tools.gtp import SimpleGoGame


class MockSimpleGoGame:
    """Mock game for testing without heavy dependencies."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, board_size: int = 9) -> None:
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.passes = 0


class TestEndgameAnalysis:
    """Tests for EndgameAnalysis dataclass."""

    def test_analysis_is_frozen(self) -> None:
        """Verify analysis is immutable."""
        analysis = EndgameAnalysis(
            fill_ratio=0.5,
            empty_count=40,
            total_positions=81,
            human_just_passed=False,
            current_passes=0,
            should_ai_pass=False,
            reason="game_continues",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            analysis.fill_ratio = 0.9  # type: ignore[misc]

    def test_analysis_fields(self) -> None:
        """Verify all fields are accessible."""
        analysis = EndgameAnalysis(
            fill_ratio=0.95,
            empty_count=4,
            total_positions=81,
            human_just_passed=True,
            current_passes=1,
            should_ai_pass=True,
            reason="consecutive_pass_detection",
        )
        assert analysis.fill_ratio == 0.95
        assert analysis.empty_count == 4
        assert analysis.total_positions == 81
        assert analysis.human_just_passed is True
        assert analysis.current_passes == 1
        assert analysis.should_ai_pass is True
        assert analysis.reason == "consecutive_pass_detection"


class TestEndgameDetectorAnalyze:
    """Tests for EndgameDetector.analyze() method."""

    def test_empty_board_no_pass(self) -> None:
        """Empty board should not trigger pass."""
        config = EndgameConfig()
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)

        analysis = detector.analyze(game, human_just_passed=True)

        assert analysis.should_ai_pass is False
        assert analysis.fill_ratio == 0.0
        assert analysis.empty_count == 81
        assert analysis.reason == "game_continues"

    def test_full_board_human_pass_triggers_ai_pass(self) -> None:
        """Board >90% full + human passed → AI should pass."""
        config = EndgameConfig(fill_threshold=0.90)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        # Fill 95% of board
        filled = int(81 * 0.95)
        game.board.flat[:filled] = 1

        analysis = detector.analyze(game, human_just_passed=True)

        assert analysis.should_ai_pass is True
        assert analysis.fill_ratio >= 0.90
        assert analysis.reason == "board_fill_threshold_reached"

    def test_partial_board_no_pass(self) -> None:
        """60% filled board should not trigger pass."""
        config = EndgameConfig(fill_threshold=0.90)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        # Fill 60% of board
        filled = int(81 * 0.60)
        game.board.flat[:filled] = 1

        analysis = detector.analyze(game, human_just_passed=True)

        assert analysis.should_ai_pass is False
        assert 0.55 < analysis.fill_ratio < 0.65
        assert analysis.reason == "game_continues"

    def test_consecutive_pass_detection(self) -> None:
        """If game.passes >= 1 and human passes → AI pass."""
        config = EndgameConfig(pass_on_consecutive=True)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        game.passes = 1  # Previous pass already made

        analysis = detector.analyze(game, human_just_passed=True)

        assert analysis.should_ai_pass is True
        assert analysis.reason == "consecutive_pass_detection"

    def test_consecutive_pass_disabled(self) -> None:
        """Consecutive pass detection can be disabled."""
        config = EndgameConfig(pass_on_consecutive=False, fill_threshold=0.99)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        game.passes = 1

        analysis = detector.analyze(game, human_just_passed=True)

        # Should not trigger because fill threshold not met
        assert analysis.should_ai_pass is False
        assert analysis.reason == "game_continues"

    def test_auto_pass_disabled(self) -> None:
        """When enable_auto_pass=False, never trigger pass."""
        config = EndgameConfig(enable_auto_pass=False)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        game.board.fill(1)  # Completely full
        game.passes = 1

        analysis = detector.analyze(game, human_just_passed=True)

        assert analysis.should_ai_pass is False
        assert analysis.reason == "auto_pass_disabled"

    def test_human_did_not_pass(self) -> None:
        """If human played a move (not pass), don't auto-pass."""
        config = EndgameConfig()
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        game.board.fill(1)  # Completely full

        analysis = detector.analyze(game, human_just_passed=False)

        assert analysis.should_ai_pass is False
        assert analysis.reason == "human_did_not_pass"

    def test_configurable_threshold(self) -> None:
        """Custom threshold (e.g., 0.95) works correctly."""
        config = EndgameConfig(fill_threshold=0.95)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)

        # 92% fill - should NOT trigger
        game.board.fill(0)
        filled_92 = int(81 * 0.92)
        game.board.flat[:filled_92] = 1
        analysis_92 = detector.analyze(game, human_just_passed=True)
        assert analysis_92.should_ai_pass is False

        # 96% fill - should trigger
        game.board.fill(0)
        filled_96 = int(81 * 0.96)
        game.board.flat[:filled_96] = 1
        analysis_96 = detector.analyze(game, human_just_passed=True)
        assert analysis_96.should_ai_pass is True


class TestEndgameDetectorOverride:
    """Tests for should_override_to_pass() method."""

    def test_mcts_already_pass_no_override(self) -> None:
        """If MCTS already chose pass, no override needed."""
        config = EndgameConfig()
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        game.board.fill(1)  # Full board

        pass_action = 81  # 9*9
        result = detector.should_override_to_pass(
            game, pass_action, human_just_passed=True
        )

        assert result is False

    def test_override_mcts_to_pass(self) -> None:
        """Override MCTS move action to pass in endgame."""
        config = EndgameConfig(fill_threshold=0.90)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        # 95% fill
        filled = int(81 * 0.95)
        game.board.flat[:filled] = 1

        move_action = 40  # Some board position
        result = detector.should_override_to_pass(
            game, move_action, human_just_passed=True
        )

        assert result is True

    def test_no_override_if_conditions_not_met(self) -> None:
        """Don't override if endgame conditions not met."""
        config = EndgameConfig(fill_threshold=0.90)
        detector = EndgameDetector(config)
        game = MockSimpleGoGame(9)
        # 50% fill
        filled = int(81 * 0.50)
        game.board.flat[:filled] = 1

        move_action = 40
        result = detector.should_override_to_pass(
            game, move_action, human_just_passed=True
        )

        assert result is False


class TestEndgameDetectorHelpers:
    """Tests for helper methods."""

    def test_get_pass_action_9x9(self) -> None:
        """Pass action for 9x9 is 81."""
        detector = EndgameDetector(EndgameConfig())
        assert detector.get_pass_action(9) == 81

    def test_get_pass_action_13x13(self) -> None:
        """Pass action for 13x13 is 169."""
        detector = EndgameDetector(EndgameConfig())
        assert detector.get_pass_action(13) == 169

    def test_get_pass_action_19x19(self) -> None:
        """Pass action for 19x19 is 361."""
        detector = EndgameDetector(EndgameConfig())
        assert detector.get_pass_action(19) == 361


class TestEndgameIntegration:
    """Integration tests with real SimpleGoGame."""

    @pytest.fixture
    def real_game(self) -> "SimpleGoGame":
        """Create a real SimpleGoGame instance."""
        from src.tools.gtp import SimpleGoGame

        return SimpleGoGame(9)

    def test_real_game_empty(self, real_game: "SimpleGoGame") -> None:
        """Test with real game - empty board."""
        config = EndgameConfig()
        detector = EndgameDetector(config)

        analysis = detector.analyze(real_game, human_just_passed=True)

        assert analysis.should_ai_pass is False
        assert analysis.empty_count == 81

    def test_real_game_consecutive_passes(self, real_game: "SimpleGoGame") -> None:
        """Test with real game - consecutive passes."""
        config = EndgameConfig()
        detector = EndgameDetector(config)

        real_game.play_pass()  # First pass
        analysis = detector.analyze(real_game, human_just_passed=True)

        assert analysis.should_ai_pass is True
        assert analysis.reason == "consecutive_pass_detection"
        assert real_game.passes == 1

    def test_real_game_terminal_after_two_passes(
        self, real_game: "SimpleGoGame"
    ) -> None:
        """Verify game is terminal after two passes."""
        real_game.play_pass()
        real_game.play_pass()

        assert real_game.is_terminal() is True
        assert real_game.passes == 2
