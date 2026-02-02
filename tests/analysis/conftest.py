"""Pytest fixtures for game analysis tests."""

from __future__ import annotations

import pytest

from src.analysis.config import AnalysisConfig, AnalysisMode, AnnotationLevel
from src.analysis.evaluator import EvaluationResult, PositionEvaluator
from src.analysis.patterns import Pattern, PatternLibrary, PatternMatcher, PatternType
from src.analysis.reviewer import GameReviewer
from src.analysis.statistics import GameStatistics, StatisticsCollector


@pytest.fixture
def default_config() -> AnalysisConfig:
    """Create default analysis configuration."""
    return AnalysisConfig()


@pytest.fixture
def quick_config() -> AnalysisConfig:
    """Create quick analysis configuration."""
    return AnalysisConfig(
        name="quick",
        mode=AnalysisMode.QUICK,
        mcts_simulations=50,
        max_variations=1,
    )


@pytest.fixture
def deep_config() -> AnalysisConfig:
    """Create deep analysis configuration."""
    return AnalysisConfig(
        name="deep",
        mode=AnalysisMode.DEEP,
        mcts_simulations=1600,
        max_variations=5,
        annotation_level=AnnotationLevel.DETAILED,
    )


@pytest.fixture
def sample_evaluation() -> EvaluationResult:
    """Create sample evaluation result."""
    return EvaluationResult(
        win_rate=0.65,
        best_moves=[
            ((3, 3), 0.25),
            ((15, 3), 0.20),
            ((3, 15), 0.15),
        ],
        policy={
            (3, 3): 0.25,
            (15, 3): 0.20,
            (3, 15): 0.15,
            (10, 10): 0.10,
        },
        value=0.3,
        depth=1,
    )


@pytest.fixture
def position_evaluator(default_config: AnalysisConfig) -> PositionEvaluator:
    """Create position evaluator without model."""
    return PositionEvaluator(config=default_config)


@pytest.fixture
def game_reviewer(default_config: AnalysisConfig) -> GameReviewer:
    """Create game reviewer."""
    evaluator = PositionEvaluator(config=default_config)
    return GameReviewer(evaluator=evaluator, config=default_config)


@pytest.fixture
def sample_game_moves() -> list[tuple[str, int, int]]:
    """Create sample game moves."""
    return [
        ("B", 3, 3),  # Move 1
        ("W", 15, 3),  # Move 2
        ("B", 3, 15),  # Move 3
        ("W", 15, 15),  # Move 4
        ("B", 10, 10),  # Move 5
    ]


@pytest.fixture
def game_statistics() -> GameStatistics:
    """Create sample game statistics."""
    return GameStatistics(
        game_id="test_game_001",
        board_size=19,
        total_moves=100,
        result="B+2.5",
        black_player="Player A",
        white_player="Player B",
    )


@pytest.fixture
def statistics_collector() -> StatisticsCollector:
    """Create statistics collector."""
    return StatisticsCollector()


@pytest.fixture
def sample_pattern() -> Pattern:
    """Create sample pattern."""
    return Pattern(
        name="test_pattern",
        pattern_type=PatternType.SHAPE,
        stones={
            (0, 0): "B",
            (1, 0): "B",
            (0, 1): "B",
        },
        description="Test pattern",
        tags=["test"],
        difficulty=1,
    )


@pytest.fixture
def pattern_library() -> PatternLibrary:
    """Create pattern library."""
    return PatternLibrary()


@pytest.fixture
def pattern_matcher(pattern_library: PatternLibrary) -> PatternMatcher:
    """Create pattern matcher."""
    return PatternMatcher(library=pattern_library)


@pytest.fixture
def sample_board() -> list[list[int]]:
    """Create sample board state."""
    # 9x9 board with some stones
    board = [[0] * 9 for _ in range(9)]
    # Add black stones
    board[3][3] = 1
    board[3][4] = 1
    board[4][3] = 1
    # Add white stones
    board[5][5] = 2
    board[5][6] = 2
    return board
