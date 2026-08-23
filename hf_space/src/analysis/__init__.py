"""Game Analysis module for AlphaGalerkin.

Provides:
- Position evaluation and best move analysis
- Game review with move-by-move analysis
- Pattern recognition and statistics
- Integration with SGF for annotations
"""

from __future__ import annotations

from src.analysis.config import (
    AnalysisConfig,
    AnalysisMode,
    AnnotationLevel,
)
from src.analysis.evaluator import EvaluationResult, PositionEvaluator
from src.analysis.patterns import PatternLibrary, PatternMatcher
from src.analysis.reviewer import GameAnalysis, GameReviewer, MoveAnalysis
from src.analysis.statistics import GameStatistics, StatisticsCollector

__all__ = [
    # Configuration
    "AnalysisConfig",
    "AnalysisMode",
    "AnnotationLevel",
    # Position evaluation
    "PositionEvaluator",
    "EvaluationResult",
    # Game review
    "GameReviewer",
    "MoveAnalysis",
    "GameAnalysis",
    # Statistics
    "GameStatistics",
    "StatisticsCollector",
    # Patterns
    "PatternMatcher",
    "PatternLibrary",
]
