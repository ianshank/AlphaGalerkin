"""Configuration schemas for Game Analysis.

Provides Pydantic-validated configuration with:
- No hardcoded values
- Validation constraints
- Multiple analysis modes
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisMode(str, Enum):
    """Analysis mode determining depth and scope."""

    QUICK = "quick"  # Fast analysis, limited depth
    STANDARD = "standard"  # Balanced analysis
    DEEP = "deep"  # Comprehensive analysis
    VARIATION = "variation"  # Focus on alternative lines


class AnnotationLevel(str, Enum):
    """Level of detail for annotations."""

    MINIMAL = "minimal"  # Only critical moves
    NORMAL = "normal"  # Notable moves and mistakes
    DETAILED = "detailed"  # All moves with analysis
    VERBOSE = "verbose"  # Full analysis with variations


class MoveClassification(str, Enum):
    """Classification of move quality."""

    EXCELLENT = "excellent"  # Best or near-best move
    GOOD = "good"  # Strong move
    INACCURACY = "inaccuracy"  # Suboptimal but not bad
    MISTAKE = "mistake"  # Clear error
    BLUNDER = "blunder"  # Severe error
    NEUTRAL = "neutral"  # Neither good nor bad


class AnalysisConfig(BaseModel):
    """Configuration for game analysis.

    Controls analysis depth, scope, and output format.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        default="default",
        min_length=1,
        description="Configuration name for identification",
    )

    # Analysis depth
    mode: AnalysisMode = Field(
        default=AnalysisMode.STANDARD,
        description="Analysis mode determining depth",
    )
    mcts_simulations: int = Field(
        default=800,
        ge=1,
        le=100000,
        description="MCTS simulations per position",
    )
    max_variations: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum alternative variations to explore",
    )
    variation_depth: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum depth for variation exploration",
    )

    # Thresholds for classification
    excellent_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Win rate difference for excellent moves",
    )
    good_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Win rate difference for good moves",
    )
    inaccuracy_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Win rate drop threshold for inaccuracy",
    )
    mistake_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Win rate drop threshold for mistake",
    )
    blunder_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Win rate drop threshold for blunder",
    )

    # Output options
    annotation_level: AnnotationLevel = Field(
        default=AnnotationLevel.NORMAL,
        description="Detail level for annotations",
    )
    include_policy: bool = Field(
        default=True,
        description="Include policy distribution in analysis",
    )
    include_variations: bool = Field(
        default=True,
        description="Include alternative variations",
    )
    include_statistics: bool = Field(
        default=True,
        description="Include game statistics",
    )

    # Performance options
    batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Batch size for position evaluation",
    )
    cache_evaluations: bool = Field(
        default=True,
        description="Cache position evaluations",
    )
    max_cache_size: int = Field(
        default=10000,
        ge=100,
        description="Maximum cache size",
    )

    def compute_hash(self) -> str:
        """Compute unique hash of configuration.

        Returns:
            Hexadecimal hash string.
        """
        data = self.model_dump(mode="json")
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def get_simulations_for_mode(self) -> int:
        """Get recommended simulations for analysis mode.

        Returns:
            Number of MCTS simulations.
        """
        mode_simulations = {
            AnalysisMode.QUICK: min(100, self.mcts_simulations),
            AnalysisMode.STANDARD: self.mcts_simulations,
            AnalysisMode.DEEP: self.mcts_simulations * 2,
            AnalysisMode.VARIATION: self.mcts_simulations,
        }
        return mode_simulations.get(self.mode, self.mcts_simulations)

    def classify_move(
        self,
        actual_win_rate: float,
        best_win_rate: float,
    ) -> MoveClassification:
        """Classify a move based on win rate difference.

        Args:
            actual_win_rate: Win rate after the played move.
            best_win_rate: Win rate of the best move.

        Returns:
            Move classification.
        """
        diff = best_win_rate - actual_win_rate

        if diff <= self.excellent_threshold:
            return MoveClassification.EXCELLENT
        elif diff <= self.good_threshold:
            return MoveClassification.GOOD
        elif diff <= self.inaccuracy_threshold:
            return MoveClassification.NEUTRAL
        elif diff <= self.mistake_threshold:
            return MoveClassification.INACCURACY
        elif diff <= self.blunder_threshold:
            return MoveClassification.MISTAKE
        else:
            return MoveClassification.BLUNDER


class QuickAnalysisConfig(AnalysisConfig):
    """Quick analysis configuration preset."""

    name: str = "quick"
    mode: AnalysisMode = AnalysisMode.QUICK
    mcts_simulations: int = 100
    max_variations: int = 1
    annotation_level: AnnotationLevel = AnnotationLevel.MINIMAL


class DeepAnalysisConfig(AnalysisConfig):
    """Deep analysis configuration preset."""

    name: str = "deep"
    mode: AnalysisMode = AnalysisMode.DEEP
    mcts_simulations: int = 1600
    max_variations: int = 5
    variation_depth: int = 10
    annotation_level: AnnotationLevel = AnnotationLevel.DETAILED


def create_analysis_config(
    mode: str = "standard",
    simulations: int | None = None,
    **kwargs: Any,
) -> AnalysisConfig:
    """Factory function to create analysis config.

    Args:
        mode: Analysis mode ("quick", "standard", "deep", "variation").
        simulations: Override MCTS simulations.
        **kwargs: Additional configuration options.

    Returns:
        Configured AnalysisConfig.
    """
    if mode == "quick":
        config = QuickAnalysisConfig(**kwargs)
    elif mode == "deep":
        config = DeepAnalysisConfig(**kwargs)
    else:
        config = AnalysisConfig(mode=AnalysisMode(mode), **kwargs)

    if simulations is not None:
        config.mcts_simulations = simulations

    return config
