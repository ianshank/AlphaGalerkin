"""Tests for analysis configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.config import (
    AnalysisConfig,
    AnalysisMode,
    AnnotationLevel,
    MoveClassification,
    QuickAnalysisConfig,
    DeepAnalysisConfig,
    create_analysis_config,
)


class TestAnalysisMode:
    """Tests for AnalysisMode enum."""

    def test_all_modes_exist(self) -> None:
        """Test all analysis modes exist."""
        assert AnalysisMode.QUICK.value == "quick"
        assert AnalysisMode.STANDARD.value == "standard"
        assert AnalysisMode.DEEP.value == "deep"
        assert AnalysisMode.VARIATION.value == "variation"


class TestAnnotationLevel:
    """Tests for AnnotationLevel enum."""

    def test_all_levels_exist(self) -> None:
        """Test all annotation levels exist."""
        assert AnnotationLevel.MINIMAL.value == "minimal"
        assert AnnotationLevel.NORMAL.value == "normal"
        assert AnnotationLevel.DETAILED.value == "detailed"
        assert AnnotationLevel.VERBOSE.value == "verbose"


class TestMoveClassification:
    """Tests for MoveClassification enum."""

    def test_all_classifications_exist(self) -> None:
        """Test all classifications exist."""
        assert MoveClassification.EXCELLENT.value == "excellent"
        assert MoveClassification.GOOD.value == "good"
        assert MoveClassification.INACCURACY.value == "inaccuracy"
        assert MoveClassification.MISTAKE.value == "mistake"
        assert MoveClassification.BLUNDER.value == "blunder"
        assert MoveClassification.NEUTRAL.value == "neutral"


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_values(self, default_config: AnalysisConfig) -> None:
        """Test default configuration values."""
        assert default_config.name == "default"
        assert default_config.mode == AnalysisMode.STANDARD
        assert default_config.mcts_simulations == 800
        assert default_config.max_variations == 3

    def test_validation_mcts_simulations(self) -> None:
        """Test MCTS simulations validation."""
        # Valid
        AnalysisConfig(mcts_simulations=100)
        AnalysisConfig(mcts_simulations=100000)

        # Invalid
        with pytest.raises(ValidationError):
            AnalysisConfig(mcts_simulations=0)

    def test_validation_thresholds(self) -> None:
        """Test threshold validation."""
        # Valid
        AnalysisConfig(excellent_threshold=0.01)

        # Invalid - threshold > 1
        with pytest.raises(ValidationError):
            AnalysisConfig(excellent_threshold=1.5)

    def test_compute_hash(self, default_config: AnalysisConfig) -> None:
        """Test configuration hash computation."""
        hash1 = default_config.compute_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Same config produces same hash
        hash2 = default_config.compute_hash()
        assert hash1 == hash2

        # Different config produces different hash
        other = AnalysisConfig(name="other")
        assert default_config.compute_hash() != other.compute_hash()

    def test_get_simulations_for_mode(self) -> None:
        """Test getting simulations based on mode."""
        config = AnalysisConfig(mcts_simulations=800)

        config.mode = AnalysisMode.QUICK
        assert config.get_simulations_for_mode() <= 100

        config.mode = AnalysisMode.STANDARD
        assert config.get_simulations_for_mode() == 800

        config.mode = AnalysisMode.DEEP
        assert config.get_simulations_for_mode() == 1600

    def test_classify_move_excellent(self, default_config: AnalysisConfig) -> None:
        """Test move classification - excellent."""
        classification = default_config.classify_move(0.65, 0.66)
        assert classification == MoveClassification.EXCELLENT

    def test_classify_move_good(self, default_config: AnalysisConfig) -> None:
        """Test move classification - good."""
        classification = default_config.classify_move(0.62, 0.66)
        assert classification == MoveClassification.GOOD

    def test_classify_move_inaccuracy(self, default_config: AnalysisConfig) -> None:
        """Test move classification - inaccuracy."""
        classification = default_config.classify_move(0.55, 0.66)
        assert classification == MoveClassification.INACCURACY

    def test_classify_move_mistake(self, default_config: AnalysisConfig) -> None:
        """Test move classification - mistake."""
        classification = default_config.classify_move(0.45, 0.66)
        assert classification == MoveClassification.MISTAKE

    def test_classify_move_blunder(self, default_config: AnalysisConfig) -> None:
        """Test move classification - blunder."""
        classification = default_config.classify_move(0.30, 0.66)
        assert classification == MoveClassification.BLUNDER


class TestQuickAnalysisConfig:
    """Tests for QuickAnalysisConfig preset."""

    def test_quick_config_values(self) -> None:
        """Test quick config preset values."""
        config = QuickAnalysisConfig()
        assert config.mode == AnalysisMode.QUICK
        assert config.mcts_simulations == 100
        assert config.max_variations == 1


class TestDeepAnalysisConfig:
    """Tests for DeepAnalysisConfig preset."""

    def test_deep_config_values(self) -> None:
        """Test deep config preset values."""
        config = DeepAnalysisConfig()
        assert config.mode == AnalysisMode.DEEP
        assert config.mcts_simulations == 1600
        assert config.max_variations == 5


class TestCreateAnalysisConfig:
    """Tests for create_analysis_config factory."""

    def test_create_default(self) -> None:
        """Test creating default config."""
        config = create_analysis_config()
        assert config.mode == AnalysisMode.STANDARD

    def test_create_quick(self) -> None:
        """Test creating quick config."""
        config = create_analysis_config(mode="quick")
        assert config.mode == AnalysisMode.QUICK

    def test_create_deep(self) -> None:
        """Test creating deep config."""
        config = create_analysis_config(mode="deep")
        assert config.mode == AnalysisMode.DEEP

    def test_override_simulations(self) -> None:
        """Test overriding simulations."""
        config = create_analysis_config(simulations=500)
        assert config.mcts_simulations == 500

    def test_additional_kwargs(self) -> None:
        """Test passing additional kwargs."""
        config = create_analysis_config(
            max_variations=5,
            include_policy=False,
        )
        assert config.max_variations == 5
        assert config.include_policy is False
