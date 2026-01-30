"""Tests for training stability monitoring."""

from __future__ import annotations

import pytest
import torch
from torch.optim import SGD

from src.training.stability import (
    EarlyStopping,
    EarlyStoppingConfig,
    GradientMonitor,
    GradientStatus,
    PlateauConfig,
    PlateauDetector,
    TrainingStabilityMonitor,
)


class TestEarlyStopping:
    """Tests for EarlyStopping."""

    def test_initialization(self) -> None:
        """Test early stopping initialization."""
        config = EarlyStoppingConfig(patience=5)
        es = EarlyStopping(config)
        assert es.best_value is None
        assert es.counter == 0
        assert not es.should_stop

    def test_first_step_sets_best(self) -> None:
        """Test first step sets best value."""
        es = EarlyStopping(EarlyStoppingConfig())
        es.step(0.5)
        assert es.best_value == 0.5
        assert es.counter == 0

    def test_improvement_resets_counter(self) -> None:
        """Test improvement resets patience counter."""
        es = EarlyStopping(EarlyStoppingConfig(mode="max"))
        es.step(0.5)
        es.step(0.4)  # No improvement
        assert es.counter == 1
        es.step(0.6)  # Improvement
        assert es.counter == 0
        assert es.best_value == 0.6

    def test_triggers_after_patience(self) -> None:
        """Test early stopping triggers after patience exceeded."""
        config = EarlyStoppingConfig(patience=3, mode="max")
        es = EarlyStopping(config)

        es.step(0.5)  # Best
        assert not es.step(0.4)  # 1
        assert not es.step(0.4)  # 2
        assert es.step(0.4)  # 3 - triggers
        assert es.should_stop

    def test_min_mode(self) -> None:
        """Test early stopping with min mode (lower is better)."""
        config = EarlyStoppingConfig(patience=2, mode="min")
        es = EarlyStopping(config)

        es.step(1.0)  # Best
        assert not es.step(1.1)  # Worse
        assert es.step(1.2)  # Triggers
        assert es.should_stop

    def test_min_delta(self) -> None:
        """Test min_delta threshold."""
        config = EarlyStoppingConfig(patience=2, mode="max", min_delta=0.1)
        es = EarlyStopping(config)

        es.step(0.5)
        es.step(0.55)  # Not enough improvement (< 0.1)
        assert es.counter == 1
        es.step(0.7)  # Enough improvement
        assert es.counter == 0

    def test_reset(self) -> None:
        """Test reset functionality."""
        es = EarlyStopping(EarlyStoppingConfig(patience=2))
        es.step(0.5)
        es.step(0.4)
        es.step(0.3)

        es.reset()
        assert es.best_value is None
        assert es.counter == 0
        assert not es.should_stop


class TestPlateauDetector:
    """Tests for PlateauDetector."""

    @pytest.fixture
    def optimizer(self) -> SGD:
        """Create test optimizer."""
        model = torch.nn.Linear(10, 2)
        return SGD(model.parameters(), lr=0.1)

    def test_initialization(self, optimizer: SGD) -> None:
        """Test plateau detector initialization."""
        config = PlateauConfig(patience=3)
        detector = PlateauDetector(config, optimizer)
        assert detector.best_value is None
        assert detector.counter == 0

    def test_no_reduction_on_improvement(self, optimizer: SGD) -> None:
        """Test no LR reduction when improving."""
        config = PlateauConfig(patience=2, mode="min", threshold=0.05)
        detector = PlateauDetector(config, optimizer)

        assert not detector.step(1.0)
        assert not detector.step(0.9)  # Improvement
        assert not detector.step(0.8)  # Improvement
        assert optimizer.param_groups[0]["lr"] == 0.1

    def test_reduction_on_plateau(self, optimizer: SGD) -> None:
        """Test LR reduction on plateau."""
        config = PlateauConfig(patience=2, factor=0.5, mode="min")
        detector = PlateauDetector(config, optimizer)

        detector.step(1.0)
        detector.step(1.0)  # No improvement, count=1
        reduced = detector.step(1.0)  # No improvement, count=2, reduce

        assert reduced
        assert optimizer.param_groups[0]["lr"] == 0.05

    def test_min_lr_respected(self, optimizer: SGD) -> None:
        """Test minimum LR is respected."""
        config = PlateauConfig(patience=1, factor=0.1, min_lr=0.01)
        detector = PlateauDetector(config, optimizer)

        # Reduce multiple times
        detector.step(1.0)
        detector.step(1.0)  # Reduce to 0.01
        detector.step(1.0)  # Should stay at 0.01

        assert optimizer.param_groups[0]["lr"] >= config.min_lr

    def test_get_current_lr(self, optimizer: SGD) -> None:
        """Test getting current LR."""
        config = PlateauConfig()
        detector = PlateauDetector(config, optimizer)
        assert detector.get_current_lr() == 0.1


class TestGradientMonitor:
    """Tests for GradientMonitor."""

    def test_healthy_gradient(self) -> None:
        """Test healthy gradient detection."""
        monitor = GradientMonitor(exploding_threshold=100, vanishing_threshold=1e-7)
        status = monitor.check(1.0)

        assert status.is_healthy
        assert not status.is_exploding
        assert not status.is_vanishing
        assert not status.is_nan

    def test_exploding_gradient(self) -> None:
        """Test exploding gradient detection."""
        monitor = GradientMonitor(exploding_threshold=10.0)
        status = monitor.check(50.0)

        assert not status.is_healthy
        assert status.is_exploding
        assert not status.is_vanishing

    def test_vanishing_gradient(self) -> None:
        """Test vanishing gradient detection."""
        monitor = GradientMonitor(vanishing_threshold=1e-5)
        status = monitor.check(1e-8)

        assert not status.is_healthy
        assert not status.is_exploding
        assert status.is_vanishing

    def test_nan_gradient(self) -> None:
        """Test NaN gradient detection."""
        monitor = GradientMonitor()
        status = monitor.check(float("nan"))

        assert not status.is_healthy
        assert status.is_nan

    def test_tensor_input(self) -> None:
        """Test with tensor input."""
        monitor = GradientMonitor()
        status = monitor.check(torch.tensor(5.0))

        assert status.is_healthy
        assert status.gradient_norm == 5.0

    def test_history_tracking(self) -> None:
        """Test gradient history tracking."""
        monitor = GradientMonitor(history_size=5)

        for i in range(1, 6):
            monitor.check(float(i))

        stats = monitor.get_statistics()
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_empty_statistics(self) -> None:
        """Test statistics with empty history."""
        monitor = GradientMonitor()
        stats = monitor.get_statistics()

        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0


class TestTrainingStabilityMonitor:
    """Tests for TrainingStabilityMonitor."""

    @pytest.fixture
    def optimizer(self) -> SGD:
        """Create test optimizer."""
        model = torch.nn.Linear(10, 2)
        return SGD(model.parameters(), lr=0.1)

    def test_gradient_only(self) -> None:
        """Test with gradient monitoring only."""
        monitor = TrainingStabilityMonitor()
        status = monitor.check_gradient(1.0)
        assert status.is_healthy

    def test_with_early_stopping(self) -> None:
        """Test with early stopping."""
        es = EarlyStopping(EarlyStoppingConfig(patience=2))
        monitor = TrainingStabilityMonitor(early_stopping=es)

        monitor.check_early_stopping(0.5)
        monitor.check_early_stopping(0.4)
        result = monitor.check_early_stopping(0.3)

        assert result
        assert monitor.should_stop

    def test_with_plateau_detector(self, optimizer: SGD) -> None:
        """Test with plateau detector."""
        pd = PlateauDetector(PlateauConfig(patience=1), optimizer)
        monitor = TrainingStabilityMonitor(plateau_detector=pd)

        monitor.check_plateau(1.0)
        reduced = monitor.check_plateau(1.0)

        assert reduced

    def test_all_components(self, optimizer: SGD) -> None:
        """Test with all components."""
        es = EarlyStopping(EarlyStoppingConfig(patience=5))
        pd = PlateauDetector(PlateauConfig(patience=3), optimizer)
        gm = GradientMonitor()

        monitor = TrainingStabilityMonitor(
            early_stopping=es,
            plateau_detector=pd,
            gradient_monitor=gm,
        )

        # Check gradient
        assert monitor.check_gradient(1.0).is_healthy

        # Check plateau (not triggered yet)
        assert not monitor.check_plateau(1.0)

        # Check early stopping (not triggered yet)
        assert not monitor.check_early_stopping(0.5)
        assert not monitor.should_stop
