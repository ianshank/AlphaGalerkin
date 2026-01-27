"""Tests for quick trainer."""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from src.prototyping.config import QuickTrainConfig, PresetType
from src.prototyping.builder import PrototypeModel, ModelBuilder
from src.prototyping.trainer import (
    QuickTrainer,
    TrainResult,
    create_quick_trainer,
)


class TestTrainResult:
    """Tests for TrainResult dataclass."""

    def test_initialization(self, train_result: TrainResult) -> None:
        """Test result initialization."""
        assert train_result.result_id == "test123"
        assert train_result.model_id == "model123"
        assert train_result.n_epochs == 5
        assert train_result.final_loss == 0.1

    def test_to_dict(self, train_result: TrainResult) -> None:
        """Test serialization to dict."""
        data = train_result.to_dict()

        assert data["result_id"] == "test123"
        assert data["n_epochs"] == 5
        assert data["final_loss"] == 0.1

    def test_summary(self, train_result: TrainResult) -> None:
        """Test summary generation."""
        summary = train_result.summary()

        assert "test123" in summary
        assert "model123" in summary
        assert "Epochs: 5" in summary


class TestQuickTrainer:
    """Tests for QuickTrainer."""

    def test_initialization(self, quick_trainer: QuickTrainer) -> None:
        """Test trainer initialization."""
        assert quick_trainer.config is not None
        assert len(quick_trainer.results) == 0

    def test_train_basic(self, quick_trainer: QuickTrainer) -> None:
        """Test basic training."""
        # Mock model
        class MockModel:
            pass

        model = MockModel()
        step = [0]

        def train_fn(m: Any, batch: Any) -> float:
            step[0] += 1
            return 1.0 / (step[0] + 1)

        def data_iterator() -> Iterator[list[int]]:
            for _ in range(5):
                yield [1, 2, 3]

        result = quick_trainer.train(
            model=model,
            train_fn=train_fn,
            data_iterator=data_iterator,
        )

        assert result is not None
        assert result.n_epochs > 0
        assert result.final_loss is not None
        assert len(quick_trainer.results) == 1

    def test_train_with_prototype_model(
        self,
        quick_trainer: QuickTrainer,
        prototype_model: PrototypeModel,
    ) -> None:
        """Test training with PrototypeModel."""
        def train_fn(m: Any, batch: Any) -> float:
            return 0.1

        def data_iterator() -> Iterator[list[int]]:
            for _ in range(3):
                yield [1]

        result = quick_trainer.train(
            model=prototype_model,
            train_fn=train_fn,
            data_iterator=data_iterator,
        )

        assert result.model_id == prototype_model.model_id

    def test_train_with_callbacks(self, quick_trainer: QuickTrainer) -> None:
        """Test training with callbacks."""
        callback_calls: dict[str, int] = {
            "start": 0,
            "end": 0,
            "epoch": 0,
        }

        quick_trainer.register_callback(
            "on_train_start",
            lambda m: callback_calls.__setitem__("start", callback_calls["start"] + 1),
        )
        quick_trainer.register_callback(
            "on_train_end",
            lambda r, m: callback_calls.__setitem__("end", callback_calls["end"] + 1),
        )
        quick_trainer.register_callback(
            "on_epoch_end",
            lambda e, l, m: callback_calls.__setitem__("epoch", callback_calls["epoch"] + 1),
        )

        def train_fn(m: Any, batch: Any) -> float:
            return 0.1

        def data_iterator() -> Iterator[list[int]]:
            yield [1]

        quick_trainer.train(
            model=None,
            train_fn=train_fn,
            data_iterator=data_iterator,
        )

        assert callback_calls["start"] == 1
        assert callback_calls["end"] == 1
        assert callback_calls["epoch"] >= 1

    def test_early_stopping(self) -> None:
        """Test early stopping."""
        trainer = QuickTrainer(
            config=QuickTrainConfig(
                n_epochs=100,
                early_stopping_patience=2,
            )
        )

        # Loss that doesn't improve
        def train_fn(m: Any, batch: Any) -> float:
            return 1.0  # Constant loss

        def data_iterator() -> Iterator[list[int]]:
            yield [1]

        result = trainer.train(
            model=None,
            train_fn=train_fn,
            data_iterator=data_iterator,
        )

        assert result.stopped_early
        assert result.n_epochs < 100

    def test_get_best_result(self, quick_trainer: QuickTrainer) -> None:
        """Test getting best result."""
        # Train multiple times
        for loss in [0.3, 0.1, 0.2]:
            final_loss = loss

            def train_fn(m: Any, batch: Any) -> float:
                return final_loss

            def data_iterator() -> Iterator[list[int]]:
                yield [1]

            quick_trainer.train(
                model=None,
                train_fn=train_fn,
                data_iterator=data_iterator,
            )

        best = quick_trainer.get_best_result()
        assert best is not None
        # Best should be the one with lowest loss

    def test_clear(self, quick_trainer: QuickTrainer) -> None:
        """Test clearing results."""
        def train_fn(m: Any, batch: Any) -> float:
            return 0.1

        def data_iterator() -> Iterator[list[int]]:
            yield [1]

        quick_trainer.train(
            model=None,
            train_fn=train_fn,
            data_iterator=data_iterator,
        )

        assert len(quick_trainer.results) == 1
        quick_trainer.clear()
        assert len(quick_trainer.results) == 0


class TestCreateQuickTrainer:
    """Tests for create_quick_trainer factory."""

    def test_create_default(self) -> None:
        """Test creating default trainer."""
        trainer = create_quick_trainer()
        assert trainer.config.n_epochs == 10

    def test_create_with_preset(self) -> None:
        """Test creating with preset."""
        trainer = create_quick_trainer(preset="debug")
        assert trainer.config.n_epochs == 2

    def test_create_with_overrides(self) -> None:
        """Test creating with overrides."""
        trainer = create_quick_trainer(
            preset="small",
            n_epochs=50,
        )
        assert trainer.config.n_epochs == 50
