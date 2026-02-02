"""Pytest fixtures for prototyping tests."""

from __future__ import annotations

import pytest

from src.prototyping.builder import ModelBuilder, PrototypeModel
from src.prototyping.config import (
    PresetType,
    PrototypeConfig,
    QuickEvalConfig,
    QuickTrainConfig,
)
from src.prototyping.data import DataGenerator, SyntheticData
from src.prototyping.evaluator import EvalResult, MetricResult, QuickEvaluator
from src.prototyping.templates import TemplateRegistry
from src.prototyping.trainer import QuickTrainer, TrainResult
from src.prototyping.visualizer import Visualizer


@pytest.fixture
def default_prototype_config() -> PrototypeConfig:
    """Create default prototype config."""
    return PrototypeConfig(
        name="test_prototype",
        preset=PresetType.SMALL,
        board_sizes=[9],
    )


@pytest.fixture
def default_train_config() -> QuickTrainConfig:
    """Create default train config."""
    return QuickTrainConfig(
        n_epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        warmup_steps=5,
        log_interval=2,
        eval_interval=10,
    )


@pytest.fixture
def default_eval_config() -> QuickEvalConfig:
    """Create default eval config."""
    return QuickEvalConfig(
        n_samples=100,
        batch_size=32,
        metrics=["mse", "mae"],
        compute_confidence=False,
    )


@pytest.fixture
def model_builder() -> ModelBuilder:
    """Create model builder."""
    return ModelBuilder()


@pytest.fixture
def prototype_model(model_builder: ModelBuilder) -> PrototypeModel:
    """Create prototype model."""
    return model_builder.build(name="test_model")


@pytest.fixture
def quick_trainer(default_train_config: QuickTrainConfig) -> QuickTrainer:
    """Create quick trainer."""
    return QuickTrainer(config=default_train_config)


@pytest.fixture
def quick_evaluator(default_eval_config: QuickEvalConfig) -> QuickEvaluator:
    """Create quick evaluator."""
    return QuickEvaluator(config=default_eval_config)


@pytest.fixture
def data_generator() -> DataGenerator:
    """Create data generator."""
    return DataGenerator(seed=42)


@pytest.fixture
def synthetic_data(data_generator: DataGenerator) -> SyntheticData:
    """Create synthetic data."""
    return data_generator.generate("linear", n_samples=100)


@pytest.fixture
def train_result() -> TrainResult:
    """Create a train result."""
    return TrainResult(
        result_id="test123",
        model_id="model123",
        n_epochs=5,
        n_steps=100,
        final_loss=0.1,
        best_loss=0.05,
        metrics={"loss": [0.5, 0.3, 0.2, 0.15, 0.1]},
        duration_seconds=10.0,
    )


@pytest.fixture
def eval_result() -> EvalResult:
    """Create an eval result."""
    return EvalResult(
        result_id="eval123",
        model_id="model123",
        n_samples=100,
        metrics={
            "mse": MetricResult(name="mse", value=0.01),
            "mae": MetricResult(name="mae", value=0.05),
        },
        duration_seconds=1.0,
    )


@pytest.fixture
def visualizer() -> Visualizer:
    """Create visualizer."""
    return Visualizer(width=40, height=10)


@pytest.fixture
def template_registry() -> TemplateRegistry:
    """Create template registry."""
    return TemplateRegistry()
