"""Pytest fixtures for research tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.research.config import (
    BenchmarkConfig,
    ComparisonConfig,
    ExperimentConfig,
    ExperimentType,
    TransferConfig,
)
from src.research.experiment import Experiment, ExperimentRun, ExperimentTracker
from src.research.benchmark import Benchmark, BenchmarkResult
from src.research.validator import TransferMetrics, TransferResult, TransferValidator
from src.research.comparison import ComparisonResult, ModelComparison, ModelMetrics
from src.research.reporter import Reporter, ReportFormat


@pytest.fixture
def default_experiment_config() -> ExperimentConfig:
    """Create default experiment config."""
    return ExperimentConfig(name="test_experiment")


@pytest.fixture
def default_benchmark_config() -> BenchmarkConfig:
    """Create default benchmark config."""
    return BenchmarkConfig(
        name="test_benchmark",
        sizes=[9, 13],
        n_warmup=1,
        n_iterations=10,
    )


@pytest.fixture
def default_transfer_config() -> TransferConfig:
    """Create default transfer config."""
    return TransferConfig(
        source_size=9,
        target_sizes=[9, 13],
        n_train_samples=100,
        n_eval_samples=10,
        n_epochs=1,
    )


@pytest.fixture
def default_comparison_config() -> ComparisonConfig:
    """Create default comparison config."""
    return ComparisonConfig(
        eval_sizes=[9, 13],
        n_eval_samples=10,
        n_bootstrap=1000,
    )


@pytest.fixture
def experiment(default_experiment_config: ExperimentConfig) -> Experiment:
    """Create an experiment."""
    return Experiment(config=default_experiment_config)


@pytest.fixture
def experiment_run() -> ExperimentRun:
    """Create an experiment run."""
    return ExperimentRun(run_id="test123")


@pytest.fixture
def experiment_tracker() -> ExperimentTracker:
    """Create an experiment tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ExperimentTracker(base_dir=tmpdir)


@pytest.fixture
def benchmark(default_benchmark_config: BenchmarkConfig) -> Benchmark:
    """Create a benchmark."""
    return Benchmark(config=default_benchmark_config)


@pytest.fixture
def benchmark_result() -> BenchmarkResult:
    """Create a benchmark result."""
    return BenchmarkResult(
        name="test_benchmark",
        size=9,
        batch_size=32,
        n_iterations=100,
        mean_time_ms=10.0,
        std_time_ms=1.0,
        min_time_ms=8.0,
        max_time_ms=12.0,
        total_time_s=1.0,
        throughput=3200.0,
    )


@pytest.fixture
def transfer_validator(
    default_transfer_config: TransferConfig,
) -> TransferValidator:
    """Create a transfer validator."""
    return TransferValidator(config=default_transfer_config)


@pytest.fixture
def transfer_metrics() -> TransferMetrics:
    """Create transfer metrics."""
    return TransferMetrics(
        target_size=19,
        source_size=9,
        n_samples=100,
        mse=0.01,
        mae=0.05,
        rmse=0.1,
        max_error=0.2,
        threshold=0.05,
        passed=True,
    )


@pytest.fixture
def transfer_result() -> TransferResult:
    """Create a transfer result."""
    result = TransferResult(
        result_id="test123",
        source_size=9,
        primary_target=19,
    )
    result.target_metrics = {
        9: TransferMetrics(
            target_size=9,
            source_size=9,
            n_samples=100,
            mse=0.005,
            mae=0.03,
            rmse=0.07,
            max_error=0.15,
            threshold=0.05,
            passed=True,
        ),
        19: TransferMetrics(
            target_size=19,
            source_size=9,
            n_samples=100,
            mse=0.02,
            mae=0.08,
            rmse=0.14,
            max_error=0.3,
            threshold=0.05,
            passed=True,
        ),
    }
    result.passed = True
    result.all_passed = True
    return result


@pytest.fixture
def model_comparison(
    default_comparison_config: ComparisonConfig,
) -> ModelComparison:
    """Create a model comparison."""
    return ModelComparison(config=default_comparison_config)


@pytest.fixture
def model_metrics() -> ModelMetrics:
    """Create model metrics."""
    metrics = ModelMetrics(model_name="test_model")
    metrics.add_metrics(9, {"mse": 0.01, "mae": 0.05})
    metrics.add_metrics(13, {"mse": 0.02, "mae": 0.07})
    metrics.compute_aggregates()
    return metrics


@pytest.fixture
def comparison_result() -> ComparisonResult:
    """Create a comparison result."""
    result = ComparisonResult(comparison_id="compare123")

    model1 = ModelMetrics(model_name="model_a")
    model1.add_metrics(9, {"mse": 0.01, "mae": 0.05})
    model1.compute_aggregates()

    model2 = ModelMetrics(model_name="model_b")
    model2.add_metrics(9, {"mse": 0.02, "mae": 0.08})
    model2.compute_aggregates()

    result.model_metrics = {
        "model_a": model1,
        "model_b": model2,
    }
    result.rankings = {"mse": ["model_a", "model_b"]}

    return result


@pytest.fixture
def reporter() -> Reporter:
    """Create a reporter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Reporter(output_dir=tmpdir)
