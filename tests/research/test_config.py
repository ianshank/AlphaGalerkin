"""Tests for research configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.research.config import (
    BenchmarkConfig,
    ComparisonConfig,
    ExperimentConfig,
    ExperimentType,
    MetricType,
    TransferConfig,
    create_experiment_config,
    create_transfer_config,
)


class TestExperimentType:
    """Tests for ExperimentType enum."""

    def test_all_types_exist(self) -> None:
        """Test all experiment types exist."""
        assert ExperimentType.ABLATION.value == "ablation"
        assert ExperimentType.TRANSFER.value == "transfer"
        assert ExperimentType.SCALING.value == "scaling"
        assert ExperimentType.COMPARISON.value == "comparison"
        assert ExperimentType.HYPERPARAMETER.value == "hyperparameter"
        assert ExperimentType.CUSTOM.value == "custom"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types_exist(self) -> None:
        """Test all metric types exist."""
        assert MetricType.MSE.value == "mse"
        assert MetricType.MAE.value == "mae"
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.WIN_RATE.value == "win_rate"
        assert MetricType.THROUGHPUT.value == "throughput"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_values(self, default_experiment_config: ExperimentConfig) -> None:
        """Test default configuration values."""
        assert default_experiment_config.name == "test_experiment"
        assert default_experiment_config.experiment_type == ExperimentType.CUSTOM
        assert default_experiment_config.seed == 42
        assert default_experiment_config.n_runs == 1
        assert default_experiment_config.deterministic is True

    def test_required_name(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError):
            ExperimentConfig()  # type: ignore

    def test_n_runs_validation(self) -> None:
        """Test n_runs validation."""
        # Valid
        ExperimentConfig(name="test", n_runs=1)
        ExperimentConfig(name="test", n_runs=50)

        # Invalid
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", n_runs=0)
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", n_runs=101)

    def test_compute_hash(self, default_experiment_config: ExperimentConfig) -> None:
        """Test configuration hash computation."""
        hash1 = default_experiment_config.compute_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Same config produces same hash
        hash2 = default_experiment_config.compute_hash()
        assert hash1 == hash2

        # Different config produces different hash
        other = ExperimentConfig(name="other")
        assert default_experiment_config.compute_hash() != other.compute_hash()


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_values(self, default_benchmark_config: BenchmarkConfig) -> None:
        """Test default configuration values."""
        assert default_benchmark_config.name == "test_benchmark"
        assert default_benchmark_config.batch_size == 32
        assert default_benchmark_config.use_gpu is True

    def test_sizes_validation(self) -> None:
        """Test sizes validation."""
        # Valid
        BenchmarkConfig(sizes=[9, 19])

        # Invalid - empty
        with pytest.raises(ValidationError):
            BenchmarkConfig(sizes=[])

    def test_sizes_sorted(self) -> None:
        """Test sizes are sorted and deduplicated."""
        config = BenchmarkConfig(sizes=[19, 9, 13, 9])
        assert config.sizes == [9, 13, 19]


class TestTransferConfig:
    """Tests for TransferConfig."""

    def test_default_values(self, default_transfer_config: TransferConfig) -> None:
        """Test default configuration values."""
        assert default_transfer_config.source_size == 9
        assert 9 in default_transfer_config.target_sizes
        assert default_transfer_config.mse_threshold == 0.05

    def test_source_size_validation(self) -> None:
        """Test source size validation."""
        # Valid
        TransferConfig(source_size=5)
        TransferConfig(source_size=25)

        # Invalid
        with pytest.raises(ValidationError):
            TransferConfig(source_size=2)
        with pytest.raises(ValidationError):
            TransferConfig(source_size=30)

    def test_target_sizes_validation(self) -> None:
        """Test target sizes validation."""
        # Valid
        TransferConfig(target_sizes=[9, 19])

        # Invalid - empty
        with pytest.raises(ValidationError):
            TransferConfig(target_sizes=[])


class TestComparisonConfig:
    """Tests for ComparisonConfig."""

    def test_default_values(self, default_comparison_config: ComparisonConfig) -> None:
        """Test default configuration values."""
        assert default_comparison_config.n_bootstrap == 1000
        assert default_comparison_config.alpha == 0.05

    def test_alpha_validation(self) -> None:
        """Test alpha validation."""
        # Valid
        ComparisonConfig(alpha=0.01)
        ComparisonConfig(alpha=0.1)

        # Invalid
        with pytest.raises(ValidationError):
            ComparisonConfig(alpha=0)
        with pytest.raises(ValidationError):
            ComparisonConfig(alpha=0.5)


class TestCreateExperimentConfig:
    """Tests for create_experiment_config factory."""

    def test_create_default(self) -> None:
        """Test creating default config."""
        config = create_experiment_config(name="test")
        assert config.name == "test"
        assert config.experiment_type == ExperimentType.CUSTOM

    def test_create_with_type(self) -> None:
        """Test creating with type."""
        config = create_experiment_config(
            name="ablation",
            experiment_type="ablation",
        )
        assert config.experiment_type == ExperimentType.ABLATION

    def test_create_with_n_runs(self) -> None:
        """Test creating with n_runs."""
        config = create_experiment_config(name="test", n_runs=5)
        assert config.n_runs == 5


class TestCreateTransferConfig:
    """Tests for create_transfer_config factory."""

    def test_create_default(self) -> None:
        """Test creating default config."""
        config = create_transfer_config()
        assert config.source_size == 9
        assert 19 in config.target_sizes

    def test_create_with_custom_sizes(self) -> None:
        """Test creating with custom sizes."""
        config = create_transfer_config(
            source_size=5,
            target_sizes=[5, 9, 13],
        )
        assert config.source_size == 5
        assert config.target_sizes == [5, 9, 13]
