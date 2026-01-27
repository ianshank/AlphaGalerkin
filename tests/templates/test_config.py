"""Tests for the templates.config module."""

from __future__ import annotations

import pytest
from pydantic import Field, ValidationError

from src.templates.config import (
    BaseModuleConfig,
    BoardSizeConfig,
    MetricDefinition,
    ThresholdOperator,
    TrainableModuleConfig,
    create_config_class,
)


class TestMetricDefinition:
    """Tests for MetricDefinition."""

    def test_evaluate_less_than(self) -> None:
        """Test < operator evaluation."""
        metric = MetricDefinition(name="mse", threshold=0.05)
        assert metric.evaluate(0.04)
        assert not metric.evaluate(0.05)
        assert not metric.evaluate(0.06)

    def test_evaluate_less_equal(self) -> None:
        """Test <= operator evaluation."""
        metric = MetricDefinition(
            name="mse",
            operator=ThresholdOperator.LESS_EQUAL,
            threshold=0.05,
        )
        assert metric.evaluate(0.04)
        assert metric.evaluate(0.05)
        assert not metric.evaluate(0.06)

    def test_evaluate_greater_than(self) -> None:
        """Test > operator evaluation."""
        metric = MetricDefinition(
            name="accuracy",
            operator=ThresholdOperator.GREATER_THAN,
            threshold=0.9,
        )
        assert metric.evaluate(0.95)
        assert not metric.evaluate(0.9)
        assert not metric.evaluate(0.85)

    def test_evaluate_greater_equal(self) -> None:
        """Test >= operator evaluation."""
        metric = MetricDefinition(
            name="accuracy",
            operator=ThresholdOperator.GREATER_EQUAL,
            threshold=0.9,
        )
        assert metric.evaluate(0.95)
        assert metric.evaluate(0.9)
        assert not metric.evaluate(0.85)

    def test_evaluate_equal_with_tolerance(self) -> None:
        """Test == operator with floating point tolerance."""
        metric = MetricDefinition(
            name="score",
            operator=ThresholdOperator.EQUAL,
            threshold=1.0,
        )
        assert metric.evaluate(1.0)
        assert metric.evaluate(1.0 + 1e-10)  # Within tolerance
        assert not metric.evaluate(1.1)

    def test_evaluate_not_equal(self) -> None:
        """Test != operator."""
        metric = MetricDefinition(
            name="error",
            operator=ThresholdOperator.NOT_EQUAL,
            threshold=0.0,
        )
        assert metric.evaluate(0.1)
        assert not metric.evaluate(0.0)

    def test_format_result(self) -> None:
        """Test result formatting."""
        metric = MetricDefinition(name="mse", threshold=0.05, unit="units")
        result = metric.format_result(0.03)
        assert "[PASS]" in result
        assert "mse" in result
        assert "0.03" in result

        result = metric.format_result(0.10)
        assert "[FAIL]" in result

    def test_immutable(self) -> None:
        """Test that MetricDefinition is immutable."""
        metric = MetricDefinition(name="test", threshold=1.0)
        with pytest.raises(ValidationError):
            metric.threshold = 2.0  # type: ignore[misc]


class TestBaseModuleConfig:
    """Tests for BaseModuleConfig."""

    def test_required_fields(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError):
            BaseModuleConfig()  # type: ignore[call-arg]

    def test_default_values(self) -> None:
        """Test default values are applied."""
        config = BaseModuleConfig(name="test")
        assert config.seed == 42
        assert config.timeout_seconds == 3600
        assert config.debug is False
        assert config.description == ""

    def test_constraint_validation(self) -> None:
        """Test Field constraints are enforced."""
        # seed must be >= 0
        with pytest.raises(ValidationError):
            BaseModuleConfig(name="test", seed=-1)

        # timeout must be >= 1
        with pytest.raises(ValidationError):
            BaseModuleConfig(name="test", timeout_seconds=0)

        # timeout must be <= 86400
        with pytest.raises(ValidationError):
            BaseModuleConfig(name="test", timeout_seconds=100000)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise errors."""
        with pytest.raises(ValidationError):
            BaseModuleConfig(name="test", unknown_field="value")  # type: ignore[call-arg]

    def test_compute_hash_deterministic(self) -> None:
        """Test that hash is deterministic."""
        config1 = BaseModuleConfig(name="test", seed=42)
        config2 = BaseModuleConfig(name="test", seed=42)
        assert config1.compute_hash() == config2.compute_hash()

    def test_compute_hash_excludes_created_at(self) -> None:
        """Test that hash excludes volatile fields."""
        import time

        config1 = BaseModuleConfig(name="test")
        time.sleep(0.01)  # Ensure different created_at
        config2 = BaseModuleConfig(name="test")

        # Different created_at, but same hash
        assert config1.created_at != config2.created_at
        assert config1.compute_hash() == config2.compute_hash()

    def test_compute_hash_different_values(self) -> None:
        """Test that different values produce different hashes."""
        config1 = BaseModuleConfig(name="test", seed=42)
        config2 = BaseModuleConfig(name="test", seed=123)
        assert config1.compute_hash() != config2.compute_hash()

    def test_to_yaml_dict(self) -> None:
        """Test YAML-friendly dictionary conversion."""
        config = BaseModuleConfig(name="test", debug=True)
        yaml_dict = config.to_yaml_dict()

        assert yaml_dict["name"] == "test"
        assert yaml_dict["debug"] is True
        assert isinstance(yaml_dict["created_at"], str)  # ISO format

    def test_with_overrides(self) -> None:
        """Test creating config with overrides."""
        config = BaseModuleConfig(name="test", seed=42)
        new_config = config.with_overrides(seed=123, debug=True)

        assert new_config.name == "test"
        assert new_config.seed == 123
        assert new_config.debug is True
        # Original unchanged
        assert config.seed == 42

    def test_string_strip_whitespace(self) -> None:
        """Test that whitespace is stripped from strings."""
        config = BaseModuleConfig(name="  test  ", description="  desc  ")
        assert config.name == "test"
        assert config.description == "desc"


class TestTrainableModuleConfig:
    """Tests for TrainableModuleConfig."""

    def test_default_training_values(self) -> None:
        """Test default training parameters."""
        config = TrainableModuleConfig(name="test")
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.total_steps == 10000
        assert config.device == "auto"

    def test_learning_rate_constraints(self) -> None:
        """Test learning rate must be in valid range."""
        with pytest.raises(ValidationError):
            TrainableModuleConfig(name="test", learning_rate=0.0)

        with pytest.raises(ValidationError):
            TrainableModuleConfig(name="test", learning_rate=1.0)

    def test_warmup_vs_total_steps_validation(self) -> None:
        """Test warmup_steps must be < total_steps."""
        with pytest.raises(ValidationError):
            TrainableModuleConfig(name="test", warmup_steps=1000, total_steps=500)

    def test_device_options(self) -> None:
        """Test valid device options."""
        for device in ["auto", "cpu", "cuda", "mps"]:
            config = TrainableModuleConfig(name="test", device=device)  # type: ignore[arg-type]
            assert config.device == device


class TestBoardSizeConfig:
    """Tests for BoardSizeConfig."""

    def test_default_sizes(self) -> None:
        """Test default board sizes."""
        config = BoardSizeConfig()
        assert config.sizes == [9, 13, 19]

    def test_sizes_sorted_and_deduplicated(self) -> None:
        """Test that sizes are sorted and deduplicated."""
        config = BoardSizeConfig(sizes=[19, 9, 13, 9])
        assert config.sizes == [9, 13, 19]

    def test_empty_sizes_rejected(self) -> None:
        """Test that empty sizes list is rejected."""
        with pytest.raises(ValidationError):
            BoardSizeConfig(sizes=[])

    def test_size_bounds_validation(self) -> None:
        """Test size must be between 3 and 25."""
        with pytest.raises(ValidationError):
            BoardSizeConfig(sizes=[2])

        with pytest.raises(ValidationError):
            BoardSizeConfig(sizes=[26])

    def test_sizes_within_range_validation(self) -> None:
        """Test that sizes must be within min/max range."""
        with pytest.raises(ValidationError):
            BoardSizeConfig(min_size=9, max_size=13, sizes=[5, 9, 13])


class TestCreateConfigClass:
    """Tests for create_config_class factory."""

    def test_create_simple_config(self) -> None:
        """Test creating a simple config class."""
        MyConfig = create_config_class(
            "MyConfig",
            my_int=(int, Field(default=10, ge=1)),
            my_str=(str, Field(default="hello")),
        )

        config = MyConfig(name="test")
        assert config.my_int == 10
        assert config.my_str == "hello"

    def test_created_config_validates(self) -> None:
        """Test that created config class validates."""
        MyConfig = create_config_class(
            "MyConfig",
            my_int=(int, Field(default=10, ge=1)),
        )

        with pytest.raises(ValidationError):
            MyConfig(name="test", my_int=0)

    def test_inherits_from_base(self) -> None:
        """Test that created class inherits from base."""
        MyConfig = create_config_class("MyConfig")

        assert issubclass(MyConfig, BaseModuleConfig)

        config = MyConfig(name="test")
        assert hasattr(config, "compute_hash")
        assert hasattr(config, "seed")
