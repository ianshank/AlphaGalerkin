"""Tests for the templates.config module."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import Field, ValidationError

from src.templates.config import (
    CONFIG_HASH_LENGTH,
    VOLATILE_CONFIG_FIELDS,
    BaseModuleConfig,
    BoardSizeConfig,
    MetricDefinition,
    ThresholdOperator,
    TrainableModuleConfig,
    config_hash,
    create_config_class,
    stable_config_payload,
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


#: Two construction times far enough apart that no clock could round them together.
_EARLIER = datetime(2026, 1, 1, tzinfo=timezone.utc)
_LATER = _EARLIER + timedelta(days=30)


class _Leaf(BaseModuleConfig):
    """A module config, as a scenario's nested ``SubstrateConfig`` is one."""

    knob: int = Field(default=1, ge=0)


class _Parent(BaseModuleConfig):
    """A module config nesting another, directly and inside a list."""

    child: _Leaf = Field(default_factory=lambda: _Leaf(name="leaf"))
    children: list[_Leaf] = Field(default_factory=list)


def _pre_change_hash(data: dict[str, Any]) -> str:
    """The formula every config hash used before volatile fields were stripped in depth."""
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:CONFIG_HASH_LENGTH]


_JSON_SCALARS = st.none() | st.booleans() | st.integers() | st.text(max_size=8)
_NON_VOLATILE_KEYS = st.text(min_size=1, max_size=8).filter(
    lambda key: key not in VOLATILE_CONFIG_FIELDS
)
_JSON_PAYLOADS = st.recursive(
    _JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(_NON_VOLATILE_KEYS, children, max_size=4)
    ),
    max_leaves=12,
)


class TestConfigHashStability:
    """A config hash is a function of what the config configures, not of when it was built.

    Killed mutations, each by a named test here:
    - stripping ``created_at`` only at the top level (the pre-fix ``exclude=``)
      -> ``test_a_nested_module_config_does_not_leak_its_construction_time``;
    - not recursing into lists -> ``test_module_configs_inside_lists_are_stripped_too``;
    - an empty ``VOLATILE_CONFIG_FIELDS`` -> both of the above;
    - stripping any other key -> ``test_payloads_without_volatile_keys_hash_as_before``.
    """

    def test_a_nested_module_config_does_not_leak_its_construction_time(self) -> None:
        early = _Parent(name="p", created_at=_EARLIER, child=_Leaf(name="l", created_at=_EARLIER))
        late = _Parent(name="p", created_at=_LATER, child=_Leaf(name="l", created_at=_LATER))
        assert early.compute_hash() == late.compute_hash()

    def test_module_configs_inside_lists_are_stripped_too(self) -> None:
        early = _Parent(name="p", children=[_Leaf(name="l", created_at=_EARLIER)])
        late = _Parent(name="p", children=[_Leaf(name="l", created_at=_LATER)])
        assert early.compute_hash() == late.compute_hash()

    def test_a_nested_value_still_changes_the_hash(self) -> None:
        one = _Parent(name="p", child=_Leaf(name="l", knob=1))
        two = _Parent(name="p", child=_Leaf(name="l", knob=2))
        assert one.compute_hash() != two.compute_hash()

    def test_a_flat_module_config_hashes_exactly_as_before(self) -> None:
        config = BaseModuleConfig(name="flat", seed=7)
        assert config.compute_hash() == _pre_change_hash(config.model_dump(exclude={"created_at"}))

    @given(_JSON_PAYLOADS)
    def test_payloads_without_volatile_keys_hash_as_before(self, value: Any) -> None:
        payload = {"value": value}
        assert stable_config_payload(payload) == json.loads(json.dumps(payload))
        assert config_hash(payload) == _pre_change_hash(payload)

    def test_volatile_keys_are_removed_at_every_depth(self) -> None:
        payload = {
            "created_at": 1,
            "a": {"created_at": 2, "b": [{"created_at": 3, "c": 4}], "t": ({"created_at": 5},)},
        }
        assert stable_config_payload(payload) == {"a": {"b": [{"c": 4}], "t": [{}]}}
