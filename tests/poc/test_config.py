"""Tests for PoC configuration schemas.

Validates:
    - Pydantic model validation
    - Default value behavior
    - Constraint enforcement
    - Serialization/deserialization
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.poc.config import (
    BaseScenarioConfig,
    ComplexityScenarioConfig,
    MetricThreshold,
    ScenarioResult,
    ScenarioStatus,
    ScenarioTier,
    StabilityScenarioConfig,
    TransferScenarioConfig,
    load_config_from_dict,
)


class TestMetricThreshold:
    """Tests for MetricThreshold model."""

    def test_evaluate_less_than(self) -> None:
        """Test < operator evaluation."""
        threshold = MetricThreshold(name="mse", operator="<", value=0.05)
        assert threshold.evaluate(0.04)
        assert not threshold.evaluate(0.05)
        assert not threshold.evaluate(0.06)

    def test_evaluate_less_equal(self) -> None:
        """Test <= operator evaluation."""
        threshold = MetricThreshold(name="mse", operator="<=", value=0.05)
        assert threshold.evaluate(0.04)
        assert threshold.evaluate(0.05)
        assert not threshold.evaluate(0.06)

    def test_evaluate_greater_than(self) -> None:
        """Test > operator evaluation."""
        threshold = MetricThreshold(name="speedup", operator=">", value=1.5)
        assert threshold.evaluate(2.0)
        assert not threshold.evaluate(1.5)
        assert not threshold.evaluate(1.0)

    def test_evaluate_equal(self) -> None:
        """Test == operator with floating point tolerance."""
        threshold = MetricThreshold(name="accuracy", operator="==", value=1.0)
        assert threshold.evaluate(1.0)
        assert threshold.evaluate(1.0 + 1e-12)  # Within tolerance
        assert not threshold.evaluate(1.1)


class TestBaseScenarioConfig:
    """Tests for BaseScenarioConfig model."""

    def test_required_fields(self) -> None:
        """Test that name and description are required."""
        with pytest.raises(ValidationError):
            BaseScenarioConfig()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            BaseScenarioConfig(name="test")  # type: ignore[call-arg]

        # Should work with both
        config = BaseScenarioConfig(name="test", description="A test scenario")
        assert config.name == "test"
        assert config.description == "A test scenario"

    def test_defaults(self) -> None:
        """Test default values."""
        config = BaseScenarioConfig(name="test", description="Test")

        assert config.tier == ScenarioTier.FUNCTIONAL
        assert config.enabled is True
        assert config.timeout_seconds == 3600
        assert config.retry_count == 0
        assert config.seed == 42
        assert config.requires_gpu is False

    def test_constraint_validation(self) -> None:
        """Test constraint enforcement."""
        # timeout must be >= 1
        with pytest.raises(ValidationError):
            BaseScenarioConfig(
                name="test",
                description="Test",
                timeout_seconds=0,
            )

        # retry_count must be 0-5
        with pytest.raises(ValidationError):
            BaseScenarioConfig(
                name="test",
                description="Test",
                retry_count=10,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise errors."""
        with pytest.raises(ValidationError):
            BaseScenarioConfig(
                name="test",
                description="Test",
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_compute_hash_deterministic(self) -> None:
        """Test that hash is deterministic."""
        config1 = BaseScenarioConfig(name="test", description="Test", seed=42)
        config2 = BaseScenarioConfig(name="test", description="Test", seed=42)
        config3 = BaseScenarioConfig(name="test", description="Test", seed=43)

        assert config1.compute_hash() == config2.compute_hash()
        assert config1.compute_hash() != config3.compute_hash()


class TestTransferScenarioConfig:
    """Tests for TransferScenarioConfig model."""

    def test_defaults(self) -> None:
        """Test transfer scenario defaults."""
        config = TransferScenarioConfig()

        assert config.name == "transfer"
        assert config.train_resolution == 9
        assert config.eval_resolutions == [9, 13, 19]
        assert config.primary_eval_resolution == 19
        assert config.mse_threshold == 0.05

    def test_eval_resolutions_validation(self) -> None:
        """Test eval_resolutions validation."""
        # Empty list not allowed
        with pytest.raises(ValidationError):
            TransferScenarioConfig(eval_resolutions=[])

        # Invalid resolution
        with pytest.raises(ValidationError):
            TransferScenarioConfig(eval_resolutions=[1, 9, 19])

    def test_primary_added_to_eval(self) -> None:
        """Test that primary resolution is added to eval list."""
        config = TransferScenarioConfig(
            eval_resolutions=[9, 13],
            primary_eval_resolution=19,
        )

        assert 19 in config.eval_resolutions

    def test_eval_resolutions_sorted_unique(self) -> None:
        """Test that eval resolutions are sorted and unique."""
        config = TransferScenarioConfig(
            eval_resolutions=[19, 9, 13, 9],  # Duplicate and unsorted
        )

        assert config.eval_resolutions == [9, 13, 19]

    def test_get_default_thresholds(self) -> None:
        """Test threshold generation."""
        config = TransferScenarioConfig(
            eval_resolutions=[9, 19],
            mse_threshold=0.1,
        )

        thresholds = config.get_default_thresholds()

        assert len(thresholds) == 2
        assert thresholds[0].name == "mse_9x9"
        assert thresholds[0].value == 0.1
        assert thresholds[1].name == "mse_19x19"


class TestComplexityScenarioConfig:
    """Tests for ComplexityScenarioConfig model."""

    def test_defaults(self) -> None:
        """Test complexity scenario defaults."""
        config = ComplexityScenarioConfig()

        assert config.name == "complexity"
        assert config.grid_sizes == [9, 13, 19, 25]
        assert config.fnet_scaling_exponent_max == 1.5
        assert config.requires_gpu is True

    def test_grid_sizes_validation(self) -> None:
        """Test grid_sizes must have at least 3 elements."""
        with pytest.raises(ValidationError):
            ComplexityScenarioConfig(grid_sizes=[9, 19])

    def test_min_speedup_validation(self) -> None:
        """Test min_speedup_factor must be > 1.0."""
        with pytest.raises(ValidationError):
            ComplexityScenarioConfig(min_speedup_factor=0.5)


class TestStabilityScenarioConfig:
    """Tests for StabilityScenarioConfig model."""

    def test_defaults(self) -> None:
        """Test stability scenario defaults."""
        config = StabilityScenarioConfig()

        assert config.name == "stability"
        assert config.resolutions == [5, 9, 13, 19]
        assert config.lbb_threshold == 1e-6
        assert config.max_lbb_violations == 0


class TestScenarioResult:
    """Tests for ScenarioResult model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        from datetime import datetime

        result = ScenarioResult(
            scenario_name="test",
            config_hash="abc123",
            status=ScenarioStatus.PASSED,
            passed=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0,
        )

        assert result.scenario_name == "test"
        assert result.passed is True

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed for scenario-specific data."""
        from datetime import datetime

        result = ScenarioResult(
            scenario_name="test",
            config_hash="abc123",
            status=ScenarioStatus.PASSED,
            passed=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0,
            custom_field="custom_value",
        )

        assert result.custom_field == "custom_value"  # type: ignore[attr-defined]

    def test_summary_generation(self) -> None:
        """Test summary string generation."""
        from datetime import datetime

        result = ScenarioResult(
            scenario_name="transfer",
            config_hash="abc123",
            status=ScenarioStatus.PASSED,
            passed=True,
            metrics={"mse_19x19": 0.03},
            threshold_results={"mse_19x19": True},
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=10.5,
        )

        summary = result.summary()

        assert "PASS" in summary
        assert "transfer" in summary
        assert "mse_19x19" in summary


class TestLoadConfigFromDict:
    """Tests for load_config_from_dict function."""

    def test_load_transfer_config(self) -> None:
        """Test loading transfer config from dict."""
        data = {
            "name": "transfer",
            "description": "Test transfer",
            "train_resolution": 13,
            "mse_threshold": 0.1,
        }

        config = load_config_from_dict(data)

        assert isinstance(config, TransferScenarioConfig)
        assert config.train_resolution == 13
        assert config.mse_threshold == 0.1

    def test_load_with_type_hint(self) -> None:
        """Test loading with explicit type hint."""
        data = {
            "name": "custom_transfer",
            "description": "Test",
        }

        config = load_config_from_dict(data, scenario_type="transfer")

        assert isinstance(config, TransferScenarioConfig)

    def test_load_unknown_type(self) -> None:
        """Test loading unknown type falls back to base."""
        data = {
            "name": "unknown",
            "description": "Unknown scenario",
        }

        config = load_config_from_dict(data)

        assert isinstance(config, BaseScenarioConfig)

    def test_load_noyron_hx_config(self) -> None:
        """Loader must dispatch noyron_hx YAML to NoyronHXScenarioConfig.

        Regression for a bug where ``load_config_from_dict`` only knew about
        transfer/complexity/stability and silently fell back to
        ``BaseScenarioConfig`` for ``name="noyron_hx"``. With ``extra="forbid"``
        on the base, this raised 24 Pydantic validation errors at runtime
        when the headline YAML was loaded via the CLI runner.
        """
        from src.poc.config_noyron import NoyronHXScenarioConfig

        data = {
            "name": "noyron_hx",
            "description": "regression",
            "helix_R_major": 0.05,
            "helix_r_minor": 0.012,
            "helix_pitch": 0.02,
            "helix_n_turns": 5,
            "n_train_pts": 4096,
            "n_eval_pts": 16384,
            "device": "cpu",
        }

        config = load_config_from_dict(data)

        assert isinstance(config, NoyronHXScenarioConfig)
        assert config.helix_R_major == 0.05
        assert config.n_eval_pts == 16384


class TestConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        original = TransferScenarioConfig(
            train_resolution=13,
            eval_resolutions=[9, 13, 19, 25],
            mse_threshold=0.1,
        )

        # Serialize
        json_str = original.model_dump_json()

        # Deserialize
        data = json.loads(json_str)
        restored = TransferScenarioConfig(**data)

        assert original.train_resolution == restored.train_resolution
        assert original.eval_resolutions == restored.eval_resolutions
        assert original.compute_hash() == restored.compute_hash()

    def test_dict_roundtrip(self) -> None:
        """Test dict serialization roundtrip."""
        original = ComplexityScenarioConfig(
            grid_sizes=[5, 9, 13, 19, 25],
            batch_size=64,
        )

        # Serialize
        data = original.model_dump()

        # Deserialize
        restored = ComplexityScenarioConfig(**data)

        assert original.grid_sizes == restored.grid_sizes
        assert original.batch_size == restored.batch_size
