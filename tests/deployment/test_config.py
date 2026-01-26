"""Tests for deployment configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.deployment.config import (
    DeploymentConfig,
    ExecutionProvider,
    ExportConfig,
    QuantizationConfig,
    QuantizationMode,
    RuntimeConfig,
)


class TestExportConfig:
    """Tests for ExportConfig."""

    def test_default_values(self) -> None:
        """Test default export configuration."""
        config = ExportConfig()

        assert config.opset_version == 17
        assert config.input_names == ["board_state"]
        assert config.output_names == ["policy", "value"]
        assert config.do_constant_folding is True

    def test_opset_version_validation(self) -> None:
        """Test opset version bounds."""
        # Valid range
        config = ExportConfig(opset_version=9)
        assert config.opset_version == 9

        config = ExportConfig(opset_version=20)
        assert config.opset_version == 20

        # Invalid range
        with pytest.raises(ValidationError):
            ExportConfig(opset_version=8)

        with pytest.raises(ValidationError):
            ExportConfig(opset_version=21)

    def test_dynamic_axes(self) -> None:
        """Test dynamic axes configuration."""
        config = ExportConfig(
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            }
        )

        assert "input" in config.dynamic_axes
        assert config.dynamic_axes["input"][0] == "batch"

    def test_input_name_validation(self) -> None:
        """Test input name must be valid identifier."""
        with pytest.raises(ValidationError):
            ExportConfig(input_names=["123invalid"])


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_values(self) -> None:
        """Test default quantization configuration."""
        config = QuantizationConfig()

        assert config.enabled is True
        assert config.mode == QuantizationMode.DYNAMIC
        assert config.weight_type == "int8"
        assert config.per_channel is True

    def test_mode_options(self) -> None:
        """Test all quantization modes."""
        for mode in QuantizationMode:
            config = QuantizationConfig(mode=mode)
            assert config.mode == mode

    def test_calibration_samples(self) -> None:
        """Test calibration samples validation."""
        config = QuantizationConfig(calibration_samples=100)
        assert config.calibration_samples == 100

        with pytest.raises(ValidationError):
            QuantizationConfig(calibration_samples=5)  # Below minimum


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_default_providers(self) -> None:
        """Test default execution providers."""
        config = RuntimeConfig()

        assert ExecutionProvider.CUDA in config.execution_providers
        assert ExecutionProvider.CPU in config.execution_providers

    def test_custom_providers(self) -> None:
        """Test custom provider list."""
        config = RuntimeConfig(
            execution_providers=[
                ExecutionProvider.TENSORRT,
                ExecutionProvider.CUDA,
                ExecutionProvider.CPU,
            ]
        )

        assert config.execution_providers[0] == ExecutionProvider.TENSORRT

    def test_threading_config(self) -> None:
        """Test threading configuration."""
        config = RuntimeConfig(
            intra_op_threads=4,
            inter_op_threads=2,
        )

        assert config.intra_op_threads == 4
        assert config.inter_op_threads == 2

    def test_profiling_config(self) -> None:
        """Test profiling configuration."""
        config = RuntimeConfig(
            enable_profiling=True,
            profile_output_path="/tmp/profile",
        )

        assert config.enable_profiling is True
        assert config.profile_output_path == "/tmp/profile"


class TestDeploymentConfig:
    """Tests for combined DeploymentConfig."""

    def test_nested_configs(self) -> None:
        """Test nested configuration access."""
        config = DeploymentConfig()

        assert config.export.opset_version == 17
        assert config.quantization.mode == QuantizationMode.DYNAMIC
        assert ExecutionProvider.CPU in config.runtime.execution_providers

    def test_validation_settings(self) -> None:
        """Test validation configuration."""
        config = DeploymentConfig(
            validate_export=True,
            validation_tolerance=1e-4,
            validation_samples=20,
        )

        assert config.validate_export is True
        assert config.validation_tolerance == 1e-4
        assert config.validation_samples == 20
