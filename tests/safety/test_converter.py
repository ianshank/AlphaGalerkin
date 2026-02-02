"""Tests for the safety.converter module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.safety.config import ConverterConfig, ValidationConfig
from src.safety.converter import (
    ConversionResult,
    SafeTensorsConverter,
    convert_to_safetensors,
    extract_metadata,
    extract_state_dict,
    load_safetensors,
    SAFETENSORS_AVAILABLE,
)


# Skip all tests if safetensors is not installed
pytestmark = pytest.mark.skipif(
    not SAFETENSORS_AVAILABLE,
    reason="safetensors not installed",
)


class TestExtractStateDict:
    """Tests for extract_state_dict function."""

    def test_direct_state_dict(self) -> None:
        """Test extraction from direct state dict."""
        state_dict = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }

        result = extract_state_dict(state_dict)

        assert "weight" in result
        assert "bias" in result
        assert isinstance(result["weight"], torch.Tensor)

    def test_model_state_dict_wrapper(self) -> None:
        """Test extraction from model_state_dict wrapper."""
        checkpoint = {
            "model_state_dict": {
                "weight": torch.randn(10, 10),
            },
            "step": 1000,
        }

        result = extract_state_dict(checkpoint)

        assert "weight" in result
        assert "step" not in result  # Non-tensor should be excluded

    def test_state_dict_wrapper(self) -> None:
        """Test extraction from state_dict wrapper."""
        checkpoint = {
            "state_dict": {
                "weight": torch.randn(10, 10),
            },
        }

        result = extract_state_dict(checkpoint)

        assert "weight" in result

    def test_nested_state_dict(self) -> None:
        """Test extraction from nested state dict."""
        checkpoint = {
            "encoder": {
                "weight": torch.randn(10, 10),
            },
            "decoder": {
                "weight": torch.randn(10, 10),
            },
        }

        result = extract_state_dict(checkpoint)

        assert "encoder.weight" in result
        assert "decoder.weight" in result

    def test_non_tensor_excluded(self) -> None:
        """Test that non-tensor values are excluded."""
        checkpoint = {
            "weight": torch.randn(10, 10),
            "config": {"d_model": 64},
            "step": 1000,
        }

        result = extract_state_dict(checkpoint)

        assert "weight" in result
        assert "config" not in result
        assert "step" not in result


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extract_common_fields(self) -> None:
        """Test extraction of common metadata fields."""
        checkpoint = {
            "step": 1000,
            "version": "1.0.0",
            "epoch": 5,
        }

        metadata = extract_metadata(checkpoint)

        assert metadata["step"] == "1000"
        assert metadata["version"] == "1.0.0"
        assert metadata["epoch"] == "5"

    def test_extract_config_hash(self) -> None:
        """Test extraction of config hash."""
        checkpoint = {
            "config": {"d_model": 64, "n_layers": 4},
        }

        metadata = extract_metadata(checkpoint)

        assert "config_hash" in metadata
        assert len(metadata["config_hash"]) == 16

    def test_only_string_convertible(self) -> None:
        """Test that only string-convertible values are included."""
        checkpoint = {
            "step": 1000,
            "complex_object": object(),  # Should be skipped
        }

        metadata = extract_metadata(checkpoint)

        assert "step" in metadata
        assert "complex_object" not in metadata


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = ConversionResult(
            success=True,
            original_path="/path/to/model.pt",
            safetensors_path="/path/to/model.safetensors",
            num_tensors=10,
            metadata={"step": "1000"},
            original_hash="abc123",
            roundtrip_verified=True,
            errors=[],
            backup_path=None,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["original_path"] == "/path/to/model.pt"
        assert d["num_tensors"] == 10
        assert d["roundtrip_verified"] is True


class TestSafeTensorsConverter:
    """Tests for SafeTensorsConverter class."""

    def test_convert_valid_checkpoint(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test conversion of valid checkpoint."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", verify_roundtrip=True)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert result.num_tensors > 0
        assert result.roundtrip_verified
        assert output_path.exists()

    def test_convert_with_metadata(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test that metadata is preserved."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", include_metadata=True)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert "step" in result.metadata

    def test_convert_without_metadata(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test conversion without metadata."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", include_metadata=False)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert len(result.metadata) == 0

    def test_convert_nonexistent_source(self, temp_dir: Path) -> None:
        """Test conversion of nonexistent source file."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test")
        converter = SafeTensorsConverter(config)
        result = converter.convert("/nonexistent/model.pt", output_path)

        assert not result.success
        assert len(result.errors) > 0

    def test_convert_empty_checkpoint(
        self, empty_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test conversion of checkpoint with no tensors."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test")
        converter = SafeTensorsConverter(config)
        result = converter.convert(empty_checkpoint, output_path)

        assert not result.success
        assert any("No tensors" in e for e in result.errors)

    def test_convert_with_validation(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test conversion with source validation."""
        output_path = temp_dir / "model.safetensors"

        converter_config = ConverterConfig(name="test")
        validation_config = ValidationConfig(name="test")
        converter = SafeTensorsConverter(converter_config, validation_config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success

    def test_convert_with_validation_failure(
        self, checkpoint_with_nan: Path, temp_dir: Path
    ) -> None:
        """Test that conversion fails with validation errors."""
        output_path = temp_dir / "model.safetensors"

        converter_config = ConverterConfig(name="test")
        validation_config = ValidationConfig(name="test", check_nan_inf=True)
        converter = SafeTensorsConverter(converter_config, validation_config)
        result = converter.convert(checkpoint_with_nan, output_path)

        assert not result.success
        assert any("validation" in e.lower() for e in result.errors)

    def test_roundtrip_verification(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test that roundtrip verification works."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(
            name="test",
            verify_roundtrip=True,
            tolerance_atol=1e-6,
            tolerance_rtol=1e-5,
        )
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert result.roundtrip_verified

    def test_skip_roundtrip_verification(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test skipping roundtrip verification."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", verify_roundtrip=False)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert not result.roundtrip_verified


class TestConvertToSafetensors:
    """Tests for convert_to_safetensors convenience function."""

    def test_basic_conversion(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test basic conversion with defaults."""
        output_path = temp_dir / "model.safetensors"

        result = convert_to_safetensors(valid_checkpoint, output_path)

        assert result.success
        assert output_path.exists()

    def test_conversion_with_options(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test conversion with custom options."""
        output_path = temp_dir / "model.safetensors"

        result = convert_to_safetensors(
            valid_checkpoint,
            output_path,
            include_metadata=True,
            verify=True,
        )

        assert result.success
        assert result.roundtrip_verified


class TestLoadSafetensors:
    """Tests for load_safetensors function."""

    def test_load_converted_file(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test loading a converted SafeTensors file."""
        output_path = temp_dir / "model.safetensors"

        # First convert
        result = convert_to_safetensors(valid_checkpoint, output_path)
        assert result.success

        # Then load
        state_dict = load_safetensors(output_path)

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        assert all(isinstance(v, torch.Tensor) for v in state_dict.values())

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_safetensors("/nonexistent/model.safetensors")

    def test_load_to_device(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test loading to specific device."""
        output_path = temp_dir / "model.safetensors"

        result = convert_to_safetensors(valid_checkpoint, output_path)
        assert result.success

        state_dict = load_safetensors(output_path, device="cpu")

        for tensor in state_dict.values():
            assert tensor.device.type == "cpu"


class TestTensorTypes:
    """Tests for handling different tensor types."""

    def test_different_dtypes(self, temp_dir: Path) -> None:
        """Test conversion of different tensor dtypes."""
        checkpoint_path = temp_dir / "dtypes.pt"
        output_path = temp_dir / "dtypes.safetensors"

        # Create checkpoint with various dtypes
        state_dict = {
            "float32": torch.randn(10, 10, dtype=torch.float32),
            "float16": torch.randn(10, 10, dtype=torch.float16),
            "int64": torch.randint(0, 100, (10, 10), dtype=torch.int64),
            "bool": torch.randint(0, 2, (10, 10), dtype=torch.bool),
        }
        torch.save(state_dict, checkpoint_path)

        result = convert_to_safetensors(checkpoint_path, output_path)

        assert result.success
        assert result.num_tensors == 4

        # Verify dtypes preserved
        loaded = load_safetensors(output_path)
        assert loaded["float32"].dtype == torch.float32
        assert loaded["float16"].dtype == torch.float16
        assert loaded["int64"].dtype == torch.int64
        assert loaded["bool"].dtype == torch.bool

    def test_different_shapes(self, temp_dir: Path) -> None:
        """Test conversion of different tensor shapes."""
        checkpoint_path = temp_dir / "shapes.pt"
        output_path = temp_dir / "shapes.safetensors"

        state_dict = {
            "1d": torch.randn(100),
            "2d": torch.randn(10, 10),
            "3d": torch.randn(4, 10, 10),
            "4d": torch.randn(2, 4, 10, 10),
        }
        torch.save(state_dict, checkpoint_path)

        result = convert_to_safetensors(checkpoint_path, output_path)

        assert result.success

        loaded = load_safetensors(output_path)
        assert loaded["1d"].shape == (100,)
        assert loaded["2d"].shape == (10, 10)
        assert loaded["3d"].shape == (4, 10, 10)
        assert loaded["4d"].shape == (2, 4, 10, 10)
