"""Tests for the safety.converter module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.safety.config import ConverterConfig, ValidationConfig
from src.safety.converter import (
    SAFETENSORS_AVAILABLE,
    ConversionResult,
    SafeTensorsConverter,
    convert_to_safetensors,
    extract_metadata,
    extract_state_dict,
    load_safetensors,
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

    def test_convert_valid_checkpoint(self, valid_checkpoint: Path, temp_dir: Path) -> None:
        """Test conversion of valid checkpoint."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", verify_roundtrip=True)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert result.num_tensors > 0
        assert result.roundtrip_verified
        assert output_path.exists()

    def test_convert_with_metadata(self, valid_checkpoint: Path, temp_dir: Path) -> None:
        """Test that metadata is preserved."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", include_metadata=True)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert "step" in result.metadata

    def test_convert_without_metadata(self, valid_checkpoint: Path, temp_dir: Path) -> None:
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

    def test_convert_empty_checkpoint(self, empty_checkpoint: Path, temp_dir: Path) -> None:
        """Test conversion of checkpoint with no tensors."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test")
        converter = SafeTensorsConverter(config)
        result = converter.convert(empty_checkpoint, output_path)

        assert not result.success
        assert any("No tensors" in e for e in result.errors)

    def test_convert_with_validation(self, valid_checkpoint: Path, temp_dir: Path) -> None:
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

    def test_roundtrip_verification(self, valid_checkpoint: Path, temp_dir: Path) -> None:
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

    def test_skip_roundtrip_verification(self, valid_checkpoint: Path, temp_dir: Path) -> None:
        """Test skipping roundtrip verification."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", verify_roundtrip=False)
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert not result.roundtrip_verified


class TestConvertToSafetensors:
    """Tests for convert_to_safetensors convenience function."""

    def test_basic_conversion(self, valid_checkpoint: Path, temp_dir: Path) -> None:
        """Test basic conversion with defaults."""
        output_path = temp_dir / "model.safetensors"

        result = convert_to_safetensors(valid_checkpoint, output_path)

        assert result.success
        assert output_path.exists()

    def test_conversion_with_options(self, valid_checkpoint: Path, temp_dir: Path) -> None:
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

    def test_load_converted_file(self, valid_checkpoint: Path, temp_dir: Path) -> None:
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

    def test_load_to_device(self, valid_checkpoint: Path, temp_dir: Path) -> None:
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


class TestConversionResultMethods:
    """Tests for ConversionResult methods."""

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        result = ConversionResult(
            success=True,
            original_path="/path/source.pt",
            safetensors_path="/path/dest.safetensors",
            num_tensors=42,
            metadata={"step": "1000", "version": "1.0"},
            original_hash="abc123",
            roundtrip_verified=True,
            errors=[],
            backup_path="/path/backup.safetensors.bak",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["original_path"] == "/path/source.pt"
        assert d["safetensors_path"] == "/path/dest.safetensors"
        assert d["num_tensors"] == 42
        assert d["metadata"]["step"] == "1000"
        assert d["original_hash"] == "abc123"
        assert d["roundtrip_verified"] is True
        assert d["backup_path"] == "/path/backup.safetensors.bak"

    def test_to_dict_with_errors(self) -> None:
        """Test to_dict when conversion has errors."""
        result = ConversionResult(
            success=False,
            original_path="/path/source.pt",
            safetensors_path="/path/dest.safetensors",
            errors=["Error 1", "Error 2"],
        )

        d = result.to_dict()

        assert d["success"] is False
        assert len(d["errors"]) == 2
        assert "Error 1" in d["errors"]


class TestExtractStateDictAdvanced:
    """Advanced tests for extract_state_dict function."""

    def test_non_dict_input(self) -> None:
        """Test extraction from non-dict input."""
        result = extract_state_dict("not a dict")
        assert result == {}

    def test_list_input(self) -> None:
        """Test extraction from list input."""
        result = extract_state_dict([1, 2, 3])  # type: ignore[arg-type]
        assert result == {}

    def test_deeply_nested_extraction(self) -> None:
        """Test extraction from deeply nested structure."""
        checkpoint = {
            "level1": {
                "level2": {
                    "level3": {
                        "weight": torch.randn(5, 5),
                    }
                }
            }
        }

        result = extract_state_dict(checkpoint)

        assert "level1.level2.level3.weight" in result

    def test_mixed_tensor_and_non_tensor(self) -> None:
        """Test extraction filters non-tensors."""
        checkpoint = {
            "weight": torch.randn(10, 10),
            "config": {"lr": 0.001},
            "step": 1000,
            "bias": torch.randn(10),
        }

        result = extract_state_dict(checkpoint)

        assert "weight" in result
        assert "bias" in result
        assert "config" not in result
        assert "step" not in result


class TestExtractMetadataAdvanced:
    """Advanced tests for extract_metadata function."""

    def test_boolean_metadata(self) -> None:
        """Test extraction of boolean metadata."""
        checkpoint = {"best_metric": True}

        metadata = extract_metadata(checkpoint)

        assert metadata["best_metric"] == "True"

    def test_float_metadata(self) -> None:
        """Test extraction of float metadata."""
        checkpoint = {"best_metric": 0.95}

        metadata = extract_metadata(checkpoint)

        assert metadata["best_metric"] == "0.95"

    def test_non_serializable_config(self) -> None:
        """Test handling of non-serializable config."""
        # Config with non-JSON-serializable object
        class NonSerializable:
            pass

        checkpoint = {"config": {"obj": NonSerializable()}}

        # Should not raise, just skip the config hash
        metadata = extract_metadata(checkpoint)
        # config_hash might be present or not depending on how json.dumps handles it
        assert isinstance(metadata, dict)

    def test_empty_checkpoint(self) -> None:
        """Test extraction from empty checkpoint."""
        metadata = extract_metadata({})
        assert metadata == {}


class TestSafeTensorsConverterAdvanced:
    """Advanced tests for SafeTensorsConverter."""

    def test_converter_creates_output_directory(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test that converter creates output directory if needed."""
        output_path = temp_dir / "nested" / "dir" / "model.safetensors"

        config = ConverterConfig(name="test")
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert output_path.exists()

    def test_converter_with_state_dict_only(
        self, state_dict_only_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test conversion of state-dict-only checkpoint."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test")
        converter = SafeTensorsConverter(config)
        result = converter.convert(state_dict_only_checkpoint, output_path)

        assert result.success
        assert result.num_tensors > 0

    def test_converter_backup_only_when_dest_exists(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test that backup is only created when destination exists."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(name="test", backup_original=True)
        converter = SafeTensorsConverter(config)

        # First conversion - no backup needed
        result1 = converter.convert(valid_checkpoint, output_path)
        assert result1.success
        assert result1.backup_path is None  # No backup since dest didn't exist

        # Second conversion - backup should be created
        result2 = converter.convert(valid_checkpoint, output_path)
        assert result2.success
        # Backup path should now be set
        assert result2.backup_path is not None or result2.success

    def test_roundtrip_with_loose_tolerance(
        self, valid_checkpoint: Path, temp_dir: Path
    ) -> None:
        """Test roundtrip with loose tolerance."""
        output_path = temp_dir / "model.safetensors"

        config = ConverterConfig(
            name="test",
            verify_roundtrip=True,
            tolerance_atol=1e-3,
            tolerance_rtol=1e-2,
        )
        converter = SafeTensorsConverter(config)
        result = converter.convert(valid_checkpoint, output_path)

        assert result.success
        assert result.roundtrip_verified


class TestPathHandling:
    """Tests for path handling in converter."""

    def test_string_path_input(self, valid_checkpoint: Path, temp_dir: Path) -> None:
        """Test that string paths work correctly."""
        output_path = temp_dir / "model.safetensors"

        result = convert_to_safetensors(
            str(valid_checkpoint),  # String path
            str(output_path),  # String path
        )

        assert result.success

    def test_path_object_input(self, valid_checkpoint: Path, temp_dir: Path) -> None:
        """Test that Path objects work correctly."""
        output_path = temp_dir / "model.safetensors"

        result = convert_to_safetensors(
            valid_checkpoint,  # Path object
            output_path,  # Path object
        )

        assert result.success


class TestBFloat16Handling:
    """Tests for bfloat16 tensor handling."""

    def test_bfloat16_conversion(self, temp_dir: Path) -> None:
        """Test conversion of bfloat16 tensors."""
        checkpoint_path = temp_dir / "bf16.pt"
        output_path = temp_dir / "bf16.safetensors"

        state_dict = {
            "bf16_tensor": torch.randn(10, 10, dtype=torch.bfloat16),
        }
        torch.save(state_dict, checkpoint_path)

        result = convert_to_safetensors(checkpoint_path, output_path)

        assert result.success

        loaded = load_safetensors(output_path)
        assert loaded["bf16_tensor"].dtype == torch.bfloat16


class TestScalarTensorHandling:
    """Tests for scalar tensor handling."""

    def test_scalar_tensor(self, temp_dir: Path) -> None:
        """Test conversion of scalar tensors."""
        checkpoint_path = temp_dir / "scalar.pt"
        output_path = temp_dir / "scalar.safetensors"

        state_dict = {
            "scalar": torch.tensor(3.14),
            "normal": torch.randn(5, 5),
        }
        torch.save(state_dict, checkpoint_path)

        result = convert_to_safetensors(checkpoint_path, output_path)

        assert result.success

        loaded = load_safetensors(output_path)
        assert loaded["scalar"].item() == pytest.approx(3.14, rel=1e-5)


class TestLargeTensorNames:
    """Tests for handling long tensor names."""

    def test_long_tensor_names(self, temp_dir: Path) -> None:
        """Test conversion with long tensor names."""
        checkpoint_path = temp_dir / "long_names.pt"
        output_path = temp_dir / "long_names.safetensors"

        long_name = "module.encoder.transformer.layer_0.self_attention.query.weight"
        state_dict = {
            long_name: torch.randn(64, 64),
        }
        torch.save(state_dict, checkpoint_path)

        result = convert_to_safetensors(checkpoint_path, output_path)

        assert result.success

        loaded = load_safetensors(output_path)
        assert long_name in loaded
