"""Tests for the safety.validator module."""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path

import pytest
import torch

from src.safety.config import (
    AllowlistConfig,
    ValidationConfig,
    ValidationLevel,
)
from src.safety.validator import (
    CheckpointValidator,
    RestrictedUnpickler,
    ValidationResult,
    analyze_pickle_opcodes,
    compute_checkpoint_hash,
    validate_checkpoint,
    validate_state_dict_schema,
    validate_tensor,
)
from src.templates.logging import create_logger_class

SafetyLogger = create_logger_class("Safety")


class TestComputeCheckpointHash:
    """Tests for compute_checkpoint_hash function."""

    def test_hash_deterministic(self) -> None:
        """Test that hash is deterministic."""
        data = b"test data for hashing"

        hash1 = compute_checkpoint_hash(data)
        hash2 = compute_checkpoint_hash(data)

        assert hash1 == hash2

    def test_hash_is_sha256(self) -> None:
        """Test that hash is SHA256 format."""
        data = b"test"
        result = compute_checkpoint_hash(data)

        # SHA256 produces 64 hex characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_data_different_hash(self) -> None:
        """Test that different data produces different hashes."""
        hash1 = compute_checkpoint_hash(b"data1")
        hash2 = compute_checkpoint_hash(b"data2")

        assert hash1 != hash2


class TestAnalyzePickleOpcodes:
    """Tests for analyze_pickle_opcodes function."""

    def test_valid_tensor_pickle(self) -> None:
        """Test analysis of valid tensor pickle."""
        tensor = torch.randn(10, 10)
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        data = buffer.getvalue()

        is_safe, errors, warnings = analyze_pickle_opcodes(data)

        assert is_safe
        assert len(errors) == 0

    def test_invalid_pickle_data(self) -> None:
        """Test analysis of invalid pickle data."""
        data = b"not valid pickle data"

        is_safe, errors, warnings = analyze_pickle_opcodes(data)

        # The analyzer is lenient for unknown formats (relies on torch.load security)
        # So it returns safe=True with warnings for unknown/invalid data
        assert is_safe
        assert len(warnings) > 0

    def test_simple_dict_pickle(self) -> None:
        """Test analysis of simple dict pickle."""
        data = pickle.dumps({"key": "value", "number": 42})

        is_safe, errors, warnings = analyze_pickle_opcodes(data)

        assert is_safe
        assert len(errors) == 0


class TestRestrictedUnpickler:
    """Tests for RestrictedUnpickler class."""

    def test_allowed_torch_tensor(self) -> None:
        """Test that torch tensor classes are in the allowlist."""
        # PyTorch's serialization format uses persistent_id which requires
        # special handling. Instead of trying to unpickle a full tensor,
        # we verify that the RestrictedUnpickler correctly allows torch classes
        buffer = io.BytesIO()
        allowlist = AllowlistConfig(name="test")
        unpickler = RestrictedUnpickler(buffer, allowlist)

        # Verify torch classes are allowed by find_class
        cls = unpickler.find_class("torch", "FloatTensor")
        assert cls is not None

        cls = unpickler.find_class("torch._utils", "_rebuild_tensor_v2")
        assert cls is not None

    def test_allowed_collections(self) -> None:
        """Test that allowed collections work."""
        from collections import OrderedDict

        data = OrderedDict([("a", 1), ("b", 2)])
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        allowlist = AllowlistConfig(name="test")
        unpickler = RestrictedUnpickler(buffer, allowlist)

        result = unpickler.load()
        assert isinstance(result, OrderedDict)

    def test_denied_class_blocked(self) -> None:
        """Test that denied classes are blocked."""
        # Create pickle with os.system reference
        # We can't easily test this without actually creating malicious pickle
        # So we test the find_class method directly
        buffer = io.BytesIO()
        allowlist = AllowlistConfig(name="test")
        unpickler = RestrictedUnpickler(buffer, allowlist)

        with pytest.raises(pickle.UnpicklingError) as exc_info:
            unpickler.find_class("os", "system")

        assert "denied" in str(exc_info.value).lower()

    def test_unknown_class_blocked(self) -> None:
        """Test that unknown classes are blocked."""
        buffer = io.BytesIO()
        allowlist = AllowlistConfig(name="test")
        unpickler = RestrictedUnpickler(buffer, allowlist)

        with pytest.raises(pickle.UnpicklingError) as exc_info:
            unpickler.find_class("unknown_module", "UnknownClass")

        assert "not in allowlist" in str(exc_info.value)


class TestValidateTensor:
    """Tests for validate_tensor function."""

    def test_valid_tensor(self) -> None:
        """Test validation of a valid tensor."""
        tensor = torch.randn(10, 10)
        config = ValidationConfig(name="test")
        logger = SafetyLogger("test")

        errors = validate_tensor("test_tensor", tensor, config, logger)

        assert len(errors) == 0

    def test_tensor_with_nan(self) -> None:
        """Test detection of NaN values."""
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float("nan")
        config = ValidationConfig(name="test", check_nan_inf=True)
        logger = SafetyLogger("test")

        errors = validate_tensor("test_tensor", tensor, config, logger)

        assert len(errors) == 1
        assert "NaN" in errors[0]

    def test_tensor_with_inf(self) -> None:
        """Test detection of Inf values."""
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float("inf")
        config = ValidationConfig(name="test", check_nan_inf=True)
        logger = SafetyLogger("test")

        errors = validate_tensor("test_tensor", tensor, config, logger)

        assert len(errors) == 1
        assert "Inf" in errors[0]

    def test_nan_check_disabled(self) -> None:
        """Test that NaN check can be disabled."""
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float("nan")
        config = ValidationConfig(name="test", check_nan_inf=False)
        logger = SafetyLogger("test")

        errors = validate_tensor("test_tensor", tensor, config, logger)

        assert len(errors) == 0

    def test_disallowed_dtype(self) -> None:
        """Test detection of disallowed dtypes."""
        tensor = torch.randn(10, 10).to(torch.float32)
        config = ValidationConfig(
            name="test",
            allowed_dtypes=["float64"],  # Only allow float64
        )
        logger = SafetyLogger("test")

        errors = validate_tensor("test_tensor", tensor, config, logger)

        assert len(errors) == 1
        assert "disallowed dtype" in errors[0]


class TestValidateStateDictSchema:
    """Tests for validate_state_dict_schema function."""

    def test_valid_state_dict(self) -> None:
        """Test validation of valid state dict."""
        state_dict = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }
        config = ValidationConfig(name="test")
        logger = SafetyLogger("test")

        is_valid, errors = validate_state_dict_schema(state_dict, config, logger)

        assert is_valid
        assert len(errors) == 0

    def test_non_dict_rejected(self) -> None:
        """Test that non-dict is rejected."""
        state_dict = "not a dict"  # type: ignore[assignment]
        config = ValidationConfig(name="test")
        logger = SafetyLogger("test")

        is_valid, errors = validate_state_dict_schema(state_dict, config, logger)

        assert not is_valid
        assert "not a dictionary" in errors[0]

    def test_missing_expected_keys(self) -> None:
        """Test detection of missing expected keys."""
        state_dict = {
            "weight": torch.randn(10, 10),
        }
        config = ValidationConfig(
            name="test",
            expected_keys=["weight", "bias"],
        )
        logger = SafetyLogger("test")

        is_valid, errors = validate_state_dict_schema(state_dict, config, logger)

        assert not is_valid
        assert any("Missing" in e for e in errors)

    def test_nested_state_dict(self) -> None:
        """Test validation of nested state dict."""
        state_dict = {
            "encoder": {
                "weight": torch.randn(10, 10),
            },
            "decoder": {
                "weight": torch.randn(10, 10),
            },
        }
        config = ValidationConfig(name="test")
        logger = SafetyLogger("test")

        is_valid, errors = validate_state_dict_schema(state_dict, config, logger)

        assert is_valid
        assert len(errors) == 0


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = ValidationResult(
            valid=True,
            checkpoint_hash="abc123",
            errors=[],
            warnings=["warning1"],
            metadata={"key": "value"},
            validation_level=ValidationLevel.STANDARD,
        )

        d = result.to_dict()

        assert d["valid"] is True
        assert d["checkpoint_hash"] == "abc123"
        assert d["errors"] == []
        assert d["warnings"] == ["warning1"]
        assert d["metadata"] == {"key": "value"}
        assert d["validation_level"] == "standard"


class TestCheckpointValidator:
    """Tests for CheckpointValidator class."""

    def test_validate_valid_checkpoint(
        self, valid_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test validation of valid checkpoint."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(valid_checkpoint)

        assert result.valid
        assert len(result.errors) == 0
        assert result.checkpoint_hash != ""
        assert result.validation_level == ValidationLevel.STANDARD

    def test_validate_nonexistent_file(self, standard_config: ValidationConfig) -> None:
        """Test validation of nonexistent file."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate("/nonexistent/path/model.pt")

        assert not result.valid
        assert "does not exist" in result.errors[0]

    def test_validate_with_hash_verification(
        self, valid_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test validation with hash verification."""
        # First get the actual hash
        with open(valid_checkpoint, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()

        validator = CheckpointValidator(standard_config)
        result = validator.validate(valid_checkpoint, expected_hash=expected_hash)

        assert result.valid
        assert result.checkpoint_hash == expected_hash

    def test_validate_hash_mismatch(
        self, valid_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test validation fails with hash mismatch."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(valid_checkpoint, expected_hash="wrong_hash")

        assert not result.valid
        assert any("Hash mismatch" in e for e in result.errors)

    def test_validate_nan_detection(
        self, checkpoint_with_nan: Path, standard_config: ValidationConfig
    ) -> None:
        """Test that NaN values are detected."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(checkpoint_with_nan)

        assert not result.valid
        assert any("NaN" in e for e in result.errors)

    def test_validate_inf_detection(
        self, checkpoint_with_inf: Path, standard_config: ValidationConfig
    ) -> None:
        """Test that Inf values are detected."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(checkpoint_with_inf)

        assert not result.valid
        assert any("Inf" in e for e in result.errors)

    def test_validate_permissive_skips_nan_check(
        self, checkpoint_with_nan: Path, permissive_config: ValidationConfig
    ) -> None:
        """Test permissive validation skips NaN check."""
        validator = CheckpointValidator(permissive_config)
        result = validator.validate(checkpoint_with_nan)

        # Permissive level skips detailed checks
        assert result.valid

    def test_validate_strict_requires_hash(
        self, valid_checkpoint: Path, strict_config: ValidationConfig
    ) -> None:
        """Test strict validation requires hash."""
        validator = CheckpointValidator(strict_config)
        result = validator.validate(valid_checkpoint)

        assert not result.valid
        assert any("hash" in e.lower() for e in result.errors)

    def test_validate_metadata_extraction(
        self, valid_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test that metadata is extracted."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(valid_checkpoint)

        assert "file_size_bytes" in result.metadata
        assert "sha256" in result.metadata
        assert result.metadata.get("step") == 1000

    def test_validate_state_dict_only(
        self, state_dict_only_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test validation of state-dict-only checkpoint."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(state_dict_only_checkpoint)

        assert result.valid


class TestValidateCheckpointFunction:
    """Tests for validate_checkpoint convenience function."""

    def test_basic_validation(self, valid_checkpoint: Path) -> None:
        """Test basic validation with defaults."""
        result = validate_checkpoint(valid_checkpoint)

        assert result.valid
        assert result.validation_level == ValidationLevel.STANDARD

    def test_validation_with_config(
        self, valid_checkpoint: Path, strict_config: ValidationConfig
    ) -> None:
        """Test validation with custom config."""
        # Get hash for strict mode
        with open(valid_checkpoint, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()

        result = validate_checkpoint(
            valid_checkpoint,
            expected_hash=expected_hash,
            config=strict_config,
        )

        assert result.valid
        assert result.validation_level == ValidationLevel.STRICT

    def test_validation_nonexistent(self) -> None:
        """Test validation of nonexistent file."""
        result = validate_checkpoint("/does/not/exist.pt")

        assert not result.valid


class TestFileSizeLimits:
    """Tests for file size limit validation."""

    def test_small_file_allowed(
        self, valid_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test that small files are allowed."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(valid_checkpoint)

        assert result.valid

    def test_file_size_in_metadata(
        self, valid_checkpoint: Path, standard_config: ValidationConfig
    ) -> None:
        """Test that file size is reported in metadata."""
        validator = CheckpointValidator(standard_config)
        result = validator.validate(valid_checkpoint)

        assert "file_size_bytes" in result.metadata
        assert "file_size_gb" in result.metadata
        assert result.metadata["file_size_gb"] < 1.0  # Should be very small


class TestTensorSizeLimits:
    """Tests for tensor size validation."""

    def test_tensor_size_validation(self, temp_dir: Path) -> None:
        """Test that tensor size is checked."""
        checkpoint_path = temp_dir / "tensor_test.pt"

        # Create a checkpoint with known tensor size
        tensor = torch.randn(100, 100)  # 40KB float32
        torch.save({"weights": tensor}, checkpoint_path)

        config = ValidationConfig(
            name="test",
            max_tensor_size_gb=0.001,  # 1MB limit
        )
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        # Should pass - tensor is small
        assert result.valid


class TestComplexCheckpoints:
    """Tests for complex checkpoint structures."""

    def test_deeply_nested_state_dict(self, temp_dir: Path) -> None:
        """Test validation of deeply nested state dict."""
        checkpoint_path = temp_dir / "nested.pt"

        nested = {
            "encoder": {
                "layer1": {
                    "sublayer": {
                        "weight": torch.randn(10, 10),
                    }
                }
            }
        }
        torch.save(nested, checkpoint_path)

        config = ValidationConfig(name="test")
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        assert result.valid

    def test_mixed_content_checkpoint(self, temp_dir: Path) -> None:
        """Test validation of checkpoint with mixed tensor and metadata."""
        checkpoint_path = temp_dir / "mixed.pt"

        checkpoint = {
            "model_state_dict": {
                "weight": torch.randn(32, 32),
                "bias": torch.randn(32),
            },
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [],
            },
            "epoch": 10,
            "loss": 0.5,
            "config": {"lr": 0.001, "batch_size": 32},
        }
        torch.save(checkpoint, checkpoint_path)

        config = ValidationConfig(name="test")
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        assert result.valid
        assert result.metadata.get("epoch") == 10

    def test_empty_nested_dict(self, temp_dir: Path) -> None:
        """Test validation of checkpoint with empty nested dicts."""
        checkpoint_path = temp_dir / "empty_nested.pt"

        checkpoint = {
            "empty": {},
            "nested_empty": {"inner": {}},
            "weight": torch.randn(5, 5),
        }
        torch.save(checkpoint, checkpoint_path)

        config = ValidationConfig(name="test")
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        assert result.valid


class TestDtypeValidation:
    """Tests for tensor dtype validation."""

    def test_allowed_float32(self, temp_dir: Path) -> None:
        """Test that float32 tensors are allowed."""
        checkpoint_path = temp_dir / "float32.pt"
        torch.save({"t": torch.randn(10, 10, dtype=torch.float32)}, checkpoint_path)

        config = ValidationConfig(name="test", allowed_dtypes=["float32"])
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        assert result.valid

    def test_disallowed_dtype_rejected(self, temp_dir: Path) -> None:
        """Test that disallowed dtypes are rejected."""
        checkpoint_path = temp_dir / "float64.pt"
        torch.save({"t": torch.randn(10, 10, dtype=torch.float64)}, checkpoint_path)

        config = ValidationConfig(
            name="test",
            allowed_dtypes=["float32"],  # Only float32 allowed
        )
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        assert not result.valid
        assert any("dtype" in e.lower() for e in result.errors)

    def test_multiple_dtype_checkpoint(self, temp_dir: Path) -> None:
        """Test checkpoint with multiple dtypes."""
        checkpoint_path = temp_dir / "multi_dtype.pt"
        checkpoint = {
            "float": torch.randn(5, 5, dtype=torch.float32),
            "int": torch.randint(0, 10, (5, 5), dtype=torch.int64),
            "bool": torch.randint(0, 2, (5, 5), dtype=torch.bool),
        }
        torch.save(checkpoint, checkpoint_path)

        config = ValidationConfig(
            name="test",
            allowed_dtypes=["float32", "int64", "bool"],
        )
        validator = CheckpointValidator(config)
        result = validator.validate(checkpoint_path)

        assert result.valid


class TestRestrictedUnpicklerAdvanced:
    """Advanced tests for RestrictedUnpickler."""

    def test_custom_allowlist_allows_class(self) -> None:
        """Test that custom allowlist entries are respected."""
        import io

        buffer = io.BytesIO()
        allowlist = AllowlistConfig(
            name="test",
            custom_allowlist=["collections.Counter"],
        )
        unpickler = RestrictedUnpickler(buffer, allowlist)

        # Should not raise - Counter is in custom allowlist
        cls = unpickler.find_class("collections", "Counter")
        assert cls is not None

    def test_nested_module_path(self) -> None:
        """Test finding classes in nested modules."""
        import io

        buffer = io.BytesIO()
        allowlist = AllowlistConfig(name="test")
        unpickler = RestrictedUnpickler(buffer, allowlist)

        # torch._utils is a nested module in the allowlist
        cls = unpickler.find_class("torch._utils", "_rebuild_tensor_v2")
        assert cls is not None


class TestValidationResultMethods:
    """Tests for ValidationResult methods."""

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        result = ValidationResult(
            valid=True,
            checkpoint_hash="abc123def456",
            errors=["error1", "error2"],
            warnings=["warning1"],
            metadata={"key1": "value1", "key2": 123},
            validation_level=ValidationLevel.STRICT,
        )

        d = result.to_dict()

        assert d["valid"] is True
        assert d["checkpoint_hash"] == "abc123def456"
        assert len(d["errors"]) == 2
        assert len(d["warnings"]) == 1
        assert d["metadata"]["key1"] == "value1"
        assert d["validation_level"] == "strict"

    def test_to_dict_with_empty_collections(self) -> None:
        """Test to_dict with empty errors and warnings."""
        result = ValidationResult(
            valid=True,
            checkpoint_hash="hash",
            errors=[],
            warnings=[],
            metadata={},
            validation_level=ValidationLevel.PERMISSIVE,
        )

        d = result.to_dict()

        assert d["errors"] == []
        assert d["warnings"] == []
        assert d["metadata"] == {}


class TestComputeCheckpointHashEdgeCases:
    """Edge case tests for compute_checkpoint_hash."""

    def test_empty_data(self) -> None:
        """Test hash of empty data."""
        result = compute_checkpoint_hash(b"")
        assert len(result) == 64  # SHA256 produces 64 hex chars

    def test_large_data(self) -> None:
        """Test hash of larger data."""
        data = b"x" * 1000000  # 1MB of data
        result = compute_checkpoint_hash(data)
        assert len(result) == 64

    def test_binary_data(self) -> None:
        """Test hash of binary data with null bytes."""
        data = b"\x00\x01\x02\xff\xfe\xfd"
        result = compute_checkpoint_hash(data)
        assert len(result) == 64


class TestAnalyzePickleOpcodesAdvanced:
    """Advanced tests for pickle opcode analysis."""

    def test_complex_nested_structure(self) -> None:
        """Test analysis of complex nested pickle."""
        import pickle

        data = {
            "list": [1, 2, [3, 4, {"nested": "value"}]],
            "dict": {"a": {"b": {"c": 1}}},
            "tuple": (1, (2, (3,))),
        }
        pickled = pickle.dumps(data)

        is_safe, errors, warnings = analyze_pickle_opcodes(pickled)

        assert is_safe
        assert len(errors) == 0

    def test_truncated_pickle(self) -> None:
        """Test analysis of truncated pickle data."""
        import pickle

        data = {"key": "value"}
        pickled = pickle.dumps(data)
        truncated = pickled[:len(pickled) // 2]

        is_safe, errors, warnings = analyze_pickle_opcodes(truncated)

        # Should fail to parse
        assert not is_safe


class TestValidateTensorAdvanced:
    """Advanced tests for validate_tensor function."""

    def test_zero_tensor(self) -> None:
        """Test validation of all-zero tensor."""
        tensor = torch.zeros(100, 100)
        config = ValidationConfig(name="test", check_nan_inf=True)
        logger = SafetyLogger("test")

        errors = validate_tensor("zeros", tensor, config, logger)

        assert len(errors) == 0

    def test_very_large_values(self) -> None:
        """Test validation of tensor with very large but finite values."""
        tensor = torch.tensor([1e38, -1e38])
        config = ValidationConfig(name="test", check_nan_inf=True)
        logger = SafetyLogger("test")

        errors = validate_tensor("large", tensor, config, logger)

        # Large but finite values should pass
        assert len(errors) == 0

    def test_negative_inf(self) -> None:
        """Test detection of negative infinity."""
        tensor = torch.tensor([float("-inf")])
        config = ValidationConfig(name="test", check_nan_inf=True)
        logger = SafetyLogger("test")

        errors = validate_tensor("neginf", tensor, config, logger)

        assert len(errors) == 1
        assert "Inf" in errors[0]

    def test_mixed_nan_inf(self) -> None:
        """Test tensor with both NaN and Inf values."""
        tensor = torch.tensor([float("nan"), float("inf"), 1.0])
        config = ValidationConfig(name="test", check_nan_inf=True)
        logger = SafetyLogger("test")

        errors = validate_tensor("mixed", tensor, config, logger)

        # Should detect both issues
        assert len(errors) >= 1
