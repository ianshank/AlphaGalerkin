"""Tests for the safety.config module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.safety.config import (
    AllowlistConfig,
    ConverterConfig,
    ValidationConfig,
    ValidationLevel,
    get_permissive_config,
    get_standard_config,
    get_strict_config,
)


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert ValidationLevel.PERMISSIVE.value == "permissive"
        assert ValidationLevel.STANDARD.value == "standard"
        assert ValidationLevel.STRICT.value == "strict"

    def test_enum_membership(self) -> None:
        """Test enum membership."""
        assert len(ValidationLevel) == 3


class TestAllowlistConfig:
    """Tests for AllowlistConfig."""

    def test_default_torch_modules(self) -> None:
        """Test default torch modules are present."""
        config = AllowlistConfig(name="test")
        assert "torch.FloatTensor" in config.torch_modules
        assert "torch._utils._rebuild_tensor_v2" in config.torch_modules

    def test_default_numpy_modules(self) -> None:
        """Test default numpy modules are present."""
        config = AllowlistConfig(name="test")
        assert "numpy.ndarray" in config.numpy_modules
        assert "numpy.dtype" in config.numpy_modules

    def test_default_stdlib_modules(self) -> None:
        """Test default stdlib modules are present."""
        config = AllowlistConfig(name="test")
        assert "collections.OrderedDict" in config.stdlib_modules

    def test_default_denylist(self) -> None:
        """Test default denylist contains dangerous functions."""
        config = AllowlistConfig(name="test")
        assert "os.system" in config.custom_denylist
        assert "subprocess.Popen" in config.custom_denylist
        assert "builtins.eval" in config.custom_denylist

    def test_get_allowlist_set(self) -> None:
        """Test allowlist set generation."""
        config = AllowlistConfig(name="test")
        allowlist = config.get_allowlist_set()

        assert isinstance(allowlist, set)
        assert ("torch", "FloatTensor") in allowlist
        assert ("numpy", "ndarray") in allowlist
        assert ("collections", "OrderedDict") in allowlist

    def test_get_denylist_set(self) -> None:
        """Test denylist set generation."""
        config = AllowlistConfig(name="test")
        denylist = config.get_denylist_set()

        assert isinstance(denylist, set)
        assert ("os", "system") in denylist
        assert ("builtins", "eval") in denylist

    def test_custom_allowlist(self) -> None:
        """Test adding custom modules to allowlist."""
        config = AllowlistConfig(
            name="test",
            custom_allowlist=["my_module.MyClass", "other.Thing"],
        )
        allowlist = config.get_allowlist_set()

        assert ("my_module", "MyClass") in allowlist
        assert ("other", "Thing") in allowlist

    def test_custom_denylist(self) -> None:
        """Test adding custom modules to denylist."""
        config = AllowlistConfig(
            name="test",
            custom_denylist=["bad.Evil", "os.system"],
        )
        denylist = config.get_denylist_set()

        assert ("bad", "Evil") in denylist
        assert ("os", "system") in denylist

    def test_is_allowed_torch(self) -> None:
        """Test is_allowed for torch modules."""
        config = AllowlistConfig(name="test")

        assert config.is_allowed("torch", "FloatTensor")
        assert config.is_allowed("torch._utils", "_rebuild_tensor_v2")

    def test_is_allowed_denied(self) -> None:
        """Test is_allowed returns False for denied modules."""
        config = AllowlistConfig(name="test")

        assert not config.is_allowed("os", "system")
        assert not config.is_allowed("builtins", "eval")

    def test_is_allowed_unknown(self) -> None:
        """Test is_allowed returns False for unknown modules."""
        config = AllowlistConfig(name="test")

        assert not config.is_allowed("unknown", "module")
        assert not config.is_allowed("malicious", "code")

    def test_denylist_takes_precedence(self) -> None:
        """Test that denylist takes precedence over allowlist."""
        config = AllowlistConfig(
            name="test",
            custom_allowlist=["danger.Evil"],
            custom_denylist=["danger.Evil"],
        )

        # Even though it's in allowlist, denylist should take precedence
        assert not config.is_allowed("danger", "Evil")


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_required_name(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError):
            ValidationConfig()  # type: ignore[call-arg]

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ValidationConfig(name="test")

        assert config.level == ValidationLevel.STANDARD
        assert config.max_file_size_gb == 50.0
        assert config.max_tensor_size_gb == 10.0
        assert config.require_hash_verification is False
        assert config.check_nan_inf is True

    def test_level_validation(self) -> None:
        """Test that level must be valid enum."""
        config = ValidationConfig(name="test", level=ValidationLevel.STRICT)
        assert config.level == ValidationLevel.STRICT

    def test_max_file_size_constraints(self) -> None:
        """Test max_file_size_gb constraints."""
        # Valid values
        config = ValidationConfig(name="test", max_file_size_gb=100.0)
        assert config.max_file_size_gb == 100.0

        # Too small
        with pytest.raises(ValidationError):
            ValidationConfig(name="test", max_file_size_gb=0.0)

        # Too large
        with pytest.raises(ValidationError):
            ValidationConfig(name="test", max_file_size_gb=2000.0)

    def test_max_tensor_size_constraints(self) -> None:
        """Test max_tensor_size_gb constraints."""
        config = ValidationConfig(name="test", max_tensor_size_gb=5.0)
        assert config.max_tensor_size_gb == 5.0

        with pytest.raises(ValidationError):
            ValidationConfig(name="test", max_tensor_size_gb=0.0)

    def test_tensor_size_cannot_exceed_file_size(self) -> None:
        """Test that tensor size can't exceed file size."""
        with pytest.raises(ValidationError) as exc_info:
            ValidationConfig(
                name="test",
                max_file_size_gb=10.0,
                max_tensor_size_gb=20.0,
            )

        assert "cannot exceed" in str(exc_info.value)

    def test_expected_keys(self) -> None:
        """Test expected_keys configuration."""
        config = ValidationConfig(
            name="test",
            expected_keys=["model_state_dict", "step", "config"],
        )
        assert config.expected_keys == ["model_state_dict", "step", "config"]

    def test_allowed_dtypes_default(self) -> None:
        """Test default allowed dtypes."""
        config = ValidationConfig(name="test")

        assert "float32" in config.allowed_dtypes
        assert "float16" in config.allowed_dtypes
        assert "int64" in config.allowed_dtypes
        assert "bool" in config.allowed_dtypes

    def test_allowed_dtypes_validation(self) -> None:
        """Test that invalid dtypes are rejected."""
        with pytest.raises(ValidationError):
            ValidationConfig(
                name="test",
                allowed_dtypes=["float32", "unknown_dtype"],
            )

    def test_sandbox_timeout_constraints(self) -> None:
        """Test sandbox_timeout_seconds constraints."""
        config = ValidationConfig(name="test", sandbox_timeout_seconds=60)
        assert config.sandbox_timeout_seconds == 60

        with pytest.raises(ValidationError):
            ValidationConfig(name="test", sandbox_timeout_seconds=0)

        with pytest.raises(ValidationError):
            ValidationConfig(name="test", sandbox_timeout_seconds=10000)

    def test_allowlist_config_nested(self) -> None:
        """Test that allowlist config is properly nested."""
        config = ValidationConfig(name="test")
        assert isinstance(config.allowlist, AllowlistConfig)
        assert config.allowlist.name == "default"


class TestConverterConfig:
    """Tests for ConverterConfig."""

    def test_default_values(self) -> None:
        """Test default converter configuration."""
        config = ConverterConfig(name="test")

        assert config.include_metadata is True
        assert config.verify_roundtrip is True
        assert config.compression is None
        assert config.backup_original is True

    def test_tolerance_values(self) -> None:
        """Test tolerance configuration."""
        config = ConverterConfig(
            name="test",
            tolerance_atol=1e-5,
            tolerance_rtol=1e-4,
        )

        assert config.tolerance_atol == 1e-5
        assert config.tolerance_rtol == 1e-4

    def test_compression_validation(self) -> None:
        """Test compression algorithm validation."""
        # Valid compression
        config = ConverterConfig(name="test", compression="lz4")
        assert config.compression == "lz4"

        config = ConverterConfig(name="test", compression="zstd")
        assert config.compression == "zstd"

        # Invalid compression
        with pytest.raises(ValidationError):
            ConverterConfig(name="test", compression="gzip")


class TestConfigPresets:
    """Tests for configuration preset functions."""

    def test_get_permissive_config(self) -> None:
        """Test permissive config preset."""
        config = get_permissive_config()

        assert config.level == ValidationLevel.PERMISSIVE
        assert config.max_file_size_gb == 100.0
        assert config.check_nan_inf is False

    def test_get_standard_config(self) -> None:
        """Test standard config preset."""
        config = get_standard_config()

        assert config.level == ValidationLevel.STANDARD
        assert config.max_file_size_gb == 50.0
        assert config.check_nan_inf is True

    def test_get_strict_config(self) -> None:
        """Test strict config preset."""
        config = get_strict_config()

        assert config.level == ValidationLevel.STRICT
        assert config.max_file_size_gb == 10.0
        assert config.max_tensor_size_gb == 5.0
        assert config.check_nan_inf is True
        assert config.require_hash_verification is True

    def test_preset_custom_name(self) -> None:
        """Test that presets accept custom names."""
        config = get_strict_config(name="my_strict")
        assert config.name == "my_strict"


class TestConfigHashing:
    """Tests for configuration hashing."""

    def test_compute_hash_deterministic(self) -> None:
        """Test that hash is deterministic."""
        config1 = ValidationConfig(name="test", seed=42)
        config2 = ValidationConfig(name="test", seed=42)

        assert config1.compute_hash() == config2.compute_hash()

    def test_compute_hash_different_values(self) -> None:
        """Test that different values produce different hashes."""
        config1 = ValidationConfig(name="test", max_file_size_gb=10.0)
        config2 = ValidationConfig(name="test", max_file_size_gb=20.0)

        assert config1.compute_hash() != config2.compute_hash()

    def test_with_overrides(self) -> None:
        """Test creating config with overrides."""
        config = ValidationConfig(name="test", max_file_size_gb=10.0)
        new_config = config.with_overrides(max_file_size_gb=20.0)

        assert config.max_file_size_gb == 10.0
        assert new_config.max_file_size_gb == 20.0
