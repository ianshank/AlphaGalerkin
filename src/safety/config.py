"""Configuration schemas for the safety module.

This module provides Pydantic-validated configuration for checkpoint
validation with no hardcoded values. All security-sensitive parameters
are configurable and validated.

Example:
    from src.safety.config import ValidationConfig, ValidationLevel

    config = ValidationConfig(
        name="production",
        level=ValidationLevel.STRICT,
        max_file_size_gb=10.0,
    )

"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.templates.config import BaseModuleConfig


class ValidationLevel(str, Enum):
    """Validation strictness levels.

    Attributes:
        PERMISSIVE: Basic checks only (file exists, size limits).
        STANDARD: Static analysis + schema validation.
        STRICT: All checks including hash verification.

    """

    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"


class AllowlistConfig(BaseModuleConfig):
    """Configuration for pickle allowlist.

    Defines which modules and classes are safe to unpickle.
    This is the primary defense against arbitrary code execution.

    Attributes:
        torch_modules: Allowed torch module paths.
        numpy_modules: Allowed numpy module paths.
        stdlib_modules: Allowed stdlib module paths.
        custom_allowlist: Additional module.class pairs to allow.
        custom_denylist: Module.class pairs to explicitly deny.

    """

    # PyTorch core tensors and utilities
    torch_modules: list[str] = Field(
        default_factory=lambda: [
            "torch.FloatTensor",
            "torch.LongTensor",
            "torch.IntTensor",
            "torch.DoubleTensor",
            "torch.HalfTensor",
            "torch.BFloat16Tensor",
            "torch.BoolTensor",
            "torch.CharTensor",
            "torch.ShortTensor",
            "torch.ByteTensor",
            "torch._utils._rebuild_tensor_v2",
            "torch._utils._rebuild_parameter",
            "torch._utils._rebuild_parameter_with_state",
            "torch._utils._rebuild_device_tensor_v2",
            "torch.storage._load_from_bytes",
            "torch.storage.TypedStorage",
            "torch.storage.UntypedStorage",
            "torch.nn.parameter.Parameter",
        ],
        description="Allowed torch module.class paths",
    )

    # NumPy support (read-only operations)
    numpy_modules: list[str] = Field(
        default_factory=lambda: [
            "numpy.ndarray",
            "numpy.dtype",
            "numpy.core.multiarray._reconstruct",
            "numpy.core.multiarray.scalar",
            "numpy._core.multiarray._reconstruct",
            "numpy._core.multiarray.scalar",
        ],
        description="Allowed numpy module.class paths",
    )

    # Standard library collections
    stdlib_modules: list[str] = Field(
        default_factory=lambda: [
            "collections.OrderedDict",
            "collections.defaultdict",
            "builtins.dict",
            "builtins.list",
            "builtins.tuple",
            "builtins.set",
            "builtins.frozenset",
        ],
        description="Allowed stdlib module.class paths",
    )

    # User-defined additions
    custom_allowlist: list[str] = Field(
        default_factory=list,
        description="Additional module.class paths to allow",
    )

    # Explicit denials (takes precedence over allowlist)
    custom_denylist: list[str] = Field(
        default_factory=lambda: [
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.run",
            "builtins.eval",
            "builtins.exec",
            "builtins.__import__",
        ],
        description="Module.class paths to explicitly deny",
    )

    def get_allowlist_set(self) -> set[tuple[str, str]]:
        """Get the complete allowlist as module/name tuples.

        Returns:
            Set of (module, name) tuples for allowed classes.

        """
        all_paths = (
            self.torch_modules
            + self.numpy_modules
            + self.stdlib_modules
            + self.custom_allowlist
        )

        result: set[tuple[str, str]] = set()
        for path in all_paths:
            if "." in path:
                parts = path.rsplit(".", 1)
                result.add((parts[0], parts[1]))
            else:
                result.add(("builtins", path))

        return result

    def get_denylist_set(self) -> set[tuple[str, str]]:
        """Get the denylist as module/name tuples.

        Returns:
            Set of (module, name) tuples for denied classes.

        """
        result: set[tuple[str, str]] = set()
        for path in self.custom_denylist:
            if "." in path:
                parts = path.rsplit(".", 1)
                result.add((parts[0], parts[1]))
            else:
                result.add(("builtins", path))

        return result

    def is_allowed(self, module: str, name: str) -> bool:
        """Check if a module.name is allowed.

        Args:
            module: Module path (e.g., 'torch._utils').
            name: Class/function name (e.g., '_rebuild_tensor_v2').

        Returns:
            True if allowed and not denied.

        """
        key = (module, name)

        # Check denylist first (takes precedence)
        if key in self.get_denylist_set():
            return False

        return key in self.get_allowlist_set()


class ValidationConfig(BaseModuleConfig):
    """Configuration for checkpoint validation.

    Attributes:
        level: Validation strictness level.
        max_file_size_gb: Maximum allowed checkpoint file size.
        max_tensor_size_gb: Maximum allowed individual tensor size.
        require_hash_verification: Whether hash verification is mandatory.
        expected_keys: Optional set of required state dict keys.
        allowlist: Pickle allowlist configuration.
        check_nan_inf: Whether to check for NaN/Inf values.
        allowed_dtypes: List of allowed tensor dtypes.
        sandbox_timeout_seconds: Timeout for sandboxed operations.

    """

    level: ValidationLevel = Field(
        default=ValidationLevel.STANDARD,
        description="Validation strictness level",
    )

    max_file_size_gb: float = Field(
        default=50.0,
        gt=0.0,
        le=1000.0,
        description="Maximum checkpoint file size in GB",
    )

    max_tensor_size_gb: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Maximum individual tensor size in GB",
    )

    require_hash_verification: bool = Field(
        default=False,
        description="Whether SHA256 hash verification is mandatory",
    )

    expected_keys: list[str] | None = Field(
        default=None,
        description="Required state dict keys (None to skip check)",
    )

    allowlist: AllowlistConfig = Field(
        default_factory=lambda: AllowlistConfig(name="default"),
        description="Pickle allowlist configuration",
    )

    check_nan_inf: bool = Field(
        default=True,
        description="Check tensors for NaN/Inf values",
    )

    allowed_dtypes: list[str] = Field(
        default_factory=lambda: [
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
            "complex64",
            "complex128",
        ],
        description="Allowed tensor data types",
    )

    sandbox_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Timeout for sandboxed deserialization",
    )

    @field_validator("allowed_dtypes")
    @classmethod
    def validate_dtypes(cls, v: list[str]) -> list[str]:
        """Validate that dtypes are known PyTorch dtypes."""
        known_dtypes = {
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
            "complex64",
            "complex128",
        }
        for dtype in v:
            if dtype not in known_dtypes:
                raise ValueError(f"Unknown dtype: {dtype}. Known: {known_dtypes}")
        return v

    @model_validator(mode="after")
    def validate_size_constraints(self) -> ValidationConfig:
        """Ensure tensor size doesn't exceed file size."""
        if self.max_tensor_size_gb > self.max_file_size_gb:
            raise ValueError(
                f"max_tensor_size_gb ({self.max_tensor_size_gb}) cannot exceed "
                f"max_file_size_gb ({self.max_file_size_gb})"
            )
        return self


class ConverterConfig(BaseModuleConfig):
    """Configuration for SafeTensors conversion.

    Attributes:
        include_metadata: Whether to include checkpoint metadata.
        verify_roundtrip: Whether to verify conversion with roundtrip test.
        compression: Optional compression for SafeTensors file.
        backup_original: Whether to create backup of original file.

    """

    include_metadata: bool = Field(
        default=True,
        description="Include checkpoint metadata in SafeTensors",
    )

    verify_roundtrip: bool = Field(
        default=True,
        description="Verify conversion with roundtrip test",
    )

    compression: str | None = Field(
        default=None,
        description="Compression algorithm (None, 'lz4', 'zstd')",
    )

    backup_original: bool = Field(
        default=True,
        description="Create backup of original file before conversion",
    )

    tolerance_atol: float = Field(
        default=1e-6,
        ge=0.0,
        description="Absolute tolerance for roundtrip verification",
    )

    tolerance_rtol: float = Field(
        default=1e-5,
        ge=0.0,
        description="Relative tolerance for roundtrip verification",
    )

    @field_validator("compression")
    @classmethod
    def validate_compression(cls, v: str | None) -> str | None:
        """Validate compression algorithm."""
        if v is not None and v not in ("lz4", "zstd"):
            raise ValueError(f"Unknown compression: {v}. Supported: lz4, zstd")
        return v


# Pre-defined configuration presets
def get_permissive_config(name: str = "permissive") -> ValidationConfig:
    """Get a permissive validation configuration.

    Use for trusted internal checkpoints only.

    Args:
        name: Configuration name.

    Returns:
        Permissive ValidationConfig.

    """
    return ValidationConfig(
        name=name,
        level=ValidationLevel.PERMISSIVE,
        max_file_size_gb=100.0,
        check_nan_inf=False,
        require_hash_verification=False,
    )


def get_standard_config(name: str = "standard") -> ValidationConfig:
    """Get standard validation configuration.

    Balanced security for most use cases.

    Args:
        name: Configuration name.

    Returns:
        Standard ValidationConfig.

    """
    return ValidationConfig(
        name=name,
        level=ValidationLevel.STANDARD,
        max_file_size_gb=50.0,
        check_nan_inf=True,
        require_hash_verification=False,
    )


def get_strict_config(name: str = "strict") -> ValidationConfig:
    """Get strict validation configuration.

    Maximum security for untrusted checkpoints.

    Args:
        name: Configuration name.

    Returns:
        Strict ValidationConfig.

    """
    return ValidationConfig(
        name=name,
        level=ValidationLevel.STRICT,
        max_file_size_gb=10.0,
        max_tensor_size_gb=5.0,
        check_nan_inf=True,
        require_hash_verification=True,
        sandbox_timeout_seconds=60,
    )
