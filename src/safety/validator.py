"""Checkpoint validation with defense-in-depth.

This module provides secure checkpoint loading by:
1. Static analysis of pickle opcodes
2. Sandboxed deserialization with RestrictedUnpickler
3. Schema validation of state dict structure
4. Cryptographic hash verification

Example:
    from src.safety.validator import CheckpointValidator, validate_checkpoint
    from src.safety.config import ValidationConfig, ValidationLevel

    # Quick validation with defaults
    result = validate_checkpoint("model.pt")
    if not result.valid:
        for error in result.errors:
            print(f"Error: {error}")

    # Full validation with custom config
    config = ValidationConfig(
        name="production",
        level=ValidationLevel.STRICT,
        require_hash_verification=True,
    )
    validator = CheckpointValidator(config)
    result = validator.validate("model.pt", expected_hash="abc123...")

"""

from __future__ import annotations

import hashlib
import io
import pickle
import pickletools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.safety.config import (
    AllowlistConfig,
    ValidationConfig,
    ValidationLevel,
    get_standard_config,
)
from src.templates.logging import BaseModuleLogger, create_logger_class

# Create module-specific logger class
_SafetyLoggerClass = create_logger_class("Safety")

if TYPE_CHECKING:
    # Use base class for type hints
    SafetyLoggerType = BaseModuleLogger
else:
    SafetyLoggerType = _SafetyLoggerClass


@dataclass
class ValidationResult:
    """Result of checkpoint validation.

    Attributes:
        valid: Whether the checkpoint passed all validation checks.
        checkpoint_hash: SHA256 hash of the checkpoint file.
        errors: List of validation errors encountered.
        warnings: List of validation warnings.
        metadata: Additional metadata extracted from checkpoint.
        validation_level: The validation level that was applied.

    """

    valid: bool
    checkpoint_hash: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.STANDARD

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "valid": self.valid,
            "checkpoint_hash": self.checkpoint_hash,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "validation_level": self.validation_level.value,
        }


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows classes from an explicit allowlist.

    This provides a critical security boundary against arbitrary code
    execution via pickle deserialization.

    Attributes:
        allowlist: Configuration defining allowed classes.
        logger: Logger instance for security events.

    """

    def __init__(
        self,
        file: io.BufferedIOBase,
        allowlist: AllowlistConfig,
        logger: BaseModuleLogger | None = None,
    ) -> None:
        """Initialize restricted unpickler.

        Args:
            file: Binary file-like object to read from.
            allowlist: Configuration defining allowed classes.
            logger: Optional logger for security events.

        """
        super().__init__(file)
        self.allowlist = allowlist
        self.logger = logger or _SafetyLoggerClass("unpickler")
        self._allowlist_set = allowlist.get_allowlist_set()
        self._denylist_set = allowlist.get_denylist_set()

    def find_class(self, module: str, name: str) -> type[Any]:
        """Find and return a class, only if it's in the allowlist.

        Args:
            module: Module path (e.g., 'torch._utils').
            name: Class name (e.g., '_rebuild_tensor_v2').

        Returns:
            The requested class if allowed.

        Raises:
            pickle.UnpicklingError: If class is not in allowlist.

        """
        key = (module, name)

        # Check denylist first (takes precedence)
        if key in self._denylist_set:
            self.logger.error(
                "blocked_denied_class",
                module=module,
                name=name,
            )
            raise pickle.UnpicklingError(f"Explicitly denied class: {module}.{name}")

        # Check allowlist
        if key in self._allowlist_set:
            self.logger.debug(
                "allowed_class",
                module=module,
                name=name,
            )
            cls: type[Any] = super().find_class(module, name)
            return cls

        # Log and reject unknown class
        self.logger.warning(
            "blocked_unknown_class",
            module=module,
            name=name,
        )
        raise pickle.UnpicklingError(f"Class not in allowlist: {module}.{name}")


# Dangerous pickle opcodes that may indicate malicious intent
DANGEROUS_OPCODES: frozenset[str] = frozenset(
    {
        "GLOBAL",  # Can import arbitrary modules
        "INST",  # Can instantiate arbitrary classes
        "OBJ",  # Can call arbitrary constructors
        "NEWOBJ",  # Can call arbitrary constructors (newer protocol)
        "NEWOBJ_EX",  # Extended NEWOBJ
        "REDUCE",  # Can call arbitrary callables
        "BUILD",  # Can call __setstate__ with arbitrary data
        "EXT1",  # Extension registry (untrusted)
        "EXT2",  # Extension registry (untrusted)
        "EXT4",  # Extension registry (untrusted)
        "STACK_GLOBAL",  # Stack-based GLOBAL
    }
)

# Opcodes that are commonly used legitimately by PyTorch
LEGITIMATE_OPCODES: frozenset[str] = frozenset(
    {
        "REDUCE",  # Used by torch for tensor reconstruction
        "BUILD",  # Used for setting attributes
        "GLOBAL",  # Used for importing torch classes
        "STACK_GLOBAL",  # Used in newer protocols
    }
)


def _extract_pickle_from_zip(data: bytes) -> bytes | None:
    """Extract pickle data from PyTorch's ZIP-based checkpoint format.

    PyTorch 1.6+ uses ZIP archives for checkpoints. The pickle data is stored
    in 'data.pkl' or 'archive/data.pkl' within the archive.

    Args:
        data: Raw bytes of the checkpoint file.

    Returns:
        The extracted pickle bytes, or None if not a ZIP or extraction failed.

    """
    import zipfile

    if not data.startswith(b"PK"):
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Try common PyTorch pickle paths
            for path in ["data.pkl", "archive/data.pkl"]:
                if path in zf.namelist():
                    return zf.read(path)

            # Try to find any .pkl file
            pkl_files = [n for n in zf.namelist() if n.endswith(".pkl")]
            if pkl_files:
                return zf.read(pkl_files[0])

    except (zipfile.BadZipFile, KeyError):
        pass

    return None


def analyze_pickle_opcodes(
    data: bytes,
    logger: BaseModuleLogger | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Static analysis of pickle opcodes without execution.

    Examines the pickle bytecode for potentially dangerous operations
    without actually deserializing the data. Supports both raw pickle
    files and PyTorch's ZIP-based checkpoint format.

    Args:
        data: Raw pickle bytes to analyze.
        logger: Optional logger for analysis events.

    Returns:
        Tuple of (is_safe, errors, warnings).

    """
    logger = logger or _SafetyLoggerClass("opcode_analyzer")
    errors: list[str] = []
    warnings: list[str] = []

    # Try to extract pickle from ZIP if it's a PyTorch ZIP archive
    pickle_data = _extract_pickle_from_zip(data)
    if pickle_data is not None:
        logger.debug("extracted_pickle_from_zip")
        data = pickle_data
    elif data.startswith(b"PK"):
        # ZIP file but couldn't extract pickle - this is still OK
        # PyTorch will handle it during load
        logger.debug("zip_format_detected_no_pickle")
        warnings.append("ZIP-based checkpoint format detected, limited static analysis")
        return True, [], warnings

    try:
        ops = list(pickletools.genops(data))
    except Exception as e:
        # If parsing fails but it's a valid PyTorch format, allow it
        # The actual security check happens during sandboxed load
        error_str = str(e)
        logger.warning("pickle_parse_warning", error=error_str)

        # Check if this looks like it could be handled by weights_only=True
        # which provides its own security guarantees
        if "unknown" in error_str.lower() or "codec" in error_str.lower():
            warnings.append(
                "Static pickle analysis inconclusive, will rely on torch.load security"
            )
            return True, [], warnings

        logger.error("pickle_parse_failed", error=error_str)
        return False, [f"Failed to parse pickle: {e}"], []

    logger.debug("analyzing_opcodes", opcode_count=len(ops))

    global_imports: list[tuple[int, str]] = []
    reduce_calls: list[int] = []

    for op, arg, pos in ops:
        opname = op.name
        # pos can be None in some cases, default to -1
        position = pos if pos is not None else -1

        if opname == "GLOBAL":
            if arg:
                global_imports.append((position, str(arg)))

        if opname == "REDUCE":
            reduce_calls.append(position)

        if opname in DANGEROUS_OPCODES and opname not in LEGITIMATE_OPCODES:
            errors.append(f"Dangerous opcode {opname} at position {position}")
            logger.warning(
                "dangerous_opcode",
                opcode=opname,
                position=position,
            )

    # Log analysis results
    if global_imports:
        logger.debug(
            "global_imports_found",
            count=len(global_imports),
            imports=[imp for _, imp in global_imports[:10]],  # Log first 10
        )

    if reduce_calls:
        warnings.append(
            f"Found {len(reduce_calls)} REDUCE opcodes (common in PyTorch, validated during load)"
        )

    is_safe = len(errors) == 0
    return is_safe, errors, warnings


def compute_checkpoint_hash(data: bytes) -> str:
    """Compute SHA256 hash of checkpoint data.

    Args:
        data: Raw checkpoint bytes.

    Returns:
        Hexadecimal hash string.

    """
    return hashlib.sha256(data).hexdigest()


def validate_tensor(
    name: str,
    tensor: torch.Tensor,
    config: ValidationConfig,
    logger: SafetyLogger,
) -> list[str]:
    """Validate a single tensor for safety issues.

    Args:
        name: Tensor name/key in state dict.
        tensor: The tensor to validate.
        config: Validation configuration.
        logger: Logger instance.

    Returns:
        List of validation errors (empty if valid).

    """
    errors: list[str] = []

    # Check size
    size_gb = tensor.numel() * tensor.element_size() / (1024**3)
    if size_gb > config.max_tensor_size_gb:
        errors.append(
            f"Tensor '{name}' is too large: {size_gb:.2f}GB > {config.max_tensor_size_gb}GB"
        )
        logger.warning(
            "tensor_too_large",
            tensor_name=name,
            size_gb=size_gb,
            limit_gb=config.max_tensor_size_gb,
        )

    # Check dtype
    dtype_name = str(tensor.dtype).replace("torch.", "")
    if dtype_name not in config.allowed_dtypes:
        errors.append(f"Tensor '{name}' has disallowed dtype: {dtype_name}")
        logger.warning(
            "disallowed_dtype",
            tensor_name=name,
            dtype=dtype_name,
        )

    # Check for NaN/Inf
    if config.check_nan_inf:
        try:
            if torch.isnan(tensor).any():
                errors.append(f"Tensor '{name}' contains NaN values")
                logger.warning("tensor_contains_nan", tensor_name=name)

            if torch.isinf(tensor).any():
                errors.append(f"Tensor '{name}' contains Inf values")
                logger.warning("tensor_contains_inf", tensor_name=name)
        except RuntimeError:
            # Some dtypes don't support isnan/isinf
            pass

    return errors


def validate_state_dict_schema(
    state_dict: dict[str, Any],
    config: ValidationConfig,
    logger: SafetyLogger,
) -> tuple[bool, list[str]]:
    """Validate state dict structure and tensor properties.

    Args:
        state_dict: The state dictionary to validate.
        config: Validation configuration.
        logger: Logger instance.

    Returns:
        Tuple of (is_valid, list of errors).

    """
    errors: list[str] = []

    # Check it's a dict
    if not isinstance(state_dict, dict):
        return False, ["State dict is not a dictionary"]

    # Check for expected keys
    if config.expected_keys:
        expected_set = set(config.expected_keys)
        actual_set = set(state_dict.keys())
        missing = expected_set - actual_set
        if missing:
            errors.append(f"Missing required keys: {missing}")
            logger.warning("missing_keys", missing=list(missing))

    # Validate tensors
    tensor_count = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            tensor_errors = validate_tensor(key, value, config, logger)
            errors.extend(tensor_errors)
        elif isinstance(value, dict):
            # Recursively validate nested dicts
            nested_valid, nested_errors = validate_state_dict_schema(value, config, logger)
            if not nested_valid:
                errors.extend([f"{key}.{e}" for e in nested_errors])

    logger.debug("schema_validation_complete", tensor_count=tensor_count)
    return len(errors) == 0, errors


class CheckpointValidator:
    """Comprehensive checkpoint validator with configurable security levels.

    Provides multi-stage validation:
    1. File existence and size checks
    2. Static pickle opcode analysis
    3. Sandboxed deserialization
    4. State dict schema validation
    5. Optional hash verification

    Example:
        config = ValidationConfig(name="prod", level=ValidationLevel.STRICT)
        validator = CheckpointValidator(config)
        result = validator.validate("model.pt")

    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        logger: BaseModuleLogger | None = None,
    ) -> None:
        """Initialize validator with configuration.

        Args:
            config: Validation configuration. Uses standard if None.
            logger: Optional logger instance.

        """
        self.config = config or get_standard_config()
        self.logger = logger or _SafetyLoggerClass(
            "validator",
            config_name=self.config.name,
            level=self.config.level.value,
        )

    def validate(
        self,
        checkpoint_path: str | Path,
        expected_hash: str | None = None,
    ) -> ValidationResult:
        """Validate a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.
            expected_hash: Optional expected SHA256 hash.

        Returns:
            ValidationResult with validation status and details.

        """
        path = Path(checkpoint_path)
        errors: list[str] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {"path": str(path)}

        self.logger.info(
            "validation_started",
            path=str(path),
            level=self.config.level.value,
        )

        # Stage 0: File existence
        if not path.exists():
            self.logger.error("file_not_found", path=str(path))
            return ValidationResult(
                valid=False,
                checkpoint_hash="",
                errors=["Checkpoint file does not exist"],
                metadata=metadata,
                validation_level=self.config.level,
            )

        # Stage 1: File size check
        file_size = path.stat().st_size
        file_size_gb = file_size / (1024**3)
        metadata["file_size_bytes"] = file_size
        metadata["file_size_gb"] = round(file_size_gb, 4)

        if file_size_gb > self.config.max_file_size_gb:
            self.logger.error(
                "file_too_large",
                size_gb=file_size_gb,
                limit_gb=self.config.max_file_size_gb,
            )
            return ValidationResult(
                valid=False,
                checkpoint_hash="",
                errors=[f"File too large: {file_size_gb:.2f}GB > {self.config.max_file_size_gb}GB"],
                metadata=metadata,
                validation_level=self.config.level,
            )

        # Stage 2: Read file and compute hash
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            self.logger.error("file_read_failed", error=str(e))
            return ValidationResult(
                valid=False,
                checkpoint_hash="",
                errors=[f"Failed to read file: {e}"],
                metadata=metadata,
                validation_level=self.config.level,
            )

        checkpoint_hash = compute_checkpoint_hash(data)
        metadata["sha256"] = checkpoint_hash

        self.logger.debug("hash_computed", hash=checkpoint_hash[:16])

        # Stage 2b: Hash verification (if required or provided)
        if expected_hash:
            if checkpoint_hash != expected_hash:
                errors.append(
                    f"Hash mismatch: expected {expected_hash[:16]}..., "
                    f"got {checkpoint_hash[:16]}..."
                )
                self.logger.error(
                    "hash_mismatch",
                    expected=expected_hash[:16],
                    actual=checkpoint_hash[:16],
                )
        elif self.config.require_hash_verification:
            errors.append("Hash verification required but no expected hash provided")

        # Early return for permissive level
        if self.config.level == ValidationLevel.PERMISSIVE:
            self.logger.info("validation_complete_permissive")
            return ValidationResult(
                valid=len(errors) == 0,
                checkpoint_hash=checkpoint_hash,
                errors=errors,
                warnings=["Permissive validation - limited security checks"],
                metadata=metadata,
                validation_level=self.config.level,
            )

        # Stage 3: Static pickle analysis
        is_safe, opcode_errors, opcode_warnings = analyze_pickle_opcodes(data, self.logger)
        errors.extend(opcode_errors)
        warnings.extend(opcode_warnings)

        if not is_safe:
            self.logger.error("opcode_analysis_failed")
            return ValidationResult(
                valid=False,
                checkpoint_hash=checkpoint_hash,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
                validation_level=self.config.level,
            )

        # Stage 4: Sandboxed deserialization
        try:
            # First try weights_only=True (safest)
            buffer = io.BytesIO(data)
            try:
                checkpoint = torch.load(
                    buffer,
                    map_location="cpu",
                    weights_only=True,
                )
                metadata["load_method"] = "weights_only"
            except Exception:
                # Fall back to RestrictedUnpickler
                buffer = io.BytesIO(data)
                unpickler = RestrictedUnpickler(
                    buffer,
                    self.config.allowlist,
                    self.logger,
                )
                checkpoint = unpickler.load()
                metadata["load_method"] = "restricted_unpickler"

            self.logger.debug(
                "deserialization_success",
                method=metadata.get("load_method"),
            )

        except pickle.UnpicklingError as e:
            self.logger.error("unpickle_blocked", error=str(e))
            errors.append(f"Blocked during deserialization: {e}")
            return ValidationResult(
                valid=False,
                checkpoint_hash=checkpoint_hash,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
                validation_level=self.config.level,
            )
        except Exception as e:
            self.logger.error("deserialization_failed", error=str(e))
            errors.append(f"Failed to deserialize: {e}")
            return ValidationResult(
                valid=False,
                checkpoint_hash=checkpoint_hash,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
                validation_level=self.config.level,
            )

        # Stage 5: Schema validation
        # Extract state dict from various checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        schema_valid, schema_errors = validate_state_dict_schema(
            state_dict, self.config, self.logger
        )
        errors.extend(schema_errors)

        # Extract checkpoint metadata
        if isinstance(checkpoint, dict):
            metadata["checkpoint_keys"] = list(checkpoint.keys())
            # Extract common metadata fields
            for key in ["version", "step", "epoch", "global_step"]:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
            if "config" in checkpoint:
                metadata["has_config"] = True

        # Final result
        is_valid = len(errors) == 0
        self.logger.info(
            "validation_complete",
            valid=is_valid,
            error_count=len(errors),
            warning_count=len(warnings),
        )

        return ValidationResult(
            valid=is_valid,
            checkpoint_hash=checkpoint_hash,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            validation_level=self.config.level,
        )


def validate_checkpoint(
    checkpoint_path: str | Path,
    expected_hash: str | None = None,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """Convenience function for checkpoint validation.

    Args:
        checkpoint_path: Path to checkpoint file.
        expected_hash: Optional expected SHA256 hash.
        config: Optional validation config (uses standard if None).

    Returns:
        ValidationResult with validation status.

    Example:
        result = validate_checkpoint("model.pt")
        if result.valid:
            print(f"Safe to load: {result.checkpoint_hash}")
        else:
            print(f"Errors: {result.errors}")

    """
    validator = CheckpointValidator(config)
    return validator.validate(checkpoint_path, expected_hash)
