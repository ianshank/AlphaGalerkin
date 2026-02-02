"""SafeTensors conversion utilities for secure checkpoint serialization.

SafeTensors provides memory-safe serialization without arbitrary code
execution risks. This module provides utilities for converting between
PyTorch and SafeTensors formats.

Example:
    from src.safety.converter import SafeTensorsConverter, convert_to_safetensors

    # Quick conversion
    result = convert_to_safetensors("model.pt", "model.safetensors")
    print(f"Converted {result['num_tensors']} tensors")

    # Full converter with config
    from src.safety.config import ConverterConfig

    config = ConverterConfig(name="prod", verify_roundtrip=True)
    converter = SafeTensorsConverter(config)
    result = converter.convert("model.pt", "model.safetensors")

"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.safety.config import ConverterConfig, ValidationConfig
from src.safety.validator import CheckpointValidator, ValidationResult
from src.templates.logging import BaseModuleLogger, create_logger_class

# Create module-specific logger class
_SafetyLoggerClass = create_logger_class("Safety")


# Optional SafeTensors import
try:
    from safetensors.torch import load_file, save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

    def load_file(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Stub for safetensors.load_file when not installed."""
        raise ImportError("safetensors not installed. Run: pip install safetensors")

    def save_file(*args: Any, **kwargs: Any) -> None:
        """Stub for safetensors.save_file when not installed."""
        raise ImportError("safetensors not installed. Run: pip install safetensors")


@dataclass
class ConversionResult:
    """Result of SafeTensors conversion.

    Attributes:
        success: Whether conversion succeeded.
        original_path: Path to original checkpoint.
        safetensors_path: Path to SafeTensors output.
        num_tensors: Number of tensors converted.
        metadata: Metadata preserved in SafeTensors.
        original_hash: SHA256 hash of original file.
        roundtrip_verified: Whether roundtrip verification passed.
        errors: List of errors encountered.
        backup_path: Path to backup file (if created).

    """

    success: bool
    original_path: str
    safetensors_path: str
    num_tensors: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    original_hash: str = ""
    roundtrip_verified: bool = False
    errors: list[str] = field(default_factory=list)
    backup_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "original_path": self.original_path,
            "safetensors_path": self.safetensors_path,
            "num_tensors": self.num_tensors,
            "metadata": self.metadata,
            "original_hash": self.original_hash,
            "roundtrip_verified": self.roundtrip_verified,
            "errors": self.errors,
            "backup_path": self.backup_path,
        }


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract state dict from various checkpoint formats.

    Handles common checkpoint structures:
    - Direct state dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - Nested structures

    Args:
        checkpoint: Loaded checkpoint (dict or state dict).

    Returns:
        Flattened state dictionary with tensor values.

    """
    if not isinstance(checkpoint, dict):
        return {}

    # Try common keys
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Filter to only tensors
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value
        elif isinstance(value, dict):
            # Flatten nested dicts
            nested = extract_state_dict(value)
            for nested_key, nested_value in nested.items():
                result[f"{key}.{nested_key}"] = nested_value

    return result


def extract_metadata(checkpoint: dict[str, Any]) -> dict[str, str]:
    """Extract serializable metadata from checkpoint.

    SafeTensors metadata must be string key-value pairs.

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        Dictionary with string values suitable for SafeTensors.

    """
    metadata: dict[str, str] = {}

    # Common metadata fields
    metadata_fields = [
        "step",
        "version",
        "epoch",
        "global_step",
        "best_metric",
        "timestamp",
    ]

    for field_name in metadata_fields:
        if field_name in checkpoint:
            value = checkpoint[field_name]
            # Convert to string
            if isinstance(value, (int, float, str, bool)):
                metadata[field_name] = str(value)

    # Add config hash if present
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        import hashlib
        import json

        try:
            config_str = json.dumps(checkpoint["config"], sort_keys=True, default=str)
            metadata["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except (TypeError, ValueError):
            pass

    return metadata


class SafeTensorsConverter:
    """Converter for PyTorch checkpoints to SafeTensors format.

    SafeTensors provides:
    - No arbitrary code execution during loading
    - Memory-mapped loading for efficiency
    - Cross-framework compatibility

    Example:
        config = ConverterConfig(name="prod", verify_roundtrip=True)
        converter = SafeTensorsConverter(config)
        result = converter.convert("model.pt", "model.safetensors")

    """

    def __init__(
        self,
        config: ConverterConfig | None = None,
        validation_config: ValidationConfig | None = None,
        logger: BaseModuleLogger | None = None,
    ) -> None:
        """Initialize converter with configuration.

        Args:
            config: Converter configuration.
            validation_config: Optional validation config for source checkpoint.
            logger: Optional logger instance.

        """
        self.config = config or ConverterConfig(name="default")
        self.validation_config = validation_config
        self.logger = logger or _SafetyLoggerClass(
            "converter",
            config_name=self.config.name,
        )

        if not SAFETENSORS_AVAILABLE:
            self.logger.warning(
                "safetensors_not_available",
                message="SafeTensors not installed. Install with: pip install safetensors",
            )

    def convert(
        self,
        source_path: str | Path,
        dest_path: str | Path,
    ) -> ConversionResult:
        """Convert a PyTorch checkpoint to SafeTensors format.

        Args:
            source_path: Path to source PyTorch checkpoint.
            dest_path: Path for output SafeTensors file.

        Returns:
            ConversionResult with conversion status and details.

        """
        source = Path(source_path)
        dest = Path(dest_path)
        errors: list[str] = []

        self.logger.info(
            "conversion_started",
            source=str(source),
            dest=str(dest),
        )

        if not SAFETENSORS_AVAILABLE:
            return ConversionResult(
                success=False,
                original_path=str(source),
                safetensors_path=str(dest),
                errors=["SafeTensors not installed"],
            )

        # Validate source checkpoint first
        validation_result: ValidationResult | None = None
        if self.validation_config:
            validator = CheckpointValidator(self.validation_config)
            validation_result = validator.validate(source)
            if not validation_result.valid:
                self.logger.error(
                    "source_validation_failed",
                    errors=validation_result.errors,
                )
                return ConversionResult(
                    success=False,
                    original_path=str(source),
                    safetensors_path=str(dest),
                    errors=["Source validation failed"] + validation_result.errors,
                    original_hash=validation_result.checkpoint_hash,
                )

        # Create backup if configured
        backup_path: str | None = None
        if self.config.backup_original and dest.exists():
            backup_path = str(dest.with_suffix(".safetensors.bak"))
            try:
                shutil.copy2(dest, backup_path)
                self.logger.debug("backup_created", path=backup_path)
            except Exception as e:
                self.logger.warning("backup_failed", error=str(e))

        # Load source checkpoint
        try:
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        except Exception:
            # Fall back to full load if weights_only fails
            try:
                checkpoint = torch.load(source, map_location="cpu")
            except Exception as e:
                self.logger.error("load_failed", error=str(e))
                return ConversionResult(
                    success=False,
                    original_path=str(source),
                    safetensors_path=str(dest),
                    errors=[f"Failed to load checkpoint: {e}"],
                )

        # Extract state dict and metadata
        state_dict = extract_state_dict(checkpoint)
        metadata: dict[str, str] = {}

        if self.config.include_metadata and isinstance(checkpoint, dict):
            metadata = extract_metadata(checkpoint)

        if not state_dict:
            self.logger.error("no_tensors_found")
            return ConversionResult(
                success=False,
                original_path=str(source),
                safetensors_path=str(dest),
                errors=["No tensors found in checkpoint"],
            )

        self.logger.debug(
            "extracted_state_dict",
            num_tensors=len(state_dict),
            metadata_keys=list(metadata.keys()),
        )

        # Save as SafeTensors
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_file(state_dict, dest, metadata=metadata if metadata else None)
            self.logger.debug("safetensors_saved", path=str(dest))
        except Exception as e:
            self.logger.error("save_failed", error=str(e))
            return ConversionResult(
                success=False,
                original_path=str(source),
                safetensors_path=str(dest),
                errors=[f"Failed to save SafeTensors: {e}"],
                backup_path=backup_path,
            )

        # Verify roundtrip if configured
        roundtrip_verified = False
        if self.config.verify_roundtrip:
            try:
                loaded = load_file(dest, device="cpu")
                roundtrip_verified = self._verify_roundtrip(state_dict, loaded)
                if not roundtrip_verified:
                    errors.append("Roundtrip verification failed")
                    self.logger.error("roundtrip_verification_failed")
                else:
                    self.logger.debug("roundtrip_verified")
            except Exception as e:
                errors.append(f"Roundtrip verification error: {e}")
                self.logger.error("roundtrip_error", error=str(e))

        # Get original hash
        original_hash = ""
        if validation_result:
            original_hash = validation_result.checkpoint_hash

        success = len(errors) == 0
        self.logger.info(
            "conversion_complete",
            success=success,
            num_tensors=len(state_dict),
        )

        return ConversionResult(
            success=success,
            original_path=str(source),
            safetensors_path=str(dest),
            num_tensors=len(state_dict),
            metadata=metadata,
            original_hash=original_hash,
            roundtrip_verified=roundtrip_verified,
            errors=errors,
            backup_path=backup_path,
        )

    def _verify_roundtrip(
        self,
        original: dict[str, torch.Tensor],
        loaded: dict[str, torch.Tensor],
    ) -> bool:
        """Verify that loaded tensors match original.

        Args:
            original: Original state dict.
            loaded: Loaded state dict from SafeTensors.

        Returns:
            True if all tensors match within tolerance.

        """
        if set(original.keys()) != set(loaded.keys()):
            self.logger.warning(
                "key_mismatch",
                original_keys=len(original),
                loaded_keys=len(loaded),
            )
            return False

        for key in original:
            orig_tensor = original[key]
            load_tensor = loaded[key]

            if orig_tensor.shape != load_tensor.shape:
                self.logger.warning(
                    "shape_mismatch",
                    key=key,
                    original=orig_tensor.shape,
                    loaded=load_tensor.shape,
                )
                return False

            if orig_tensor.dtype != load_tensor.dtype:
                self.logger.warning(
                    "dtype_mismatch",
                    key=key,
                    original=str(orig_tensor.dtype),
                    loaded=str(load_tensor.dtype),
                )
                return False

            # Check values with tolerance
            if not torch.allclose(
                orig_tensor.float(),
                load_tensor.float(),
                atol=self.config.tolerance_atol,
                rtol=self.config.tolerance_rtol,
            ):
                self.logger.warning(
                    "value_mismatch",
                    key=key,
                )
                return False

        return True


def convert_to_safetensors(
    source_path: str | Path,
    dest_path: str | Path,
    include_metadata: bool = True,
    verify: bool = True,
) -> ConversionResult:
    """Convenience function for SafeTensors conversion.

    Args:
        source_path: Path to source PyTorch checkpoint.
        dest_path: Path for output SafeTensors file.
        include_metadata: Whether to include checkpoint metadata.
        verify: Whether to verify conversion with roundtrip test.

    Returns:
        ConversionResult with conversion status.

    Example:
        result = convert_to_safetensors("model.pt", "model.safetensors")
        if result.success:
            print(f"Converted {result.num_tensors} tensors")

    """
    config = ConverterConfig(
        name="quick_convert",
        include_metadata=include_metadata,
        verify_roundtrip=verify,
    )
    converter = SafeTensorsConverter(config)
    return converter.convert(source_path, dest_path)


def load_safetensors(
    path: str | Path,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a SafeTensors checkpoint safely.

    This is a thin wrapper around safetensors.torch.load_file
    with logging.

    Args:
        path: Path to SafeTensors file.
        device: Device to load tensors to.

    Returns:
        State dictionary with tensors.

    Raises:
        ImportError: If safetensors is not installed.
        FileNotFoundError: If file doesn't exist.

    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not installed. Run: pip install safetensors")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SafeTensors file not found: {path}")

    logger = _SafetyLoggerClass("loader")
    logger.debug("loading_safetensors", path=str(path), device=device)

    state_dict: dict[str, torch.Tensor] = load_file(path, device=device)

    logger.debug(
        "loaded_safetensors",
        num_tensors=len(state_dict),
    )

    return state_dict
