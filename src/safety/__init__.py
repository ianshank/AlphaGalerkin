"""Safety module for AlphaGalerkin checkpoint validation and secure loading.

This module provides defense-in-depth mechanisms for handling potentially
untrusted model checkpoints:

- Static analysis of pickle opcodes
- Sandboxed deserialization with RestrictedUnpickler
- Schema validation of state dictionaries
- Cryptographic hash verification
- SafeTensors format conversion

Example:
    from src.safety import validate_checkpoint, CheckpointValidator
    from src.safety.config import ValidationConfig

    # Quick validation
    result = validate_checkpoint("model.pt")
    if result.valid:
        print(f"Checkpoint is safe: {result.checkpoint_hash}")

    # Full validation with config
    config = ValidationConfig(
        name="production",
        max_file_size_gb=10.0,
        require_hash_verification=True,
    )
    validator = CheckpointValidator(config)
    result = validator.validate("model.pt", expected_hash="abc123...")

"""

from src.safety.config import (
    AllowlistConfig,
    ValidationConfig,
    ValidationLevel,
)
from src.safety.converter import (
    SafeTensorsConverter,
    convert_to_safetensors,
    load_safetensors,
)
from src.safety.validator import (
    CheckpointValidator,
    ValidationResult,
    validate_checkpoint,
)

__all__ = [
    # Config
    "ValidationConfig",
    "ValidationLevel",
    "AllowlistConfig",
    # Validator
    "CheckpointValidator",
    "ValidationResult",
    "validate_checkpoint",
    # Converter
    "SafeTensorsConverter",
    "convert_to_safetensors",
    "load_safetensors",
]
