"""Validation framework for AlphaGalerkin next steps.

This module provides a configuration-driven validation framework for:
1. Test tolerance/precision issue resolution
2. GPU training validation
3. Zero-shot transfer verification
4. PR merge readiness checking

Design Principles:
    - No hardcoded values: All thresholds and parameters are configurable
    - Backwards compatible: New fields have defaults, deprecated fields are handled
    - Reusable: Components can be used independently or composed
    - Observable: Comprehensive logging and debugging support
"""

from src.validation.config import (
    GPUTrainingConfig,
    MergeReadinessConfig,
    ToleranceConfig,
    TransferValidationConfig,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from src.validation.runner import ValidationRunner
from src.validation.tolerance import (
    ToleranceChecker,
    assert_allclose,
    assert_tensor_allclose,
    get_tolerance_for_dtype,
)

__all__ = [
    # Configuration
    "ValidationConfig",
    "ToleranceConfig",
    "GPUTrainingConfig",
    "TransferValidationConfig",
    "MergeReadinessConfig",
    "ValidationResult",
    "ValidationStatus",
    # Tolerance helpers
    "ToleranceChecker",
    "assert_allclose",
    "assert_tensor_allclose",
    "get_tolerance_for_dtype",
    # Runner
    "ValidationRunner",
]
