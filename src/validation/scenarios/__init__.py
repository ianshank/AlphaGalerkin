"""Validation scenarios for AlphaGalerkin next steps.

This module provides reusable validation scenarios for:
1. GPU training validation
2. Zero-shot transfer verification
3. PR merge readiness checking
"""

from src.validation.scenarios.gpu_training import GPUTrainingValidator
from src.validation.scenarios.merge_readiness import MergeReadinessChecker
from src.validation.scenarios.tolerance_fixer import ToleranceTestFixer
from src.validation.scenarios.transfer import TransferValidator

__all__ = [
    "GPUTrainingValidator",
    "TransferValidator",
    "MergeReadinessChecker",
    "ToleranceTestFixer",
]
