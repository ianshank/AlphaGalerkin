"""Pytest fixtures for validation tests."""

from __future__ import annotations

import pytest

from src.validation.config import (
    GPUTrainingConfig,
    MergeReadinessConfig,
    ToleranceConfig,
    ToleranceLevel,
    TransferValidationConfig,
    ValidationConfig,
)
from src.validation.tolerance import ToleranceChecker


@pytest.fixture
def tolerance_config() -> ToleranceConfig:
    """Provide default tolerance configuration."""
    return ToleranceConfig(level=ToleranceLevel.STANDARD)


@pytest.fixture
def relaxed_tolerance_config() -> ToleranceConfig:
    """Provide relaxed tolerance configuration."""
    return ToleranceConfig(level=ToleranceLevel.RELAXED)


@pytest.fixture
def strict_tolerance_config() -> ToleranceConfig:
    """Provide strict tolerance configuration."""
    return ToleranceConfig(level=ToleranceLevel.STRICT)


@pytest.fixture
def tolerance_checker(tolerance_config: ToleranceConfig) -> ToleranceChecker:
    """Provide tolerance checker with standard config."""
    return ToleranceChecker(config=tolerance_config)


@pytest.fixture
def gpu_training_config() -> GPUTrainingConfig:
    """Provide GPU training configuration for testing."""
    return GPUTrainingConfig(
        device="cpu",  # Use CPU for testing
        require_gpu=False,
        n_steps=10,
        batch_size=4,
        d_model=32,
        n_heads=2,
        n_layers=2,
        log_interval=5,
    )


@pytest.fixture
def transfer_config() -> TransferValidationConfig:
    """Provide transfer validation configuration for testing."""
    return TransferValidationConfig(
        train_resolution=5,
        eval_resolutions=[5, 7],
        primary_eval_resolution=7,
        n_train_samples=100,
        n_eval_samples=20,
        n_epochs=5,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )


@pytest.fixture
def merge_config() -> MergeReadinessConfig:
    """Provide merge readiness configuration for testing."""
    return MergeReadinessConfig(
        pr_number=7,
        require_all_tests_pass=False,  # Don't run actual tests in unit tests
        require_lint_pass=False,
        require_type_check_pass=False,
    )


@pytest.fixture
def validation_config(
    tolerance_config: ToleranceConfig,
    gpu_training_config: GPUTrainingConfig,
    transfer_config: TransferValidationConfig,
    merge_config: MergeReadinessConfig,
) -> ValidationConfig:
    """Provide full validation configuration for testing."""
    return ValidationConfig(
        run_tolerance_fix=True,
        run_gpu_training=False,  # Skip by default in tests
        run_transfer_validation=False,
        run_merge_readiness=False,
        tolerance=tolerance_config,
        gpu_training=gpu_training_config,
        transfer_validation=transfer_config,
        merge_readiness=merge_config,
        output_dir="outputs/test_validation",
        save_results=False,
    )
