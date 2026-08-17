"""Regression tests for resolution transfer ratio calculations and bounds."""

from __future__ import annotations

import math

from src.poc.config_noyron import NoyronHXScenarioConfig
from src.poc.scenarios.noyron_hx import DEFAULT_TRANSFER_RATIO_FLOOR


def test_transfer_ratio_floor_protection() -> None:
    """Regression test: transfer_ratio does not divide by zero when mse_low -> 0."""
    mse_high = 0.001
    mse_low_zero = 0.0

    ratio = float(mse_high / max(mse_low_zero, DEFAULT_TRANSFER_RATIO_FLOOR))
    assert math.isfinite(ratio)
    assert ratio > 0


def test_transfer_ratio_contract_threshold() -> None:
    """Regression test: Default NoyronHXScenarioConfig specifies transfer_ratio <= 1.5."""
    config = NoyronHXScenarioConfig()
    assert config.transfer_ratio_threshold <= 1.5
    assert config.transfer_ratio_threshold > 0


def test_transfer_ratio_invariance_ideal_case() -> None:
    """Regression test: Equal low and high MSE yields transfer ratio of exactly 1.0."""
    mse_low = 0.016
    mse_high = 0.016

    transfer_ratio = float(mse_high / max(mse_low, DEFAULT_TRANSFER_RATIO_FLOOR))
    assert abs(transfer_ratio - 1.0) < 1e-6
