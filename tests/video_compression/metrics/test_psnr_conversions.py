"""Tests for :mod:`src.video_compression.metrics.psnr_conversions`."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.video_compression.metrics.psnr_conversions import (
    DEFAULT_MAX_SIGNAL,
    PSNR_DB_LOG_BASE,
    psnr_db_to_mse_surrogate,
)


class TestConstants:
    def test_log_base_is_decibel_definition(self) -> None:
        assert PSNR_DB_LOG_BASE == 10.0

    def test_default_max_signal_is_unit_normalized(self) -> None:
        assert DEFAULT_MAX_SIGNAL == 1.0


class TestPsnrConversion:
    def test_zero_db_yields_max_signal_squared(self) -> None:
        # PSNR = 0 dB <=> MSE = MAX^2
        assert psnr_db_to_mse_surrogate(0.0) == pytest.approx(1.0)

    def test_round_trip_at_30_db(self) -> None:
        # Closed form: MSE at 30 dB with MAX=1 is 1e-3.
        assert psnr_db_to_mse_surrogate(30.0) == pytest.approx(1.0e-3)

    def test_round_trip_at_60_db(self) -> None:
        assert psnr_db_to_mse_surrogate(60.0) == pytest.approx(1.0e-6)

    def test_max_signal_scales_quadratically(self) -> None:
        baseline = psnr_db_to_mse_surrogate(30.0)
        scaled = psnr_db_to_mse_surrogate(30.0, max_signal=2.0)
        # MAX doubles -> MSE quadruples (MAX^2 in the formula).
        assert scaled == pytest.approx(baseline * 4.0)

    def test_inf_psnr_yields_zero_mse(self) -> None:
        assert psnr_db_to_mse_surrogate(float("inf")) == 0.0

    def test_negative_psnr_yields_mse_above_max_squared(self) -> None:
        # PSNR < 0 means MSE > MAX^2, i.e. catastrophically bad.
        assert psnr_db_to_mse_surrogate(-3.0) > 1.0

    def test_max_signal_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            psnr_db_to_mse_surrogate(30.0, max_signal=0.0)
        with pytest.raises(ValueError):
            psnr_db_to_mse_surrogate(30.0, max_signal=-1.0)


@given(psnr_db=st.floats(min_value=0.1, max_value=80.0, allow_nan=False, allow_infinity=False))
def test_property_strictly_decreasing_in_psnr(psnr_db: float) -> None:
    # Higher PSNR -> lower MSE.
    assert psnr_db_to_mse_surrogate(psnr_db) > psnr_db_to_mse_surrogate(psnr_db + 0.1)


@given(
    psnr_db=st.floats(min_value=10.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    max_signal=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_property_invertible_to_canonical_psnr_formula(
    psnr_db: float,
    max_signal: float,
) -> None:
    mse = psnr_db_to_mse_surrogate(psnr_db, max_signal=max_signal)
    # Recompute PSNR from MSE: 10 * log10(MAX^2 / MSE)
    recovered = 10.0 * math.log10((max_signal**2) / mse)
    assert recovered == pytest.approx(psnr_db, rel=1e-6, abs=1e-6)
