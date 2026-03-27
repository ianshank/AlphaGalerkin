"""Tests for zero-shot transfer verification.

Validates TransferResult dataclass, evaluate_transfer with a mocked model,
verify_resolution_independence, and various board sizes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.experiments.physics_model import PhysicsOperator
from src.experiments.verify_transfer import (
    DEFAULT_EVAL_SEED_OFFSET,
    DEFAULT_EVAL_SIZES,
    DEFAULT_RESOLUTION_TEST_SIZES,
    TransferResult,
    evaluate_transfer,
    verify_resolution_independence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model() -> PhysicsOperator:
    """Small PhysicsOperator for fast tests.

    Uses torch.no_grad context in fixture so that downstream calls to
    verify_resolution_independence (which builds tensors from numpy
    without detaching) work correctly.
    """
    model = PhysicsOperator(
        d_model=32,
        n_heads=2,
        n_layers=2,
        n_fourier_features=16,
        use_fnet=False,
    )
    model.eval()
    return model


@pytest.fixture(params=[5, 7, 9])
def eval_size(request: pytest.FixtureRequest) -> int:
    """Parametrized evaluation grid sizes."""
    return request.param


# ---------------------------------------------------------------------------
# Tests: TransferResult dataclass
# ---------------------------------------------------------------------------


class TestTransferResult:
    """Tests for the TransferResult dataclass."""

    def test_fields_stored(self) -> None:
        """All fields are accessible."""
        r = TransferResult(
            train_size=9,
            eval_size=19,
            mse=0.001,
            mae=0.02,
            rmse=0.0316,
            max_error=0.1,
            n_samples=100,
            passed=True,
        )
        assert r.train_size == 9
        assert r.eval_size == 19
        assert r.mse == pytest.approx(0.001)
        assert r.mae == pytest.approx(0.02)
        assert r.rmse == pytest.approx(0.0316)
        assert r.max_error == pytest.approx(0.1)
        assert r.n_samples == 100
        assert r.passed is True

    def test_passed_reflects_threshold(self) -> None:
        """passed=True when mse < threshold, False otherwise."""
        passing = TransferResult(
            train_size=9, eval_size=19, mse=0.01, mae=0.0,
            rmse=0.0, max_error=0.0, n_samples=10, passed=True,
        )
        failing = TransferResult(
            train_size=9, eval_size=19, mse=0.10, mae=0.0,
            rmse=0.0, max_error=0.0, n_samples=10, passed=False,
        )
        assert passing.passed is True
        assert failing.passed is False

    def test_equality(self) -> None:
        """Two TransferResults with same fields are equal."""
        kwargs = dict(
            train_size=9, eval_size=13, mse=0.02, mae=0.01,
            rmse=0.14, max_error=0.5, n_samples=50, passed=True,
        )
        assert TransferResult(**kwargs) == TransferResult(**kwargs)


# ---------------------------------------------------------------------------
# Tests: evaluate_transfer
# ---------------------------------------------------------------------------


class TestEvaluateTransfer:
    """Tests for evaluate_transfer with real tiny model."""

    def test_returns_transfer_result(self, tiny_model: PhysicsOperator) -> None:
        """evaluate_transfer returns a TransferResult."""
        result = evaluate_transfer(
            model=tiny_model,
            train_size=5,
            eval_size=5,
            n_samples=4,
            device=torch.device("cpu"),
            seed=0,
            threshold=100.0,  # generous threshold so it passes
            n_charges=2,
            batch_size=4,
        )
        assert isinstance(result, TransferResult)

    def test_metrics_are_non_negative(self, tiny_model: PhysicsOperator) -> None:
        """MSE, MAE, RMSE, max_error are non-negative."""
        result = evaluate_transfer(
            model=tiny_model,
            train_size=5,
            eval_size=5,
            n_samples=4,
            device=torch.device("cpu"),
            seed=0,
            n_charges=2,
            batch_size=4,
        )
        assert result.mse >= 0
        assert result.mae >= 0
        assert result.rmse >= 0
        assert result.max_error >= 0

    def test_rmse_consistent_with_mse(self, tiny_model: PhysicsOperator) -> None:
        """RMSE equals sqrt(MSE)."""
        result = evaluate_transfer(
            model=tiny_model,
            train_size=5,
            eval_size=5,
            n_samples=4,
            device=torch.device("cpu"),
            seed=0,
            n_charges=2,
            batch_size=4,
        )
        assert result.rmse == pytest.approx(np.sqrt(result.mse), rel=1e-4)

    def test_n_samples_recorded(self, tiny_model: PhysicsOperator) -> None:
        """n_samples in result matches requested count."""
        n = 6
        result = evaluate_transfer(
            model=tiny_model,
            train_size=5,
            eval_size=5,
            n_samples=n,
            device=torch.device("cpu"),
            seed=0,
            n_charges=2,
            batch_size=4,
        )
        assert result.n_samples == n

    @pytest.mark.parametrize("threshold", [0.001, 1.0, 100.0])
    def test_passed_respects_threshold(
        self, tiny_model: PhysicsOperator, threshold: float
    ) -> None:
        """passed field is consistent with mse vs threshold."""
        result = evaluate_transfer(
            model=tiny_model,
            train_size=5,
            eval_size=5,
            n_samples=4,
            device=torch.device("cpu"),
            seed=0,
            threshold=threshold,
            n_charges=2,
            batch_size=4,
        )
        assert result.passed == (result.mse < threshold)

    def test_different_eval_sizes(self, tiny_model: PhysicsOperator, eval_size: int) -> None:
        """evaluate_transfer works with various eval grid sizes."""
        result = evaluate_transfer(
            model=tiny_model,
            train_size=5,
            eval_size=eval_size,
            n_samples=4,
            device=torch.device("cpu"),
            seed=0,
            n_charges=2,
            batch_size=4,
        )
        assert result.eval_size == eval_size
        assert result.mse >= 0


# ---------------------------------------------------------------------------
# Tests: verify_resolution_independence
# ---------------------------------------------------------------------------


class TestVerifyResolutionIndependence:
    """Tests for cross-resolution consistency verification."""

    def test_returns_dict_with_expected_keys(
        self, tiny_model: PhysicsOperator
    ) -> None:
        """Result dict contains consistency error metrics."""
        with torch.no_grad():
            result = verify_resolution_independence(
                model=tiny_model,
                device=torch.device("cpu"),
                resolutions=[5, 9],
                n_samples=3,
                n_charges=2,
                seed=0,
            )
        assert "mean_consistency_error" in result
        assert "std_consistency_error" in result
        assert "max_consistency_error" in result
        assert "resolutions_tested" in result

    def test_errors_are_non_negative(self, tiny_model: PhysicsOperator) -> None:
        """Consistency errors are non-negative."""
        with torch.no_grad():
            result = verify_resolution_independence(
                model=tiny_model,
                device=torch.device("cpu"),
                resolutions=[5, 7],
                n_samples=3,
                n_charges=2,
                seed=0,
            )
        assert result["mean_consistency_error"] >= 0
        assert result["max_consistency_error"] >= 0

    def test_resolutions_recorded(self, tiny_model: PhysicsOperator) -> None:
        """Tested resolutions are recorded in the result."""
        resolutions = [5, 9, 13]
        with torch.no_grad():
            result = verify_resolution_independence(
                model=tiny_model,
                device=torch.device("cpu"),
                resolutions=resolutions,
                n_samples=2,
                n_charges=2,
                seed=0,
            )
        assert result["resolutions_tested"] == resolutions

    def test_default_resolutions_used(self, tiny_model: PhysicsOperator) -> None:
        """When resolutions=None, default list is used."""
        with torch.no_grad():
            result = verify_resolution_independence(
                model=tiny_model,
                device=torch.device("cpu"),
                resolutions=None,
                n_samples=2,
                n_charges=2,
                seed=0,
            )
        assert result["resolutions_tested"] == DEFAULT_RESOLUTION_TEST_SIZES

    def test_deterministic_with_same_seed(self, tiny_model: PhysicsOperator) -> None:
        """Same seed produces identical results."""
        kwargs = dict(
            model=tiny_model,
            device=torch.device("cpu"),
            resolutions=[5, 9],
            n_samples=3,
            n_charges=2,
            seed=12345,
        )
        with torch.no_grad():
            r1 = verify_resolution_independence(**kwargs)
            r2 = verify_resolution_independence(**kwargs)
        assert r1["mean_consistency_error"] == pytest.approx(
            r2["mean_consistency_error"]
        )


# ---------------------------------------------------------------------------
# Tests: module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for documented module-level constants."""

    def test_eval_seed_offset_positive(self) -> None:
        """Seed offset is positive and large enough to avoid overlap."""
        assert DEFAULT_EVAL_SEED_OFFSET > 0
        assert DEFAULT_EVAL_SEED_OFFSET >= 1000

    def test_default_eval_sizes_sorted(self) -> None:
        """Default eval sizes are in ascending order."""
        assert DEFAULT_EVAL_SIZES == sorted(DEFAULT_EVAL_SIZES)

    def test_default_resolution_sizes_include_standard(self) -> None:
        """Default resolution test sizes include 9 and 19 (Go boards)."""
        assert 9 in DEFAULT_RESOLUTION_TEST_SIZES
        assert 19 in DEFAULT_RESOLUTION_TEST_SIZES
