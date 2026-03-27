"""Tests for the zero-shot transfer scenario.

Validates:
    - TransferScenario initialization and config
    - Config validation for resolutions and thresholds
    - Execute with mocked model training (no GPU required)
    - Result contains expected metrics and status
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from src.poc.config import (
    ScenarioResult,
    ScenarioStatus,
    TransferScenarioConfig,
)
from src.poc.registry import BaseScenario, ScenarioRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Ensure registry is clean before each test."""
    ScenarioRegistry().clear()


@pytest.fixture()
def default_config() -> TransferScenarioConfig:
    return TransferScenarioConfig(seed=SEED)


@pytest.fixture()
def small_config() -> TransferScenarioConfig:
    """Minimal config for fast mocked tests."""
    return TransferScenarioConfig(
        train_resolution=5,
        eval_resolutions=[5, 9],
        primary_eval_resolution=9,
        n_train_samples=4,
        n_eval_samples=2,
        n_epochs=1,
        batch_size=2,
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_fourier_features=8,
        mse_threshold=10.0,  # generous to pass with random weights
        seed=SEED,
    )


def _import_transfer_scenario() -> type[BaseScenario]:
    """Import TransferScenario, triggering registration."""
    from src.poc.scenarios.transfer import TransferScenario

    return TransferScenario


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestTransferScenarioInit:
    """Tests for TransferScenario initialization."""

    def test_registration(self) -> None:
        """TransferScenario should register under 'transfer'."""
        cls = _import_transfer_scenario()
        assert ScenarioRegistry().get("transfer") is cls

    def test_default_config_applied(self) -> None:
        """Scenario should use TransferScenarioConfig defaults."""
        cls = _import_transfer_scenario()
        instance = cls(name="transfer", description="test")
        assert instance.config.train_resolution == 9
        assert instance.config.mse_threshold == 0.05

    def test_custom_config(self, small_config: TransferScenarioConfig) -> None:
        """Scenario should accept a custom config."""
        cls = _import_transfer_scenario()
        instance = cls(config=small_config)
        assert instance.config.train_resolution == 5
        assert instance.config.n_epochs == 1


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestTransferConfigValidation:
    """Validation rules specific to transfer config."""

    def test_primary_eval_added(self) -> None:
        """primary_eval_resolution should be added to eval_resolutions."""
        config = TransferScenarioConfig(
            eval_resolutions=[5, 9],
            primary_eval_resolution=19,
        )
        assert 19 in config.eval_resolutions

    def test_eval_resolutions_sorted(self) -> None:
        config = TransferScenarioConfig(eval_resolutions=[19, 5, 9])
        assert config.eval_resolutions == sorted(set(config.eval_resolutions))

    @pytest.mark.parametrize("threshold", [0.001, 0.01, 0.05, 0.1, 1.0])
    def test_various_thresholds(self, threshold: float) -> None:
        """Different thresholds should be accepted."""
        config = TransferScenarioConfig(mse_threshold=threshold)
        assert config.mse_threshold == threshold


# ---------------------------------------------------------------------------
# Execute with mocked training
# ---------------------------------------------------------------------------


class TestTransferScenarioExecute:
    """Tests for execute() with heavy computation mocked."""

    def _make_fake_model(self, mse_value: float) -> nn.Module:
        """Create a mock model returning predictions that yield a given MSE."""
        model = MagicMock(spec=nn.Module)
        model.eval = MagicMock(return_value=model)
        model.train = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=iter([torch.zeros(1, requires_grad=True)]))
        model.state_dict = MagicMock(return_value={})

        def forward(coords: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
            batch_size = coords.shape[0]
            n_points = coords.shape[1]
            # Return zeros; the MSE depends on target magnitude
            return torch.zeros(batch_size, n_points)

        model.__call__ = forward
        model.to = MagicMock(return_value=model)
        return model

    @patch("src.poc.scenarios.transfer.TransferScenario._train_model")
    @patch("src.poc.scenarios.transfer.TransferScenario._evaluate_at_resolution")
    def test_execute_passes_when_below_threshold(
        self,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        small_config: TransferScenarioConfig,
    ) -> None:
        """Result should pass when all MSEs are below threshold."""
        cls = _import_transfer_scenario()
        instance = cls(config=small_config)

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {}
        mock_train.return_value = fake_model

        mock_eval.return_value = {
            "mse": 0.001,
            "mae": 0.01,
            "rmse": 0.032,
            "max_error": 0.05,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is True
        assert result.status == ScenarioStatus.PASSED
        assert mock_eval.call_count == len(small_config.eval_resolutions)

    @patch("src.poc.scenarios.transfer.TransferScenario._train_model")
    @patch("src.poc.scenarios.transfer.TransferScenario._evaluate_at_resolution")
    def test_execute_fails_when_above_threshold(
        self,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        small_config: TransferScenarioConfig,
    ) -> None:
        """Result should fail when MSE exceeds threshold."""
        cls = _import_transfer_scenario()
        # Use a very tight threshold
        small_config_dict = small_config.model_dump()
        small_config_dict["mse_threshold"] = 0.0001
        tight_config = TransferScenarioConfig(**small_config_dict)

        instance = cls(config=tight_config)

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {}
        mock_train.return_value = fake_model

        mock_eval.return_value = {
            "mse": 0.5,
            "mae": 0.4,
            "rmse": 0.7,
            "max_error": 1.0,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is False
        assert result.status == ScenarioStatus.FAILED


# ---------------------------------------------------------------------------
# Result format
# ---------------------------------------------------------------------------


class TestTransferResultFormat:
    """Tests for the structure of returned ScenarioResult."""

    @patch("src.poc.scenarios.transfer.TransferScenario._train_model")
    @patch("src.poc.scenarios.transfer.TransferScenario._evaluate_at_resolution")
    def test_result_contains_expected_metrics(
        self,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        small_config: TransferScenarioConfig,
    ) -> None:
        """Result metrics should contain per-resolution MSE entries."""
        cls = _import_transfer_scenario()
        instance = cls(config=small_config)

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {}
        mock_train.return_value = fake_model

        mock_eval.return_value = {
            "mse": 0.01,
            "mae": 0.05,
            "rmse": 0.1,
            "max_error": 0.2,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        for res in small_config.eval_resolutions:
            assert f"mse_{res}x{res}" in result.metrics
            assert f"mae_{res}x{res}" in result.metrics

    @patch("src.poc.scenarios.transfer.TransferScenario._train_model")
    @patch("src.poc.scenarios.transfer.TransferScenario._evaluate_at_resolution")
    def test_result_has_threshold_results(
        self,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        small_config: TransferScenarioConfig,
    ) -> None:
        """threshold_results should contain one entry per eval resolution."""
        cls = _import_transfer_scenario()
        instance = cls(config=small_config)

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {}
        mock_train.return_value = fake_model

        mock_eval.return_value = {"mse": 0.01, "mae": 0.05, "rmse": 0.1, "max_error": 0.2}

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert len(result.threshold_results) == len(small_config.eval_resolutions)
        for res in small_config.eval_resolutions:
            key = f"mse_{res}x{res}"
            assert key in result.threshold_results
