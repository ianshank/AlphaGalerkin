"""Tests for the LBB stability monitoring scenario.

Validates:
    - StabilityScenario initialization and registration
    - LBB constant computation for valid models
    - Stability check pass/fail with mocked projections
    - Result format with violation counts
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.poc.config import (
    ScenarioResult,
    ScenarioStatus,
    StabilityScenarioConfig,
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
def default_config() -> StabilityScenarioConfig:
    return StabilityScenarioConfig(seed=SEED)


@pytest.fixture()
def small_config() -> StabilityScenarioConfig:
    """Minimal config for fast mocked tests."""
    return StabilityScenarioConfig(
        resolutions=[3, 5, 7],
        d_model=16,
        d_key=16,
        d_value=16,
        batch_size=1,
        n_forward_passes=10,
        n_training_steps=100,
        lbb_threshold=1e-6,
        max_lbb_violations=0,
        learning_rate=1e-3,
        seed=SEED,
    )


def _import_stability_scenario() -> type[BaseScenario]:
    """Import StabilityScenario and ensure it is registered.

    The @scenario decorator only fires on first import. If the registry
    was cleared (e.g. by autouse fixture), we must re-register manually.
    """
    from src.poc.scenarios.stability import StabilityScenario

    registry = ScenarioRegistry()
    if registry.get("stability") is None:
        registry.register("stability", StabilityScenario)

    return StabilityScenario


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestStabilityScenarioInit:
    """Tests for StabilityScenario initialization."""

    def test_registration(self) -> None:
        """StabilityScenario should register under 'stability'."""
        cls = _import_stability_scenario()
        assert ScenarioRegistry().get("stability") is cls

    def test_default_config_applied(self) -> None:
        """Scenario should use StabilityScenarioConfig defaults."""
        cls = _import_stability_scenario()
        instance = cls(name="stability", description="test")
        assert instance.config.resolutions == [5, 9, 13, 19]
        assert instance.config.lbb_threshold == 1e-6

    def test_custom_config(self, small_config: StabilityScenarioConfig) -> None:
        """Scenario should accept a custom config."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)
        assert instance.config.resolutions == [3, 5, 7]
        assert instance.config.n_training_steps == 100


# ---------------------------------------------------------------------------
# Execute with mocked projections
# ---------------------------------------------------------------------------


class TestStabilityScenarioExecute:
    """Tests for execute() with heavy computation mocked."""

    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_initialization_stability"
    )
    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_training_stability"
    )
    def test_execute_passes_with_stable_model(
        self,
        mock_training: MagicMock,
        mock_init: MagicMock,
        small_config: StabilityScenarioConfig,
    ) -> None:
        """Result should pass when LBB constants are above threshold."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)

        # All LBB values well above threshold
        lbb_value = 0.1
        mock_init.return_value = {
            res: [lbb_value] * small_config.n_forward_passes
            for res in small_config.resolutions
        }
        mock_training.return_value = {
            "lbb_values": [lbb_value] * small_config.n_training_steps,
            "n_violations": 0,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is True
        assert result.status == ScenarioStatus.PASSED

    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_initialization_stability"
    )
    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_training_stability"
    )
    def test_execute_fails_with_violations(
        self,
        mock_training: MagicMock,
        mock_init: MagicMock,
        small_config: StabilityScenarioConfig,
    ) -> None:
        """Result should fail when LBB violations exceed threshold."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)

        mock_init.return_value = {
            res: [0.1] * small_config.n_forward_passes
            for res in small_config.resolutions
        }
        # Many violations
        mock_training.return_value = {
            "lbb_values": [1e-10] * small_config.n_training_steps,
            "n_violations": small_config.n_training_steps,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is False
        assert result.status == ScenarioStatus.FAILED

    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_initialization_stability"
    )
    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_training_stability"
    )
    def test_execute_fails_on_init_violations(
        self,
        mock_training: MagicMock,
        mock_init: MagicMock,
        small_config: StabilityScenarioConfig,
    ) -> None:
        """Result should fail when initialization LBB is below threshold."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)

        # LBB values below threshold at init
        mock_init.return_value = {
            res: [1e-10] * small_config.n_forward_passes
            for res in small_config.resolutions
        }
        mock_training.return_value = {
            "lbb_values": [0.1] * small_config.n_training_steps,
            "n_violations": 0,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is False
        assert result.threshold_results["init_stability"] is False

    def test_setup_required_before_execute(self) -> None:
        """execute() should raise if setup() was not called."""
        cls = _import_stability_scenario()
        instance = cls(name="stability", description="test")
        instance._start_time = datetime.now()

        with pytest.raises(RuntimeError, match="setup"):
            instance.execute()


# ---------------------------------------------------------------------------
# LBB constant computation
# ---------------------------------------------------------------------------


class TestLBBConstant:
    """Tests for LBB constant behavior with mocked values."""

    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_initialization_stability"
    )
    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_training_stability"
    )
    @pytest.mark.parametrize("lbb_value", [0.001, 0.01, 0.1, 1.0])
    def test_various_lbb_values_pass(
        self,
        mock_training: MagicMock,
        mock_init: MagicMock,
        small_config: StabilityScenarioConfig,
        lbb_value: float,
    ) -> None:
        """LBB values above threshold should yield passing result."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)

        mock_init.return_value = {
            res: [lbb_value] * small_config.n_forward_passes
            for res in small_config.resolutions
        }
        mock_training.return_value = {
            "lbb_values": [lbb_value] * small_config.n_training_steps,
            "n_violations": 0,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is True
        assert result.metrics["lbb_training_mean"] == pytest.approx(lbb_value)
        assert result.metrics["lbb_training_min"] == pytest.approx(lbb_value)


# ---------------------------------------------------------------------------
# Result format
# ---------------------------------------------------------------------------


class TestStabilityResultFormat:
    """Tests for the structure of returned ScenarioResult."""

    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_initialization_stability"
    )
    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_training_stability"
    )
    def test_result_contains_expected_metrics(
        self,
        mock_training: MagicMock,
        mock_init: MagicMock,
        small_config: StabilityScenarioConfig,
    ) -> None:
        """Result metrics should contain per-resolution LBB entries."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)

        mock_init.return_value = {
            res: [0.1] * small_config.n_forward_passes
            for res in small_config.resolutions
        }
        mock_training.return_value = {
            "lbb_values": [0.1] * small_config.n_training_steps,
            "n_violations": 0,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        for res in small_config.resolutions:
            assert f"lbb_init_mean_{res}x{res}" in result.metrics
            assert f"lbb_init_min_{res}x{res}" in result.metrics

        assert "lbb_training_mean" in result.metrics
        assert "lbb_training_min" in result.metrics
        assert "lbb_violations" in result.metrics

    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_initialization_stability"
    )
    @patch(
        "src.poc.scenarios.stability.StabilityScenario._test_training_stability"
    )
    def test_result_has_threshold_results(
        self,
        mock_training: MagicMock,
        mock_init: MagicMock,
        small_config: StabilityScenarioConfig,
    ) -> None:
        """threshold_results should have init_stability and training_stability."""
        cls = _import_stability_scenario()
        instance = cls(config=small_config)

        mock_init.return_value = {
            res: [0.1] * small_config.n_forward_passes
            for res in small_config.resolutions
        }
        mock_training.return_value = {
            "lbb_values": [0.1] * small_config.n_training_steps,
            "n_violations": 0,
        }

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert "init_stability" in result.threshold_results
        assert "training_stability" in result.threshold_results
