"""Tests for the LBB stability monitoring scenario (src/poc/scenarios/stability.py).

Validates:
    - StabilityScenario initialization and config
    - setup() and teardown() lifecycle
    - execute() with mocked GalerkinProjection for fast CPU tests
    - Initialization stability testing across resolutions
    - Training stability testing with LBB monitoring
    - Pass/fail logic based on LBB threshold and violations
    - Error handling when setup() is not called
    - Edge cases: empty resolutions, single resolution
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.poc.config import (
    ScenarioStatus,
    StabilityScenarioConfig,
)
from src.poc.registry import ScenarioRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test to avoid duplicate registrations."""
    ScenarioRegistry().clear()
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.poc.scenarios"):
            del sys.modules[mod_name]


@pytest.fixture
def small_config() -> StabilityScenarioConfig:
    """Create a small fast config suitable for CPU testing.

    Respects Pydantic validators: n_forward_passes >= 10, n_training_steps >= 100.
    """
    return StabilityScenarioConfig(
        name="stability",
        description="Test stability scenario",
        d_model=16,
        d_key=8,
        d_value=8,
        resolutions=[3, 5],
        n_forward_passes=10,
        batch_size=2,
        n_training_steps=100,
        learning_rate=1e-3,
        lbb_threshold=1e-6,
        max_lbb_violations=0,
        seed=42,
    )


def _make_fake_projection(lbb_value: float = 0.5):
    """Create a fake GalerkinProjection that returns a fixed LBB constant.

    Returns a class that mimics GalerkinProjection: it is an nn.Module
    with a forward() method and compute_lbb_constant() method.
    """

    class FakeGalerkinProjection(nn.Module):
        def __init__(self, d_model: int, d_key: int, d_value: int) -> None:
            super().__init__()
            # Need at least one parameter for the optimizer
            self.linear = nn.Linear(d_model, d_model)
            self._lbb_value = lbb_value

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

        def compute_lbb_constant(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.full((batch_size,), self._lbb_value)

    return FakeGalerkinProjection


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestStabilityScenarioConfig:
    """Tests for StabilityScenarioConfig validation."""

    def test_default_config(self) -> None:
        """Default config has valid values."""
        config = StabilityScenarioConfig(name="stability", description="test")
        assert config.d_model == 64
        assert config.d_key == 32
        assert config.d_value == 32
        assert config.resolutions == [5, 9, 13, 19]
        assert config.lbb_threshold == 1e-6
        assert config.max_lbb_violations == 0

    def test_custom_config(self, small_config: StabilityScenarioConfig) -> None:
        """Custom config values are preserved."""
        assert small_config.d_model == 16
        assert small_config.d_key == 8
        assert small_config.resolutions == [3, 5]
        assert small_config.n_forward_passes == 10
        assert small_config.n_training_steps == 100

    def test_config_hash_deterministic(self, small_config: StabilityScenarioConfig) -> None:
        """Config hash is deterministic."""
        h1 = small_config.compute_hash()
        h2 = small_config.compute_hash()
        assert h1 == h2

    def test_config_hash_changes_on_different_params(self) -> None:
        """Different configs produce different hashes."""
        c1 = StabilityScenarioConfig(name="s1", description="a")
        c2 = StabilityScenarioConfig(name="s2", description="b")
        assert c1.compute_hash() != c2.compute_hash()

    def test_config_invalid_d_model(self) -> None:
        """d_model below minimum raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StabilityScenarioConfig(name="bad", description="bad", d_model=4)

    def test_config_invalid_d_key(self) -> None:
        """d_key below minimum raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StabilityScenarioConfig(name="bad", description="bad", d_key=2)

    def test_config_invalid_n_training_steps(self) -> None:
        """n_training_steps below minimum raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StabilityScenarioConfig(name="bad", description="bad", n_training_steps=5)


# ---------------------------------------------------------------------------
# Scenario lifecycle tests
# ---------------------------------------------------------------------------


class TestStabilityScenarioLifecycle:
    """Tests for setup/teardown and lifecycle."""

    def test_init_with_config(self, small_config: StabilityScenarioConfig) -> None:
        """Scenario can be initialized with a config object."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)
        assert s.config.d_model == 16
        assert s._device is None

    def test_init_with_kwargs(self) -> None:
        """Scenario can be initialized with keyword arguments."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(
            name="stability",
            description="kw test",
            d_model=32,
            d_key=16,
            d_value=16,
        )
        assert s.config.d_model == 32

    def test_setup_sets_device(self, small_config: StabilityScenarioConfig) -> None:
        """setup() initializes the device."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)
        s.setup()

        assert s._device is not None
        # In CI without GPU, should be CPU
        assert s._device == torch.device("cpu") or s._device == torch.device("cuda")

    def test_setup_sets_scenario_logger(self, small_config: StabilityScenarioConfig) -> None:
        """setup() initializes the scenario logger."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)
        s.setup()

        assert s._scenario_logger is not None

    def test_teardown_without_gpu(self, small_config: StabilityScenarioConfig) -> None:
        """teardown() runs without error on CPU."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)
        s.setup()
        s.teardown()  # Should not raise

    def test_execute_without_setup_raises(self, small_config: StabilityScenarioConfig) -> None:
        """execute() raises RuntimeError if setup() was not called."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)
        # _start_time needs to be set for the assertion in execute
        s._start_time = datetime.now()

        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s.execute()


# ---------------------------------------------------------------------------
# Execution tests with mocked GalerkinProjection
# ---------------------------------------------------------------------------


class TestStabilityScenarioExecution:
    """Tests for scenario execution with mocked heavy computation."""

    def test_execute_passes_with_high_lbb(self, small_config: StabilityScenarioConfig) -> None:
        """Scenario passes when LBB constant is well above threshold."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            result = s.run()

        assert result.status == ScenarioStatus.PASSED
        assert result.passed is True
        assert result.metrics["lbb_violations"] == 0
        assert result.metrics["lbb_training_min"] == pytest.approx(0.5)

    def test_execute_fails_with_low_lbb(self) -> None:
        """Scenario fails when LBB constant is below threshold."""
        from src.poc.scenarios.stability import StabilityScenario

        # Set threshold high so that our fake LBB value fails
        config = StabilityScenarioConfig(
            name="stability",
            description="low lbb test",
            d_model=16,
            d_key=8,
            d_value=8,
            resolutions=[3],
            n_forward_passes=10,
            batch_size=2,
            n_training_steps=100,
            lbb_threshold=1.0,  # threshold > fake value
            max_lbb_violations=0,
            seed=42,
        )

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=config)
            result = s.run()

        assert result.status == ScenarioStatus.FAILED
        assert result.passed is False

    def test_execute_records_init_metrics(self, small_config: StabilityScenarioConfig) -> None:
        """Execute records per-resolution LBB init metrics."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.42)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            result = s.run()

        # Check init metrics for each resolution
        for res in small_config.resolutions:
            mean_key = f"lbb_init_mean_{res}x{res}"
            min_key = f"lbb_init_min_{res}x{res}"
            assert mean_key in result.metrics
            assert min_key in result.metrics
            assert result.metrics[mean_key] == pytest.approx(0.42)
            assert result.metrics[min_key] == pytest.approx(0.42)

    def test_execute_records_training_metrics(self, small_config: StabilityScenarioConfig) -> None:
        """Execute records training LBB metrics."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.1)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            result = s.run()

        assert "lbb_training_mean" in result.metrics
        assert "lbb_training_min" in result.metrics
        assert "lbb_violations" in result.metrics
        assert result.metrics["lbb_training_mean"] == pytest.approx(0.1)
        assert result.metrics["lbb_training_min"] == pytest.approx(0.1)

    def test_execute_counts_violations(self) -> None:
        """Violations are counted when LBB dips below threshold."""
        from src.poc.scenarios.stability import StabilityScenario

        config = StabilityScenarioConfig(
            name="stability",
            description="violation test",
            d_model=16,
            d_key=8,
            d_value=8,
            resolutions=[3],
            n_forward_passes=10,
            batch_size=2,
            n_training_steps=100,
            lbb_threshold=0.3,  # threshold above fake value
            max_lbb_violations=200,  # allow many violations so training passes
            seed=42,
        )

        fake_cls = _make_fake_projection(lbb_value=0.1)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=config)
            result = s.run()

        # All 100 training steps should have violations (0.1 < 0.3)
        assert result.metrics["lbb_violations"] == 100
        # Init also violates (0.1 < 0.3), so init_stability fails
        assert result.passed is False

    def test_execute_allows_max_violations(self) -> None:
        """Scenario passes if violations <= max_lbb_violations."""
        from src.poc.scenarios.stability import StabilityScenario

        config = StabilityScenarioConfig(
            name="stability",
            description="max violations test",
            d_model=16,
            d_key=8,
            d_value=8,
            resolutions=[3],
            n_forward_passes=10,
            batch_size=2,
            n_training_steps=100,
            lbb_threshold=0.3,
            max_lbb_violations=100,  # Exactly allow 100 violations
            seed=42,
        )

        # Use a high LBB for init (passes) but low for training
        high_lbb = 0.5
        low_lbb = 0.1

        class SwitchingProjection(nn.Module):
            """Returns high LBB for first calls (init), low for later (training)."""

            def __init__(self, d_model: int, d_key: int, d_value: int) -> None:
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)
                self._call_count = 0
                # After init phase (1 resolution * 10 forward passes = 10 calls),
                # switch to low LBB
                self._init_calls = 1 * 10

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

            def compute_lbb_constant(self, x: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                self._call_count += 1
                if self._call_count <= self._init_calls:
                    return torch.full((batch_size,), high_lbb)
                return torch.full((batch_size,), low_lbb)

        with patch(
            "src.math_kernel.integral.GalerkinProjection",
            SwitchingProjection,
        ):
            s = StabilityScenario(config=config)
            result = s.run()

        # Init passes (high LBB), training has 100 violations <= max 100
        assert result.threshold_results["init_stability"] is True
        assert result.threshold_results["training_stability"] is True
        assert result.passed is True

    def test_result_contains_expected_fields(self, small_config: StabilityScenarioConfig) -> None:
        """Result has all expected fields including custom extras."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            result = s.run()

        assert result.scenario_name == "stability"
        assert result.config_hash == small_config.compute_hash()
        # StabilityScenario.setup() picks "cuda" when available else "cpu";
        # the test asserts the recorded device matches that contract on
        # whatever host runs it (was hardcoded "cpu" and broke CUDA hosts).
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert result.device == expected_device
        assert result.python_version != ""
        assert result.torch_version != ""
        assert result.duration_seconds >= 0
        assert result.start_time is not None
        assert result.end_time is not None

    def test_execute_with_single_resolution(self) -> None:
        """Scenario works with a single resolution."""
        from src.poc.scenarios.stability import StabilityScenario

        config = StabilityScenarioConfig(
            name="stability",
            description="single resolution",
            d_model=16,
            d_key=8,
            d_value=8,
            resolutions=[5],
            n_forward_passes=10,
            batch_size=2,
            n_training_steps=100,
            seed=42,
        )

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=config)
            result = s.run()

        assert result.status == ScenarioStatus.PASSED
        assert "lbb_init_mean_5x5" in result.metrics

    def test_execute_empty_resolutions_raises(self) -> None:
        """Execute raises ValueError for empty resolutions list."""
        from src.poc.scenarios.stability import StabilityScenario

        # We need to bypass Pydantic validation for resolutions to test
        # the runtime check in execute()
        config = StabilityScenarioConfig(
            name="stability",
            description="empty res",
            d_model=16,
            d_key=8,
            d_value=8,
            resolutions=[3],
            n_forward_passes=10,
            batch_size=2,
            n_training_steps=100,
            seed=42,
        )
        # Manually empty the resolutions after creation
        object.__setattr__(config, "resolutions", [])

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=config)
            # run() catches exceptions and returns ERROR status
            result = s.run()

        assert result.status == ScenarioStatus.ERROR
        assert result.passed is False


# ---------------------------------------------------------------------------
# Tests for internal methods
# ---------------------------------------------------------------------------


class TestStabilityInternalMethods:
    """Tests for _test_initialization_stability and _test_training_stability."""

    def test_initialization_stability_returns_per_resolution(
        self, small_config: StabilityScenarioConfig
    ) -> None:
        """_test_initialization_stability returns a dict keyed by resolution."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.25)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            s.setup()

            torch.manual_seed(small_config.seed)
            results = s._test_initialization_stability()

        assert set(results.keys()) == {3, 5}
        for res, values in results.items():
            assert len(values) > 0
            assert all(v == pytest.approx(0.25) for v in values)

    def test_training_stability_returns_lbb_values_and_violations(
        self, small_config: StabilityScenarioConfig
    ) -> None:
        """_test_training_stability returns dict with lbb_values and n_violations."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            s.setup()

            results = s._test_training_stability()

        assert "lbb_values" in results
        assert "n_violations" in results
        assert len(results["lbb_values"]) == small_config.n_training_steps
        assert results["n_violations"] == 0

    def test_training_stability_uses_middle_resolution(
        self, small_config: StabilityScenarioConfig
    ) -> None:
        """Training stability uses the middle resolution from the list."""
        from src.poc.scenarios.stability import StabilityScenario

        captured_n_tokens: list[int] = []

        class TrackingProjection(nn.Module):
            def __init__(self, d_model: int, d_key: int, d_value: int) -> None:
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                captured_n_tokens.append(x.shape[1])
                return self.linear(x)

            def compute_lbb_constant(self, x: torch.Tensor) -> torch.Tensor:
                return torch.full((x.shape[0],), 1.0)

        with patch(
            "src.math_kernel.integral.GalerkinProjection",
            TrackingProjection,
        ):
            s = StabilityScenario(config=small_config)
            s.setup()

            s._test_training_stability()

        # small_config.resolutions = [3, 5], middle = index 1 -> resolution 5
        expected_n_tokens = 5 * 5
        assert all(n == expected_n_tokens for n in captured_n_tokens)

    def test_init_stability_without_setup_raises(
        self, small_config: StabilityScenarioConfig
    ) -> None:
        """_test_initialization_stability raises if setup not called."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)

        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s._test_initialization_stability()

    def test_training_stability_without_setup_raises(
        self, small_config: StabilityScenarioConfig
    ) -> None:
        """_test_training_stability raises if setup not called."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(config=small_config)

        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s._test_training_stability()


# ---------------------------------------------------------------------------
# Integration test: full run() lifecycle
# ---------------------------------------------------------------------------


class TestStabilityScenarioIntegration:
    """Integration tests using run() which calls setup/execute/teardown."""

    def test_full_run_lifecycle(self, small_config: StabilityScenarioConfig) -> None:
        """Full run lifecycle completes without error."""
        from src.poc.scenarios.stability import StabilityScenario

        fake_cls = _make_fake_projection(lbb_value=0.5)

        with patch("src.math_kernel.integral.GalerkinProjection", fake_cls):
            s = StabilityScenario(config=small_config)
            result = s.run()

        assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
        assert result.duration_seconds >= 0
        assert result.scenario_name == "stability"

    def test_run_error_handling(self, small_config: StabilityScenarioConfig) -> None:
        """run() catches unexpected errors and returns ERROR status."""
        from src.poc.scenarios.stability import StabilityScenario

        class ExplodingProjection(nn.Module):
            def __init__(self, d_model: int, d_key: int, d_value: int) -> None:
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("boom")

            def compute_lbb_constant(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("boom")

        with patch(
            "src.math_kernel.integral.GalerkinProjection",
            ExplodingProjection,
        ):
            s = StabilityScenario(config=small_config)
            result = s.run()

        assert result.status == ScenarioStatus.ERROR
        assert result.passed is False
        assert "boom" in (result.error_message or "")

    def test_scenario_name_from_decorator(self) -> None:
        """StabilityScenario has the correct name from @scenario decorator."""
        from src.poc.scenarios.stability import StabilityScenario

        s = StabilityScenario(name="stability", description="test")
        assert s.name == "stability"
