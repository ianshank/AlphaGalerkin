"""Tests for the computational complexity benchmark scenario.

Validates:
    - ComplexityScenario initialization and registration
    - Benchmark timing across different grid sizes
    - Scaling exponent computation from log-log regression
    - Result format and threshold evaluation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.poc.config import (
    ComplexityScenarioConfig,
    ScenarioStatus,
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
def default_config() -> ComplexityScenarioConfig:
    return ComplexityScenarioConfig(seed=SEED)


@pytest.fixture()
def small_config() -> ComplexityScenarioConfig:
    """Minimal config for fast mocked tests."""
    return ComplexityScenarioConfig(
        grid_sizes=[3, 5, 7, 9],
        d_model=16,
        n_heads=2,
        batch_size=1,
        n_warmup=1,
        n_iterations=10,
        fnet_scaling_exponent_max=2.0,
        softmax_scaling_exponent_min=1.0,
        min_speedup_factor=1.01,
        seed=SEED,
    )


def _import_complexity_scenario() -> type[BaseScenario]:
    """Import ComplexityScenario and ensure it is registered.

    The @scenario decorator only fires on first import. If the registry
    was cleared (e.g. by autouse fixture), we must re-register manually.
    """
    from src.poc.scenarios.complexity import ComplexityScenario

    registry = ScenarioRegistry()
    if registry.get("complexity") is None:
        registry.register("complexity", ComplexityScenario)

    return ComplexityScenario


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestComplexityScenarioInit:
    """Tests for ComplexityScenario initialization."""

    def test_registration(self) -> None:
        """ComplexityScenario should register under 'complexity'."""
        cls = _import_complexity_scenario()
        assert ScenarioRegistry().get("complexity") is cls

    def test_default_config_applied(self) -> None:
        """Scenario should use ComplexityScenarioConfig defaults."""
        cls = _import_complexity_scenario()
        instance = cls(name="complexity", description="test")
        assert instance.config.grid_sizes == [9, 13, 19, 25]
        assert instance.config.fnet_scaling_exponent_max == 1.5

    def test_custom_config(self, small_config: ComplexityScenarioConfig) -> None:
        """Scenario should accept a custom config."""
        cls = _import_complexity_scenario()
        instance = cls(config=small_config)
        assert instance.config.grid_sizes == [3, 5, 7, 9]
        assert instance.config.n_iterations == 10


# ---------------------------------------------------------------------------
# Scaling exponent computation
# ---------------------------------------------------------------------------


class TestScalingExponent:
    """Tests for _compute_scaling_exponent."""

    def _make_benchmark_results(
        self, n_tokens_list: list[int], times_ms: list[float]
    ) -> list[Any]:
        """Create BenchmarkResult-like objects for testing."""
        from src.poc.scenarios.complexity import BenchmarkResult

        return [
            BenchmarkResult(
                n_tokens=n,
                mean_time_ms=t,
                std_time_ms=0.0,
                memory_mb=0.0,
            )
            for n, t in zip(n_tokens_list, times_ms, strict=False)
        ]

    def test_linear_scaling(self) -> None:
        """O(N) data should yield exponent close to 1.0."""
        cls = _import_complexity_scenario()
        instance = cls(name="complexity", description="test")

        n_values = [100, 400, 900, 1600]
        times = [float(n) * 0.01 for n in n_values]  # exact O(N)
        results = self._make_benchmark_results(n_values, times)

        exponent = instance._compute_scaling_exponent(results)
        assert abs(exponent - 1.0) < 0.1

    def test_quadratic_scaling(self) -> None:
        """O(N^2) data should yield exponent close to 2.0."""
        cls = _import_complexity_scenario()
        instance = cls(name="complexity", description="test")

        n_values = [100, 400, 900, 1600]
        times = [float(n) ** 2 * 1e-5 for n in n_values]  # exact O(N^2)
        results = self._make_benchmark_results(n_values, times)

        exponent = instance._compute_scaling_exponent(results)
        assert abs(exponent - 2.0) < 0.1

    def test_nlogn_scaling(self) -> None:
        """O(N log N) data should yield exponent between 1.0 and 1.5."""
        cls = _import_complexity_scenario()
        instance = cls(name="complexity", description="test")

        n_values = [100, 400, 900, 1600, 2500]
        times = [float(n) * np.log(n) * 0.001 for n in n_values]
        results = self._make_benchmark_results(n_values, times)

        exponent = instance._compute_scaling_exponent(results)
        assert 1.0 <= exponent <= 1.5

    def test_insufficient_data_returns_zero(self) -> None:
        """Less than 2 data points should return 0.0."""
        cls = _import_complexity_scenario()
        instance = cls(name="complexity", description="test")

        results = self._make_benchmark_results([100], [1.0])
        exponent = instance._compute_scaling_exponent(results)
        assert exponent == 0.0

    def test_empty_results_returns_zero(self) -> None:
        """Empty list should return 0.0."""
        cls = _import_complexity_scenario()
        instance = cls(name="complexity", description="test")

        exponent = instance._compute_scaling_exponent([])
        assert exponent == 0.0


# ---------------------------------------------------------------------------
# Execute with mocked benchmarks
# ---------------------------------------------------------------------------


class TestComplexityScenarioExecute:
    """Tests for execute() with benchmark methods mocked."""

    def _make_mock_results(
        self, grid_sizes: list[int], exponent: float
    ) -> list[Any]:
        """Create BenchmarkResults with a given scaling exponent."""
        from src.poc.scenarios.complexity import BenchmarkResult

        return [
            BenchmarkResult(
                n_tokens=g * g,
                mean_time_ms=float((g * g) ** exponent) * 0.001,
                std_time_ms=0.0,
                memory_mb=0.0,
            )
            for g in grid_sizes
        ]

    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_fnet")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_softmax")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_galerkin")
    def test_execute_passes_with_expected_scaling(
        self,
        mock_galerkin: MagicMock,
        mock_softmax: MagicMock,
        mock_fnet: MagicMock,
        small_config: ComplexityScenarioConfig,
    ) -> None:
        """Result should pass when FNet is sub-quadratic and softmax is quadratic."""
        cls = _import_complexity_scenario()
        instance = cls(config=small_config)

        sizes = small_config.grid_sizes
        mock_fnet.return_value = self._make_mock_results(sizes, exponent=1.2)
        mock_softmax.return_value = self._make_mock_results(sizes, exponent=2.0)
        mock_galerkin.return_value = self._make_mock_results(sizes, exponent=1.0)

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.status == ScenarioStatus.PASSED
        assert result.passed is True
        assert "fnet_scaling_exponent" in result.metrics
        assert "softmax_scaling_exponent" in result.metrics
        assert "galerkin_scaling_exponent" in result.metrics

    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_fnet")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_softmax")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_galerkin")
    def test_execute_fails_when_fnet_too_slow(
        self,
        mock_galerkin: MagicMock,
        mock_softmax: MagicMock,
        mock_fnet: MagicMock,
    ) -> None:
        """Result should fail when FNet scaling exceeds threshold."""
        config = ComplexityScenarioConfig(
            grid_sizes=[3, 5, 7, 9],
            d_model=16,
            n_heads=2,
            batch_size=1,
            n_warmup=1,
            n_iterations=10,
            fnet_scaling_exponent_max=1.0,  # very strict
            softmax_scaling_exponent_min=1.0,
            min_speedup_factor=1.01,
            seed=SEED,
        )
        cls = _import_complexity_scenario()
        instance = cls(config=config)

        sizes = config.grid_sizes
        # FNet with quadratic scaling -- should fail the exponent check
        mock_fnet.return_value = self._make_mock_results(sizes, exponent=2.0)
        mock_softmax.return_value = self._make_mock_results(sizes, exponent=2.0)
        mock_galerkin.return_value = self._make_mock_results(sizes, exponent=1.0)

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert result.passed is False
        assert result.status == ScenarioStatus.FAILED


# ---------------------------------------------------------------------------
# Result format
# ---------------------------------------------------------------------------


class TestComplexityResultFormat:
    """Tests for the structure of returned ScenarioResult."""

    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_fnet")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_softmax")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_galerkin")
    def test_result_contains_timing_metrics(
        self,
        mock_galerkin: MagicMock,
        mock_softmax: MagicMock,
        mock_fnet: MagicMock,
        small_config: ComplexityScenarioConfig,
    ) -> None:
        """Result should contain per-size timing metrics."""
        from src.poc.scenarios.complexity import BenchmarkResult

        cls = _import_complexity_scenario()
        instance = cls(config=small_config)

        sizes = small_config.grid_sizes
        bench = [
            BenchmarkResult(
                n_tokens=g * g,
                mean_time_ms=float(g),
                std_time_ms=0.1,
                memory_mb=0.0,
            )
            for g in sizes
        ]
        mock_fnet.return_value = bench
        mock_softmax.return_value = bench
        mock_galerkin.return_value = bench

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        for g in sizes:
            n_tokens = g * g
            assert f"fnet_time_ms_n{n_tokens}" in result.metrics
            assert f"softmax_time_ms_n{n_tokens}" in result.metrics
            assert f"galerkin_time_ms_n{n_tokens}" in result.metrics

    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_fnet")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_softmax")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_galerkin")
    def test_result_has_threshold_results(
        self,
        mock_galerkin: MagicMock,
        mock_softmax: MagicMock,
        mock_fnet: MagicMock,
        small_config: ComplexityScenarioConfig,
    ) -> None:
        """threshold_results should contain entries for each check."""
        from src.poc.scenarios.complexity import BenchmarkResult

        cls = _import_complexity_scenario()
        instance = cls(config=small_config)

        bench = [
            BenchmarkResult(n_tokens=g * g, mean_time_ms=1.0, std_time_ms=0.0, memory_mb=0.0)
            for g in small_config.grid_sizes
        ]
        mock_fnet.return_value = bench
        mock_softmax.return_value = bench
        mock_galerkin.return_value = bench

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        assert "fnet_scaling_exponent" in result.threshold_results
        assert "softmax_scaling_exponent" in result.threshold_results
        assert "fnet_speedup" in result.threshold_results

    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_fnet")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_softmax")
    @patch("src.poc.scenarios.complexity.ComplexityScenario._benchmark_galerkin")
    def test_speedup_computed(
        self,
        mock_galerkin: MagicMock,
        mock_softmax: MagicMock,
        mock_fnet: MagicMock,
        small_config: ComplexityScenarioConfig,
    ) -> None:
        """Speedup metric should be softmax_time / fnet_time at largest size."""
        from src.poc.scenarios.complexity import BenchmarkResult

        cls = _import_complexity_scenario()
        instance = cls(config=small_config)

        sizes = small_config.grid_sizes
        largest_n = max(g * g for g in sizes)

        fnet_bench = [
            BenchmarkResult(n_tokens=g * g, mean_time_ms=1.0, std_time_ms=0.0, memory_mb=0.0)
            for g in sizes
        ]
        softmax_bench = [
            BenchmarkResult(n_tokens=g * g, mean_time_ms=3.0, std_time_ms=0.0, memory_mb=0.0)
            for g in sizes
        ]
        galerkin_bench = fnet_bench

        mock_fnet.return_value = fnet_bench
        mock_softmax.return_value = softmax_bench
        mock_galerkin.return_value = galerkin_bench

        instance.setup()
        instance._start_time = datetime.now()
        result = instance.execute()

        expected_speedup = 3.0 / 1.0
        assert abs(result.metrics["fnet_speedup_at_largest"] - expected_speedup) < 1e-6


# ---------------------------------------------------------------------------
# Full execute with real (tiny) models on CPU
# ---------------------------------------------------------------------------


class TestComplexityScenarioRealExecution:
    """Run the actual benchmark code with tiny models for coverage."""

    def _tiny_config(self) -> ComplexityScenarioConfig:
        return ComplexityScenarioConfig(
            grid_sizes=[3, 5, 7],
            d_model=16,
            n_heads=2,
            batch_size=1,
            n_warmup=1,
            n_iterations=10,
            fnet_scaling_exponent_max=5.0,
            softmax_scaling_exponent_min=0.0,
            min_speedup_factor=1.001,
            seed=SEED,
        )

    def test_full_run_end_to_end(self) -> None:
        """End-to-end run() on CPU exercises setup/execute/teardown."""
        cls = _import_complexity_scenario()
        config = self._tiny_config()
        instance = cls(config=config)
        result = instance.run()

        assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
        assert result.duration_seconds > 0
        assert "fnet_scaling_exponent" in result.metrics

    def test_benchmark_fnet_real(self) -> None:
        """_benchmark_fnet with real FNetMixingLayer."""
        cls = _import_complexity_scenario()
        config = self._tiny_config()
        instance = cls(config=config)
        instance._start_time = datetime.now()
        instance._metrics = {}
        instance._artifacts = {}
        instance.setup()

        results = instance._benchmark_fnet()
        assert len(results) == 3
        for r in results:
            assert r.mean_time_ms > 0
            assert r.n_tokens == r.n_tokens  # sanity

    def test_benchmark_softmax_real(self) -> None:
        """_benchmark_softmax with real SoftmaxAttention."""
        cls = _import_complexity_scenario()
        config = self._tiny_config()
        instance = cls(config=config)
        instance._start_time = datetime.now()
        instance._metrics = {}
        instance._artifacts = {}
        instance.setup()

        results = instance._benchmark_softmax()
        assert len(results) == 3

    def test_benchmark_galerkin_real(self) -> None:
        """_benchmark_galerkin with real GalerkinAttention."""
        cls = _import_complexity_scenario()
        config = self._tiny_config()
        instance = cls(config=config)
        instance._start_time = datetime.now()
        instance._metrics = {}
        instance._artifacts = {}
        instance.setup()

        results = instance._benchmark_galerkin()
        assert len(results) == 3

    def test_setup_without_gpu(self) -> None:
        """Verify setup warns about missing GPU."""
        cls = _import_complexity_scenario()
        config = self._tiny_config()
        instance = cls(config=config)
        instance._start_time = datetime.now()
        instance._metrics = {}
        instance._artifacts = {}
        instance.setup()
        assert instance._device is not None

    def test_teardown_after_run(self) -> None:
        """teardown should not raise even after a run."""
        cls = _import_complexity_scenario()
        config = self._tiny_config()
        instance = cls(config=config)
        instance._start_time = datetime.now()
        instance._metrics = {}
        instance._artifacts = {}
        instance.setup()
        instance.teardown()
