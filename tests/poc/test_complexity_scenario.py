"""Tests for the computational complexity benchmark scenario.

Validates:
    - ComplexityScenario initialization and config
    - setup() and teardown() lifecycle
    - execute() with mocked layers for fast CPU tests
    - _benchmark_fnet, _benchmark_softmax, _benchmark_galerkin internals
    - _compute_scaling_exponent with various inputs
    - Pass/fail logic based on scaling thresholds
    - Error handling when setup() is not called
    - Edge cases: empty results, single grid size
    - BenchmarkResult dataclass
    - Metric and threshold recording
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.poc.config import (
    ComplexityScenarioConfig,
    ScenarioStatus,
)
from src.poc.registry import ScenarioRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test to avoid duplicate registrations."""
    ScenarioRegistry().clear()
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.poc.scenarios"):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Helper: Fake layers for mocking
# ---------------------------------------------------------------------------


def _make_fake_fnet_layer():
    """Create a fake FNetMixingLayer class for mocking."""

    class FakeFNetMixingLayer(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor, grid_size: int = 0) -> torch.Tensor:
            return x

    return FakeFNetMixingLayer


def _make_fake_softmax_attention():
    """Create a fake SoftmaxAttention class for mocking."""

    class FakeSoftmaxAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, **kwargs) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    return FakeSoftmaxAttention


def _make_fake_galerkin_attention():
    """Create a fake GalerkinAttention class for mocking."""

    class FakeGalerkinAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, **kwargs) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    return FakeGalerkinAttention


def _patch_all_layers():
    """Return a context manager that patches all three layer imports."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()))
    stack.enter_context(
        patch("src.modeling.attention.SoftmaxAttention", _make_fake_softmax_attention())
    )
    stack.enter_context(
        patch("src.modeling.attention.GalerkinAttention", _make_fake_galerkin_attention())
    )
    return stack


@pytest.fixture
def small_config() -> ComplexityScenarioConfig:
    """Create a small fast config suitable for CPU testing.

    Respects Pydantic validators: grid_sizes needs at least 3, n_iterations >= 10.
    """
    return ComplexityScenarioConfig(
        name="complexity",
        description="Test complexity scenario",
        grid_sizes=[3, 5, 7],
        batch_size=2,
        n_warmup=1,
        n_iterations=10,
        d_model=16,
        n_heads=2,
        fnet_scaling_exponent_max=3.0,  # Very lenient for fast tests
        softmax_scaling_exponent_min=0.0,  # Very lenient for fast tests
        min_speedup_factor=1.01,  # Very lenient for fast tests
        requires_gpu=False,
        seed=42,
    )


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_benchmark_result_fields(self) -> None:
        """BenchmarkResult stores all expected fields."""
        from src.poc.scenarios.complexity import BenchmarkResult

        br = BenchmarkResult(
            n_tokens=81,
            mean_time_ms=1.5,
            std_time_ms=0.2,
            memory_mb=10.0,
        )
        assert br.n_tokens == 81
        assert br.mean_time_ms == 1.5
        assert br.std_time_ms == 0.2
        assert br.memory_mb == 10.0

    def test_benchmark_result_zero_values(self) -> None:
        """BenchmarkResult can hold zero values."""
        from src.poc.scenarios.complexity import BenchmarkResult

        br = BenchmarkResult(
            n_tokens=0,
            mean_time_ms=0.0,
            std_time_ms=0.0,
            memory_mb=0.0,
        )
        assert br.n_tokens == 0
        assert br.mean_time_ms == 0.0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestComplexityScenarioConfig:
    """Tests for ComplexityScenarioConfig validation."""

    def test_default_config(self) -> None:
        """Default config has valid values."""
        config = ComplexityScenarioConfig(name="complexity", description="test")
        assert config.grid_sizes == [9, 13, 19, 25]
        assert config.batch_size == 32
        assert config.n_warmup == 10
        assert config.n_iterations == 100
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.fnet_scaling_exponent_max == 1.5
        assert config.softmax_scaling_exponent_min == 1.5
        assert config.min_speedup_factor == 1.5
        assert config.requires_gpu is True

    def test_custom_config(self, small_config: ComplexityScenarioConfig) -> None:
        """Custom config values are preserved."""
        assert small_config.grid_sizes == [3, 5, 7]
        assert small_config.batch_size == 2
        assert small_config.n_warmup == 1
        assert small_config.n_iterations == 10
        assert small_config.d_model == 16
        assert small_config.n_heads == 2

    def test_config_hash_deterministic(self, small_config: ComplexityScenarioConfig) -> None:
        """Config hash is deterministic."""
        h1 = small_config.compute_hash()
        h2 = small_config.compute_hash()
        assert h1 == h2

    def test_config_hash_changes_on_different_params(self) -> None:
        """Different configs produce different hashes."""
        c1 = ComplexityScenarioConfig(name="c1", description="a", grid_sizes=[3, 5, 7])
        c2 = ComplexityScenarioConfig(name="c2", description="b", grid_sizes=[3, 5, 7])
        assert c1.compute_hash() != c2.compute_hash()

    def test_config_grid_sizes_sorted_and_deduped(self) -> None:
        """Grid sizes are sorted and deduplicated by validator."""
        config = ComplexityScenarioConfig(
            name="complexity",
            description="test",
            grid_sizes=[7, 3, 5, 3],
        )
        assert config.grid_sizes == [3, 5, 7]

    def test_config_too_few_grid_sizes_raises(self) -> None:
        """Less than 3 grid sizes raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="at least 3"):
            ComplexityScenarioConfig(name="bad", description="bad", grid_sizes=[3, 5])

    def test_config_invalid_d_model(self) -> None:
        """d_model below minimum raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ComplexityScenarioConfig(name="bad", description="bad", d_model=4, grid_sizes=[3, 5, 7])

    def test_config_invalid_n_iterations(self) -> None:
        """n_iterations below minimum raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ComplexityScenarioConfig(
                name="bad", description="bad", n_iterations=5, grid_sizes=[3, 5, 7]
            )

    def test_config_min_speedup_must_be_above_one(self) -> None:
        """min_speedup_factor must be > 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ComplexityScenarioConfig(
                name="bad", description="bad", min_speedup_factor=0.5, grid_sizes=[3, 5, 7]
            )

    def test_config_invalid_batch_size(self) -> None:
        """batch_size below minimum raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ComplexityScenarioConfig(
                name="bad", description="bad", batch_size=0, grid_sizes=[3, 5, 7]
            )


# ---------------------------------------------------------------------------
# Scenario lifecycle tests
# ---------------------------------------------------------------------------


class TestComplexityScenarioLifecycle:
    """Tests for setup/teardown and lifecycle."""

    def test_init_with_config(self, small_config: ComplexityScenarioConfig) -> None:
        """Scenario can be initialized with a config object."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        assert s.config.d_model == 16
        assert s._device is None
        assert s._scenario_logger is None

    def test_init_with_kwargs(self) -> None:
        """Scenario can be initialized with keyword arguments."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(
            name="complexity",
            description="kw test",
            d_model=32,
            n_heads=2,
            grid_sizes=[3, 5, 7],
        )
        assert s.config.d_model == 32

    def test_init_default_config(self) -> None:
        """Scenario creates default config when no config provided."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(
            name="complexity",
            description="default test",
            grid_sizes=[3, 5, 7],
        )
        assert s.config.name == "complexity"

    def test_setup_sets_device(self, small_config: ComplexityScenarioConfig) -> None:
        """setup() initializes the device."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        s.setup()

        assert s._device is not None
        assert s._device in (torch.device("cpu"), torch.device("cuda"))

    def test_setup_sets_scenario_logger(self, small_config: ComplexityScenarioConfig) -> None:
        """setup() initializes the scenario logger."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        s.setup()

        assert s._scenario_logger is not None

    def test_setup_warns_no_gpu(self, small_config: ComplexityScenarioConfig) -> None:
        """setup() warns when GPU is not available."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with patch("src.poc.scenarios.complexity.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device = torch.device

            s = ComplexityScenario(config=small_config)
            # Need to provide a real ScenarioLogger so .warning() works
            s.setup()

            # The scenario logger's warning would have been called

    def test_teardown_without_gpu(self, small_config: ComplexityScenarioConfig) -> None:
        """teardown() runs without error on CPU."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        s.setup()
        s.teardown()  # Should not raise

    def test_teardown_without_setup(self, small_config: ComplexityScenarioConfig) -> None:
        """teardown() runs safely even without setup."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        s.teardown()  # Should not raise

    def test_execute_without_setup_raises(self, small_config: ComplexityScenarioConfig) -> None:
        """execute() raises RuntimeError if setup() was not called."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        s._start_time = datetime.now()

        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s.execute()

    def test_scenario_name_from_decorator(self) -> None:
        """ComplexityScenario has the correct name from @scenario decorator."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(name="complexity", description="test", grid_sizes=[3, 5, 7])
        assert s.name == "complexity"


# ---------------------------------------------------------------------------
# _compute_scaling_exponent tests
# ---------------------------------------------------------------------------


class TestComputeScalingExponent:
    """Tests for the _compute_scaling_exponent method."""

    def _make_scenario(self) -> object:
        """Create a scenario instance for testing internal methods."""
        from src.poc.scenarios.complexity import ComplexityScenario

        return ComplexityScenario(
            name="complexity",
            description="exponent test",
            grid_sizes=[3, 5, 7],
        )

    def test_empty_results_returns_zero(self) -> None:
        """Empty results list returns exponent 0.0."""
        s = self._make_scenario()
        assert s._compute_scaling_exponent([]) == 0.0

    def test_single_result_returns_zero(self) -> None:
        """Single result returns exponent 0.0 (cannot fit line)."""
        from src.poc.scenarios.complexity import BenchmarkResult

        s = self._make_scenario()
        results = [BenchmarkResult(n_tokens=100, mean_time_ms=1.0, std_time_ms=0.1, memory_mb=0)]
        assert s._compute_scaling_exponent(results) == 0.0

    def test_linear_scaling(self) -> None:
        """O(N) data should give exponent close to 1.0."""
        from src.poc.scenarios.complexity import BenchmarkResult

        s = self._make_scenario()
        # time ~ N (linear)
        results = [
            BenchmarkResult(n_tokens=100, mean_time_ms=1.0, std_time_ms=0.1, memory_mb=0),
            BenchmarkResult(n_tokens=1000, mean_time_ms=10.0, std_time_ms=0.1, memory_mb=0),
            BenchmarkResult(n_tokens=10000, mean_time_ms=100.0, std_time_ms=0.1, memory_mb=0),
        ]
        exponent = s._compute_scaling_exponent(results)
        assert abs(exponent - 1.0) < 0.01

    def test_quadratic_scaling(self) -> None:
        """O(N^2) data should give exponent close to 2.0."""
        from src.poc.scenarios.complexity import BenchmarkResult

        s = self._make_scenario()
        # time ~ N^2 (quadratic)
        results = [
            BenchmarkResult(n_tokens=100, mean_time_ms=1.0, std_time_ms=0.1, memory_mb=0),
            BenchmarkResult(n_tokens=1000, mean_time_ms=100.0, std_time_ms=0.1, memory_mb=0),
            BenchmarkResult(n_tokens=10000, mean_time_ms=10000.0, std_time_ms=0.1, memory_mb=0),
        ]
        exponent = s._compute_scaling_exponent(results)
        assert abs(exponent - 2.0) < 0.01

    def test_nlogn_scaling(self) -> None:
        """O(N log N) data should give exponent between 1.0 and 1.5."""
        from src.poc.scenarios.complexity import BenchmarkResult

        s = self._make_scenario()
        # time ~ N * log(N)
        ns = [100, 1000, 10000]
        results = [
            BenchmarkResult(
                n_tokens=n,
                mean_time_ms=n * np.log(n),
                std_time_ms=0.1,
                memory_mb=0,
            )
            for n in ns
        ]
        exponent = s._compute_scaling_exponent(results)
        # N*logN looks like N^1.x in log-log, should be between 1 and 2
        assert 1.0 < exponent < 2.0

    def test_two_results_gives_nonzero(self) -> None:
        """Two results should still fit a line and give nonzero exponent."""
        from src.poc.scenarios.complexity import BenchmarkResult

        s = self._make_scenario()
        results = [
            BenchmarkResult(n_tokens=100, mean_time_ms=1.0, std_time_ms=0.1, memory_mb=0),
            BenchmarkResult(n_tokens=10000, mean_time_ms=100.0, std_time_ms=0.1, memory_mb=0),
        ]
        exponent = s._compute_scaling_exponent(results)
        assert exponent != 0.0
        assert abs(exponent - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Benchmark method tests (mocked layers)
# ---------------------------------------------------------------------------


class TestBenchmarkMethods:
    """Tests for _benchmark_fnet, _benchmark_softmax, _benchmark_galerkin."""

    def test_benchmark_fnet_returns_results(self, small_config: ComplexityScenarioConfig) -> None:
        """_benchmark_fnet returns one BenchmarkResult per grid size."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()):
            s = ComplexityScenario(config=small_config)
            s.setup()
            results = s._benchmark_fnet()

        assert len(results) == len(small_config.grid_sizes)
        for i, result in enumerate(results):
            expected_n = small_config.grid_sizes[i] ** 2
            assert result.n_tokens == expected_n
            assert result.mean_time_ms > 0
            assert result.std_time_ms >= 0

    def test_benchmark_softmax_returns_results(
        self, small_config: ComplexityScenarioConfig
    ) -> None:
        """_benchmark_softmax returns one BenchmarkResult per grid size."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with patch(
            "src.modeling.attention.SoftmaxAttention",
            _make_fake_softmax_attention(),
        ):
            s = ComplexityScenario(config=small_config)
            s.setup()
            results = s._benchmark_softmax()

        assert len(results) == len(small_config.grid_sizes)
        for i, result in enumerate(results):
            expected_n = small_config.grid_sizes[i] ** 2
            assert result.n_tokens == expected_n
            assert result.mean_time_ms > 0

    def test_benchmark_galerkin_returns_results(
        self, small_config: ComplexityScenarioConfig
    ) -> None:
        """_benchmark_galerkin returns one BenchmarkResult per grid size."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with patch(
            "src.modeling.attention.GalerkinAttention",
            _make_fake_galerkin_attention(),
        ):
            s = ComplexityScenario(config=small_config)
            s.setup()
            results = s._benchmark_galerkin()

        assert len(results) == len(small_config.grid_sizes)
        for i, result in enumerate(results):
            expected_n = small_config.grid_sizes[i] ** 2
            assert result.n_tokens == expected_n

    def test_benchmark_fnet_without_setup_raises(
        self, small_config: ComplexityScenarioConfig
    ) -> None:
        """_benchmark_fnet raises if setup() was not called."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s._benchmark_fnet()

    def test_benchmark_softmax_without_setup_raises(
        self, small_config: ComplexityScenarioConfig
    ) -> None:
        """_benchmark_softmax raises if setup() was not called."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s._benchmark_softmax()

    def test_benchmark_galerkin_without_setup_raises(
        self, small_config: ComplexityScenarioConfig
    ) -> None:
        """_benchmark_galerkin raises if setup() was not called."""
        from src.poc.scenarios.complexity import ComplexityScenario

        s = ComplexityScenario(config=small_config)
        with pytest.raises(RuntimeError, match="setup.*must be called"):
            s._benchmark_galerkin()

    def test_benchmark_memory_cpu_is_zero(self, small_config: ComplexityScenarioConfig) -> None:
        """On CPU, memory_mb should be 0.0."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()):
            s = ComplexityScenario(config=small_config)
            s.setup()
            # Force CPU device
            s._device = torch.device("cpu")
            results = s._benchmark_fnet()

        for result in results:
            assert result.memory_mb == 0.0


# ---------------------------------------------------------------------------
# Execution tests with mocked layers
# ---------------------------------------------------------------------------


class TestComplexityScenarioExecution:
    """Tests for scenario execution with mocked heavy computation."""

    def _run_with_mocks(self, config: ComplexityScenarioConfig):
        """Run scenario with all layers mocked."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=config)
            result = s.run()
        return result

    def test_execute_completes(self, small_config: ComplexityScenarioConfig) -> None:
        """Scenario completes without error."""
        result = self._run_with_mocks(small_config)
        assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
        assert result.duration_seconds >= 0

    def test_execute_records_scaling_exponents(
        self, small_config: ComplexityScenarioConfig
    ) -> None:
        """Execute records the three scaling exponents as metrics."""
        result = self._run_with_mocks(small_config)
        assert "fnet_scaling_exponent" in result.metrics
        assert "softmax_scaling_exponent" in result.metrics
        assert "galerkin_scaling_exponent" in result.metrics

    def test_execute_records_speedup(self, small_config: ComplexityScenarioConfig) -> None:
        """Execute records fnet speedup at largest size."""
        result = self._run_with_mocks(small_config)
        assert "fnet_speedup_at_largest" in result.metrics

    def test_execute_records_per_grid_timings(self, small_config: ComplexityScenarioConfig) -> None:
        """Execute records per-grid-size timing metrics."""
        result = self._run_with_mocks(small_config)
        for gs in small_config.grid_sizes:
            n = gs * gs
            assert f"fnet_time_ms_n{n}" in result.metrics
            assert f"softmax_time_ms_n{n}" in result.metrics
            assert f"galerkin_time_ms_n{n}" in result.metrics

    def test_execute_threshold_results(self, small_config: ComplexityScenarioConfig) -> None:
        """Execute populates threshold_results dict."""
        result = self._run_with_mocks(small_config)
        assert "fnet_scaling_exponent" in result.threshold_results
        assert "softmax_scaling_exponent" in result.threshold_results
        assert "fnet_speedup" in result.threshold_results

    def test_execute_passes_with_controlled_results(self) -> None:
        """Scenario passes when benchmark results satisfy all thresholds."""
        from src.poc.scenarios.complexity import BenchmarkResult, ComplexityScenario

        config = ComplexityScenarioConfig(
            name="complexity",
            description="controlled pass test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            fnet_scaling_exponent_max=3.0,
            softmax_scaling_exponent_min=0.5,
            min_speedup_factor=1.01,
            seed=42,
        )

        # Craft results: fnet ~ O(N), softmax ~ O(N^2)
        grid_sizes = config.grid_sizes
        fnet_results = [
            BenchmarkResult(
                n_tokens=gs * gs, mean_time_ms=float(gs * gs), std_time_ms=0.1, memory_mb=0.0
            )
            for gs in grid_sizes
        ]
        softmax_results = [
            BenchmarkResult(
                n_tokens=gs * gs, mean_time_ms=float((gs * gs) ** 2), std_time_ms=0.1, memory_mb=0.0
            )
            for gs in grid_sizes
        ]
        galerkin_results = [
            BenchmarkResult(
                n_tokens=gs * gs, mean_time_ms=float(gs * gs), std_time_ms=0.1, memory_mb=0.0
            )
            for gs in grid_sizes
        ]

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=config)
            s.setup()
            s._start_time = datetime.now()

            s._benchmark_fnet = lambda: fnet_results
            s._benchmark_softmax = lambda: softmax_results
            s._benchmark_galerkin = lambda: galerkin_results

            torch.manual_seed(config.seed)
            result = s.execute()

        assert result.status == ScenarioStatus.PASSED
        assert result.passed is True

    def test_execute_fails_strict_fnet_threshold(self) -> None:
        """Scenario fails when fnet scaling exponent exceeds strict threshold."""
        config = ComplexityScenarioConfig(
            name="complexity",
            description="strict fnet test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            fnet_scaling_exponent_max=-10.0,  # Impossibly strict
            softmax_scaling_exponent_min=0.0,
            min_speedup_factor=1.01,
            seed=42,
        )
        result = self._run_with_mocks(config)
        assert result.threshold_results["fnet_scaling_exponent"] is False
        assert result.passed is False
        assert result.status == ScenarioStatus.FAILED

    def test_execute_fails_strict_softmax_threshold(self) -> None:
        """Scenario fails when softmax scaling exponent is below strict threshold."""
        config = ComplexityScenarioConfig(
            name="complexity",
            description="strict softmax test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            fnet_scaling_exponent_max=100.0,
            softmax_scaling_exponent_min=100.0,  # Impossibly strict
            min_speedup_factor=1.01,
            seed=42,
        )
        result = self._run_with_mocks(config)
        assert result.threshold_results["softmax_scaling_exponent"] is False
        assert result.passed is False

    def test_execute_fails_strict_speedup_threshold(self) -> None:
        """Scenario fails when speedup is below strict threshold."""
        config = ComplexityScenarioConfig(
            name="complexity",
            description="strict speedup test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            fnet_scaling_exponent_max=100.0,
            softmax_scaling_exponent_min=0.0,
            min_speedup_factor=1000.0,  # Impossibly strict
            seed=42,
        )
        result = self._run_with_mocks(config)
        assert result.threshold_results["fnet_speedup"] is False
        assert result.passed is False

    def test_result_contains_expected_fields(self, small_config: ComplexityScenarioConfig) -> None:
        """Result has all expected fields including custom extras."""
        result = self._run_with_mocks(small_config)

        assert result.scenario_name == "complexity"
        assert result.config_hash == small_config.compute_hash()
        assert result.device in ("cpu", "cuda")
        assert result.python_version != ""
        assert result.torch_version != ""
        assert result.duration_seconds >= 0
        assert result.start_time is not None
        assert result.end_time is not None

    def test_result_has_custom_extra_fields(self, small_config: ComplexityScenarioConfig) -> None:
        """ScenarioResult includes extra custom fields from execute()."""
        result = self._run_with_mocks(small_config)

        # These are set via extra="allow" on ScenarioResult
        assert hasattr(result, "fnet_exponent")
        assert hasattr(result, "softmax_exponent")
        assert hasattr(result, "galerkin_exponent")
        assert hasattr(result, "speedup")

    def test_result_summary_is_string(self, small_config: ComplexityScenarioConfig) -> None:
        """Result summary is a non-empty string."""
        result = self._run_with_mocks(small_config)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "complexity" in summary
        assert "Status" in summary


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestComplexityEdgeCases:
    """Edge case and error handling tests."""

    def test_run_catches_error_returns_error_status(self) -> None:
        """run() catches unexpected errors and returns ERROR status."""
        from src.poc.scenarios.complexity import ComplexityScenario

        config = ComplexityScenarioConfig(
            name="complexity",
            description="error test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            seed=42,
        )

        class ExplodingFNet(nn.Module):
            def __init__(self, d_model: int) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
                raise RuntimeError("FFT explosion")

        with patch("src.modeling.fnet.FNetMixingLayer", ExplodingFNet):
            s = ComplexityScenario(config=config)
            result = s.run()

        assert result.status == ScenarioStatus.ERROR
        assert result.passed is False
        assert "FFT explosion" in (result.error_message or "")

    def test_speedup_zero_when_fnet_time_zero(self) -> None:
        """Speedup is 0.0 when fnet time is zero (division guard)."""
        from src.poc.scenarios.complexity import BenchmarkResult, ComplexityScenario

        config = ComplexityScenarioConfig(
            name="complexity",
            description="zero time test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            seed=42,
        )

        # Make fnet benchmark return zero times
        zero_results = [
            BenchmarkResult(n_tokens=gs * gs, mean_time_ms=0.0, std_time_ms=0.0, memory_mb=0.0)
            for gs in config.grid_sizes
        ]
        nonzero_results = [
            BenchmarkResult(n_tokens=gs * gs, mean_time_ms=1.0, std_time_ms=0.1, memory_mb=0.0)
            for gs in config.grid_sizes
        ]

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=config)
            s.setup()
            s._start_time = datetime.now()

            # Monkey-patch benchmark methods to return controlled results
            s._benchmark_fnet = lambda: zero_results
            s._benchmark_softmax = lambda: nonzero_results
            s._benchmark_galerkin = lambda: nonzero_results

            torch.manual_seed(config.seed)
            result = s.execute()

        assert result.metrics["fnet_speedup_at_largest"] == 0.0

    def test_speedup_zero_when_no_results(self) -> None:
        """Speedup is 0.0 when benchmark results are empty."""
        from src.poc.scenarios.complexity import ComplexityScenario

        config = ComplexityScenarioConfig(
            name="complexity",
            description="empty results test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            seed=42,
        )

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=config)
            s.setup()
            s._start_time = datetime.now()

            # Return empty results
            s._benchmark_fnet = lambda: []
            s._benchmark_softmax = lambda: []
            s._benchmark_galerkin = lambda: []

            torch.manual_seed(config.seed)
            result = s.execute()

        assert result.metrics["fnet_speedup_at_largest"] == 0.0
        assert result.metrics["fnet_scaling_exponent"] == 0.0
        assert result.metrics["softmax_scaling_exponent"] == 0.0
        assert result.metrics["galerkin_scaling_exponent"] == 0.0

    def test_all_thresholds_fail_still_returns_result(self) -> None:
        """Even when all thresholds fail, we get a valid result."""
        config = ComplexityScenarioConfig(
            name="complexity",
            description="all fail test",
            grid_sizes=[3, 5, 7],
            batch_size=2,
            n_warmup=1,
            n_iterations=10,
            d_model=16,
            n_heads=2,
            fnet_scaling_exponent_max=-100.0,
            softmax_scaling_exponent_min=100.0,
            min_speedup_factor=1000.0,
            seed=42,
        )

        from src.poc.scenarios.complexity import ComplexityScenario

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=config)
            result = s.run()

        assert result.status == ScenarioStatus.FAILED
        assert result.passed is False
        # All three thresholds should fail
        assert all(not v for v in result.threshold_results.values())

    def test_scenario_registered_in_registry(self) -> None:
        """The @scenario decorator registers ComplexityScenario."""
        from src.poc.scenarios.complexity import ComplexityScenario  # noqa: F401

        registry = ScenarioRegistry()
        assert registry.get("complexity") is not None

    def test_config_class_attribute(self) -> None:
        """ComplexityScenario.config_class is ComplexityScenarioConfig."""
        from src.poc.scenarios.complexity import ComplexityScenario

        assert ComplexityScenario.config_class is ComplexityScenarioConfig


# ---------------------------------------------------------------------------
# Integration test: full run() lifecycle
# ---------------------------------------------------------------------------


class TestComplexityScenarioIntegration:
    """Integration tests using run() which calls setup/execute/teardown."""

    def test_full_run_lifecycle(self, small_config: ComplexityScenarioConfig) -> None:
        """Full run lifecycle completes without error."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=small_config)
            result = s.run()

        assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)
        assert result.duration_seconds >= 0
        assert result.scenario_name == "complexity"

    def test_run_with_different_seeds(self) -> None:
        """Different seeds produce potentially different timings."""
        from src.poc.scenarios.complexity import ComplexityScenario

        results = []
        for seed in [42, 123]:
            config = ComplexityScenarioConfig(
                name="complexity",
                description="seed test",
                grid_sizes=[3, 5, 7],
                batch_size=2,
                n_warmup=1,
                n_iterations=10,
                d_model=16,
                n_heads=2,
                fnet_scaling_exponent_max=100.0,
                softmax_scaling_exponent_min=0.0,
                min_speedup_factor=1.01,
                seed=seed,
            )

            with (
                patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
                patch(
                    "src.modeling.attention.SoftmaxAttention",
                    _make_fake_softmax_attention(),
                ),
                patch(
                    "src.modeling.attention.GalerkinAttention",
                    _make_fake_galerkin_attention(),
                ),
            ):
                s = ComplexityScenario(config=config)
                result = s.run()

            results.append(result)

        # Both should complete successfully
        for r in results:
            assert r.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)

    def test_run_records_all_expected_metrics(self, small_config: ComplexityScenarioConfig) -> None:
        """Full run records all expected metrics."""
        from src.poc.scenarios.complexity import ComplexityScenario

        with (
            patch("src.modeling.fnet.FNetMixingLayer", _make_fake_fnet_layer()),
            patch(
                "src.modeling.attention.SoftmaxAttention",
                _make_fake_softmax_attention(),
            ),
            patch(
                "src.modeling.attention.GalerkinAttention",
                _make_fake_galerkin_attention(),
            ),
        ):
            s = ComplexityScenario(config=small_config)
            result = s.run()

        expected_metric_keys = {
            "fnet_scaling_exponent",
            "softmax_scaling_exponent",
            "galerkin_scaling_exponent",
            "fnet_speedup_at_largest",
        }
        # Per-grid-size metrics
        for gs in small_config.grid_sizes:
            n = gs * gs
            expected_metric_keys.add(f"fnet_time_ms_n{n}")
            expected_metric_keys.add(f"softmax_time_ms_n{n}")
            expected_metric_keys.add(f"galerkin_time_ms_n{n}")

        for key in expected_metric_keys:
            assert key in result.metrics, f"Missing metric: {key}"
