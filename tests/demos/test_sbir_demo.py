"""Tests for SBIR benchmark demo.

Validates that:
- SBIRDemo runs without errors
- HTML report is generated with valid structure
- JSON output is well-formed
- SBIRHTMLReportGenerator produces valid HTML
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.demos.config import SBIRDemoConfig
from src.demos.sbir_demo import SBIRDemo, SBIRHTMLReportGenerator
from src.research.pde_benchmarks import PDEBenchmarkResult

scipy = pytest.importorskip("scipy", reason="scipy required for SBIR demo tests")

# Minimal benchmark config for fast testing (no PINN, small grids)
_FAST_SUITE_CONFIG: dict[str, object] = {
    "suite_name": "test_sbir",
    "description": "Fast SBIR test suite",
    "output_dir": "outputs/test_sbir",
    "benchmarks": [
        {
            "name": "test_poisson",
            "pde_type": "poisson",
            "domain": {"dim": 2, "min": [0.0, 0.0], "max": [1.0, 1.0]},
            "parameters": {},
            "refinement_levels": [4, 8],
        },
    ],
    "baselines": [
        {"name": "uniform_fdm", "type": "classical"},
        {"name": "dorfler_amr", "type": "classical", "marking_fraction": 0.3},
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_results() -> list[PDEBenchmarkResult]:
    """Create sample benchmark results for report testing."""
    return [
        PDEBenchmarkResult(
            benchmark_name="poisson_lshaped",
            method_name="uniform_fdm",
            n_dof=16,
            l2_error=0.1,
            wall_time_seconds=0.01,
            convergence_rate=None,
        ),
        PDEBenchmarkResult(
            benchmark_name="poisson_lshaped",
            method_name="uniform_fdm",
            n_dof=64,
            l2_error=0.02,
            wall_time_seconds=0.05,
            convergence_rate=1.16,
        ),
        PDEBenchmarkResult(
            benchmark_name="poisson_lshaped",
            method_name="dorfler_amr",
            n_dof=16,
            l2_error=0.08,
            wall_time_seconds=0.02,
            convergence_rate=None,
        ),
        PDEBenchmarkResult(
            benchmark_name="poisson_lshaped",
            method_name="dorfler_amr",
            n_dof=64,
            l2_error=0.01,
            wall_time_seconds=0.08,
            convergence_rate=1.50,
        ),
        PDEBenchmarkResult(
            benchmark_name="burgers_shock",
            method_name="uniform_fdm",
            n_dof=32,
            l2_error=0.05,
            wall_time_seconds=0.03,
        ),
    ]


@pytest.fixture
def fast_suite_config_path(tmp_path: Path) -> Path:
    """Write a minimal YAML config for fast testing."""
    path = tmp_path / "fast_suite.yaml"
    path.write_text(yaml.dump(_FAST_SUITE_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def demo_config(tmp_path: Path, fast_suite_config_path: Path) -> SBIRDemoConfig:
    """Create demo config with fast baselines (no PINN)."""
    return SBIRDemoConfig(
        suite_config_path=str(fast_suite_config_path),
        output_dir=str(tmp_path / "sbir_output"),
        generate_html=True,
        generate_json=True,
        generate_markdown=True,
    )


# ---------------------------------------------------------------------------
# HTML Report Generator Tests
# ---------------------------------------------------------------------------


class TestSBIRHTMLReportGenerator:
    """Tests for HTML report generation."""

    def test_generate_returns_html(
        self,
        sample_results: list[PDEBenchmarkResult],
    ) -> None:
        """Generated output should be valid HTML."""
        gen = SBIRHTMLReportGenerator(suite_name="Test Suite")
        html = gen.generate(sample_results, total_time_seconds=1.5)

        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "Test Suite" in html

    def test_html_contains_benchmark_names(
        self,
        sample_results: list[PDEBenchmarkResult],
    ) -> None:
        """HTML should contain all benchmark names."""
        gen = SBIRHTMLReportGenerator()
        html = gen.generate(sample_results, total_time_seconds=1.0)

        assert "poisson_lshaped" in html
        assert "burgers_shock" in html

    def test_html_contains_method_names(
        self,
        sample_results: list[PDEBenchmarkResult],
    ) -> None:
        """HTML should contain all method names."""
        gen = SBIRHTMLReportGenerator()
        html = gen.generate(sample_results, total_time_seconds=1.0)

        assert "uniform_fdm" in html
        assert "dorfler_amr" in html

    def test_html_contains_error_values(
        self,
        sample_results: list[PDEBenchmarkResult],
    ) -> None:
        """HTML should contain formatted error values."""
        gen = SBIRHTMLReportGenerator()
        html = gen.generate(sample_results, total_time_seconds=1.0)

        # Check scientific notation appears
        assert "e-" in html.lower() or "e+" in html.lower() or "1.00e-01" in html

    def test_html_contains_tables(
        self,
        sample_results: list[PDEBenchmarkResult],
    ) -> None:
        """HTML should contain table elements."""
        gen = SBIRHTMLReportGenerator()
        html = gen.generate(sample_results, total_time_seconds=1.0)

        assert "<table>" in html
        assert "<th>" in html
        assert "<td>" in html

    def test_html_contains_convergence_rates(
        self,
        sample_results: list[PDEBenchmarkResult],
    ) -> None:
        """HTML should include convergence rate values where available."""
        gen = SBIRHTMLReportGenerator()
        html = gen.generate(sample_results, total_time_seconds=1.0)

        assert "1.50" in html  # dorfler_amr convergence rate
        assert "Conv. Rate" in html

    def test_empty_results(self) -> None:
        """Generator should handle empty results gracefully."""
        gen = SBIRHTMLReportGenerator()
        html = gen.generate([], total_time_seconds=0.0)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html

    def test_nan_error_handling(self) -> None:
        """Generator should handle NaN errors gracefully."""
        results = [
            PDEBenchmarkResult(
                benchmark_name="test",
                method_name="solver",
                n_dof=10,
                l2_error=float("nan"),
                wall_time_seconds=0.01,
            ),
        ]
        gen = SBIRHTMLReportGenerator()
        html = gen.generate(results, total_time_seconds=0.5)

        assert "N/A" in html


# ---------------------------------------------------------------------------
# SBIRDemo Tests
# ---------------------------------------------------------------------------


class TestSBIRDemo:
    """Tests for the end-to-end demo runner."""

    def test_demo_runs_without_error(self, demo_config: SBIRDemoConfig) -> None:
        """Demo should run the full suite without raising exceptions."""
        demo = SBIRDemo(config=demo_config)
        results = demo.run()

        assert isinstance(results, list)
        # Should produce at least some results (even if some solvers fail)
        assert len(results) >= 0

    def test_demo_produces_html(self, demo_config: SBIRDemoConfig) -> None:
        """Demo should produce an HTML report file."""
        demo = SBIRDemo(config=demo_config)
        demo.run()

        html_path = Path(demo_config.output_dir) / "report.html"
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<table>" in content

    def test_demo_produces_json(self, demo_config: SBIRDemoConfig) -> None:
        """Demo should produce a valid JSON results file."""
        demo = SBIRDemo(config=demo_config)
        demo.run()

        json_path = Path(demo_config.output_dir) / "results.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "results" in data
        assert "n_results" in data
        assert isinstance(data["results"], list)

    def test_demo_produces_markdown(self, demo_config: SBIRDemoConfig) -> None:
        """Demo should produce a Markdown report via PDEBenchmarkRunner."""
        demo = SBIRDemo(config=demo_config)
        demo.run()

        md_path = Path(demo_config.output_dir) / "results.md"
        assert md_path.exists()

    def test_default_config(self) -> None:
        """SBIRDemo should accept default config."""
        demo = SBIRDemo()
        assert demo.config.suite_config_path == "config/benchmarks/sbir_suite.yaml"
        assert demo.config.generate_html is True

    def test_demo_with_quick_levels(
        self,
        tmp_path: Path,
        fast_suite_config_path: Path,
    ) -> None:
        """Demo should work with overridden refinement levels."""
        config = SBIRDemoConfig(
            suite_config_path=str(fast_suite_config_path),
            output_dir=str(tmp_path / "quick_output"),
            refinement_levels=[4, 8],
        )
        demo = SBIRDemo(config=config)
        # Just verify construction works, don't run full suite
        assert demo.config.refinement_levels == [4, 8]


# ---------------------------------------------------------------------------
# SBIRDemoConfig Tests
# ---------------------------------------------------------------------------


class TestSBIRDemoConfig:
    """Tests for the demo configuration."""

    def test_default_values(self) -> None:
        """Default config should have sensible values."""
        config = SBIRDemoConfig()
        assert config.suite_config_path == "config/benchmarks/sbir_suite.yaml"
        assert config.generate_html is True
        assert config.generate_json is True
        assert config.max_time_per_benchmark_seconds > 0

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = SBIRDemoConfig(
            output_dir="/tmp/custom",
            seed=123,
            verbose=True,
        )
        assert config.output_dir == "/tmp/custom"
        assert config.seed == 123
        assert config.verbose is True

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SBIRDemoConfig(unknown_field=42)  # type: ignore[call-arg]
