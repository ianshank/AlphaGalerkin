"""Tests for src/research/pde_benchmarks.py.

Covers PDEBenchmarkResult, PDEBenchmarkRunner, report generation,
convergence rate calculation, and operator/solver creation helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from src.research.pde_benchmarks import PDEBenchmarkResult, PDEBenchmarkRunner

scipy = pytest.importorskip("scipy", reason="scipy required for FDM solver")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Minimal YAML config with one benchmark and the FDM baseline."""
    config = {
        "suite_name": "test_suite",
        "benchmarks": [
            {
                "name": "test_poisson",
                "pde_type": "poisson",
                "description": "1D Poisson for testing",
                "domain": {"dim": 1, "min": [0.0], "max": [1.0]},
                "parameters": {},
                "refinement_levels": [8, 16],
            }
        ],
        "baselines": [
            {"name": "uniform_fdm", "type": "classical"},
        ],
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


@pytest.fixture()
def two_baselines_config(tmp_path: Path) -> Path:
    """Config with two baselines (FDM + Dorfler AMR)."""
    config = {
        "suite_name": "two_baseline_suite",
        "benchmarks": [
            {
                "name": "bench_1d",
                "pde_type": "poisson",
                "domain": {"dim": 1, "min": [0.0], "max": [1.0]},
                "parameters": {},
                "refinement_levels": [10, 20],
            }
        ],
        "baselines": [
            {"name": "uniform_fdm"},
            {"name": "dorfler_amr", "marking_fraction": 0.3},
        ],
    }
    config_path = tmp_path / "two_baseline.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


@pytest.fixture()
def empty_baselines_config(tmp_path: Path) -> Path:
    """Config with no baselines (runner should fall back to default)."""
    config = {
        "suite_name": "empty_baselines",
        "benchmarks": [
            {
                "name": "bench_empty",
                "pde_type": "poisson",
                "domain": {"dim": 1, "min": [0.0], "max": [1.0]},
                "parameters": {},
                "refinement_levels": [8],
            }
        ],
        "baselines": [],
    }
    config_path = tmp_path / "empty_baselines.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# PDEBenchmarkResult
# ---------------------------------------------------------------------------


class TestPDEBenchmarkResult:
    def test_to_dict_basic(self):
        r = PDEBenchmarkResult(
            benchmark_name="poisson",
            method_name="fdm",
            n_dof=100,
            l2_error=0.001,
            wall_time_seconds=0.05,
        )
        d = r.to_dict()
        assert d["benchmark_name"] == "poisson"
        assert d["method_name"] == "fdm"
        assert d["n_dof"] == 100
        assert d["l2_error"] == pytest.approx(0.001)
        assert d["wall_time_seconds"] == pytest.approx(0.05)
        assert d["convergence_rate"] is None

    def test_to_dict_with_convergence_rate(self):
        r = PDEBenchmarkResult(
            benchmark_name="b",
            method_name="m",
            n_dof=64,
            l2_error=1e-3,
            wall_time_seconds=0.1,
            convergence_rate=2.0,
        )
        d = r.to_dict()
        assert d["convergence_rate"] == pytest.approx(2.0)

    def test_to_dict_with_metadata(self):
        r = PDEBenchmarkResult(
            benchmark_name="b",
            method_name="m",
            n_dof=10,
            l2_error=0.01,
            wall_time_seconds=0.0,
            metadata={"key": "value"},
        )
        assert r.to_dict()["metadata"] == {"key": "value"}

    def test_metadata_defaults_empty(self):
        r = PDEBenchmarkResult(
            benchmark_name="b", method_name="m", n_dof=1, l2_error=0.0, wall_time_seconds=0.0
        )
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# PDEBenchmarkRunner.__init__
# ---------------------------------------------------------------------------


class TestPDEBenchmarkRunnerInit:
    def test_loads_config(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        assert runner._config["suite_name"] == "test_suite"

    def test_accepts_string_path(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(str(minimal_config_yaml))
        assert runner._config is not None

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PDEBenchmarkRunner(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            PDEBenchmarkRunner(bad_yaml)


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_run_benchmark_returns_results(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = runner._config["benchmarks"][0]
        results = runner.run_benchmark(bench_cfg)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_results_have_correct_benchmark_name(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = runner._config["benchmarks"][0]
        results = runner.run_benchmark(bench_cfg)
        for r in results:
            assert r.benchmark_name == "test_poisson"

    def test_results_have_positive_dof(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = runner._config["benchmarks"][0]
        results = runner.run_benchmark(bench_cfg)
        for r in results:
            assert r.n_dof > 0

    def test_results_have_finite_time(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = runner._config["benchmarks"][0]
        results = runner.run_benchmark(bench_cfg)
        for r in results:
            assert math.isfinite(r.wall_time_seconds)
            assert r.wall_time_seconds >= 0.0

    def test_two_baselines_produce_results(self, two_baselines_config: Path):
        runner = PDEBenchmarkRunner(two_baselines_config)
        bench_cfg = runner._config["benchmarks"][0]
        results = runner.run_benchmark(bench_cfg)
        method_names = {r.method_name for r in results}
        # At least one of the registered solvers should appear
        assert len(method_names) >= 1


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_run_all_returns_list(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = runner.run_all()
        assert isinstance(results, list)

    def test_run_all_empty_benchmarks(self, tmp_path: Path):
        config = {"suite_name": "empty", "benchmarks": [], "baselines": []}
        path = tmp_path / "empty.yaml"
        path.write_text(yaml.dump(config), encoding="utf-8")
        runner = PDEBenchmarkRunner(path)
        results = runner.run_all()
        assert results == []

    def test_fallback_when_no_baselines(self, empty_baselines_config: Path):
        """Runner should fall back to default solver when baselines list is empty."""
        runner = PDEBenchmarkRunner(empty_baselines_config)
        results = runner.run_all()
        # May produce results (fallback FDM) or empty list – just no crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_generates_json(self, minimal_config_yaml: Path, tmp_path: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = runner.run_all()
        out_dir = tmp_path / "reports"
        runner.generate_report(results, out_dir)
        json_path = out_dir / "results.json"
        assert json_path.exists()

    def test_json_is_valid(self, minimal_config_yaml: Path, tmp_path: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = runner.run_all()
        out_dir = tmp_path / "reports"
        runner.generate_report(results, out_dir)
        data = json.loads((out_dir / "results.json").read_text())
        assert "suite_name" in data
        assert "results" in data
        assert data["n_results"] == len(results)

    def test_generates_markdown(self, minimal_config_yaml: Path, tmp_path: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = runner.run_all()
        out_dir = tmp_path / "reports"
        runner.generate_report(results, out_dir)
        md_path = out_dir / "results.md"
        assert md_path.exists()

    def test_markdown_contains_suite_name(self, minimal_config_yaml: Path, tmp_path: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = runner.run_all()
        out_dir = tmp_path / "reports"
        runner.generate_report(results, out_dir)
        md = (out_dir / "results.md").read_text()
        assert "test_suite" in md

    def test_report_with_empty_results(self, minimal_config_yaml: Path, tmp_path: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        out_dir = tmp_path / "empty_reports"
        runner.generate_report([], out_dir)
        assert (out_dir / "results.json").exists()
        assert (out_dir / "results.md").exists()

    def test_output_dir_created(self, minimal_config_yaml: Path, tmp_path: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        out_dir = tmp_path / "new_dir" / "subdir"
        assert not out_dir.exists()
        runner.generate_report([], out_dir)
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCSV:
    def _sample_results(self) -> list[PDEBenchmarkResult]:
        return [
            PDEBenchmarkResult(
                benchmark_name="poisson",
                method_name="fdm",
                n_dof=16,
                l2_error=1e-3,
                wall_time_seconds=0.05,
                convergence_rate=2.0,
                metadata={"seed": 42, "refinement_level": 16},
            ),
            PDEBenchmarkResult(
                benchmark_name="poisson",
                method_name="fdm",
                n_dof=64,
                l2_error=2.5e-4,
                wall_time_seconds=0.18,
                convergence_rate=None,
                metadata={"seed": 42},  # no refinement_level -> falls back to n_dof
            ),
            PDEBenchmarkResult(
                benchmark_name="burgers",
                method_name="amr",
                n_dof=32,
                l2_error=math.nan,  # NaN rows still export
                wall_time_seconds=0.10,
                metadata={},
            ),
        ]

    def test_export_csv_writes_header_and_rows(
        self, minimal_config_yaml: Path, tmp_path: Path
    ) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        out = tmp_path / "out.csv"
        runner.export_csv(self._sample_results(), out)

        lines = out.read_text().splitlines()
        assert lines[0].startswith(
            "problem,method,refinement_level,n_dof,l2_error,wall_time_seconds,convergence_rate,seed"
        )
        assert len(lines) == 4  # header + 3 data rows

    def test_export_csv_uses_refinement_level_metadata(
        self, minimal_config_yaml: Path, tmp_path: Path
    ) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        out = tmp_path / "out.csv"
        runner.export_csv(self._sample_results(), out)

        rows = out.read_text().splitlines()[1:]
        # First row has explicit metadata['refinement_level']=16
        assert rows[0].split(",")[2] == "16"
        # Second row falls back to n_dof=64 because metadata lacks the key
        assert rows[1].split(",")[2] == "64"

    def test_export_csv_leaves_nan_error_blank(
        self, minimal_config_yaml: Path, tmp_path: Path
    ) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        out = tmp_path / "out.csv"
        runner.export_csv(self._sample_results(), out)

        nan_row = out.read_text().splitlines()[3]
        # Column 4 is l2_error; NaN renders as empty to match downstream
        # pandas read_csv conventions.
        assert nan_row.split(",")[4] == ""

    def test_export_csv_creates_parent_dirs(
        self, minimal_config_yaml: Path, tmp_path: Path
    ) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        out = tmp_path / "deep" / "nested" / "out.csv"
        assert not out.parent.exists()
        runner.export_csv([], out)
        assert out.exists()
        assert out.read_text().splitlines()[0].startswith("problem,method")

    def test_generate_report_writes_csv_alongside(
        self, minimal_config_yaml: Path, tmp_path: Path
    ) -> None:
        """``generate_report`` now emits results.csv in addition to json/md."""
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = runner.run_all()
        out_dir = tmp_path / "reports"
        runner.generate_report(results, out_dir)
        assert (out_dir / "results.csv").exists()


# ---------------------------------------------------------------------------
# build_pareto_plot_data
# ---------------------------------------------------------------------------


class TestBuildParetoPlotData:
    def test_groups_by_benchmark_then_method(self, minimal_config_yaml: Path) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = [
            PDEBenchmarkResult(
                benchmark_name="poisson",
                method_name="fdm",
                n_dof=16,
                l2_error=1e-3,
                wall_time_seconds=0.05,
            ),
            PDEBenchmarkResult(
                benchmark_name="poisson",
                method_name="fdm",
                n_dof=64,
                l2_error=3e-4,
                wall_time_seconds=0.2,
            ),
            PDEBenchmarkResult(
                benchmark_name="burgers",
                method_name="amr",
                n_dof=32,
                l2_error=5e-3,
                wall_time_seconds=0.1,
            ),
        ]
        by_problem = runner.build_pareto_plot_data(results)

        assert set(by_problem.keys()) == {"poisson", "burgers"}
        assert set(by_problem["poisson"].keys()) == {"fdm"}
        assert by_problem["poisson"]["fdm"]["n_dof"] == [16, 64]
        assert by_problem["poisson"]["fdm"]["error"] == [1e-3, 3e-4]
        assert by_problem["burgers"]["amr"]["wall_time"] == [0.1]

    def test_skips_nan_error_rows(self, minimal_config_yaml: Path) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        results = [
            PDEBenchmarkResult(
                benchmark_name="poisson",
                method_name="fdm",
                n_dof=16,
                l2_error=math.nan,
                wall_time_seconds=0.05,
            ),
        ]
        assert runner.build_pareto_plot_data(results) == {}

    def test_empty_input_returns_empty_dict(self, minimal_config_yaml: Path) -> None:
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        assert runner.build_pareto_plot_data([]) == {}


# ---------------------------------------------------------------------------
# _attach_convergence_rates
# ---------------------------------------------------------------------------


class TestAttachConvergenceRates:
    def test_single_result_no_rate(self):
        results = [
            PDEBenchmarkResult(
                benchmark_name="b", method_name="m", n_dof=10, l2_error=0.1, wall_time_seconds=0.0
            )
        ]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        assert updated[0].convergence_rate is None

    def test_two_results_same_method(self):
        """With two levels, the finer result should get a convergence rate."""
        results = [
            PDEBenchmarkResult(
                benchmark_name="b", method_name="m", n_dof=16, l2_error=0.1, wall_time_seconds=0.0
            ),
            PDEBenchmarkResult(
                benchmark_name="b", method_name="m", n_dof=64, l2_error=0.025, wall_time_seconds=0.0
            ),
        ]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        # First result stays None; second gets a rate
        rates = [r.convergence_rate for r in updated]
        non_none = [r for r in rates if r is not None]
        assert len(non_none) >= 1
        # Expect convergence rate ~ 1 for 4x DOF, 4x error reduction
        # log(0.1/0.025) / log(64/16) = log(4) / log(4) = 1.0
        assert non_none[0] == pytest.approx(1.0, rel=1e-5)

    def test_zero_l2_error_no_rate(self):
        """Zero error should not produce a log-log rate."""
        results = [
            PDEBenchmarkResult(
                benchmark_name="b", method_name="m", n_dof=10, l2_error=0.0, wall_time_seconds=0.0
            ),
            PDEBenchmarkResult(
                benchmark_name="b", method_name="m", n_dof=20, l2_error=0.0, wall_time_seconds=0.0
            ),
        ]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        # Neither should have a finite rate from zeros
        assert all(r.convergence_rate is None for r in updated)

    def test_nan_l2_error_no_rate(self):
        results = [
            PDEBenchmarkResult(
                benchmark_name="b",
                method_name="m",
                n_dof=10,
                l2_error=float("nan"),
                wall_time_seconds=0.0,
            ),
            PDEBenchmarkResult(
                benchmark_name="b",
                method_name="m",
                n_dof=20,
                l2_error=float("nan"),
                wall_time_seconds=0.0,
            ),
        ]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        assert all(r.convergence_rate is None for r in updated)

    def test_different_methods_independent(self):
        """Results from different methods should not interfere."""
        results = [
            PDEBenchmarkResult(
                benchmark_name="b", method_name="fdm", n_dof=16, l2_error=0.1, wall_time_seconds=0.0
            ),
            PDEBenchmarkResult(
                benchmark_name="b",
                method_name="fdm",
                n_dof=64,
                l2_error=0.025,
                wall_time_seconds=0.0,
            ),
            PDEBenchmarkResult(
                benchmark_name="b",
                method_name="amr",
                n_dof=16,
                l2_error=0.05,
                wall_time_seconds=0.0,
            ),
            PDEBenchmarkResult(
                benchmark_name="b",
                method_name="amr",
                n_dof=64,
                l2_error=0.01,
                wall_time_seconds=0.0,
            ),
        ]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates_by_method: dict[str, list[float]] = {}
        for r in updated:
            if r.convergence_rate is not None:
                rates_by_method.setdefault(r.method_name, []).append(r.convergence_rate)
        # Both methods should have at least one rate
        assert "fdm" in rates_by_method
        assert "amr" in rates_by_method


# ---------------------------------------------------------------------------
# _normalise_solver_name (static helper)
# ---------------------------------------------------------------------------


class TestNormaliseSolverName:
    @pytest.mark.parametrize(
        "input_name, expected",
        [
            ("uniform_fem", "uniform_fdm"),
            ("uniform_fdm", "uniform_fdm"),
            ("dorfler_amr", "dorfler_amr"),
            ("pinn", "pinn"),
            ("PINN", "pinn"),
            ("unknown_solver", "unknown_solver"),
        ],
    )
    def test_normalisation(self, input_name: str, expected: str):
        result = PDEBenchmarkRunner._normalise_solver_name(input_name)
        assert result == expected


# ---------------------------------------------------------------------------
# _create_operator
# ---------------------------------------------------------------------------


class TestCreateOperator:
    def test_creates_poisson_operator(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = {"name": "test", "pde_type": "poisson", "domain": {"dim": 1}}
        op = runner._create_operator(bench_cfg)
        assert op is not None
        assert op.dim == 1

    def test_unknown_pde_type_falls_back_to_poisson(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = {"name": "test", "pde_type": "unknown_type", "domain": {"dim": 1}}
        op = runner._create_operator(bench_cfg)
        assert op is not None

    def test_creates_2d_operator(self, minimal_config_yaml: Path):
        runner = PDEBenchmarkRunner(minimal_config_yaml)
        bench_cfg = {
            "name": "test_2d",
            "pde_type": "poisson",
            "domain": {"dim": 2, "min": [0.0, 0.0], "max": [1.0, 1.0]},
        }
        op = runner._create_operator(bench_cfg)
        assert op.dim == 2
