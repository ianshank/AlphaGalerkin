"""Tests for the spectral-bias benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import src.research.extra_solvers  # noqa: F401 — registry side-effects
from src.experiments.spectral_bias_benchmark import (
    SpectralBiasBenchmark,
    SpectralBiasConfig,
    _make_sinusoidal_operator,
)
from src.research.baselines import SOLVER_REGISTRY, BaseSolver, SolverResult


# ---------------------------------------------------------------------------
# Mock solver to keep the test deterministic and fast.
# ---------------------------------------------------------------------------


class _MockBiasSolver(BaseSolver):
    """L2 error grows linearly with frequency — emulates spectral bias."""

    name = "_mock_bias"
    description = "Mock baseline whose error grows with k"

    def solve(self, operator, n_dof, **kwargs):  # type: ignore[no-untyped-def]
        freq = float(getattr(operator, "frequency", 1.0))
        # Synthetic error: 0.01 * freq
        l2 = 0.01 * freq
        return SolverResult(
            solution=np.zeros(n_dof, dtype=np.float64),
            grid_points=np.zeros((n_dof, 2), dtype=np.float64),
            n_dof=n_dof,
            wall_time_seconds=1e-3,
            l2_error=l2,
        )


@pytest.fixture(autouse=True)
def _register_mock():
    SOLVER_REGISTRY.setdefault("_mock_bias", _MockBiasSolver)
    yield
    SOLVER_REGISTRY.pop("_mock_bias", None)


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class TestSpectralBiasConfig:
    def test_defaults(self) -> None:
        cfg = SpectralBiasConfig(solvers=["_mock_bias"])
        assert cfg.frequencies == [1.0, 5.0, 10.0, 50.0]
        assert cfg.n_dof >= 16

    def test_empty_solvers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpectralBiasConfig(solvers=[])

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpectralBiasConfig(solvers=["x"], unknown=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Sinusoidal operator
# ---------------------------------------------------------------------------


class TestSinusoidalOperator:
    def test_source_term_sin_product(self) -> None:
        op = _make_sinusoidal_operator(frequency=2.0, domain=(0.0, 1.0))
        # At (0.25, 0.25), source = sin(2π·0.25)·sin(2π·0.25) = 1
        coords = np.array([[0.25, 0.25]], dtype=np.float32)
        f = op.source_term(coords)
        assert f.shape == (1,)
        assert f[0] == pytest.approx(1.0, abs=1e-5)

    def test_exact_solution_satisfies_poisson(self) -> None:
        # u = sin(kπx) sin(kπy) / (2 (kπ)²)  ⇒  -Δu = sin(kπx) sin(kπy) = f
        op = _make_sinusoidal_operator(frequency=3.0, domain=(0.0, 1.0))
        coords = np.array([[0.4, 0.6]], dtype=np.float32)
        u = op.exact_solution(coords)
        assert u is not None
        assert np.all(np.isfinite(u))

    def test_frequency_attribute(self) -> None:
        op = _make_sinusoidal_operator(frequency=7.0, domain=(0.0, 1.0))
        assert op.frequency == 7.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestSpectralBiasBenchmark:
    def test_run_records_each_cell(self) -> None:
        cfg = SpectralBiasConfig(
            solvers=["_mock_bias"],
            frequencies=[1.0, 5.0, 10.0],
            n_dof=32,
        )
        report = SpectralBiasBenchmark(cfg).run()
        assert len(report.measurements) == 3
        assert all(m.success for m in report.measurements)

    def test_l2_error_grows_with_frequency(self) -> None:
        cfg = SpectralBiasConfig(
            solvers=["_mock_bias"],
            frequencies=[1.0, 5.0, 50.0],
            n_dof=32,
        )
        report = SpectralBiasBenchmark(cfg).run()
        errors = [m.l2_error for m in report.measurements]
        assert errors[0] is not None and errors[-1] is not None
        assert errors[0] < errors[-1]

    def test_unknown_solver_recorded_as_failure(self) -> None:
        cfg = SpectralBiasConfig(
            solvers=["nonexistent"],
            frequencies=[1.0],
            n_dof=32,
        )
        report = SpectralBiasBenchmark(cfg).run()
        assert report.measurements[0].success is False

    def test_save_emits_csv_and_json(self, tmp_path: Path) -> None:
        cfg = SpectralBiasConfig(
            solvers=["_mock_bias"],
            frequencies=[1.0, 5.0],
            n_dof=32,
        )
        bench = SpectralBiasBenchmark(cfg)
        report = bench.run()
        artefacts = bench.save(report, tmp_path)
        assert artefacts["csv"].exists()
        assert artefacts["json"].exists()
        data = json.loads(artefacts["json"].read_text())
        assert data["config"]["solvers"] == ["_mock_bias"]
        assert len(data["measurements"]) == 2
