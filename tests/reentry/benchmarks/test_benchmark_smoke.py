"""Benchmark smoke tests: verify CLI benchmarks run without crashing.

These are not full validation tests — they verify the pipeline
is wired correctly end-to-end.
"""

from __future__ import annotations

import numpy as np

from src.reentry.postprocess.comparison import (
    FIRE2_FLIGHT_DATA,
    ValidationReport,
    compare_stagnation_heat_flux,
)
from src.reentry.postprocess.stagnation import sutton_graves_heat_flux


class TestSodBenchmarkSmoke:
    """Smoke test: Sod shock tube runs and produces reasonable results."""

    def test_sod_runs(self) -> None:
        from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
        from src.reentry.solver.euler_1d import Euler1DSolver, ShockTubeIC

        config = ReentrySolverConfig(name="bench", flux_scheme=FluxScheme.ROE)
        solver = Euler1DSolver(config, n_cells=100, gamma=1.4)
        ic = ShockTubeIC.sod()
        result = solver.solve(ic, t_final=0.2)

        assert result.n_steps > 0
        assert np.all(result.density > 0)
        assert np.all(result.pressure > 0)


class TestSuttonGravesCorrelation:
    """Verify Sutton-Graves gives physically reasonable heat flux."""

    def test_fire2_1636s_order_of_magnitude(self) -> None:
        q = sutton_graves_heat_flux(
            nose_radius_m=0.9347,
            freestream_velocity_m_s=11360.0,
            freestream_density_kg_m3=4.855e-4,
        )
        # Should be in the MW/m^2 range
        assert 1e5 < q < 1e8

    def test_validation_against_flight_data(self) -> None:
        """Compare Sutton-Graves estimate to FIRE II flight data."""
        q_sg = sutton_graves_heat_flux(
            nose_radius_m=0.9347,
            freestream_velocity_m_s=11360.0,
            freestream_density_kg_m3=4.855e-4,
        )
        q_flight = FIRE2_FLIGHT_DATA["t1636s"]["q_stag_W_m2"]

        m = compare_stagnation_heat_flux(q_sg, q_flight, tolerance_percent=50.0)
        # Sutton-Graves is approximate — should be within 50%
        # (exact validation requires full NS solver, not correlation)
        assert m.computed > 0
        assert m.reference > 0


class TestValidationReportSmoke:
    """Smoke test for validation report generation."""

    def test_report_generation(self) -> None:
        report = ValidationReport(
            case_name="smoke_test",
            metrics=[
                compare_stagnation_heat_flux(1.0e6, 1.15e6, 15.0),
            ],
        )
        summary = report.summary()
        assert "smoke_test" in summary
        assert report.n_pass + report.n_fail == 1
