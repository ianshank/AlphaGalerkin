"""Tests for post-processing: surface extraction, stagnation, comparison."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.postprocess.comparison import (
    ComparisonMetric,
    ValidationReport,
    compare_stagnation_heat_flux,
)
from src.reentry.postprocess.stagnation import (
    fay_riddell_correction,
    find_stagnation_point,
    sutton_graves_heat_flux,
)
from src.reentry.postprocess.surface import SurfaceData, extract_surface


class TestSurfaceExtraction:
    def test_uniform_flow_surface(self) -> None:
        ny, nx = 10, 20
        rho = np.ones((ny, nx)) * 1.225
        u = np.ones((ny, nx)) * 300.0
        v = np.zeros((ny, nx))
        p = np.ones((ny, nx)) * 101325.0
        t = np.ones((ny, nx)) * 288.15
        x_cell = np.tile(np.linspace(0.05, 1.95, nx), (ny, 1))
        y_cell = np.tile(np.linspace(0.05, 0.95, ny).reshape(-1, 1), (1, nx))

        result = extract_surface(
            rho,
            u,
            v,
            p,
            t,
            x_cell,
            y_cell,
            freestream_rho=1.225,
            freestream_u=300.0,
            freestream_p=101325.0,
            freestream_t=288.15,
        )
        assert result.x.shape == (nx,)
        # Uniform flow: Cp ~ 0
        np.testing.assert_allclose(result.cp, 0.0, atol=0.01)

    def test_surface_data_fields(self) -> None:
        sd = SurfaceData(
            x=np.array([0, 1]),
            pressure=np.array([1e5, 2e5]),
            heat_flux=np.array([1e4, 2e4]),
            cp=np.array([0, 1]),
            cf=np.array([0.001, 0.002]),
            stanton=np.array([0.01, 0.02]),
        )
        assert len(sd.x) == 2


class TestStagnationAnalysis:
    def test_find_stagnation_point(self) -> None:
        ny, nx = 10, 20
        p = np.ones((ny, nx)) * 50000.0
        p[0, 10] = 100000.0  # Stagnation point at center
        x_cell = np.tile(np.linspace(0, 2, nx), (ny, 1))
        y_cell = np.tile(np.linspace(0, 1, ny).reshape(-1, 1), (1, nx))
        hf = np.ones(nx) * 1e4
        hf[10] = 5e4
        t = np.ones((ny, nx)) * 300.0
        rho = np.ones((ny, nx)) * 0.5

        result = find_stagnation_point(p, x_cell, y_cell, hf, t, rho)
        assert result.p_stag == pytest.approx(100000.0)
        assert result.q_stag == pytest.approx(5e4)

    def test_sutton_graves(self) -> None:
        # FIRE II at t=1636s approximate conditions
        q = sutton_graves_heat_flux(
            nose_radius_m=0.9347,
            freestream_velocity_m_s=11360.0,
            freestream_density_kg_m3=4.855e-4,
        )
        # Should be order of magnitude ~MW/m^2
        assert q > 1e5
        assert q < 1e8

    def test_fay_riddell(self) -> None:
        q_frozen = 1e6
        q_corrected = fay_riddell_correction(q_frozen, lewis_number=1.4)
        # Correction should increase heat flux
        assert q_corrected > q_frozen


class TestComparisonMetric:
    def test_metric_passes(self) -> None:
        m = ComparisonMetric(
            name="test",
            computed=105.0,
            reference=100.0,
            unit="W/m^2",
            tolerance_percent=10.0,
        )
        assert m.error_percent == pytest.approx(5.0)
        assert m.passes

    def test_metric_fails(self) -> None:
        m = ComparisonMetric(
            name="test",
            computed=120.0,
            reference=100.0,
            unit="W/m^2",
            tolerance_percent=10.0,
        )
        assert not m.passes

    def test_validation_report(self) -> None:
        report = ValidationReport(
            case_name="test_case",
            metrics=[
                ComparisonMetric("a", 10.0, 10.0, "Pa", 5.0),
                ComparisonMetric("b", 20.0, 15.0, "W/m^2", 15.0),
            ],
        )
        assert report.all_pass is False  # b fails (33% error)
        summary = report.summary()
        assert "FAIL" in summary

    def test_compare_stagnation(self) -> None:
        m = compare_stagnation_heat_flux(1.0e6, 1.15e6, tolerance_percent=15.0)
        assert m.error_percent == pytest.approx(13.04, rel=0.1)
        assert m.passes
