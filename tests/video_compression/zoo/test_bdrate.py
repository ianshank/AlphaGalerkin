"""Tests for :mod:`src.video_compression.zoo.bdrate`."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.video_compression.metrics.rd_curves import RDCurve, RDPoint
from src.video_compression.zoo.bdrate import (
    BD_RATE_REPORT_SCHEMA_VERSION,
    DEFAULT_PRIMARY_BD_RATE_GATE_PCT,
    BDRateAssemblyError,
    BDRateConfig,
    BDRatePoint,
    BDRateReport,
    compute_bd_rate_report,
)

# --------------------------------------------------------------------------
# Helpers — curve builders.
# --------------------------------------------------------------------------


def _build_curve(
    name: str,
    *,
    rates_psnrs: list[tuple[float, float]],
    lambdas: list[float] | None = None,
) -> RDCurve:
    """Build a curve from explicit (rate, psnr) pairs."""
    curve = RDCurve(name=name, points=[])
    n = len(rates_psnrs)
    if lambdas is None:
        lambdas = [0.001 * (i + 1) for i in range(n)]
    for (rate, psnr), lam in zip(rates_psnrs, lambdas, strict=True):
        curve.add_point(RDPoint(rate=rate, distortion=10 ** (-psnr / 10), psnr=psnr, lambda_rd=lam))
    return curve


def _shift_psnr(curve: RDCurve, *, db: float) -> RDCurve:
    """Build a parallel curve shifted up by ``db`` dB."""
    points = [
        RDPoint(
            rate=p.rate,
            distortion=p.distortion,
            psnr=(p.psnr + db) if p.psnr is not None else None,
            ssim=p.ssim,
            lambda_rd=p.lambda_rd,
        )
        for p in curve.points
    ]
    return RDCurve(name=f"{curve.name}_shifted", points=points)


# --------------------------------------------------------------------------
# BDRateConfig
# --------------------------------------------------------------------------


class TestBDRateConfig:
    def test_defaults(self) -> None:
        cfg = BDRateConfig(name="cfg")
        assert cfg.metric == "psnr"
        assert cfg.primary_lambda_rd == pytest.approx(0.015)
        assert cfg.primary_bd_rate_gate_pct == pytest.approx(DEFAULT_PRIMARY_BD_RATE_GATE_PCT)

    def test_unknown_field_ignored_for_forward_compat(self) -> None:
        cfg = BDRateConfig.model_validate(
            {"name": "cfg", "future_v2_field": 1.5},
        )
        assert cfg.name == "cfg"

    def test_gate_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            BDRateConfig(name="cfg", primary_bd_rate_gate_pct=200.0)

    def test_primary_lambda_rd_can_be_none(self) -> None:
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        assert cfg.primary_lambda_rd is None

    def test_overlap_fraction_bounds(self) -> None:
        with pytest.raises(ValidationError):
            BDRateConfig(name="cfg", min_overlap_fraction=1.5)


# --------------------------------------------------------------------------
# Self-identity: a curve compared against itself yields BD-rate == 0.
# --------------------------------------------------------------------------


class TestSelfIdentity:
    def test_identical_curves_zero_bd_rate(self) -> None:
        curve = _build_curve(
            "test",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        cfg = BDRateConfig(name="cfg")
        report = compute_bd_rate_report(curve, curve, cfg)
        assert report.bd_rate_pct == pytest.approx(0.0, abs=1e-6)
        assert report.bd_psnr_db == pytest.approx(0.0, abs=1e-6)
        # Self-vs-self is fully overlapping → gate "passed" against the
        # default −15 % gate (0.0 <= -15.0 is False, so "failed" at default).
        assert report.gate_status in {"passed", "failed"}
        assert report.overlap_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Test-better-than-reference: shifting test up should yield negative BD-rate.
# --------------------------------------------------------------------------


class TestRelativeRanking:
    def test_test_better_yields_negative_bd_rate(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        # Shift test up 3 dB: lower rate at any quality => negative BD-rate.
        test = _shift_psnr(ref, db=3.0)
        test = RDCurve(name="test", points=test.points)  # rename for the report
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        report = compute_bd_rate_report(test, ref, cfg)
        assert report.bd_rate_pct < 0.0
        assert report.bd_psnr_db is not None
        assert report.bd_psnr_db > 0.0

    def test_test_worse_yields_positive_bd_rate(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        test = _shift_psnr(ref, db=-3.0)
        test = RDCurve(name="test", points=test.points)
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        report = compute_bd_rate_report(test, ref, cfg)
        assert report.bd_rate_pct > 0.0
        assert report.bd_psnr_db is not None
        assert report.bd_psnr_db < 0.0


# --------------------------------------------------------------------------
# Gate logic
# --------------------------------------------------------------------------


class TestGateLogic:
    def test_gate_passed_when_test_well_below_threshold(self) -> None:
        # Wider PSNR range so a moderate shift still leaves enough overlap
        # for the BD-rate integration to be meaningful.
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.05, 26.0), (0.1, 30.0), (0.2, 34.0), (0.4, 38.0), (0.8, 42.0)],
        )
        test_pts = _shift_psnr(ref, db=2.0).points
        test = RDCurve(name="test", points=test_pts)
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        report = compute_bd_rate_report(test, ref, cfg)
        assert report.gate_status == "passed"
        assert report.gate_passed is True

    def test_gate_failed_when_no_improvement(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        report = compute_bd_rate_report(ref, ref, cfg)  # identical
        # 0 > -15 → fail
        assert report.gate_status == "failed"
        assert report.gate_passed is False

    def test_gate_skipped_when_curves_dont_overlap(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        # Test curve with PSNRs entirely above ref range.
        test = _build_curve(
            "test",
            rates_psnrs=[(0.05, 50.0), (0.1, 52.0), (0.2, 54.0), (0.4, 56.0)],
        )
        cfg = BDRateConfig(name="cfg", min_overlap_fraction=0.99)
        report = compute_bd_rate_report(test, ref, cfg)
        assert report.gate_status == "skipped"
        assert report.gate_passed is False

    def test_primary_lambda_rd_routes_to_neighborhood(self) -> None:
        # 8-point grid; gate must target the entry whose lambda matches.
        rates_psnrs = [(0.05 + 0.05 * i, 28.0 + i) for i in range(8)]
        lambdas = [0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045, 0.09, 0.18]
        ref = _build_curve("ref", rates_psnrs=rates_psnrs, lambdas=lambdas)
        # Test shifted by 2 dB across the board.
        test_pts = _shift_psnr(ref, db=2.0).points
        test = RDCurve(name="test", points=test_pts)
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=0.015)
        report = compute_bd_rate_report(test, ref, cfg)
        assert report.primary_lambda_rd == pytest.approx(0.015)
        assert report.primary_bd_rate_pct is not None
        assert report.primary_bd_rate_pct < 0.0

    def test_primary_lambda_rd_none_uses_full_curve(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        test = RDCurve(name="test", points=_shift_psnr(ref, db=4.0).points)
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        report = compute_bd_rate_report(test, ref, cfg)
        assert report.primary_lambda_rd is None
        assert report.primary_bd_rate_pct is None
        # Full-curve BD-rate should clear -15 % easily at +4 dB.
        assert report.gate_status == "passed"


# --------------------------------------------------------------------------
# Failures and report serialization.
# --------------------------------------------------------------------------


class TestReportFailures:
    def test_too_few_points_raises(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        sparse = RDCurve(
            name="sparse",
            points=[RDPoint(rate=0.5, distortion=0.001, psnr=33.0, lambda_rd=0.01)],
        )
        with pytest.raises(BDRateAssemblyError, match=">=2 points"):
            compute_bd_rate_report(sparse, ref)

    def test_ssim_metric_path(self) -> None:
        # Build curves with both PSNR and SSIM.
        ref = RDCurve(
            name="ref",
            points=[
                RDPoint(rate=0.1, distortion=0.01, psnr=28.0, ssim=0.90, lambda_rd=0.001),
                RDPoint(rate=0.2, distortion=0.005, psnr=30.0, ssim=0.93, lambda_rd=0.005),
                RDPoint(rate=0.4, distortion=0.001, psnr=33.0, ssim=0.96, lambda_rd=0.015),
                RDPoint(rate=0.8, distortion=0.0001, psnr=36.0, ssim=0.98, lambda_rd=0.03),
            ],
        )
        test = RDCurve(
            name="test",
            points=[
                RDPoint(rate=0.05, distortion=0.01, psnr=28.0, ssim=0.91, lambda_rd=0.001),
                RDPoint(rate=0.1, distortion=0.005, psnr=30.0, ssim=0.94, lambda_rd=0.005),
                RDPoint(rate=0.2, distortion=0.001, psnr=33.0, ssim=0.97, lambda_rd=0.015),
                RDPoint(rate=0.4, distortion=0.0001, psnr=36.0, ssim=0.99, lambda_rd=0.03),
            ],
        )
        cfg = BDRateConfig(name="cfg", metric="ssim", primary_lambda_rd=None)
        report = compute_bd_rate_report(test, ref, cfg)
        # SSIM path → bd_psnr_db is null
        assert report.bd_psnr_db is None
        assert report.metric == "ssim"


class TestReportSerialization:
    def test_round_trip_json(self) -> None:
        ref = _build_curve(
            "ref",
            rates_psnrs=[(0.1, 28.0), (0.2, 30.0), (0.4, 33.0), (0.8, 36.0)],
        )
        test = RDCurve(name="test", points=_shift_psnr(ref, db=2.0).points)
        cfg = BDRateConfig(name="cfg", primary_lambda_rd=None)
        report = compute_bd_rate_report(test, ref, cfg)

        as_json = report.model_dump_json()
        parsed = json.loads(as_json)
        round = BDRateReport.model_validate(parsed)
        assert round.test_curve_name == report.test_curve_name
        assert round.bd_rate_pct == pytest.approx(report.bd_rate_pct)
        assert round.gate_status == report.gate_status
        assert round.schema_version == BD_RATE_REPORT_SCHEMA_VERSION

    def test_unknown_field_dropped_on_load(self) -> None:
        # A future-emitted report has new fields → today's loader silently
        # drops them (extra="ignore").
        payload = {
            "name": "rep",
            "test_curve_name": "test",
            "reference_curve_name": "ref",
            "metric": "psnr",
            "bd_rate_pct": -10.0,
            "primary_bd_rate_gate_pct": -15.0,
            "gate_passed": False,
            "gate_status": "failed",
            "overlap_fraction": 1.0,
            "future_field_v2": "ignored",
        }
        report = BDRateReport.model_validate(payload)
        assert report.bd_rate_pct == pytest.approx(-10.0)


class TestBDRatePoint:
    def test_lambda_optional(self) -> None:
        p = BDRatePoint(name="p", bpp=0.5)
        assert p.lambda_rd is None
        assert p.psnr_db is None

    def test_negative_bpp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BDRatePoint(name="p", bpp=-0.1)


# --------------------------------------------------------------------------
# Property-based: monotone PSNR shift yields BD-rate of opposite sign.
# --------------------------------------------------------------------------


# Bound to the overlapping regime: a +shift larger than the reference
# PSNR span pushes the test curve entirely above the reference, at which
# point compute_bd_rate has no overlap to integrate over and (correctly)
# returns 0.0. The "positive shift -> negative BD-rate" invariant only
# holds while the curves still overlap, so we cap shift_db < ref span.
@given(shift_db=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=20, deadline=None)
def test_property_positive_shift_gives_negative_bd_rate(shift_db: float) -> None:
    # Ref spans 28..42 dB so a +4 dB shift still overlaps the top 10 dB.
    ref = _build_curve(
        "ref",
        rates_psnrs=[(0.05, 28.0), (0.1, 32.0), (0.2, 36.0), (0.4, 40.0), (0.8, 42.0)],
    )
    test = RDCurve(name="test", points=_shift_psnr(ref, db=shift_db).points)
    cfg = BDRateConfig(name="cfg", primary_lambda_rd=None, min_overlap_fraction=0.0)
    report = compute_bd_rate_report(test, ref, cfg)
    assert report.bd_rate_pct < 0.0
