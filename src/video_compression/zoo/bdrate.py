"""BD-rate report assembly for the Phase 2-E gate.

Wraps the existing :func:`compute_bd_rate` /
:func:`compute_bd_psnr` primitives in
:mod:`src.video_compression.metrics.rd_curves` with a
config-driven, JSON-serializable report. This module exists so that:

* the BD-rate gate's threshold is a Pydantic field with bounds, not a
  magic ``-15.0`` literal scattered across CLIs and CI scripts;
* the report itself is forward-compat (``extra="ignore"`` + an explicit
  schema version) so older binaries can still load reports written by
  newer ones;
* a single-call entry point (:func:`compute_bd_rate_report`) takes the
  test curve, the reference curve, and a config and returns a
  ``BDRateReport`` ready to drop into ``storage_root/bd_rate_report.json``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog
from pydantic import ConfigDict, Field, model_validator

from src.templates.config import BaseModuleConfig
from src.video_compression.metrics.rd_curves import (
    RDCurve,
    compute_bd_psnr,
    compute_bd_rate,
)

logger = structlog.get_logger(__name__)

#: Schema version for :class:`BDRateReport` JSON documents.
BD_RATE_REPORT_SCHEMA_VERSION: int = 1

#: Default primary-lambda BD-rate gate (negative percent = test better).
DEFAULT_PRIMARY_BD_RATE_GATE_PCT: float = -15.0

#: Default minimum bpp overlap fraction required between curves before BD-rate
#: is considered meaningful. Below this, the gate is reported as ``skipped``.
DEFAULT_MIN_OVERLAP_FRACTION: float = 0.5


QualityMetric = Literal["psnr", "ssim"]


class BDRateConfig(BaseModuleConfig):
    """How a BD-rate gate is computed and judged.

    Every numerical knob is a Pydantic field. The single source of truth
    for the −15 % gate is ``primary_bd_rate_gate_pct``; CLIs and CI
    scripts read it from here rather than hardcoding it.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    metric: QualityMetric = Field(
        default="psnr",
        description=(
            "Quality metric used for BD-rate. 'psnr' uses PSNR (dB); "
            "'ssim' uses SSIM (linear). Note: BD-rate vs. SSIM is less "
            "stable than vs. PSNR for codec comparison."
        ),
    )
    primary_lambda_rd: float | None = Field(
        default=0.015,
        gt=0.0,
        le=10.0,
        description=(
            "Lambda value of the primary operating point used for the "
            "gate decision. The matching test-curve point is identified "
            "by lambda_rd (not by index), so manifests with arbitrary "
            "ordering still gate correctly. Set to None to gate on the "
            "average across the full curve."
        ),
    )
    primary_bd_rate_gate_pct: float = Field(
        default=DEFAULT_PRIMARY_BD_RATE_GATE_PCT,
        ge=-100.0,
        le=100.0,
        description=(
            "Pass criterion at the primary operating point. BD-rate "
            "<= this value is a pass. Negative means test codec is "
            "competitive (lower bitrate at equal quality)."
        ),
    )
    min_overlap_fraction: float = Field(
        default=DEFAULT_MIN_OVERLAP_FRACTION,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of the reference curve's quality range "
            "that must overlap the test curve before BD-rate is "
            "considered meaningful. Below this, the gate is 'skipped'."
        ),
    )

    @model_validator(mode="after")
    def _validate_gate(self) -> BDRateConfig:
        # No further constraints today; placeholder for future cross-field
        # validation (e.g. requiring primary_lambda_rd when gate < -50).
        return self


class BDRatePoint(BaseModuleConfig):
    """A single (rate, quality) point captured in a report."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    lambda_rd: float | None = Field(
        default=None,
        description=(
            "Lambda value if the point came from a zoo entry; None for "
            "baseline curves whose points are not parameterized by lambda."
        ),
    )
    bpp: float = Field(..., ge=0.0, description="Bits per pixel.")
    psnr_db: float | None = Field(
        default=None,
        ge=0.0,
        description="PSNR in dB; null when the source curve only has SSIM.",
    )
    ms_ssim: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="MS-SSIM (linear); null when not measured.",
    )


class BDRateReport(BaseModuleConfig):
    """JSON-serializable BD-rate gate result.

    Persisted as ``bd_rate_report.json`` next to the zoo entries.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    schema_version: int = Field(
        default=BD_RATE_REPORT_SCHEMA_VERSION,
        ge=1,
        description="Schema version for migration on read.",
    )
    test_curve_name: str = Field(
        ...,
        min_length=1,
        description="Name of the test (AlphaGalerkin) curve.",
    )
    reference_curve_name: str = Field(
        ...,
        min_length=1,
        description="Name of the reference curve (typically H.265).",
    )
    metric: QualityMetric = Field(
        ...,
        description="Quality metric used for the BD-rate computation.",
    )
    bd_rate_pct: float = Field(
        ...,
        description=(
            "Average BD-rate across the overlap range, percent. "
            "Negative means test codec is more efficient."
        ),
    )
    bd_psnr_db: float | None = Field(
        default=None,
        description=(
            "Average PSNR delta in dB at equal bitrate (test - reference). "
            "Null when metric != 'psnr' or when the curves do not overlap."
        ),
    )
    primary_lambda_rd: float | None = Field(
        default=None,
        description="Lambda used for the gate decision; None for full-curve gate.",
    )
    primary_bd_rate_pct: float | None = Field(
        default=None,
        description=(
            "BD-rate evaluated at the primary lambda neighborhood. None "
            "when no entry matched primary_lambda_rd or when the gate is "
            "computed over the full curve."
        ),
    )
    primary_bd_rate_gate_pct: float = Field(
        ...,
        description="Configured gate threshold (copied from BDRateConfig).",
    )
    gate_passed: bool = Field(
        ...,
        description="True iff the primary BD-rate is <= the configured gate.",
    )
    gate_status: Literal["passed", "failed", "skipped"] = Field(
        ...,
        description=(
            "'passed' / 'failed' on a real comparison; 'skipped' when the "
            "curves don't overlap enough for a meaningful number."
        ),
    )
    overlap_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=("Fraction of the reference quality range that overlaps the test curve."),
    )
    test_points: list[BDRatePoint] = Field(
        default_factory=list,
        description="Test-curve points (in rate-ascending order).",
    )
    reference_points: list[BDRatePoint] = Field(
        default_factory=list,
        description="Reference-curve points (in rate-ascending order).",
    )


class BDRateAssemblyError(ValueError):
    """Raised when a BD-rate report cannot be assembled."""


def _filter_to_metric(curve: RDCurve, metric: QualityMetric) -> RDCurve:
    """Drop points missing the selected quality metric.

    ``RDCurve.rates`` includes every point but ``RDCurve.psnrs`` /
    ``RDCurve.ssims`` skip points whose corresponding field is ``None``.
    The downstream BD-rate code interpolates ``rates`` against
    ``qualities`` via ``np.interp`` and crashes when the two arrays
    have different lengths. We pre-filter here so the rate / quality
    arrays stay aligned by construction.
    """
    out = RDCurve(name=curve.name, points=[])
    for p in curve.points:
        value = p.psnr if metric == "psnr" else p.ssim
        if value is None:
            continue
        out.add_point(p)
    return out


def _curve_to_points(curve: RDCurve) -> list[BDRatePoint]:
    """Adapt :class:`RDCurve` points to the report-friendly schema.

    The inherited ``BaseModuleConfig.name`` field gets a unique
    ``<curve>_p<index>`` identifier so per-point records remain
    distinguishable in JSON output (and in any future per-point logging).
    """
    out: list[BDRatePoint] = []
    for i, p in enumerate(curve.points):
        out.append(
            BDRatePoint(
                name=f"{curve.name}_p{i}",
                lambda_rd=p.lambda_rd,
                bpp=float(p.rate),
                psnr_db=float(p.psnr) if p.psnr is not None else None,
                ms_ssim=float(p.ssim) if p.ssim is not None else None,
            )
        )
    return out


def _quality_overlap_fraction(
    test: RDCurve,
    reference: RDCurve,
    metric: QualityMetric,
) -> float:
    """Fraction of the reference quality range that overlaps the test.

    Returns 0.0 when either curve has no quality samples for the metric.
    """
    test_q = test.psnrs if metric == "psnr" else test.ssims
    ref_q = reference.psnrs if metric == "psnr" else reference.ssims
    if len(test_q) == 0 or len(ref_q) == 0:
        return 0.0
    ref_lo, ref_hi = float(np.min(ref_q)), float(np.max(ref_q))
    test_lo, test_hi = float(np.min(test_q)), float(np.max(test_q))
    overlap_lo = max(ref_lo, test_lo)
    overlap_hi = min(ref_hi, test_hi)
    ref_range = ref_hi - ref_lo
    if ref_range <= 0.0:
        return 0.0
    return max(0.0, (overlap_hi - overlap_lo) / ref_range)


def _bd_rate_at_primary(
    test: RDCurve,
    reference: RDCurve,
    *,
    primary_lambda_rd: float,
    metric: QualityMetric,
) -> float | None:
    """BD-rate restricted to a 3-point lambda neighborhood around the primary λ.

    Selection semantics: the test curve's points are sorted by
    ``lambda_rd`` (ascending), the entry whose lambda is **nearest** to
    ``primary_lambda_rd`` becomes the window center, and the ±1 lambda
    neighbors are added if they exist. The neighborhood is then handed
    back as a fresh :class:`RDCurve` (which re-sorts by rate, the
    convention BD-rate expects).

    Returns ``None`` when no test-curve point carries a ``lambda_rd``
    value, or when the lambda-sorted neighborhood collapses to fewer
    than 2 points.

    Note: this is a **nearest-lambda** match, not an exact one — the
    primary lambda from :class:`BDRateConfig` need not appear verbatim
    in the manifest. A future tolerance-bounded mode can be added as a
    Pydantic field if required; today's behavior is deterministic and
    documented.
    """
    candidates = [p for p in test.points if p.lambda_rd is not None]
    if not candidates:
        return None
    # Sort by lambda so the ±1 window is in lambda space, not the
    # rate-sorted order from RDCurve.add_point. Without this, a curve
    # where lambda_rd is non-monotone in rate could yield a window that
    # spans the wrong R-D operating points.
    lambda_sorted = sorted(candidates, key=lambda p: float(p.lambda_rd))  # type: ignore[arg-type]
    nearest_pos = min(
        range(len(lambda_sorted)),
        key=lambda i: abs(float(lambda_sorted[i].lambda_rd) - primary_lambda_rd),  # type: ignore[arg-type]
    )
    lo = max(0, nearest_pos - 1)
    hi = min(len(lambda_sorted), nearest_pos + 2)
    window_points = lambda_sorted[lo:hi]
    if len(window_points) < 2:
        return None
    # Insert via add_point so the resulting curve is rate-sorted, the
    # convention np.interp inside compute_bd_rate relies on. The
    # RDCurve constructor accepts a points= list verbatim and would
    # otherwise leave them in whatever order the lambda window had.
    window_curve = RDCurve(name=f"{test.name}_primary_window", points=[])
    for p in window_points:
        window_curve.add_point(p)
    return float(compute_bd_rate(reference, window_curve, metric=metric))


def compute_bd_rate_report(
    test: RDCurve,
    reference: RDCurve,
    config: BDRateConfig | None = None,
) -> BDRateReport:
    """Build a BD-rate gate report comparing ``test`` against ``reference``.

    Args:
        test: AlphaGalerkin (or other test) R-D curve.
        reference: Reference R-D curve (typically H.265).
        config: Gate config; ``None`` means defaults.

    Returns:
        :class:`BDRateReport` with the gate verdict pre-computed.

    Raises:
        BDRateAssemblyError: When either curve has fewer than 2 points
            with the configured quality metric.

    """
    cfg = config or BDRateConfig(name="bdrate_default")
    log = logger.bind(
        component="compute_bd_rate_report",
        test_curve=test.name,
        reference_curve=reference.name,
        metric=cfg.metric,
    )

    # Filter both curves to points that actually carry the selected
    # metric so downstream np.interp doesn't get mismatched-length
    # rates / qualities arrays. A curve mixing metric=present and
    # metric=missing is otherwise a silent crash inside compute_bd_rate.
    test_filtered = _filter_to_metric(test, cfg.metric)
    reference_filtered = _filter_to_metric(reference, cfg.metric)

    if len(test_filtered.points) < 2 or len(reference_filtered.points) < 2:
        raise BDRateAssemblyError(
            f"both curves need >=2 points with metric={cfg.metric!r}; got "
            f"test={len(test_filtered.points)} "
            f"reference={len(reference_filtered.points)}",
        )

    overlap = _quality_overlap_fraction(test_filtered, reference_filtered, cfg.metric)
    bd_rate_pct = float(compute_bd_rate(reference_filtered, test_filtered, metric=cfg.metric))
    bd_psnr_db: float | None = None
    if cfg.metric == "psnr":
        bd_psnr_db = float(compute_bd_psnr(reference_filtered, test_filtered))

    primary_bd_rate: float | None = None
    if cfg.primary_lambda_rd is not None:
        primary_bd_rate = _bd_rate_at_primary(
            test_filtered,
            reference_filtered,
            primary_lambda_rd=cfg.primary_lambda_rd,
            metric=cfg.metric,
        )

    if overlap < cfg.min_overlap_fraction:
        gate_status: Literal["passed", "failed", "skipped"] = "skipped"
        gate_passed = False
    else:
        # Pick the value the gate is judged on: per-primary if available,
        # otherwise the overall BD-rate.
        gate_value = primary_bd_rate if primary_bd_rate is not None else bd_rate_pct
        gate_passed = gate_value <= cfg.primary_bd_rate_gate_pct
        gate_status = "passed" if gate_passed else "failed"

    report = BDRateReport(
        name=f"bdrate_{test.name}_vs_{reference.name}",
        test_curve_name=test.name,
        reference_curve_name=reference.name,
        metric=cfg.metric,
        bd_rate_pct=bd_rate_pct,
        bd_psnr_db=bd_psnr_db,
        primary_lambda_rd=cfg.primary_lambda_rd,
        primary_bd_rate_pct=primary_bd_rate,
        primary_bd_rate_gate_pct=cfg.primary_bd_rate_gate_pct,
        gate_passed=gate_passed,
        gate_status=gate_status,
        overlap_fraction=overlap,
        test_points=_curve_to_points(test),
        reference_points=_curve_to_points(reference),
    )

    log.info(
        "bdrate.report.assembled",
        bd_rate_pct=bd_rate_pct,
        primary_bd_rate_pct=primary_bd_rate,
        gate_status=gate_status,
        overlap_fraction=overlap,
    )
    return report


__all__ = [
    "BD_RATE_REPORT_SCHEMA_VERSION",
    "DEFAULT_MIN_OVERLAP_FRACTION",
    "DEFAULT_PRIMARY_BD_RATE_GATE_PCT",
    "BDRateAssemblyError",
    "BDRateConfig",
    "BDRatePoint",
    "BDRateReport",
    "QualityMetric",
    "compute_bd_rate_report",
]
