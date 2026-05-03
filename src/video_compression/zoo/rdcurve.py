"""R-D curve assembly from a finished zoo sweep.

Phase 2-E (R-D curve + BD-rate gate): consume the per-entry
``metrics.json`` files written by :class:`VideoCodecZoo` during a Phase 2-D
sweep and produce a single :class:`~src.video_compression.metrics.rd_curves.RDCurve`.
The curve is **rate-sorted** (ascending bpp), matching the
:class:`RDCurve.add_point` convention used by the BD-rate code; per-entry
``lambda_rd`` is preserved on each :class:`RDPoint` so callers that need
lambda-keyed access (e.g. the primary-lambda gate in
:mod:`src.video_compression.zoo.bdrate`) can still find specific points.
Downstream :mod:`src.video_compression.zoo.bdrate` consumes the curve to
compute the BD-rate gate vs. an H.265 baseline.

Design notes
------------

* No new R-D curve type is introduced — the existing
  :class:`RDCurve`/:class:`RDPoint` already cover the field set
  (rate / distortion / psnr / ssim / lambda_rd).
* All measurement-affecting knobs (which metric keys to read, fit method,
  whether to enforce monotonicity) flow through
  :class:`RDCurveFitConfig`. There are no hardcoded metric names in the
  module body.
* The reader is permissive on extra metric keys (forward-compat with new
  metrics emitted by future trainers) but strict on missing required
  keys (so a corrupted ``metrics.json`` fails loud).
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
import structlog
from pydantic import ConfigDict, Field, model_validator

from src.templates.config import BaseModuleConfig
from src.video_compression.metrics.psnr_conversions import psnr_db_to_mse_surrogate
from src.video_compression.metrics.rd_curves import RDCurve, RDPoint
from src.video_compression.zoo.config import ModelZooManifestConfig
from src.video_compression.zoo.storage import VideoCodecZoo

logger = structlog.get_logger(__name__)

#: Default metric key emitted by :class:`ZooTrainer` for the rate axis.
DEFAULT_RATE_METRIC_KEY: str = "bpp"
#: Default metric key emitted by :class:`ZooTrainer` for the quality axis.
DEFAULT_QUALITY_METRIC_KEY: str = "psnr_db"
#: Default metric key emitted by :class:`ZooTrainer` for MS-SSIM (optional).
DEFAULT_MSSSIM_METRIC_KEY: str = "ms_ssim"


FitMethod = Literal["linear_log", "monotone_spline"]
QualityMetric = Literal["psnr", "ms_ssim"]


class RDCurveFitConfig(BaseModuleConfig):
    """How an R-D curve is assembled from per-entry metrics.

    Every parameter that influences the resulting :class:`RDCurve` is
    surfaced here so the caller can audit / override the assembly logic
    via YAML / JSON without code edits.
    """

    # Forward-compat for future fields (e.g. quality_metric=vmaf).
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    rate_metric_key: str = Field(
        default=DEFAULT_RATE_METRIC_KEY,
        min_length=1,
        description=(
            "Key in metrics.json that holds the per-entry rate. Standard "
            "ZooTrainer output uses 'bpp' (bits per pixel)."
        ),
    )
    quality_metric_key: str = Field(
        default=DEFAULT_QUALITY_METRIC_KEY,
        min_length=1,
        description=(
            "Key in metrics.json that holds the per-entry quality. "
            "Standard ZooTrainer output uses 'psnr_db'."
        ),
    )
    msssim_metric_key: str = Field(
        default=DEFAULT_MSSSIM_METRIC_KEY,
        min_length=1,
        description=(
            "Key in metrics.json that holds the per-entry MS-SSIM "
            "(optional; copied to RDPoint.ssim if present)."
        ),
    )
    fit_method: FitMethod = Field(
        default="linear_log",
        description=(
            "Reserved for downstream interpolation policy. The "
            "assembler itself does not fit anything — it returns the "
            "raw rate-sorted points and lets BD-rate's own log-domain "
            "trapezoidal/cubic interpolator do the integration. "
            "'linear_log' is the only value the BD-rate path currently "
            "honors; 'monotone_spline' is accepted, logged, and "
            "treated identically (placeholder for a future scipy-PCHIP "
            "interpolator). Setting this field does not change the "
            "returned RDCurve."
        ),
    )
    enforce_monotone: bool = Field(
        default=True,
        description=(
            "If True, reject any curve where higher rate does not yield "
            "higher quality. This is the canonical sanity check on a "
            "trained R-D sweep — non-monotonicity points to overfitting "
            "or training-loop bugs."
        ),
    )
    require_lambda_unique: bool = Field(
        default=True,
        description=(
            "If True, reject manifests with duplicate lambda_rd values "
            "(would collapse to a single curve point)."
        ),
    )
    min_points: int = Field(
        default=2,
        ge=2,
        le=64,
        description=(
            "Minimum number of (rate, quality) points required to assemble "
            "a curve. BD-rate cubic fitting requires >=4; the default of "
            "2 allows linear-log diagnostics on a partial sweep."
        ),
    )

    @model_validator(mode="after")
    def _disjoint_keys(self) -> RDCurveFitConfig:
        keys = {
            self.rate_metric_key,
            self.quality_metric_key,
            self.msssim_metric_key,
        }
        if len(keys) < 3:
            raise ValueError(
                "rate_metric_key / quality_metric_key / msssim_metric_key "
                "must all be distinct; got "
                f"{self.rate_metric_key!r}, {self.quality_metric_key!r}, "
                f"{self.msssim_metric_key!r}",
            )
        return self


class RDCurveAssemblyError(ValueError):
    """Raised when an R-D curve cannot be built from a sweep's metrics."""


def compute_rd_curve(
    zoo: VideoCodecZoo,
    manifest: ModelZooManifestConfig,
    fit_config: RDCurveFitConfig | None = None,
    *,
    name: str | None = None,
    only_entry_ids: list[str] | None = None,
) -> RDCurve:
    """Assemble an R-D curve from a finished zoo sweep.

    Args:
        zoo: Filesystem-backed registry of trained entries.
        manifest: Manifest that describes the sweep (drives entry order
            and lambda values).
        fit_config: Curve-assembly knobs. ``None`` means defaults.
        name: Optional curve name (default: manifest.name).
        only_entry_ids: Optional allow-list. ``None`` means every entry
            in the manifest. Useful when the gate is computed on a
            subset of the grid.

    Returns:
        :class:`RDCurve` with one :class:`RDPoint` per entry, sorted by
        rate (matches the ``RDCurve`` convention).

    Raises:
        RDCurveAssemblyError: When too few entries have metrics on disk,
            when a required metric key is missing, when lambda values
            are not unique (and the config requires uniqueness), or when
            the resulting curve is not monotone (and the config requires
            monotonicity).

    """
    cfg = fit_config or RDCurveFitConfig(name="rdcurve_fit_default")
    log = logger.bind(
        component="compute_rd_curve",
        manifest=manifest.name,
        n_entries=len(manifest.entries),
        rate_metric_key=cfg.rate_metric_key,
        quality_metric_key=cfg.quality_metric_key,
    )

    allow = set(only_entry_ids) if only_entry_ids else None
    points: list[RDPoint] = []
    seen_lambdas: list[float] = []
    skipped: list[str] = []

    for entry in manifest.entries:
        if allow is not None and entry.entry_id not in allow:
            continue
        if not zoo.has_entry(entry.entry_id):
            skipped.append(entry.entry_id)
            log.warning(
                "rdcurve.entry.skipped.no_checkpoint",
                entry_id=entry.entry_id,
            )
            continue

        try:
            metrics = zoo.load_metrics(entry.entry_id)
        except FileNotFoundError:
            skipped.append(entry.entry_id)
            log.warning(
                "rdcurve.entry.skipped.no_metrics",
                entry_id=entry.entry_id,
            )
            continue

        if cfg.rate_metric_key not in metrics:
            raise RDCurveAssemblyError(
                f"entry {entry.entry_id!r} metrics.json is missing required "
                f"key {cfg.rate_metric_key!r}; available keys: "
                f"{sorted(metrics)}",
            )
        if cfg.quality_metric_key not in metrics:
            raise RDCurveAssemblyError(
                f"entry {entry.entry_id!r} metrics.json is missing required "
                f"key {cfg.quality_metric_key!r}; available keys: "
                f"{sorted(metrics)}",
            )

        rate = float(metrics[cfg.rate_metric_key])
        psnr = float(metrics[cfg.quality_metric_key])
        ssim = float(metrics[cfg.msssim_metric_key]) if cfg.msssim_metric_key in metrics else None

        # ``RDPoint.distortion`` is "MSE-or-similar"; we use the closed-form
        # PSNR -> MSE conversion as a stable surrogate so downstream consumers
        # that look at distortion (rather than PSNR) stay sane. The PSNR
        # field is what the BD-rate computation actually uses.
        surrogate_distortion = psnr_db_to_mse_surrogate(psnr)

        points.append(
            RDPoint(
                rate=rate,
                distortion=surrogate_distortion,
                psnr=psnr,
                ssim=ssim,
                lambda_rd=float(entry.lambda_rd),
            )
        )
        seen_lambdas.append(float(entry.lambda_rd))

    if len(points) < cfg.min_points:
        raise RDCurveAssemblyError(
            f"need at least {cfg.min_points} entries with metrics on disk; "
            f"found {len(points)} (skipped: {skipped})",
        )

    if cfg.require_lambda_unique:
        # Counter is O(n) vs. O(n^2) for the prior list.count() in a
        # comprehension. Per gemini-code-assist on PR #82.
        counts = Counter(seen_lambdas)
        duplicates = sorted(lam for lam, n in counts.items() if n > 1)
        if duplicates:
            raise RDCurveAssemblyError(
                f"duplicate lambda_rd values across selected entries: {duplicates!r}",
            )

    curve = RDCurve(name=name or manifest.name, points=[])
    for p in points:
        curve.add_point(p)  # add_point keeps points sorted by rate

    if cfg.enforce_monotone and not curve.is_monotonic():
        psnrs = curve.psnrs.tolist()
        rates = curve.rates.tolist()
        raise RDCurveAssemblyError(
            f"R-D curve {curve.name!r} is not monotonically increasing in "
            f"PSNR with rate; rates={rates}, psnrs={psnrs}. Either fix the "
            "sweep or set RDCurveFitConfig.enforce_monotone=False.",
        )

    if cfg.fit_method == "monotone_spline":
        # The fit method does not change the stored points (the curve is
        # always sampled / interpolated lazily by the BD-rate code). We
        # log the request so users running with non-default fits can
        # confirm their config took effect.
        log.info("rdcurve.fit.requested", method=cfg.fit_method)

    log.info(
        "rdcurve.assembled",
        n_points=len(curve.points),
        rate_min=float(np.min(curve.rates)),
        rate_max=float(np.max(curve.rates)),
        psnr_min=float(np.min(curve.psnrs)),
        psnr_max=float(np.max(curve.psnrs)),
        skipped=skipped,
    )
    return curve


__all__ = [
    "DEFAULT_MSSSIM_METRIC_KEY",
    "DEFAULT_QUALITY_METRIC_KEY",
    "DEFAULT_RATE_METRIC_KEY",
    "FitMethod",
    "QualityMetric",
    "RDCurveAssemblyError",
    "RDCurveFitConfig",
    "compute_rd_curve",
]
