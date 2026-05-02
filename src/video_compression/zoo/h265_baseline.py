"""Persistent H.265 (or other reference codec) quality baselines.

Mirrors the perf-side :class:`BaselineRegistry` pattern: a versioned JSON
document carrying one entry per (sequence_id, resolution, fps) cell, with
unversioned-to-v1 migration on read and ``extra="ignore"`` forward-compat
on the schema. The registry is then converted to an
:class:`~src.video_compression.metrics.rd_curves.RDCurve` for BD-rate
computation in :mod:`src.video_compression.zoo.bdrate`.

This is intentionally separate from
:class:`src.video_compression.perf.baseline.BaselineRegistry`: that one
holds **performance** baselines (throughput, latency, VRAM); this one
holds **quality** baselines (bpp, PSNR, MS-SSIM). The two registries
share no fields, so coupling them would be a footgun.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import ConfigDict, Field

from src.templates.config import BaseModuleConfig
from src.video_compression.metrics.psnr_conversions import psnr_db_to_mse_surrogate
from src.video_compression.metrics.rd_curves import RDCurve, RDPoint

logger = structlog.get_logger(__name__)

#: Schema version for :class:`H265BaselineDocument`.
H265_BASELINE_SCHEMA_VERSION: int = 1
#: Schema version for :class:`H265BaselineEntry`.
H265_BASELINE_ENTRY_SCHEMA_VERSION: int = 1


class H265BaselineEntry(BaseModuleConfig):
    """A single (sequence × codec_setting) measurement.

    ``cell_key`` is the stable composite identifier
    ``"<sequence_id>|<resolution>|<fps>|<codec>|<crf|qp>"`` so a registry
    can host multiple baselines side-by-side without collisions.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    schema_version: int = Field(
        default=H265_BASELINE_ENTRY_SCHEMA_VERSION,
        ge=1,
        description="Entry schema version for migration.",
    )
    cell_key: str = Field(
        ...,
        min_length=1,
        description=("Stable composite key, e.g. 'akiyo|cif|30|libx265|crf28'."),
    )
    sequence_id: str = Field(..., min_length=1)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    fps: float = Field(..., gt=0.0, le=1000.0)
    codec: str = Field(
        ...,
        min_length=1,
        description="Codec name as understood by ffmpeg, e.g. 'libx265'.",
    )
    crf: int | None = Field(
        default=None,
        ge=0,
        le=63,
        description="CRF value when applicable; null for fixed-bitrate runs.",
    )
    preset: str = Field(
        default="medium",
        min_length=1,
        description="Codec preset (medium, slow, etc.).",
    )

    # Recorded measurements (canonical units)
    bpp: float = Field(..., ge=0.0, description="Bits per pixel.")
    psnr_db: float | None = Field(
        default=None,
        ge=0.0,
        description="Average PSNR in dB; null when not measured.",
    )
    ms_ssim: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Average MS-SSIM (linear); null when not measured.",
    )
    vmaf: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Average VMAF score; null when not measured.",
    )


class H265BaselineDocument(BaseModuleConfig):
    """Versioned JSON file holding a collection of baseline entries."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    schema_version: int = Field(
        default=H265_BASELINE_SCHEMA_VERSION,
        ge=1,
    )
    description: str = Field(
        default="",
        description="Free-form notes (where/how the baseline was recorded).",
    )
    hardware_tag: str = Field(
        default="",
        description=(
            "Optional tag describing the recording hardware "
            "(ffmpeg version, machine, etc.). Diagnostic only."
        ),
    )
    entries: list[H265BaselineEntry] = Field(
        default_factory=list,
        description="Per-cell baseline measurements.",
    )


class H265BaselineMigrationError(ValueError):
    """Raised when a baseline document cannot be migrated."""


def _migrate_h265_baseline_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate a raw baseline JSON dict to the current schema.

    Migration table:

    +----------------+----------------+--------------------------------+
    | from           | to             | change                         |
    +================+================+================================+
    | (unversioned)  | 1              | add ``schema_version`` field   |
    +----------------+----------------+--------------------------------+
    """
    raw = dict(raw)
    schema_version = raw.get("schema_version")
    if schema_version is None:
        logger.info(
            "h265_baseline.migration.unversioned_to_v1",
            keys=sorted(raw.keys()),
        )
        # Single source of truth for the schema version target — avoids
        # drift between the migration target and the binary's notion
        # of "current".
        raw["schema_version"] = H265_BASELINE_SCHEMA_VERSION
        schema_version = H265_BASELINE_SCHEMA_VERSION
    if not isinstance(schema_version, int):
        raise H265BaselineMigrationError(
            f"schema_version must be int; got {type(schema_version).__name__}",
        )
    if schema_version > H265_BASELINE_SCHEMA_VERSION:
        raise H265BaselineMigrationError(
            f"baseline schema_version={schema_version} is newer than this "
            f"binary ({H265_BASELINE_SCHEMA_VERSION}); upgrade the package "
            f"or pin a compatible baseline.",
        )
    return raw


class H265BaselineRegistry:
    """In-memory view of an H.265 (or other reference codec) baseline.

    Loads JSON documents written by an offline ffmpeg sweep, and converts
    them to :class:`RDCurve` for BD-rate computation. Saving rewrites
    the document under the current schema version.
    """

    def __init__(self, document: H265BaselineDocument) -> None:
        self._document = document
        self._by_key: dict[str, H265BaselineEntry] = {e.cell_key: e for e in document.entries}

    @classmethod
    def load(cls, path: str | Path) -> H265BaselineRegistry:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"h265 baseline file not found: {p}")
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"h265 baseline {p} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise H265BaselineMigrationError(
                f"h265 baseline root must be a JSON object; got {type(raw).__name__}",
            )
        migrated = _migrate_h265_baseline_document(raw)
        # ``name`` is required by BaseModuleConfig; promote a stable name
        # from the file stem when older documents lack one.
        migrated.setdefault("name", p.stem)
        document = H265BaselineDocument.model_validate(migrated)
        logger.info(
            "h265_baseline.loaded",
            path=str(p),
            schema_version=document.schema_version,
            n_entries=len(document.entries),
        )
        return cls(document)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self._document.model_dump(mode="json")
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info(
            "h265_baseline.saved",
            path=str(p),
            n_entries=len(self._document.entries),
        )
        return p

    @property
    def document(self) -> H265BaselineDocument:
        return self._document

    @property
    def entries(self) -> list[H265BaselineEntry]:
        return list(self._document.entries)

    def get(self, cell_key: str) -> H265BaselineEntry | None:
        return self._by_key.get(cell_key)

    def filter(
        self,
        *,
        sequence_id: str | None = None,
        codec: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> list[H265BaselineEntry]:
        """Return entries matching every non-None predicate."""
        out: list[H265BaselineEntry] = []
        for e in self._document.entries:
            if sequence_id is not None and e.sequence_id != sequence_id:
                continue
            if codec is not None and e.codec != codec:
                continue
            if width is not None and e.width != width:
                continue
            if height is not None and e.height != height:
                continue
            out.append(e)
        return out

    def to_curve(
        self,
        *,
        sequence_id: str,
        codec: str = "libx265",
        width: int | None = None,
        height: int | None = None,
        name: str | None = None,
    ) -> RDCurve:
        """Project a (sequence × codec [× resolution]) slice to an RDCurve.

        Args:
            sequence_id: Required sequence selector.
            codec: Codec selector (default: ``"libx265"``).
            width: Optional width selector for multi-resolution baselines.
            height: Optional height selector.
            name: Optional curve name (default: ``"<codec>_<sequence_id>"``).

        Raises:
            ValueError: When fewer than 2 matching entries have a
                non-null ``psnr_db`` (BD-rate needs two distinct points).

        """
        matches = self.filter(
            sequence_id=sequence_id,
            codec=codec,
            width=width,
            height=height,
        )
        # Require psnr; SSIM-only baselines aren't usable through the
        # default BD-rate path.
        with_psnr = [e for e in matches if e.psnr_db is not None]
        if len(with_psnr) < 2:
            raise ValueError(
                f"need >=2 baseline entries with PSNR for "
                f"sequence_id={sequence_id!r} codec={codec!r} "
                f"(filtered to width={width!r} height={height!r}); "
                f"got {len(with_psnr)}",
            )
        curve_name = name or f"{codec}_{sequence_id}"
        curve = RDCurve(name=curve_name, points=[])
        for e in with_psnr:
            psnr = float(e.psnr_db)  # type: ignore[arg-type]
            ssim = float(e.ms_ssim) if e.ms_ssim is not None else None
            curve.add_point(
                RDPoint(
                    rate=float(e.bpp),
                    distortion=psnr_db_to_mse_surrogate(psnr),
                    psnr=psnr,
                    ssim=ssim,
                )
            )
        return curve


__all__ = [
    "H265_BASELINE_ENTRY_SCHEMA_VERSION",
    "H265_BASELINE_SCHEMA_VERSION",
    "H265BaselineDocument",
    "H265BaselineEntry",
    "H265BaselineMigrationError",
    "H265BaselineRegistry",
]
