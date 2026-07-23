"""Reference-codec quality runners (Phase 2-E + Phase 6).

Wraps ``ffmpeg`` (and friends) in a Pydantic-configured runner that
encodes a test sequence with a reference codec, decodes it back, and
measures (bpp, PSNR, MS-SSIM) for use by
:class:`~src.video_compression.zoo.h265_baseline.H265BaselineRegistry`.

Design constraints
------------------

* No hardcoded paths / codec names / presets — every knob is a Pydantic
  field with bounds.
* The runner must **not** import ffmpeg at module load — its presence is
  checked at call time so CPU CI without ffmpeg installed continues to
  pass (the runner returns a "skipped" :class:`BaselineRunResult`).
* Subprocess invocation is split into a single helper
  (:func:`_run_subprocess`) so tests can monkeypatch one function instead
  of intercepting ``subprocess.run`` everywhere.
* The output entry shape matches
  :class:`~src.video_compression.zoo.h265_baseline.H265BaselineEntry`
  so a sweep over CRF values builds a registry with a single
  ``register(...)`` call per cell.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
import torch
from pydantic import ConfigDict, Field, model_validator

from src.templates.config import BaseModuleConfig
from src.video_compression.metrics.quality import MSSSIM, compute_psnr
from src.video_compression.zoo.h265_baseline import (
    H265_BASELINE_ENTRY_SCHEMA_VERSION,
    H265BaselineEntry,
)

logger = structlog.get_logger(__name__)


#: Default codec preset; ``"medium"`` is ffmpeg's documented quality/speed
#: midpoint and is what the AOMediaCodec test conditions specify.
DEFAULT_PRESET: str = "medium"
#: Default ffmpeg binary name; resolved via :func:`shutil.which`.
DEFAULT_FFMPEG_BIN: str = "ffmpeg"
#: Default subprocess timeout for a single encode/decode call (seconds).
DEFAULT_SUBPROCESS_TIMEOUT_S: float = 600.0
#: Default colorspace string for raw I/O (matches Xiph CIF sequences).
DEFAULT_PIXEL_FORMAT: str = "yuv420p"


SupportedCodec = Literal["libx265", "libaom-av1", "libvpx-vp9"]


class FFmpegBaselineConfig(BaseModuleConfig):
    """Configuration for one reference-codec encode-decode-measure cycle."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    codec: SupportedCodec = Field(
        default="libx265",
        description="ffmpeg encoder name.",
    )
    crf: int | None = Field(
        default=28,
        ge=0,
        le=63,
        description=(
            "Constant Rate Factor. Required for the libx265/libvpx-vp9 "
            "rate-control modes; libaom-av1 may use its own ``-cq-level``."
        ),
    )
    preset: str = Field(
        default=DEFAULT_PRESET,
        min_length=1,
        description="Codec preset (medium / slow / placebo / etc.).",
    )
    pixel_format: str = Field(
        default=DEFAULT_PIXEL_FORMAT,
        min_length=1,
        description=(
            "Pixel format for raw I/O. Reference codecs are most "
            "comparable in their native chroma subsampling."
        ),
    )
    extra_encode_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Extra args forwarded verbatim to the encode invocation. "
            "Enables tuning (e.g. ``['-tune', 'psnr']``) without needing "
            "schema bumps."
        ),
    )
    extra_decode_flags: list[str] = Field(
        default_factory=list,
        description="Extra args forwarded verbatim to the decode invocation.",
    )
    ffmpeg_bin: str = Field(
        default=DEFAULT_FFMPEG_BIN,
        min_length=1,
        description=(
            "ffmpeg binary name or absolute path. Resolved via shutil.which "
            "at call time so unset binaries trigger a 'skipped' result."
        ),
    )
    subprocess_timeout_s: float = Field(
        default=DEFAULT_SUBPROCESS_TIMEOUT_S,
        gt=0.0,
        le=86400.0,
        description="Per-subprocess timeout in seconds.",
    )

    @model_validator(mode="after")
    def _validate_codec_args(self) -> FFmpegBaselineConfig:
        # CRF is required for libx265 / libvpx-vp9 in the standard rate-
        # control mode used here. libaom-av1 may run without CRF via -cq-level
        # in ``extra_encode_flags`` so we only enforce it for the others.
        if self.crf is None and self.codec in {"libx265", "libvpx-vp9"}:
            raise ValueError(
                f"codec={self.codec!r} requires a CRF value; got crf=None",
            )
        return self


@dataclass(frozen=True)
class BaselineRunResult:
    """Outcome of a single :class:`FFmpegBaselineRunner` invocation.

    ``status`` is "ok" on a successful measurement, "skipped" when ffmpeg
    is unavailable (or any required input is missing), and "failed" on a
    subprocess error.
    """

    status: Literal["ok", "skipped", "failed"]
    entry: H265BaselineEntry | None
    encoded_bytes: int | None
    encode_seconds: float | None
    decode_seconds: float | None
    reason: str = ""


def _run_subprocess(
    cmd: list[str],
    *,
    timeout_s: float,
) -> subprocess.CompletedProcess[bytes]:
    """Single chokepoint for subprocess.run so tests can monkeypatch it.

    Returns the completed process; callers inspect ``returncode`` and
    decide how to handle non-zero exits (we want failures surfaced as
    ``BaselineRunResult.status == "failed"``, not as exceptions).
    """
    return subprocess.run(  # noqa: S603 - intentional subprocess
        cmd,
        check=False,
        capture_output=True,
        timeout=timeout_s,
    )


def _load_y4m_to_tensor(
    path: Path,
    *,
    width: int,
    height: int,
    n_frames: int,
) -> torch.Tensor | None:
    """Load a Y4M / YUV file into a [F, C, H, W] tensor in [0, 1] float32.

    Returns ``None`` when ``av`` / ``imageio`` is unavailable; callers
    treat this as "skip the metric, return None for psnr/ms_ssim".

    The decode path uses PyAV when available because that's the only
    pure-Python option that handles all three of YUV/Y4M/MP4 cleanly. The
    fallback is a documented "skip" — the runner never raises on missing
    optional decoders, it just emits a structured warning so the caller
    can decide whether to treat it as a hard failure.
    """
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "baseline.decode.pyav_missing",
            path=str(path),
            note="install [video] extra (pyav) to enable on-the-fly metrics",
        )
        return None

    try:
        container = av.open(str(path))
    except av.AVError as exc:
        logger.warning(
            "baseline.decode.open_failed",
            path=str(path),
            err=str(exc),
        )
        return None

    frames: list[torch.Tensor] = []
    try:
        for frame in container.decode(video=0):
            if len(frames) >= n_frames:
                break
            arr = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            # Lazy resize when the source resolution differs from the
            # caller's spec. We use bilinear because it matches what every
            # other tool in the codec pipeline does.
            if tensor.shape[-2:] != (height, width):
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            frames.append(tensor)
    finally:
        container.close()

    if not frames:
        return None
    return torch.stack(frames, dim=0)


class FFmpegBaselineRunner:
    """Encode → decode → measure with an ffmpeg-backed reference codec.

    The runner is stateless across calls; each call builds a fresh
    temporary directory, runs ffmpeg twice (encode + decode), and tears
    everything down before returning. The caller is responsible for
    aggregating multiple runs into an
    :class:`~src.video_compression.zoo.h265_baseline.H265BaselineRegistry`.
    """

    def __init__(self, config: FFmpegBaselineConfig | None = None) -> None:
        self._config = config or FFmpegBaselineConfig(name="ffmpeg_baseline_default")
        self._log = logger.bind(
            component="FFmpegBaselineRunner",
            codec=self._config.codec,
            preset=self._config.preset,
            crf=self._config.crf,
        )

    @property
    def config(self) -> FFmpegBaselineConfig:
        return self._config

    def is_ffmpeg_available(self) -> bool:
        """Return True when the configured ffmpeg binary resolves on PATH."""
        return shutil.which(self._config.ffmpeg_bin) is not None

    def run(
        self,
        *,
        sequence_path: Path,
        sequence_id: str,
        width: int,
        height: int,
        fps: float,
        n_frames: int,
        original_tensor: torch.Tensor | None = None,
    ) -> BaselineRunResult:
        """Encode then decode then measure quality vs. the original.

        Args:
            sequence_path: Local path to the Y4M / YUV / MP4 source.
            sequence_id: Stable identifier for the source (used as
                cell_key prefix).
            width: Source width in pixels.
            height: Source height in pixels.
            fps: Source frame rate.
            n_frames: Number of frames to encode/measure.
            original_tensor: Optional pre-loaded original in [F, C, H, W]
                float32 [0, 1]. When None, the runner attempts to load
                it from ``sequence_path`` to compute PSNR/MS-SSIM.

        Returns:
            :class:`BaselineRunResult`. Status "skipped" indicates the
            run was deliberately not attempted (missing ffmpeg or
            sequence file). Status "failed" indicates an error during
            execution.

        """
        cfg = self._config
        log = self._log.bind(
            sequence_id=sequence_id,
            width=width,
            height=height,
        )

        if not self.is_ffmpeg_available():
            log.warning("baseline.run.skipped.ffmpeg_missing", bin=cfg.ffmpeg_bin)
            return BaselineRunResult(
                status="skipped",
                entry=None,
                encoded_bytes=None,
                encode_seconds=None,
                decode_seconds=None,
                reason=f"ffmpeg binary {cfg.ffmpeg_bin!r} not on PATH",
            )

        if not sequence_path.exists():
            log.warning("baseline.run.skipped.sequence_missing", path=str(sequence_path))
            return BaselineRunResult(
                status="skipped",
                entry=None,
                encoded_bytes=None,
                encode_seconds=None,
                decode_seconds=None,
                reason=f"sequence file {sequence_path!s} not found",
            )

        with tempfile.TemporaryDirectory(prefix="ffmpeg_baseline_") as tmpdir:
            tmp = Path(tmpdir)
            encoded_path = tmp / "encoded.mkv"
            decoded_path = tmp / "decoded.y4m"

            encode_cmd = self._build_encode_cmd(sequence_path, encoded_path)
            decode_cmd = self._build_decode_cmd(encoded_path, decoded_path)

            log.info("baseline.run.encode.start", cmd=encode_cmd)
            try:
                enc = _run_subprocess(encode_cmd, timeout_s=cfg.subprocess_timeout_s)
            except subprocess.TimeoutExpired:
                return BaselineRunResult(
                    status="failed",
                    entry=None,
                    encoded_bytes=None,
                    encode_seconds=None,
                    decode_seconds=None,
                    reason=f"encode timeout after {cfg.subprocess_timeout_s}s",
                )
            if enc.returncode != 0 or not encoded_path.exists():
                return BaselineRunResult(
                    status="failed",
                    entry=None,
                    encoded_bytes=None,
                    encode_seconds=None,
                    decode_seconds=None,
                    reason=(
                        f"encode failed: rc={enc.returncode} "
                        f"stderr={enc.stderr[-200:].decode(errors='replace')!r}"
                    ),
                )

            encoded_bytes = encoded_path.stat().st_size
            bpp = self._bytes_to_bpp(
                encoded_bytes,
                width=width,
                height=height,
                n_frames=n_frames,
            )

            log.info("baseline.run.decode.start", cmd=decode_cmd)
            try:
                dec = _run_subprocess(decode_cmd, timeout_s=cfg.subprocess_timeout_s)
            except subprocess.TimeoutExpired:
                return BaselineRunResult(
                    status="failed",
                    entry=None,
                    encoded_bytes=encoded_bytes,
                    encode_seconds=None,
                    decode_seconds=None,
                    reason=f"decode timeout after {cfg.subprocess_timeout_s}s",
                )
            if dec.returncode != 0 or not decoded_path.exists():
                return BaselineRunResult(
                    status="failed",
                    entry=None,
                    encoded_bytes=encoded_bytes,
                    encode_seconds=None,
                    decode_seconds=None,
                    reason=(
                        f"decode failed: rc={dec.returncode} "
                        f"stderr={dec.stderr[-200:].decode(errors='replace')!r}"
                    ),
                )

            psnr_db, ms_ssim = self._measure_quality(
                original_tensor=original_tensor,
                original_path=sequence_path,
                decoded_path=decoded_path,
                width=width,
                height=height,
                n_frames=n_frames,
            )

        cell_key = self._build_cell_key(
            sequence_id=sequence_id,
            width=width,
            height=height,
            fps=fps,
        )
        entry = H265BaselineEntry(
            name=cell_key,
            schema_version=H265_BASELINE_ENTRY_SCHEMA_VERSION,
            cell_key=cell_key,
            sequence_id=sequence_id,
            codec=cfg.codec,
            crf=cfg.crf,
            preset=cfg.preset,
            width=width,
            height=height,
            fps=fps,
            bpp=bpp,
            psnr_db=psnr_db,
            ms_ssim=ms_ssim,
        )
        log.info(
            "baseline.run.completed",
            bpp=bpp,
            psnr_db=psnr_db,
            ms_ssim=ms_ssim,
            encoded_bytes=encoded_bytes,
        )
        return BaselineRunResult(
            status="ok",
            entry=entry,
            encoded_bytes=encoded_bytes,
            encode_seconds=None,
            decode_seconds=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_encode_cmd(
        self,
        sequence_path: Path,
        encoded_path: Path,
    ) -> list[str]:
        cfg = self._config
        cmd = [
            cfg.ffmpeg_bin,
            "-y",
            "-i",
            str(sequence_path),
            "-c:v",
            cfg.codec,
            "-preset",
            cfg.preset,
        ]
        if cfg.crf is not None:
            cmd += ["-crf", str(cfg.crf)]
        cmd += ["-pix_fmt", cfg.pixel_format]
        cmd += list(cfg.extra_encode_flags)
        cmd.append(str(encoded_path))
        return cmd

    def _build_decode_cmd(
        self,
        encoded_path: Path,
        decoded_path: Path,
    ) -> list[str]:
        cfg = self._config
        cmd = [
            cfg.ffmpeg_bin,
            "-y",
            "-i",
            str(encoded_path),
            "-pix_fmt",
            cfg.pixel_format,
        ]
        cmd += list(cfg.extra_decode_flags)
        cmd.append(str(decoded_path))
        return cmd

    @staticmethod
    def _bytes_to_bpp(
        encoded_bytes: int,
        *,
        width: int,
        height: int,
        n_frames: int,
    ) -> float:
        if width <= 0 or height <= 0 or n_frames <= 0:
            raise ValueError(
                f"invalid sequence dimensions: width={width} height={height} n_frames={n_frames}",
            )
        total_pixels = float(width) * float(height) * float(n_frames)
        return float(encoded_bytes * 8) / total_pixels

    @staticmethod
    def _build_cell_key(
        *,
        sequence_id: str,
        width: int,
        height: int,
        fps: float,
    ) -> str:
        # Use the same composite-key convention as
        # BaselineRegistry.cell_key on the perf side.
        return f"{sequence_id}|{width}x{height}|{fps:g}"

    def _measure_quality(
        self,
        *,
        original_tensor: torch.Tensor | None,
        original_path: Path,
        decoded_path: Path,
        width: int,
        height: int,
        n_frames: int,
    ) -> tuple[float | None, float | None]:
        """Compute (PSNR_dB, MS-SSIM) on the decoded vs. original.

        Returns ``(None, None)`` when neither source can be loaded as a
        tensor (typical when running on a CI host without PyAV).
        """
        original = (
            original_tensor
            if original_tensor is not None
            else _load_y4m_to_tensor(
                original_path,
                width=width,
                height=height,
                n_frames=n_frames,
            )
        )
        if original is None:
            return None, None

        decoded = _load_y4m_to_tensor(
            decoded_path,
            width=width,
            height=height,
            n_frames=n_frames,
        )
        if decoded is None:
            return None, None

        # Trim to the shorter of the two sequences so PSNR/MS-SSIM are
        # always defined. The mismatched-length case is normal when the
        # encoder drops a final partial frame.
        n = min(original.shape[0], decoded.shape[0])
        if n < 1:
            return None, None
        original = original[:n]
        decoded = decoded[:n]

        psnr = float(compute_psnr(decoded, original).item())
        msssim_module = MSSSIM()
        msssim = float(msssim_module(decoded, original).item())
        return psnr, msssim


__all__ = [
    "DEFAULT_FFMPEG_BIN",
    "DEFAULT_PIXEL_FORMAT",
    "DEFAULT_PRESET",
    "DEFAULT_SUBPROCESS_TIMEOUT_S",
    "BaselineRunResult",
    "FFmpegBaselineConfig",
    "FFmpegBaselineRunner",
    "SupportedCodec",
]
